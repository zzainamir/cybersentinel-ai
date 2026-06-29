"""
CyberSentinel-AI — Azure Function

Trigger: Azure Event Hubs (fires on every new threat detection)
Action:  Writes detection to Microsoft Sentinel via Log Analytics API

Data flow:
  FastAPI (high-threat alert) → Event Hubs → THIS FUNCTION → Sentinel
"""

import azure.functions as func
import logging
import json
import os
import datetime
import hashlib
import hmac
import base64
import requests

app = func.FunctionApp()

logger = logging.getLogger(__name__)


@app.event_hub_message_trigger(
    arg_name="azeventhub",
    event_hub_name="threat-detections",
    connection="EVENTHUB_CONNECTION",
)
def threat_processor(azeventhub: func.EventHubEvent):
    """
    Fires every time FastAPI sends a high-threat detection to Event Hubs.
    Enriches the detection and writes it to Microsoft Sentinel.
    """
    try:
        raw_body  = azeventhub.get_body().decode('utf-8')
        detection = json.loads(raw_body)

        # Enrich with processing metadata
        detection['processed_at']     = datetime.datetime.utcnow().isoformat() + 'Z'
        detection['source_system']    = 'CyberSentinel-AI'
        detection['function_version'] = '1.0.0'

        # Write to Sentinel
        status = _post_to_log_analytics(detection)

        if status == 200:
            logger.info(
                f"Detection written to Sentinel  "
                f"threat_score={detection.get('threat_score', 'N/A')}  "
                f"threat_level={detection.get('threat_level', 'N/A')}"
            )
        else:
            logger.error(f"Failed to write to Sentinel — HTTP {status}")

        # Log an extra warning for high-severity events
        level = detection.get('threat_level', '')
        if level in ('HIGH', 'CRITICAL'):
            logger.warning(
                f"HIGH-SEVERITY DETECTION: "
                f"score={detection.get('threat_score')}  "
                f"level={level}"
            )

    except json.JSONDecodeError as e:
        logger.error(f"Could not parse Event Hub message as JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing detection: {e}", exc_info=True)


# ── Log Analytics HTTP Data Collector API ─────────────────────────────────────

def _build_signature(workspace_id: str, workspace_key: str,
                     date: str, content_length: int,
                     method: str, content_type: str, resource: str) -> str:
    """
    Build the SharedKey HMAC-SHA256 signature required by the
    Log Analytics HTTP Data Collector API.
    """
    x_headers      = f'x-ms-date:{date}'
    string_to_hash = f'{method}\n{content_length}\n{content_type}\n{x_headers}\n{resource}'
    bytes_to_hash  = string_to_hash.encode('utf-8')
    decoded_key    = base64.b64decode(workspace_key)

    mac          = hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256)
    encoded_hash = base64.b64encode(mac.digest()).decode('utf-8')
    return f'SharedKey {workspace_id}:{encoded_hash}'


def _post_to_log_analytics(data: dict, log_type: str = 'CyberSentinelDetections') -> int:
    """
    POST a detection record to the Log Analytics HTTP Data Collector API.
    Sentinel reads from this workspace and the table appears as
    CyberSentinelDetections_CL (CL = Custom Log).
    """
    workspace_id  = os.environ['WORKSPACE_ID']
    workspace_key = os.environ['WORKSPACE_KEY']

    body           = json.dumps([data])
    method         = 'POST'
    content_type   = 'application/json'
    resource       = '/api/logs'
    rfc1123date    = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    content_length = len(body)

    signature = _build_signature(
        workspace_id, workspace_key,
        rfc1123date, content_length,
        method, content_type, resource
    )

    uri = (f'https://{workspace_id}.ods.opinsights.azure.com'
           f'{resource}?api-version=2016-04-01')

    headers = {
        'content-type':  content_type,
        'Authorization': signature,
        'Log-Type':      log_type,
        'x-ms-date':     rfc1123date,
    }

    response = requests.post(uri, data=body, headers=headers, timeout=10)
    return response.status_code