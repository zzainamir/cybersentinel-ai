"""
Sends high-threat detections from FastAPI to Azure Event Hubs.

Only HIGH and CRITICAL detections are sent — this prevents flooding
Event Hubs with low-confidence alerts and keeps costs near zero.

If EVENTHUB_CONNECTION_STRING is not set, sending is silently
disabled so the API still works during local development.

Sending runs in a background thread so it never blocks the API response.
"""

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)


class EventHubSender:

    ALERT_LEVELS = {'HIGH', 'CRITICAL'}

    def __init__(self):
        self.connection_str = os.environ.get('EVENTHUB_CONNECTION_STRING', '')
        self.hub_name       = os.environ.get('EVENTHUB_NAME', 'threat-detections')
        self.enabled        = bool(self.connection_str)

        if self.enabled:
            logger.info(f"Event Hubs sender ready — hub: {self.hub_name}")
        else:
            logger.warning(
                "EVENTHUB_CONNECTION_STRING not set. "
                "High-threat detections will NOT be sent to Azure Sentinel. "
                "Set the environment variable to enable streaming."
            )

    def send_if_threat(self, detection: dict) -> bool:
        """
        Send detection to Event Hubs only if threat level is HIGH or CRITICAL.
        Returns True if dispatch started, False if skipped.
        """
        if not self.enabled:
            return False

        level = detection.get('threat_level', 'LOW')
        if level not in self.ALERT_LEVELS:
            return False

        return self._send(detection)

    def send(self, detection: dict) -> bool:
        """Send any detection to Event Hubs regardless of threat level."""
        if not self.enabled:
            return False
        return self._send(detection)

    def _send(self, detection: dict) -> bool:
        """
        Dispatch send to a background thread so it never blocks the API.
        The daemon=True flag means the thread is killed automatically
        when the main process exits — no cleanup needed.
        """
        thread = threading.Thread(
            target=self._send_background,
            args=(detection,),
            daemon=True,
        )
        thread.start()
        return True

    def _send_background(self, detection: dict):
        """
        Actual Event Hubs send logic. Runs in background thread.
        Failures are logged but never raised — the API is unaffected.
        """
        try:
            from azure.eventhub import EventHubProducerClient, EventData

            producer = EventHubProducerClient.from_connection_string(
                conn_str=self.connection_str,
                eventhub_name=self.hub_name,
            )
            with producer:
                batch = producer.create_batch()
                batch.add(EventData(json.dumps(detection)))
                producer.send_batch(batch)

            logger.info(
                f"Sent to Event Hubs: "
                f"score={detection.get('threat_score')}  "
                f"level={detection.get('threat_level')}"
            )

        except Exception as e:
            logger.error(f"Event Hubs send failed: {e}")