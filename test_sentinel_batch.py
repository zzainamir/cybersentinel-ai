import requests
import numpy as np

r = requests.get('http://localhost:8000/health')
n = r.json()['n_features']
print(f'Sending batch of 100 flows ({n} features each)...')

normal    = np.random.randn(70, n).tolist()
anomalous = (np.random.randn(30, n) * 8 + 5).tolist()
all_flows = normal + anomalous

r = requests.post('http://localhost:8000/predict/batch',
                  json={'flows': all_flows})

print(f'Status code: {r.status_code}')
print(f'Response: {r.text[:500]}')

if r.status_code != 200:
    print('Request failed — check server logs in the other prompt')
else:
    result = r.json()
    print(f"\nBatch results:")
    print(f"  Total flows:   {result['flow_count']}")
    print(f"  Attacks found: {result['attack_count']}")
    print(f"  Attack rate:   {result['attack_rate']}%")
    for i, pred in enumerate(result['predictions']):
        level = pred['threat_level']
        score = pred['threat_score']
        tag   = ' <- SENT TO SENTINEL' if level in ('HIGH', 'CRITICAL') else ''
        print(f"  [{i+1:>3}]  score={score:.4f}  level={level}{tag}")