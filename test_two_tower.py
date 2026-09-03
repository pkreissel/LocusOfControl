import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()
ge_api_key = os.environ.get('GREEN_EARTH_API_KEY')
req = urllib.request.Request(
    'https://api.greenearth.social/candidates/generate',
    data=json.dumps({
        "generators": [
            { "name": 'two_tower', "weight": 1.0 }
        ],
        "user_did": "did:plc:z72i7hdynmk6r22z27h6tvur",
        "num_candidates": 1
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json', 'X-API-Key': ge_api_key}
)
try:
    with urllib.request.urlopen(req) as response:
        cand_data = json.loads(response.read().decode('utf-8'))
        print(json.dumps(cand_data['candidates'][0] if cand_data['candidates'] else None, indent=2))
except Exception as e:
    print(f"Failed: {e}")
