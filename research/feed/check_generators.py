import json
import urllib.request
import os
from dotenv import load_dotenv

def main():
    load_dotenv('bunny_feed/.env')
    load_dotenv('.env')
    ge_api_key = os.environ.get('GREEN_EARTH_API_KEY')
    user_did = "did:plc:xo2uz55ugf2yguhjryahasuf"

    with open('reposts_analysis.json', 'r') as f:
        reposts = json.load(f)
    
    # Sort by original post creation time, newest first
    valid_reposts = [p for p in reposts if p.get('post_created')]
    valid_reposts.sort(key=lambda x: x['post_created'], reverse=True)
    
    mid = len(valid_reposts) // 2
    recent_half = valid_reposts[:mid]
    older_half = valid_reposts[mid:]
    
    recent_uris = {p['uri'] for p in recent_half}
    older_uris = {p['uri'] for p in older_half}
    
    print(f"Total valid reposts: {len(valid_reposts)}")
    print(f"Recent half (newer than {older_half[0]['post_created']}): {len(recent_uris)}")
    print(f"Older half: {len(older_uris)}\n")

    generators = [
        'popularity',
        'random_posts',
        'followed_users',
        'network_likes',
        'two_tower',
        'two_tower_empty_history'
    ]
    
    for gen in generators:
        print(f"Testing generator: {gen}...")
        try:
            req = urllib.request.Request(
                'https://api.greenearth.social/candidates/generate',
                data=json.dumps({
                    "generators": [
                        { "name": gen, "weight": 1.0 }
                    ],
                    "user_did": user_did,
                    "num_candidates": 1000
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'X-API-Key': ge_api_key}
            )
            
            with urllib.request.urlopen(req) as response:
                cand_data = json.loads(response.read().decode('utf-8'))
                
            candidates = cand_data.get('candidates', [])
            cand_uris = {c['at_uri'] for c in candidates}
            
            recent_overlap = recent_uris.intersection(cand_uris)
            older_overlap = older_uris.intersection(cand_uris)
            total_overlap = len(recent_overlap) + len(older_overlap)
            
            print(f"  Returned {len(cand_uris)} candidates.")
            print(f"  Total Overlap: {total_overlap}")
            print(f"    -> From Recent Half: {len(recent_overlap)}")
            print(f"    -> From Older Half: {len(older_overlap)}\n")
        except Exception as e:
            print(f"  Failed: {e}\n")
    
if __name__ == '__main__':
    main()
