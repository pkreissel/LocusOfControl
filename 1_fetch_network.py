import os
import json
import time
from atproto import Client
from tqdm import tqdm
from dotenv import load_dotenv

def get_all_follows(client, actor, max_limit=None):
    follows = []
    cursor = None
    while True:
        try:
            # get_follows returns an object with 'follows' (list of profiles) and 'cursor'
            response = client.get_follows(actor=actor, cursor=cursor, limit=100)
            if not response.follows:
                break
                
            follows.extend([f.did for f in response.follows])
            
            if max_limit and len(follows) >= max_limit:
                break
                
            if not response.cursor:
                break
            cursor = response.cursor
            time.sleep(0.1) # Be nice to the API
        except Exception as e:
            print(f"\nError fetching follows for {actor}: {e}")
            break
            
    return follows[:max_limit] if max_limit else follows

def main():
    load_dotenv()
    handle = os.environ.get('BSKY_HANDLE') or os.environ.get('BLUESKY_HANDLE')
    password = os.environ.get('BSKY_APP_PASSWORD') or os.environ.get('BLUESKY_APP_PASSWORD')
    pds_url = os.environ.get('BSKY_PDS_URL', 'https://bsky.social')
    
    if not handle or not password:
        print("Please set BSKY_HANDLE and BSKY_APP_PASSWORD (or BLUESKY_HANDLE and BLUESKY_APP_PASSWORD) environment variables.")
        print("Example: export BSKY_HANDLE='your.handle.bsky.social' BSKY_APP_PASSWORD='your-app-password'")
        return

    client = Client(base_url=pds_url)
    print(f"Logging in as {handle} on {pds_url}...")
    client.login(handle, password)
    
    print("Fetching your follows (1st-degree connections)...")
    my_follows = get_all_follows(client, handle)
    print(f"Found {len(my_follows)} accounts you follow.")
    
    network_dids = set(my_follows)
    
    print("Fetching 2nd-degree follows (up to 50 per followed account) to expand network...")
    for did in tqdm(my_follows):
        # Fetching a small batch of 2nd degree follows to expand network broadly without hitting rate limits
        second_degree = get_all_follows(client, did, max_limit=50)
        network_dids.update(second_degree)
        
    network_dids_list = list(network_dids)
    print(f"\nTotal unique DIDs in collected network: {len(network_dids_list)}")
    
    with open('network_dids.json', 'w') as f:
        json.dump(network_dids_list, f, indent=2)
    print("Saved network to network_dids.json")

if __name__ == '__main__':
    main()
