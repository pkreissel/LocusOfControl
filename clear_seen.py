import os
import json
import urllib.request
from dotenv import load_dotenv

def main():
    load_dotenv('bunny_feed/.env')
    load_dotenv('.env')

    db_url = os.environ.get('BUNNY_DATABASE_URL') or os.environ.get('BUNNY_DB_URL')
    token = os.environ.get('BUNNY_DATABASE_AUTH_TOKEN') or os.environ.get('BUNNY_DB_TOKEN')

    if not db_url or not token:
        print("Missing DB credentials")
        exit(1)

    endpoint = f"{db_url.replace('libsql://', 'https://').replace('wss://', 'https://').rstrip('/')}/v2/pipeline"

    query = {
        "requests": [
            {
                "type": "execute",
                "stmt": {
                    "sql": "DELETE FROM feed_interactions WHERE event = 'app.bsky.feed.defs#interactionSeen'",
                }
            }
        ]
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(query).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    )

    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            print("Response:", json.dumps(res, indent=2))
            
            # Look for affected rows in the response
            try:
                affected = res['results'][0]['response']['result']['affected_row_count']
                print(f"\nSuccessfully deleted {affected} interactionSeen events from the database!")
            except (KeyError, IndexError):
                print("Could not parse affected_row_count, but request succeeded.")
                
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
