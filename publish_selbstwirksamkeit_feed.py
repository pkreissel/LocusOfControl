import os
from dotenv import load_dotenv
from atproto import Client, models

def main():
    load_dotenv()
    
    handle = os.environ.get('BLUESKY_HANDLE')
    app_password = os.environ.get('BLUESKY_APP_PASSWORD')
    
    if not handle or not app_password:
        print("Please set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in your .env file")
        return
        
    pds_url = os.environ.get('BLUESKY_PDS_URL') or os.environ.get('BSKY_PDS_URL') or 'https://bsky.social'
    
    print(f"Logging in as {handle} on {pds_url}...")
    client = Client(base_url=pds_url)
    client.login(handle, app_password)

    # Must be the PullZone hostname (not the *.bunny.run runtime domain), so
    # requests reach the worker through the PullZone and the Edge Cache API
    # accepts request.url.origin as a valid cache-key domain.
    hostname = "audacitytohope-bji54.b-cdn.net"
    did_web = f"did:web:{hostname}"
    
    rkey = 'selbstwirksamkeit'
    
    print(f"Publishing feed generator to {did_web}...")
    
    response = client.com.atproto.repo.put_record(
        models.ComAtprotoRepoPutRecord.Data(
            repo=client.me.did,
            collection='app.bsky.feed.generator',
            rkey=rkey,
            record=models.AppBskyFeedGenerator.Record(
                did=did_web,
                displayName='Selbstwirksamkeit',
                description='Ein Feed, der Beiträge basierend auf dem Locus of Control filtert (nicht personalisiert).',
                acceptsInteractions=True,
                createdAt=client.get_current_time_iso()
            )
        )
    )
    
    print(f"✅ Feed published successfully!")
    print(f"URI: {response.uri}")
    print(f"You can view it in the app: https://bsky.app/profile/{handle}/feed/{rkey}")

if __name__ == '__main__':
    main()
