"""Publish (or update) one app.bsky.feed.generator record.

Every feed this project serves is one Bunny deployment behind one did:web; what
distinguishes them is the record key, which arrives back as the `feed` query
parameter on getFeedSkeleton and selects the scoring vector in
bunny_feed/src/feeds.ts. So publishing a feed is exactly: write a record whose
rkey matches a key in that registry.

put_record is an upsert, so re-running this with the same rkey edits the existing
feed in place (name/description) rather than creating a duplicate.

Run:
  uv run --with atproto --with python-dotenv python publish_feed_generic.py \
      --rkey helpful-feed --name "Helpful Feed" --description "..."
"""

import argparse
import os
import sys

from atproto import Client, models
from dotenv import load_dotenv

# Must be the PullZone hostname (not the *.bunny.run runtime domain), so requests
# reach the worker through the PullZone and the Edge Cache API accepts
# request.url.origin as a valid cache-key domain.
DEFAULT_HOSTNAME = "feedforward.b-cdn.net"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rkey", required=True,
                    help="record key; must match a key in bunny_feed/src/feeds.ts")
    ap.add_argument("--name", help="displayName shown in the app (required unless --delete)")
    ap.add_argument("--description", help="required unless --delete")
    ap.add_argument("--delete", action="store_true",
                    help="unpublish: remove the feed generator record entirely")
    ap.add_argument("--hostname", default=DEFAULT_HOSTNAME)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the record that would be written and exit")
    args = ap.parse_args()

    load_dotenv()
    handle = os.environ.get("BLUESKY_HANDLE")
    app_password = os.environ.get("BLUESKY_APP_PASSWORD")
    if not handle or not app_password:
        sys.exit("Set BLUESKY_HANDLE and BLUESKY_APP_PASSWORD in your .env file")

    if not args.delete and not (args.name and args.description):
        sys.exit("--name and --description are required unless --delete is given")

    did_web = f"did:web:{args.hostname}"
    print(f"rkey:        {args.rkey}")
    if args.delete:
        print("action:      DELETE (unpublish)")
    else:
        print(f"displayName: {args.name}")
        print(f"description: {args.description}")
        print(f"service:     {did_web}")
    if args.dry_run:
        print("\n(dry run — nothing published)")
        return

    pds_url = (os.environ.get("BLUESKY_PDS_URL")
               or os.environ.get("BSKY_PDS_URL")
               or "https://bsky.social")
    print(f"\nLogging in as {handle} on {pds_url}...")
    client = Client(base_url=pds_url)
    client.login(handle, app_password)

    if args.delete:
        client.com.atproto.repo.delete_record(
            models.ComAtprotoRepoDeleteRecord.Data(
                repo=client.me.did,
                collection="app.bsky.feed.generator",
                rkey=args.rkey,
            )
        )
        print(f"\n\u2705 Feed unpublished: {args.rkey}")
        return

    response = client.com.atproto.repo.put_record(
        models.ComAtprotoRepoPutRecord.Data(
            repo=client.me.did,
            collection="app.bsky.feed.generator",
            rkey=args.rkey,
            record=models.AppBskyFeedGenerator.Record(
                did=did_web,
                displayName=args.name,
                description=args.description,
                acceptsInteractions=True,
                createdAt=client.get_current_time_iso(),
            ),
        )
    )

    print("\n✅ Feed published.")
    print(f"URI: {response.uri}")
    print(f"View: https://bsky.app/profile/{handle}/feed/{args.rkey}")


if __name__ == "__main__":
    main()
