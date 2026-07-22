import { Agent } from '@atproto/api';

declare const process: any;
declare const Bunny: any;

function getEnv(key: string): string | undefined {
  if (typeof Bunny !== 'undefined' && Bunny.env) return Bunny.env[key];
  if (typeof process !== 'undefined' && process.env) return process.env[key];
  return undefined;
}

// Source of candidate posts to score. `getFeed` requires either an authed
// agent or a *real* feed on the public appview (public.api.bsky.app); an
// unauthenticated call to bsky.social returns 401. Defaults to Bluesky's
// Discover ("what's hot") feed, overridable via the CANDIDATE_FEED_URI env var.
const CANDIDATE_FEED_URI =
  getEnv('CANDIDATE_FEED_URI') ||
  'at://did:plc:ooensn4mr5mhznzypvxelfa3/app.bsky.feed.generator/fu';

// A stream/transport error talking to the user's PDS (e.g. eurosky.social
// intermittently RST-ing an HTTP/2 stream) is often transient — one retry on a
// fresh connection usually succeeds. Deterministic failures (e.g. an oversized
// cursor the PDS rejects, or a 4xx) are not retried; they fall through to the
// caller's public-AppView fallback. Kept to a single extra attempt because each
// getFeed page is already ~1-2s and the skeleton response must stay responsive.
const MAX_FETCH_ATTEMPTS = 2;
function isTransientFetchError(err: any): boolean {
  if (typeof err?.status === 'number' && err.status >= 500) return true;
  return /sending request|http2|stream error|fetch failed|timed out|timeout|connection/i.test(err?.message || '');
}

export async function fetchCandidatePosts(agent: Agent, limit: number = 50, cursor?: string, candidateFeedUriOverride?: string) {
  let lastErr: any;
  const feedUri = candidateFeedUriOverride || CANDIDATE_FEED_URI;
  for (let attempt = 1; attempt <= MAX_FETCH_ATTEMPTS; attempt++) {
    try {
      const response = await agent.app.bsky.feed.getFeed({
        feed: feedUri,
        limit,
        ...(cursor ? { cursor } : {}),
      });

      const nextCursor = response.data.cursor;
      if (!response.data.feed) return { candidates: [], cursor: nextCursor };

      const candidates = response.data.feed.map(item => {
        return {
          uri: item.post.uri,
          cid: item.post.cid,
          // @ts-ignore
          text: item.post.record.text || '',
        };
      });

      return { candidates, cursor: nextCursor };
    } catch (err: any) {
      lastErr = err;
      if (attempt < MAX_FETCH_ATTEMPTS && isTransientFetchError(err)) {
        console.error(`[Bluesky API] getFeed attempt ${attempt}/${MAX_FETCH_ATTEMPTS} failed (transient), retrying:`, err?.message);
        await new Promise(r => setTimeout(r, 400));
        continue;
      }
      console.error(`[Bluesky API Error]: Failed to fetch candidate posts from ${feedUri}`, err);
      throw err;
    }
  }
  throw lastErr;
}
