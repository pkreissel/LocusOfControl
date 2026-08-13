import { jwtDecode } from 'jwt-decode';
import { getEmbeddings, EMBEDDING_MODEL } from './jina';
import { scoreEmbedding } from './vector';
import { recordEmbeddingUsage, recordInteractions, getSeenUris, pruneOldSeenInteractions, getCachedEmbeddings, putEmbeddings, pruneOldEmbeddings } from './db';
import * as BunnySDK from "@bunny.net/edgescript-sdk";

console.log("Bunny Edge Script module initialized");

// Drizzle wraps every driver failure in a `DrizzleQueryError` whose message is
// just "Failed query: <sql>" — the real cause (Turso HTTP status/body, or a
// runtime `fetch` error) is nested in `.cause`. Walk the chain so logs show why
// a query actually failed instead of the useless wrapper text.
function errChain(e: any): string {
  const parts: string[] = [];
  for (let cur = e, depth = 0; cur && depth < 6; cur = cur.cause, depth++) {
    parts.push(cur?.message ?? String(cur));
  }
  return parts.join(' <- ');
}

// Calibrated on the live Discover feed (447-post sample): p>=0.35 passes ~25%
// of candidates. The old 0.15 passed ~96%, effectively no filtering.
const VECTOR_PROB_THRESHOLD = 0.25;
declare const process: any;
declare const Bunny: any;

function getEnv(key: string): string | undefined {
  if (typeof Bunny !== 'undefined' && Bunny.env) return Bunny.env[key];
  if (typeof process !== 'undefined' && process.env) return process.env[key];
  return undefined;
}

// The canonical public base URL the feed is published under (see
// publish_feed.py). Used to serve the did:web document's serviceEndpoint, which
// must match the origin Bluesky reaches the feed generator at. Override via
// PUBLIC_BASE_URL if the deployment ever moves; the request origin is not used
// as a fallback since it varies by isolate warm-up.
const DEFAULT_PUBLIC_BASE_URL = 'https://feedforward.b-cdn.net';
function getPublicBaseUrl(_requestUrl: URL): string {
  const configured = getEnv('PUBLIC_BASE_URL');
  if (configured) return configured.replace(/\/+$/, '');
  return DEFAULT_PUBLIC_BASE_URL;
}

type Candidate = { uri: string; cid?: string; text: string; minilm_l12_embedding?: string; [key: string]: any };
type SkeletonBody = { feed: { post: string }[]; cursor?: string };

const SKELETON_TTL_SEC = 60;

// --- Feed request diagnostics ---------------------------------------------
// One structured line per getFeedSkeleton request (hit AND miss) so we can see
// whether each page was served personalized or fell back to the public
// (English) AppView — and, on cache hits, whether a personalized key is
// re-serving a body that was produced by an earlier public fallback.
function logFeed(fields: Record<string, unknown>) {
  try { console.log('[feed] ' + JSON.stringify(fields)); } catch { /* never break the response */ }
}
function shortDid(did: string | null): string {
  return !did ? 'anon' : (did.length > 20 ? did.slice(0, 20) + '…' : did);
}
function cursorTag(c?: string): string {
  return !c ? 'START' : (c.length > 12 ? c.slice(0, 12) + '…' : c);
}

// Prune embedding-cache rows older than this. Vectors for posts that stop
// appearing in feeds age out; a still-relevant post is simply re-embedded once.
const EMBED_CACHE_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
// Fraction of feed requests that opportunistically prune stale cache rows,
// keeping the table bounded without a separate cron.
const EMBED_PRUNE_PROBABILITY = 0.02;

// Seen-post exclusion. On each personalized cache miss we pull the user's
// most-recently-shown posts and hand them to candidate generation as
// `exclude_uris`, so paging/refreshing doesn't repeat posts they've already
// viewed. Bounded two ways: at most SEEN_EXCLUDE_LIMIT posts, and never older
// than SEEN_RETENTION_DAYS — we never go back further than the retention window.
// That same window governs the opportunistic prune of "shown" rows (engaged
// rows are kept for training), so read scope and stored data stay in lockstep.
const SEEN_EXCLUDE_LIMIT = 1000;
const SEEN_RETENTION_DAYS = 7;
const SEEN_PRUNE_PROBABILITY = 0.02;

// The skeleton cache uses the Bunny Edge Cache API. Caching is only an
// optimization, so any cache failure degrades to a miss / no-op and never
// breaks the feed response.
async function safeCacheMatch(cache: Cache, key: string): Promise<Response | undefined> {
  try {
    return await cache.match(key);
  } catch (e) {
    console.error("[cache] match failed (treating as miss):", (e as any)?.message);
    return undefined;
  }
}

async function safeCachePut(cache: Cache, key: string, res: Response): Promise<void> {
  try {
    await cache.put(key, res);
  } catch (e) {
    console.error("[cache] put failed (skipping):", (e as any)?.message);
  }
}

// Embed candidates, reusing vectors cached in the DB and only sending cache
// misses to Jina. Returns embeddings aligned to `cands` and the tokens actually
// billed (misses only). Reads are a single batched SELECT and writes a single
// batched upsert, and the cache is shared globally across edge PoPs.
async function embedCandidates(cands: Candidate[], apiKey: string): Promise<{ embeddings: number[][]; tokens: number }> {
  const embeddings: number[][] = new Array(cands.length);

  // Batch-load cached vectors. Best-effort: a DB error just means "all miss".
  const cached = await getCachedEmbeddings(cands.map(c => c.uri), EMBEDDING_MODEL)
    .catch((e: any) => { console.error("[embed cache] read failed:", errChain(e)); return new Map<string, number[]>(); });

  // De-duplicate misses by uri: Jina is billed per unique text, and SQLite
  // rejects upserting the same row twice in one statement.
  const missSlot = new Map<string, number>(); // uri -> index into miss arrays
  const missUris: string[] = [];
  const missTexts: string[] = [];
  for (let i = 0; i < cands.length; i++) {
    const hit = cached.get(cands[i].uri);
    if (hit) {
      embeddings[i] = hit;
    } else if (!missSlot.has(cands[i].uri)) {
      missSlot.set(cands[i].uri, missUris.length);
      missUris.push(cands[i].uri);
      missTexts.push(cands[i].text);
    }
  }

  let tokens = 0;
  if (missUris.length > 0) {
    const res = await getEmbeddings(missTexts, apiKey);
    tokens = res.tokens;

    // Fill every still-empty slot (including duplicate uris) from the results.
    for (let i = 0; i < cands.length; i++) {
      if (!embeddings[i]) embeddings[i] = res.embeddings[missSlot.get(cands[i].uri)!];
    }

    // One batched upsert of the freshly-computed vectors.
    const toStore = missUris.map((uri, k) => ({ uri, vec: res.embeddings[k] }));
    await putEmbeddings(toStore, EMBEDDING_MODEL)
      .catch((e: any) => console.error("[embed cache] write failed:", errChain(e)));
  }

  // Opportunistically keep the table bounded.
  if (Math.random() < EMBED_PRUNE_PROBABILITY) {
    await pruneOldEmbeddings(Date.now() - EMBED_CACHE_MAX_AGE_MS)
      .catch((e: any) => console.error("[embed cache] prune failed:", errChain(e)));
  }

  return { embeddings, tokens };
}

BunnySDK.net.http.serve(async (request: Request) => {
  try {
    return await handleRequest(request);
  } catch (err: any) {
    console.error("[Fetch Event] Unhandled error in handleRequest:", err);
    return new Response(JSON.stringify({ error: "Internal Server Error", details: err.message }), { status: 500 });
  }
});

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);

  // --- FEED GENERATOR SKELETON ENDPOINT ---
  if (url.pathname === '/xrpc/app.bsky.feed.getFeedSkeleton') {
    const t0 = Date.now();
    try {
      const apiKey = getEnv('JINA_API_KEY') || 'MISSING_API_KEY';
      if (!apiKey || apiKey === 'MISSING_API_KEY') {
        console.error("[getFeedSkeleton Error]: Missing Jina API Key in environment");
        return new Response(JSON.stringify({ error: "Missing Jina API Key" }), { status: 500 });
      }

      const geApiKey = getEnv('GREEN_EARTH_API_KEY');
      if (!geApiKey) {
        console.error("[getFeedSkeleton Error]: Missing GE API Key in environment");
        return new Response(JSON.stringify({ error: "Missing GE API Key" }), { status: 500 });
      }

      // Extract User DID from ATProto Service JWT
      const authHeader = request.headers.get('Authorization');
      let userDid = null;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const decoded: any = jwtDecode(authHeader.substring(7));
          userDid = decoded.iss; // Service JWT issuer is the user's DID
        } catch (e) { }
      }
      const feedUriParam = url.searchParams.get('feed') || '';
      const rkey = feedUriParam.split('/').pop() || 'unknown';
      const isSelbstwirksamkeit = rkey === 'selbstwirksamkeit';

      const unauthPostUri = getEnv('UNAUTHENTICATED_POST_URI');
      if (!userDid && unauthPostUri && !isSelbstwirksamkeit) {
        return new Response(JSON.stringify({ feed: [{ post: unauthPostUri }] }), {
          headers: { 'Content-Type': 'application/json' }
        });
      }

      if (isSelbstwirksamkeit) {
        userDid = null; // force unauthenticated path
      }

      const didTag = shortDid(userDid);

      // (1) Serve from the scored-skeleton cache when fresh. Keyed by feed identity +
      // user DID (or "public") + pagination cursor. A hit returns instantly and
      // skips the upstream fetches, embeddings, and the OAuth restore/lock below.
      const startCursor: string | undefined = url.searchParams.get('cursor') || undefined;
      const cacheKey = `${rkey}:${userDid || 'public'}:${startCursor || 'START'}`;

      const skeletonCacheUrl = new URL(request.url);
      skeletonCacheUrl.pathname = '/_cache/skeleton';
      skeletonCacheUrl.search = '';
      skeletonCacheUrl.searchParams.set('key', cacheKey);
      const skeletonCacheKeyStr = skeletonCacheUrl.toString();

      const cache = caches.default;
      // Personalized feeds are never cached (see the write path below): every
      // logged-in request recomputes so seen-post exclusion (interactionSeen)
      // takes effect immediately instead of lagging behind the cache TTL. Only
      // the public/unauthenticated feed is served from — and written to — cache.
      const cachedSkeleton = userDid ? undefined : await safeCacheMatch(cache, skeletonCacheKeyStr);

      if (cachedSkeleton && cachedSkeleton.status === 200) {
        const body = await cachedSkeleton.json();
        // Cache hits were previously silent. A public-fallback response cached
        // under a personalized key would be re-served as "English" with no
        // trace — surface the markers we stored on write so it shows in logs.
        logFeed({
          did: didTag, cursor: cursorTag(startCursor), cache: 'hit',
          personalized: cachedSkeleton.headers.get('X-Personalized'),
          path: cachedSkeleton.headers.get('X-Feed-Path'),
          cand: cachedSkeleton.headers.get('X-Cand-Count'),
          pass: cachedSkeleton.headers.get('X-Pass-Count'),
          feedLen: Array.isArray((body as any)?.feed) ? (body as any).feed.length : null,
          nextCursor: (body as any)?.cursor ? 'y' : 'n',
          ms: Date.now() - t0,
        });
        return new Response(JSON.stringify(body), {
          headers: {
            'Content-Type': 'application/json',
            'X-Cache': 'hit',
            // Never let the Bunny PullZone CDN cache the personalized skeleton:
            // it keys on neither the cursor nor the per-user Authorization/DID, so
            // one cached copy freezes the feed for everyone. This response is
            // already served from our own per-(user,cursor) caches.default; the
            // CDN layer on top must pass through.
            'Cache-Control': 'no-store',
          },
        });
      }

      let candidates: Candidate[] = [];
      let cursor: string | undefined = undefined; // We don't have cursor pagination yet
      const personalized = !!userDid;
      const feedPath = personalized ? 'personalized' : 'public-no-did';
      
      const tG = Date.now();
      const geHeaders = {
        'Content-Type': 'application/json',
        'X-API-Key': geApiKey
      };
      
      const primaryGenerator = userDid ? 'followed_users' : 'popularity';

      // Exclude posts this user has already been shown so paging/refreshing
      // doesn't repeat them. Best-effort: a DB failure just means no exclusion
      // this request. Only meaningful when we know who the user is.
      let excludeUris: string[] = [];
      if (userDid) {
        excludeUris = await getSeenUris(userDid, SEEN_EXCLUDE_LIMIT, SEEN_RETENTION_DAYS)
          .catch((e: any) => { console.error("[getFeedSkeleton] seen-uris read failed:", errChain(e)); return []; });
      }

      // Opportunistically age out old "shown" rows so the table stays bounded
      // without a separate cron (engaged rows are kept for training).
      if (Math.random() < SEEN_PRUNE_PROBABILITY) {
        await pruneOldSeenInteractions(SEEN_RETENTION_DAYS)
          .catch((e: any) => console.error("[getFeedSkeleton] seen prune failed:", errChain(e)));
      }

      try {
        const genBody: any = {
          generators: [
            { name: primaryGenerator, weight: 1.0 },
            { name: 'network_likes', weight: 1.0 },
            { name: 'two_tower', weight: 1.0 }
          ],
          infill: 'popularity',
          user_did: userDid || 'did:plc:z72i7hdynmk6r22z27h6tvur', // fallback DID if none
          num_candidates: 1000
        };
        // Only send the key when non-empty so the unauthenticated path stays
        // byte-identical to before and we never send an empty array.
        if (excludeUris.length > 0) genBody.exclude_uris = excludeUris;
        const candRes = await fetch('https://api.greenearth.social/candidates/generate', {
          method: 'POST',
          headers: geHeaders,
          body: JSON.stringify(genBody)
        });
        if (!candRes.ok) {
          throw new Error(`GE Generate failed: ${candRes.status}`);
        }
        const candData = await candRes.json();
        candidates = (candData.candidates || []).map((c: any) => ({
            ...c,
            uri: c.at_uri,
            text: c.content || ''
        }));
      } catch (err: any) {
        console.error("[getFeedSkeleton] GE generate fetch failed:", err?.message || String(err));
      }
      
      const gatherMs = Date.now() - tG;
      let feed: { post: string }[] = [];
      const filteredCandidates: Candidate[] = [];

      if (candidates.length > 0) {
        // (3) Reuse cached vectors; only cache-missed posts are sent to Jina.
        const { embeddings, tokens } = await embedCandidates(candidates, apiKey);
        // Best-effort cost tracking; never let it break the feed response.
        await recordEmbeddingUsage(tokens).catch((e: any) =>
          console.error("[getFeedSkeleton] failed to record token usage:", e?.cause?.message || e?.message));
        
        for (let i = 0; i < candidates.length; i++) {
          const score = scoreEmbedding(embeddings[i]);
          if (score.probability >= VECTOR_PROB_THRESHOLD) {
            filteredCandidates.push(candidates[i]);
          }
        }
        
        if (filteredCandidates.length > 0) {
          // (4) Rank via Green Earth
          try {
            const rankRes = await fetch('https://api.greenearth.social/rank/predict', {
              method: 'POST',
              headers: geHeaders,
              body: JSON.stringify({
                candidates: filteredCandidates,
                models: [{ name: 'two_tower', weight: 1.0 }],
                user_did: userDid || 'did:plc:z72i7hdynmk6r22z27h6tvur'
              })
            });
            if (rankRes.ok) {
                const rankData = await rankRes.json();
                if (rankData.rankings && rankData.rankings.length > 0) {
                    const rankedUris = rankData.rankings.map((r: any) => r.at_uri);
                    filteredCandidates.sort((a, b) => {
                        const idxA = rankedUris.indexOf(a.uri);
                        const idxB = rankedUris.indexOf(b.uri);
                        return (idxA === -1 ? 9999 : idxA) - (idxB === -1 ? 9999 : idxB);
                    });
                }
            }
          } catch (e: any) {
              console.error("[getFeedSkeleton] GE rank failed:", e?.message || String(e));
          }

          // NOTE: the GE /diversify step was removed. It re-sorts candidates by
          // the generator's popularity `score` and ignores the input order, so
          // it silently overwrote the rank order above (the reranker had no
          // visible effect). It also expects minilm embeddings, which this feed
          // doesn't produce (we embed with Jina), so it couldn't diversify
          // anyway. Rank order is now the final feed order.
          feed = filteredCandidates.map(c => ({ post: c.uri }));
        }
      }

      // Only return a cursor while the upstream feed still has more pages;
      // omitting it signals "end of feed" to the client.
      const body: SkeletonBody = { feed };
      if (cursor) body.cursor = cursor;

      // Only cache the public/unauthenticated feed. Personalized (logged-in)
      // feeds are never cached: caching them lagged interactionSeen exclusion by
      // up to SKELETON_TTL_SEC (the cached page kept re-serving already-seen
      // posts). Recomputing every personalized request keeps exclusion immediate;
      // the public feed still caches since it has no per-user state to stale.
      if (!userDid) {
        const resToCache = new Response(JSON.stringify(body), {
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': `s-maxage=${SKELETON_TTL_SEC}`,
            // Markers so a later cache HIT isn't blind: whether this cached body
            // was personalized, and how it was produced.
            'X-Personalized': String(personalized),
            'X-Feed-Path': feedPath,
            'X-Cand-Count': String(candidates.length),
            'X-Pass-Count': String(feed.length),
          }
        });
        await safeCachePut(cache, skeletonCacheKeyStr, resToCache);
      }

      logFeed({
        did: didTag, cursor: cursorTag(startCursor), cache: 'miss',
        cand: candidates.length, pass: feed.length,
        nextCursor: cursor ? 'y' : 'n',
        gatherMs, ms: Date.now() - t0,
      });

      return new Response(JSON.stringify(body), {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'miss',
          // Never let the Bunny PullZone CDN cache the personalized skeleton (see
          // the hit path above): its cache key ignores the cursor and the per-user
          // Authorization/DID, so a single cached copy freezes the feed and defeats
          // seen-post exclusion. Our own caches.default already caches per-key.
          'Cache-Control': 'no-store',
          // Diagnostics (readable off the HTTP response, no logs needed):
          // X-Feed-Version confirms which build is actually serving traffic;
          // X-Exclude-Count reports how many seen-post URIs getSeenUris returned
          // and thus how many were sent to GE as exclude_uris this request.
          'X-Feed-Version': 'exclusion-probe-3-nocache-personalized',
          'X-Exclude-Count': String(excludeUris.length),
        },
      });
    } catch (e: any) {
      console.error("[getFeedSkeleton] error:", e?.status ?? '', e?.message, e?.stack);
      return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (url.pathname === '/xrpc/app.bsky.feed.sendInteractions' && request.method === 'POST') {
    try {
      // Extract User DID from ATProto Service JWT
      const authHeader = request.headers.get('Authorization');
      let userDid = null;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const decoded: any = jwtDecode(authHeader.substring(7));
          userDid = decoded.iss; // Service JWT issuer is the user's DID
        } catch (e) { }
      }

      const body = await request.json();
      if (body.interactions && Array.isArray(body.interactions)) {
        // Record every interaction, INCLUDING `interactionSeen`: the "shown"
        // signal is what powers seen-post exclusion during candidate generation
        // (getSeenUris). Batched into one multi-row insert so a scroll burst is a
        // single DB round-trip, not one per post.
        const toRecord = body.interactions
          .filter((i: any) => i && i.item && i.event)
          .map((i: any) => ({ item: i.item, event: i.event, feedContext: i.feedContext }));
        await recordInteractions(userDid, toRecord).catch((e: any) => {
          console.error("[sendInteractions] failed to record interactions:", e?.cause?.message || e?.message);
        });
      }

      return new Response(JSON.stringify({}), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      console.error("[sendInteractions] error:", e?.message, e?.stack);
      return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (url.pathname === '/.well-known/did.json') {
    const publicBaseUrl = getPublicBaseUrl(url);
    const publicHostname = new URL(publicBaseUrl).hostname;
    return new Response(JSON.stringify({
      "@context": ["https://www.w3.org/ns/did/v1"],
      "id": `did:web:${publicHostname}`,
      "service": [
        {
          "id": "#bsky_fg",
          "type": "BskyFeedGenerator",
          "serviceEndpoint": publicBaseUrl
        }
      ]
    }), { headers: { 'Content-Type': 'application/json' } });
  }

  return new Response(`Bunny ATProto Feed Generator - Current time: ${new Date().toISOString()}`, { status: 200 });
}
