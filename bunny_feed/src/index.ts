import { Agent } from '@atproto/api';
import { jwtDecode } from 'jwt-decode';
import { getEmbeddings, EMBEDDING_MODEL } from './jina';
import { scoreEmbedding } from './vector';
import { fetchCandidatePosts } from './bsky';
import { recordEmbeddingUsage, recordInteraction, getCachedEmbeddings, putEmbeddings, pruneOldEmbeddings } from './db';
import { OAuthClient } from '@atproto/oauth-client';
import { LibSqlStateStore, LibSqlSessionStore, WebCryptoRuntimeImplementation, initOAuthTables } from './oauth-store';
import * as BunnySDK from "@bunny.net/edgescript-sdk";

console.log("Bunny Edge Script module initialized");

// Calibrated on the live Discover feed (447-post sample): p>=0.35 passes ~25%
// of candidates. The old 0.15 passed ~96%, effectively no filtering.
const VECTOR_PROB_THRESHOLD = 0.35;
declare const process: any;
declare const Bunny: any;

function getEnv(key: string): string | undefined {
  if (typeof Bunny !== 'undefined' && Bunny.env) return Bunny.env[key];
  if (typeof process !== 'undefined' && process.env) return process.env[key];
  return undefined;
}

// The OAuth client_id MUST be the exact, stable URL where client-metadata.json
// is publicly served, and it MUST be byte-identical at authorization time and
// at every later refresh — otherwise the PDS rejects the refresh with
// invalid_grant "Token was not issued to this client" (which then deletes the
// session). Deriving it from the incoming request origin is unsafe: this Worker
// is reachable via several origins (custom domain, Bunny *.b-cdn.net, http vs
// https, www vs apex), and a module-level singleton freezes to whichever origin
// first warms the isolate. So pin everything to a configured canonical base URL
// and only fall back to the request origin when unconfigured (local dev).
// Default to the PullZone hostname the feed is published under (see
// publish_feed.py) — this is the origin the PDS fetched client-metadata.json
// from at login, so it must be the client_id at refresh time too. Override via
// PUBLIC_BASE_URL if the deployment ever moves. The request origin is no longer
// used as a fallback: it varies by isolate warm-up and is what caused the
// client_id mismatch in the first place.
const DEFAULT_PUBLIC_BASE_URL = 'https://audacitytohope-bji54.b-cdn.net';
function getPublicBaseUrl(_requestUrl: URL): string {
  const configured = getEnv('PUBLIC_BASE_URL');
  if (configured) return configured.replace(/\/+$/, '');
  return DEFAULT_PUBLIC_BASE_URL;
}

// Keyed by resolved base URL so a stable PUBLIC_BASE_URL yields exactly one
// client regardless of which origin/isolate serves a given request; without it,
// each origin still gets its own client instead of cross-contaminating.
const oauthClients = new Map<string, OAuthClient>();
function getOAuthClient(requestUrl: URL) {
  const base = getPublicBaseUrl(requestUrl);
  let client = oauthClients.get(base);
  if (!client) {
    client = new OAuthClient({
      clientMetadata: {
        client_name: 'Locus of Control Feed',
        client_id: `${base}/client-metadata.json`,
        client_uri: base,
        redirect_uris: [`${base}/oauth/callback`],
        scope: 'atproto rpc:app.bsky.feed.getFeed?aud=did:web:api.bsky.app#bsky_appview rpc:app.bsky.feed.getFeedSkeleton?aud=did:web:api.bsky.app#bsky_appview',
        grant_types: ['authorization_code', 'refresh_token'],
        response_types: ['code'],
        token_endpoint_auth_method: 'none',
        application_type: 'web',
        dpop_bound_access_tokens: true
      },
      responseMode: 'query',
      stateStore: new LibSqlStateStore(),
      sessionStore: new LibSqlSessionStore(),
      runtimeImplementation: new WebCryptoRuntimeImplementation(),
      handleResolver: 'https://bsky.social'
    });
    oauthClients.set(base, client);
    console.log('[oauth] initialised client for base', base);
  }
  return client;
}

type Candidate = { uri: string; cid: string; text: string };
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
    .catch((e: any) => { console.error("[embed cache] read failed:", e?.message); return new Map<string, number[]>(); });

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
      .catch((e: any) => console.error("[embed cache] write failed:", e?.message));
  }

  // Opportunistically keep the table bounded.
  if (Math.random() < EMBED_PRUNE_PROBABILITY) {
    await pruneOldEmbeddings(Date.now() - EMBED_CACHE_MAX_AGE_MS)
      .catch((e: any) => console.error("[embed cache] prune failed:", e?.message));
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

  // Ensure tables are created (idempotent)
  await initOAuthTables();

  if (url.pathname === '/client-metadata.json') {
    const client = getOAuthClient(url);
    return new Response(JSON.stringify(client.clientMetadata), {
      headers: { 'Content-Type': 'application/json' }
    });
  }

  if (url.pathname === '/.well-known/oauth-protected-resource') {
    return new Response(JSON.stringify({
      resource: url.origin,
      authorization_servers: ["https://bsky.social"] // Simplified for now
    }), { headers: { 'Content-Type': 'application/json' } });
  }

  // --- OAUTH / LOGIN ENDPOINT ---
  if (url.pathname === '/login' && request.method === 'POST') {
    try {
      const text = await request.text();
      const params = new URLSearchParams(text);
      const handle = params.get('handle') || '';
      if (!handle) {
        return new Response("Missing handle", { status: 400 });
      }

      const client = getOAuthClient(url);
      const authUrl = await client.authorize(handle, { scope: client.clientMetadata.scope });

      return Response.redirect(authUrl.toString(), 302);
    } catch (e: any) {
      console.error("[Login Route Error]", e);
      return new Response(`<h2>Login Failed</h2><p>${e.message}</p><a href="/">Try again</a>`, { status: 401, headers: { 'Content-Type': 'text/html' } });
    }
  }

  if (url.pathname === '/oauth/callback') {
    try {
      const client = getOAuthClient(url);
      const params = new URLSearchParams(url.search);
      const result = await client.callback(params);

      return new Response(`<h2>Login Successful!</h2><p>Welcome, ${result.session.sub}. Your session is now saved via OAuth.</p>`, {
        headers: { 'Content-Type': 'text/html' }
      });
    } catch (e: any) {
      console.error("[Callback Route Error]", e);
      return new Response(`<h2>Callback Failed</h2><p>${e.message}</p>`, { status: 401, headers: { 'Content-Type': 'text/html' } });
    }
  }

  // --- FEED GENERATOR SKELETON ENDPOINT ---
  if (url.pathname === '/xrpc/app.bsky.feed.getFeedSkeleton') {
    const t0 = Date.now();
    try {
      const apiKey = getEnv('JINA_API_KEY') || 'MISSING_API_KEY';
      if (!apiKey || apiKey === 'MISSING_API_KEY') {
        console.error("[getFeedSkeleton Error]: Missing Jina API Key in environment");
        return new Response(JSON.stringify({ error: "Missing Jina API Key" }), { status: 500 });
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
      const cachedSkeleton = await safeCacheMatch(cache, skeletonCacheKeyStr);

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
          headers: { 'Content-Type': 'application/json', 'X-Cache': 'hit' },
        });
      }

      // Prefer an authenticated agent so we get the *personalized* candidate
      // feed (same as the user's own client). Restore the OAuth session saved
      // during login; fall back to the generic public appview if the user has
      // never logged in or the session can't be refreshed.
      let agent: Agent;
      let personalized = false;
      let feedPath: string = userDid ? 'personalized' : 'public-no-did';
      let failReason: string | undefined;
      let restoreMs: number | undefined;
      if (userDid) {
        const tR = Date.now();
        try {
          const oauthSession = await getOAuthClient(url).restore(userDid);
          agent = new Agent(oauthSession);
          personalized = true;
          restoreMs = Date.now() - tR;
        } catch (err: any) {
          restoreMs = Date.now() - tR;
          feedPath = 'public-fallback-restore';
          failReason = err?.message || String(err);
          console.error("[getFeedSkeleton] OAuth restore failed for", userDid, '-', failReason);
          console.error(err);
        }
      }
      if (!personalized) {
        // Public appview allows unauthenticated getFeed for real feeds
        // (bsky.social returns 401); returns the non-personalized feed.
        console.log(`[getFeedSkeleton] Using unauthenticated public AppView for user: ${userDid || 'UNKNOWN'}`);
        agent = new Agent('https://public.api.bsky.app');
      }
      // Pagination: the client passes the `cursor` we returned last time, and
      // we thread it through the upstream getFeed call. Gather several upstream
      // pages first, then embed all of their posts in a single Jina batch (one
      // network round-trip) before scoring and filtering.
      // volume in one fewer upstream round-trip, which also means one fewer
      // atproto getSession cycle (lock + session read) on the personalized path.
      const CANDIDATE_PAGES = 3;
      const PAGE_SIZE = 100;
      const candidateFeedUriOverride = isSelbstwirksamkeit 
        ? 'at://did:plc:ituoear7k6qx3smjfoxhufm4/app.bsky.feed.generator/aaanl422clvee'
        : undefined;

      const gatherCandidates = async (a: Agent) => {
        const collected: Candidate[] = [];
        let cur = startCursor;
        for (let page = 0; page < CANDIDATE_PAGES; page++) {
          const result = await fetchCandidatePosts(a, PAGE_SIZE, cur, candidateFeedUriOverride);
          cur = result.cursor;
          collected.push(...result.candidates);
          if (!cur) break; // reached the end of the upstream feed
        }
        return { collected, cur };
      };

      // If the personalized (authenticated) fetch fails — e.g. the OAuth token
      // can't be refreshed — degrade to the public AppView instead of throwing.
      // A 500 here is surfaced to the user by the app as "feed unavailable",
      // whereas the public feed still returns real (non-personalized) results.
      let candidates: Candidate[];
      let cursor: string | undefined;
      const tG = Date.now();
      try {
        ({ collected: candidates, cur: cursor } = await gatherCandidates(agent!));
      } catch (err: any) {
        if (!personalized) throw err; // already on the public path; nothing to fall back to
        feedPath = 'public-fallback-fetch';
        failReason = err?.message || String(err);
        console.error("[getFeedSkeleton] personalized fetch failed, falling back to public AppView:", failReason, err?.stack || err);
        ({ collected: candidates, cur: cursor } = await gatherCandidates(new Agent('https://public.api.bsky.app')));
      }
      const gatherMs = Date.now() - tG;

      const feed: { post: string }[] = [];
      if (candidates.length > 0) {
        // (3) Reuse cached vectors; only cache-missed posts are sent to Jina.
        const { embeddings, tokens } = await embedCandidates(candidates, apiKey);
        // Best-effort cost tracking; never let it break the feed response.
        // Must be awaited: a fire-and-forget fetch runs after the response is
        // returned, outside the live request context, where Bunny's fetch dies.
        await recordEmbeddingUsage(tokens).catch((e: any) =>
          console.error("[getFeedSkeleton] failed to record token usage:", e?.cause?.message || e?.message));
        for (let i = 0; i < candidates.length; i++) {
          const score = scoreEmbedding(embeddings[i]);
          if (score.probability >= VECTOR_PROB_THRESHOLD) {
            feed.push({ post: candidates[i].uri });
          }
        }
      }

      // Only return a cursor while the upstream feed still has more pages;
      // omitting it signals "end of feed" to the client.
      const body: SkeletonBody = { feed };
      if (cursor) body.cursor = cursor;

      // Don't cache a public fallback under a personalized key: doing so pins
      // the English (unauthenticated) feed for this user for SKELETON_TTL_SEC,
      // turning a transient upstream blip into a sticky language switch. Skip
      // the write so the next request re-attempts the personalized path.
      const servedFallback = !!userDid && !personalized;
      if (!servedFallback) {
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
        path: feedPath, personalized, cached: !servedFallback,
        cand: candidates.length, pass: feed.length,
        nextCursor: cursor ? 'y' : 'n',
        restoreMs, gatherMs, ms: Date.now() - t0,
        ...(failReason ? { fail: failReason } : {}),
      });

      return new Response(JSON.stringify(body), {
        headers: { 'Content-Type': 'application/json', 'X-Cache': 'miss' },
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
        for (const interaction of body.interactions) {
          await recordInteraction(userDid, interaction.item, interaction.event, interaction.feedContext).catch((e: any) => {
            console.error("[sendInteractions] failed to record interaction:", e?.cause?.message || e?.message);
          });
        }
      }

      return new Response(JSON.stringify({}), { headers: { 'Content-Type': 'application/json' } });
    } catch (e: any) {
      console.error("[sendInteractions] error:", e?.message, e?.stack);
      return new Response(JSON.stringify({ error: e.message }), { status: 500, headers: { 'Content-Type': 'application/json' } });
    }
  }

  if (url.pathname === '/.well-known/did.json') {
    return new Response(JSON.stringify({
      "@context": ["https://www.w3.org/ns/did/v1"],
      "id": `did:web:${url.hostname}`,
      "service": [
        {
          "id": "#bsky_fg",
          "type": "BskyFeedGenerator",
          "serviceEndpoint": `https://${url.hostname}`
        }
      ]
    }), { headers: { 'Content-Type': 'application/json' } });
  }

  if (url.pathname === '/') {
    const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Feed Generator Login</title>
  <style>
    body { font-family: sans-serif; max-width: 400px; margin: 2rem auto; padding: 1rem; }
    input, button { display: block; width: 100%; margin-bottom: 1rem; padding: 0.5rem; box-sizing: border-box; }
    label { display: block; margin-bottom: 0.25rem; font-weight: bold; }
  </style>
</head>
<body>
  <h2>Feed Generator Login</h2>
  <form action="/login" method="POST">
    <label>Bluesky Handle:</label>
    <input type="text" name="handle" required placeholder="e.g. alice.bsky.social">
    <button type="submit">Log In with Bluesky</button>
  </form>
</body>
</html>`;
    return new Response(html, { headers: { 'Content-Type': 'text/html' } });
  }

  return new Response("Bunny ATProto Feed Generator", { status: 200 });
}
