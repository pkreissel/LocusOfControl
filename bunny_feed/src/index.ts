import { Agent } from '@atproto/api';
import { jwtDecode } from 'jwt-decode';
import { getEmbeddings, EMBEDDING_MODEL } from './jina';
import { scoreEmbedding } from './vector';
import { recordEmbeddingUsage, recordInteraction, getCachedEmbeddings, putEmbeddings, pruneOldEmbeddings } from './db';
import { OAuthClient } from '@atproto/oauth-client';
import { LibSqlStateStore, LibSqlSessionStore, WebCryptoRuntimeImplementation, initOAuthTables } from './oauth-store';
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
const DEFAULT_PUBLIC_BASE_URL = 'https://feedforward.b-cdn.net';
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
      resource: getPublicBaseUrl(url),
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
      
      try {
        const candRes = await fetch('https://api.greenearth.social/candidates/generate', {
          method: 'POST',
          headers: geHeaders,
          body: JSON.stringify({
            generators: [
              { name: primaryGenerator, weight: 1.0 },
              { name: 'network_likes', weight: 1.0 },
              { name: 'two_tower', weight: 1.0 }
            ],
            infill: 'popularity',
            user_did: userDid || 'did:plc:z72i7hdynmk6r22z27h6tvur', // fallback DID if none
            num_candidates: 1000
          })
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
                models: [{ name: 'perspective', weight: 1.0 }],
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
        cand: candidates.length, pass: feed.length,
        nextCursor: cursor ? 'y' : 'n',
        gatherMs, ms: Date.now() - t0,
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
          if (interaction.event === 'app.bsky.feed.defs#interactionSeen') continue;
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
