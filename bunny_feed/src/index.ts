import { BskyAgent } from '@atproto/api';
import { jwtDecode } from 'jwt-decode';
import { getEmbeddings } from './jina';
import { scoreEmbedding } from './vector';
import { fetchCandidatePosts } from './bsky';
import { saveUserSession, getUserSession } from './db';

const VECTOR_PROB_THRESHOLD = 0.15;

declare const process: any;

// Bunny Edge Script Entrypoint
// @ts-ignore
addEventListener('fetch', (event: any) => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);

  // --- OAUTH / LOGIN ENDPOINT ---
  // Users send a POST request with { "handle": "...", "password": "..." }
  // to authenticate, representing our App Password flow (or an OAuth2 code exchange).
  if (url.pathname === '/oauth/login' && request.method === 'POST') {
    try {
      const body = await request.json() as any;
      if (!body.handle || !body.password) {
        return new Response(JSON.stringify({ error: "Missing handle or password" }), { status: 400 });
      }

      const agent = new BskyAgent({ service: 'https://bsky.social' });
      const { data } = await agent.login({
        identifier: body.handle,
        password: body.password
      });

      // Save token to Bunny DB
      await saveUserSession(data.did, data.handle, data.accessJwt, data.refreshJwt);

      return new Response(JSON.stringify({ 
        success: true, 
        message: 'Successfully authenticated and stored token in Bunny Database',
        did: data.did,
        handle: data.handle
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: "Authentication failed", details: e.message }), { status: 401 });
    }
  }

  // --- FEED GENERATOR SKELETON ENDPOINT ---
  if (url.pathname === '/xrpc/app.bsky.feed.getFeedSkeleton') {
    try {
      const apiKey = typeof process !== 'undefined' && process.env ? process.env.JINA_API_KEY : 'MISSING_API_KEY';
      if (!apiKey || apiKey === 'MISSING_API_KEY') {
         return new Response(JSON.stringify({ error: "Missing Jina API Key configuration" }), { status: 500 });
      }

      // Extract User DID from ATProto Service JWT
      const authHeader = request.headers.get('Authorization');
      let userDid = null;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const decoded: any = jwtDecode(authHeader.substring(7));
          userDid = decoded.iss; // Service JWT issuer is the user's DID
        } catch (e) {
          console.error("Invalid JWT", e);
        }
      }

      // Initialize BskyAgent. If user is logged in, use their session to avoid rate limits
      const agent = new BskyAgent({ service: 'https://bsky.social' });
      if (userDid) {
        try {
          const session = await getUserSession(userDid);
          if (session) {
            await agent.resumeSession({
              did: userDid,
              handle: session.handle,
              accessJwt: session.accessJwt,
              refreshJwt: session.refreshJwt,
              email: '',
              active: true
            });
          }
        } catch (e) {
           console.warn("Failed to load user session from DB", e);
        }
      }

      // 1. Fetch candidates from the public feed
      const candidates = await fetchCandidatePosts(agent, 50);
      if (candidates.length === 0) {
        return new Response(JSON.stringify({ feed: [] }), {
           headers: { 'Content-Type': 'application/json' }
        });
      }

      // 2. Fetch embeddings for candidate texts
      const texts = candidates.map(c => c.text);
      const embeddings = await getEmbeddings(texts, apiKey);

      // 3. Score and filter candidates
      const feed = [];
      for (let i = 0; i < candidates.length; i++) {
        const score = scoreEmbedding(embeddings[i]);
        if (score.probability >= VECTOR_PROB_THRESHOLD) {
          feed.push({ post: candidates[i].uri });
        }
      }

      // 4. Return ATProto Skeleton structure
      return new Response(JSON.stringify({ feed }), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: e.message }), { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  return new Response("Bunny ATProto Feed Generator", { status: 200 });
}
