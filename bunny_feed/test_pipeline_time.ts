import * as fs from 'fs';
import * as path from 'path';

// Parse .env manually for local testing
const envFile = fs.readFileSync(path.join(__dirname, '.env'), 'utf8');
const env = envFile.split('\n').reduce((acc, line) => {
  const match = line.match(/^([^=]+)=(.*)$/);
  if (match) acc[match[1]] = match[2].replace(/^"|"$/g, '').trim();
  return acc;
}, {} as Record<string, string>);

const JINA_API_KEY = env['JINA_API_KEY'];
const GREEN_EARTH_API_KEY = env['GREEN_EARTH_API_KEY'] || 'MISSING';
const GENERATOR_NAME = env['GREEN_EARTH_GENERATOR'] || 'following';
const RANK_MODEL = env['GREEN_EARTH_RANK_MODEL'] || 'locus-of-control';
const USER_DID = 'did:plc:z72i7hdynmk6r22z27h6tvur'; // example DID

if (!JINA_API_KEY) throw new Error("Missing JINA_API_KEY in .env");

// We'll mock the internal vector/Jina imports here for simplicity,
// or we can import them directly if the environment is set up.
import { getEmbeddings } from './src/jina';
import { scoreEmbedding } from './src/vector';

const VECTOR_PROB_THRESHOLD = 0.35;

async function runTest() {
  if (GREEN_EARTH_API_KEY === 'MISSING') {
    console.error("Please add GREEN_EARTH_API_KEY to bunny_feed/.env to run this test.");
    return;
  }

  const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': GREEN_EARTH_API_KEY
  };

  console.log("--- Starting Pipeline Timing Test ---");
  const overallStart = Date.now();

  // Discover available generators and models
  console.log("Discovering available Green Earth Generators and Models...");
  const genListRes = await fetch('https://api.greenearth.social/candidates/generators', { headers });
  const genList = genListRes.ok ? await genListRes.json() : { generators: [] };
  console.log("Available Generators:", genList.generators);

  const modelListRes = await fetch('https://api.greenearth.social/rank/models', { headers });
  const modelList = modelListRes.ok ? await modelListRes.json() : { rankers: [] };
  console.log("Available Models:", modelList.rankers);

  // Hardcode selections based on user request
  let genToUse = 'followed_users';
  let modelToUse = 'perspective';
  
  // 1. Generate Candidates
  let tStart = Date.now();
  console.log(`[1] Fetching candidates via GE /candidates/generate (Generator: ${genToUse})...`);
  const candRes = await fetch('https://api.greenearth.social/candidates/generate', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      generators: [{ name: genToUse, weight: 1.0 }],
      user_did: USER_DID,
      num_candidates: 1000
    })
  });
  if (!candRes.ok) {
    throw new Error(`GE Generate failed: ${candRes.status} ${await candRes.text()}`);
  }
  const candData = await candRes.json();
  const candidates = candData.candidates || [];
  const tGenerate = Date.now() - tStart;
  console.log(`    -> Fetched ${candidates.length} candidates in ${tGenerate}ms`);
  
  if (candidates.length > 0) {
      console.log(`    -> Example candidate keys: ${Object.keys(candidates[0]).join(', ')}`);
      console.log(`    -> Example minilm_l12_embedding present: ${!!candidates[0].minilm_l12_embedding}`);
  }

  // 2. Classify (Local Jina + Vector)
  tStart = Date.now();
  console.log(`[2] Computing Jina embeddings and scoring...`);
  // Batch in chunks if necessary, but 300 should be fine for Jina if we just map texts
  const texts = candidates.map((c: any) => c.content || '');
  
  // Note: we aren't caching here for the raw test to measure worst-case latency
  const jinaRes = await getEmbeddings(texts, JINA_API_KEY);
  const embeddings = jinaRes.embeddings;
  const filteredCandidates = [];
  
  for (let i = 0; i < candidates.length; i++) {
    const score = scoreEmbedding(embeddings[i]);
    if (score.probability >= VECTOR_PROB_THRESHOLD) {
      filteredCandidates.push(candidates[i]);
    }
  }
  const tClassify = Date.now() - tStart;
  console.log(`    -> Kept ${filteredCandidates.length}/${candidates.length} candidates in ${tClassify}ms`);

  if (filteredCandidates.length === 0) {
    console.log("No candidates survived filtering. Aborting further steps.");
    return;
  }

  // 3. Rank (GE)
  tStart = Date.now();
  console.log(`[3] Ranking via GE /rank/predict (Model: ${modelToUse})...`);
  const rankRes = await fetch('https://api.greenearth.social/rank/predict', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      candidates: filteredCandidates,
      models: [{ name: modelToUse, weight: 1.0 }],
      user_did: USER_DID
    })
  });
  if (!rankRes.ok) {
      // If it fails (e.g. model not found), log it but don't crash
      console.error(`    -> GE Rank failed: ${rankRes.status} ${await rankRes.text()}`);
  } else {
      const rankData = await rankRes.json();
      const tRank = Date.now() - tStart;
      console.log(`    -> Ranked ${rankData.rankings?.length || 0} candidates in ${tRank}ms`);
      if (rankData.rankings && rankData.rankings.length > 0) {
          console.log(`    -> Example ranker output keys: ${Object.keys(rankData.rankings[0]).join(', ')}`);
          
          // Re-order our filteredCandidates based on the returned rank
          const rankedUris = rankData.rankings.map((r: any) => r.at_uri);
          filteredCandidates.sort((a: any, b: any) => {
              const idxA = rankedUris.indexOf(a.at_uri);
              const idxB = rankedUris.indexOf(b.at_uri);
              // If not found in rankings (shouldn't happen), push to end
              return (idxA === -1 ? 9999 : idxA) - (idxB === -1 ? 9999 : idxB);
          });
      }
  }

  // 4. Diversify (GE)
  // We can only diversify if we have minilm_l12_embedding, let's check if candidates have it.
  const hasMinilm = filteredCandidates.some((c: any) => !!c.minilm_l12_embedding);
  if (hasMinilm) {
    tStart = Date.now();
    console.log(`[4] Diversifying via GE /diversify...`);
    const divRes = await fetch('https://api.greenearth.social/diversify', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        candidates: filteredCandidates
      })
    });
    if (!divRes.ok) {
        console.error(`    -> GE Diversify failed: ${divRes.status} ${await divRes.text()}`);
    } else {
        const divData = await divRes.json();
        const tDiv = Date.now() - tStart;
        console.log(`    -> Diversified ${divData.candidates?.length || 0} candidates in ${tDiv}ms`);
    }
  } else {
    console.log(`[4] Skipping GE /diversify because candidates lack minilm_l12_embedding.`);
  }

  const overallTime = Date.now() - overallStart;
  console.log(`--- Test Complete. Total Pipeline Time: ${overallTime}ms ---`);
}

runTest().catch(console.error);
