import * as fs from 'fs';
import * as path from 'path';

// Parse .env manually for local testing
const envFile = fs.readFileSync(path.join(__dirname, '.env'), 'utf8');
const env = envFile.split('\n').reduce((acc, line) => {
  const match = line.match(/^([^=]+)=(.*)$/);
  if (match) acc[match[1]] = match[2].replace(/^"|"$/g, '').trim();
  return acc;
}, {} as Record<string, string>);

const GREEN_EARTH_API_KEY = env['GREEN_EARTH_API_KEY'] || 'MISSING';
const USER_DID = 'did:plc:z72i7hdynmk6r22z27h6tvur'; // example DID

async function runTest() {
  if (GREEN_EARTH_API_KEY === 'MISSING') {
    console.error("Please add GREEN_EARTH_API_KEY to bunny_feed/.env to run this test.");
    return;
  }

  const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': GREEN_EARTH_API_KEY
  };

  const generatorsToTest = ['followed_users', 'popularity', 'network_likes', 'two_tower'];
  const NUM_CANDIDATES = 500;

  console.log(`--- Testing Individual Generators (Target: ${NUM_CANDIDATES} candidates) ---`);

  for (const gen of generatorsToTest) {
    console.log(`\nTesting generator: ${gen}...`);
    const tStart = Date.now();
    
    try {
      const res = await fetch('https://api.greenearth.social/candidates/generate', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          generators: [{ name: gen, weight: 1.0 }],
          user_did: USER_DID,
          num_candidates: NUM_CANDIDATES
        })
      });

      if (!res.ok) {
        console.error(`  -> Failed: ${res.status} ${await res.text()}`);
        continue;
      }

      const data = await res.json();
      const candidates = data.candidates || [];
      const duration = Date.now() - tStart;
      
      console.log(`  -> Fetched ${candidates.length} candidates in ${duration}ms`);
    } catch (e: any) {
      console.error(`  -> Error fetching ${gen}:`, e.message);
    }
  }
  
  console.log('\n--- Done ---');
}

runTest().catch(console.error);
