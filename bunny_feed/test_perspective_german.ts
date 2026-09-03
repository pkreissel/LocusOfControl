import * as fs from 'fs';
import * as path from 'path';

// Parse .env manually for local testing
const envFile = fs.readFileSync(path.join(process.cwd(), '.env'), 'utf8');
const env = envFile.split('\n').reduce((acc, line) => {
  const match = line.match(/^([^=]+)=(.*)$/);
  if (match) acc[match[1]] = match[2].replace(/^"|"$/g, '').trim();
  return acc;
}, {} as Record<string, string>);

const GREEN_EARTH_API_KEY = env['GREEN_EARTH_API_KEY'];

if (!GREEN_EARTH_API_KEY) {
  console.error("Missing GREEN_EARTH_API_KEY in .env");
  process.exit(1);
}

const headers = {
  'Content-Type': 'application/json',
  'X-API-Key': GREEN_EARTH_API_KEY
};

async function testPerspectiveGerman() {
  const candidates = [
    {
      at_uri: 'at://did:plc:m2uq4xp53ln6ajjhjg5putln/app.bsky.feed.post/3mqmqiofk7k24',
      content: 'Und wer von der FPÖ versorgt wird, wird von Steuergeld versorgt. WIR zahlen dafür. Und das nicht nur mit Geld.\n\nÖsterreich versagt komplett in Bezug auf den antifaschistischen Auftrag in unserer Verfassung.\n\nDanke @colettemschmidt.bsky.social und @fabianschmid.bsky.social fürs nicht locker lassen!',
      score: 1.0,
      author_did: 'did:plc:m2uq4xp53ln6ajjhjg5putln'
    },
    {
      at_uri: 'at://did:plc:aqystvoyxf2yjxdele473aan/app.bsky.feed.post/3mqmrxzr7mc2h',
      content: 'Arschloch wer behauptet Retrogames wären sein Hobby und er (gendern nicht nötig) hat dann kein Roland MT-32 im Haus um die Soundtracks abzuspielen. Meine Meinung!',
      score: 1.0,
      author_did: 'did:plc:aqystvoyxf2yjxdele473aan'
    },
    {
      at_uri: 'at://did:plc:u6wpahdxudi2unba3go2nbsv/app.bsky.feed.post/3mqmrxkfvwk2k',
      content: 'The IDIOT Was The President In 2020 ‼️😡🚩 Who Was President November 2020 ? If The Elections Weren’t Secure It Was The tDUMP’S Fault ‼️🚩😡 STOP 🛑 THE NONSENSE',
      score: 1.0,
      author_did: 'did:plc:u6wpahdxudi2unba3go2nbsv'
    }
  ];

  const requestBody = {
    candidates: candidates,
    models: [{ name: 'perspective', weight: 1.0 }],
    user_did: 'did:plc:z72i7hdynmk6r22z27h6tvur' // Valid DID format required by API
  };

  console.log("Input Candidates Order:");
  candidates.forEach((c, idx) => {
    console.log(` ${idx + 1}. [${c.at_uri}] ${c.content}`);
  });
  console.log("\nSending German candidates to /rank/predict...");
  
  const res = await fetch('https://api.greenearth.social/rank/predict', {
    method: 'POST',
    headers,
    body: JSON.stringify(requestBody)
  });

  if (!res.ok) {
    console.error(`Request failed: ${res.status} ${await res.text()}`);
    return;
  }

  const data = await res.json();
  
  if (data.rankings && data.rankings.length > 0) {
      console.log("\nOutput Rankings Order:");
      data.rankings.forEach((r: any) => {
          const original = candidates.find(c => c.at_uri === r.at_uri);
          console.log(` ${r.rank}. [${r.at_uri}] Score: ${r.rank_score} | Content: ${original?.content}`);
      });

      // Check if order changed
      const initialUris = candidates.map(c => c.at_uri);
      const outputUris = data.rankings.map((r: any) => r.at_uri);
      
      let changed = false;
      for (let i = 0; i < initialUris.length; i++) {
        if (initialUris[i] !== outputUris[i]) {
          changed = true;
          break;
        }
      }

      console.log(`\nDid perspective ranker change the order for German posts? ${changed ? 'YES' : 'NO'}`);
  } else {
      console.log("\nNo rankings returned:", JSON.stringify(data, null, 2));
  }
}

testPerspectiveGerman().catch(console.error);
