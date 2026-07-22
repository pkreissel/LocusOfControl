import * as dotenv from 'dotenv';
import * as path from 'path';
dotenv.config({ path: path.resolve(__dirname, '../.env') });
import { createClient } from '@libsql/client/web';
import * as fs from 'fs';

async function main() {
  const url = process.env.BUNNY_DATABASE_URL || process.env.BUNNY_DB_URL;
  const authToken = process.env.BUNNY_DATABASE_AUTH_TOKEN || process.env.BUNNY_DB_TOKEN;

  if (!url || !authToken) throw new Error("Missing env vars");

  const client = createClient({ url, authToken });

  const result = await client.execute("SELECT * FROM feed_interactions");
  fs.writeFileSync('interactions.json', JSON.stringify(result.rows, null, 2));
  console.log(`Saved ${result.rows.length} interactions`);
  process.exit(0);
}
main().catch(console.error);
