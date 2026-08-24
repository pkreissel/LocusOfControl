// Apply pending drizzle migrations against BunnyDB.
//
// Runs drizzle-orm's migrator directly instead of the `drizzle-kit migrate` CLI.
// The CLI proved unreliable against BunnyDB: under a latency spike it times out
// (~41s) and exits 1 with the error swallowed by its spinner, even when the
// migration set is a clean no-op — whereas the underlying migrate() below just
// takes a few extra seconds and succeeds. This is the same migrator and the same
// __drizzle_migrations bookkeeping the CLI uses, so behavior is identical; we
// only gain a real error message and a retry around BunnyDB's transient stalls.
//
// Credentials come from the environment (BUNNY_DATABASE_URL / _AUTH_TOKEN in CI;
// dotenv loads a local .env when present). Exits non-zero on persistent failure
// so a broken migration fails the deploy loudly.
import { createClient } from '@libsql/client';
import { drizzle } from 'drizzle-orm/libsql';
import { migrate } from 'drizzle-orm/libsql/migrator';
import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: resolve(__dirname, '..', '.env') });

const url = process.env.BUNNY_DATABASE_URL || process.env.BUNNY_DB_URL;
const authToken = process.env.BUNNY_DATABASE_AUTH_TOKEN || process.env.BUNNY_DB_TOKEN;
if (!url || !authToken) {
  console.error('[migrate] missing BUNNY_DATABASE_URL / BUNNY_DATABASE_AUTH_TOKEN');
  process.exit(1);
}

const MAX_ATTEMPTS = 5;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const client = createClient({ url, authToken });
const db = drizzle(client);
const migrationsFolder = resolve(__dirname, '..', 'drizzle');

let lastErr;
for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
  const t = Date.now();
  try {
    await migrate(db, { migrationsFolder });
    console.log(`[migrate] applied/no-op in ${Date.now() - t}ms (attempt ${attempt})`);
    process.exit(0);
  } catch (e) {
    lastErr = e;
    const msg = e?.cause?.message || e?.message || String(e);
    console.warn(`[migrate] attempt ${attempt}/${MAX_ATTEMPTS} failed after ${Date.now() - t}ms: ${msg}`);
    if (attempt < MAX_ATTEMPTS) await sleep(3000 * attempt);
  }
}
console.error(`[migrate] giving up after ${MAX_ATTEMPTS} attempts:`, lastErr?.cause?.message || lastErr?.message || lastErr);
process.exit(1);
