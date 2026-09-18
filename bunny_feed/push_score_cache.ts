// Warm the live feed's score cache from a locally computed cache.
//
// score_loc_vector.py scores posts with the deployed vector from local Jina
// weights; this pushes those scores into `post_scores` so the feed serves them as
// cache hits instead of paying Jina for the same embedding. A post the feed has
// never seen arrives pre-scored.
//
// Faithful to what the worker would have computed: measured on 64 Silberskeets
// posts, local weights vs the Jina API agreed to mean |Δprob| 0.0007 / max
// 0.0035 (r = 0.99983), and the 0.88 gate's verdict flipped on none of them.
// Re-measure if the embedding model or its task ever changes.
//
// The version string and the scorer fingerprint are imported from the worker's
// own modules, never restated here: (uri, version) is the cache key, and a key
// this script derived independently could drift from the one the feed reads.
//
// NOTE ON LIFETIME: the feed prunes `created_at < now - 4 days`
// (SCORE_CACHE_MAX_AGE_MS), so rows are written with the push time and a warmed
// cache decays within four days. This is a warmer, not a durable store.
//
// Dry run by default — it reports what would change and writes nothing. Pass
// --commit to write.
//
// Run from bunny_feed/:
//   npx tsx push_score_cache.ts ../scores_loc_vector_silberskeets.jsonl
//   npx tsx push_score_cache.ts ../scores_loc_vector_silberskeets.jsonl --commit
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as fs from 'fs';
import { createClient } from '@libsql/client';
import { EMBEDDING_MODEL } from './src/jina';
import { VECTOR_VERSION } from './src/vector';

// bunny_feed/.env first (it holds the PROD_* pair), repo root second.
dotenv.config({ path: path.resolve(__dirname, '.env') });
dotenv.config({ path: path.resolve(__dirname, '../.env') });

// Same chunk the worker writes in: 200 rows x 4 columns = 800 bound params,
// under SQLite's ~999 per-statement limit.
const WRITE_CHUNK = 200;

type CachedScore = {
  uri: string;
  vector_prob: number;
  vector_version: string;
  embedding_model?: string;
  /**
   * What computed `vector_prob`. Absent means the vector itself did
   * (score_loc_vector.py) — the ordinary case, and what the cache is documented
   * to hold. `typesafe_calibrated` means a teacher judgement quantile-matched
   * onto this vector's probability scale (calibrate_typesafe_to_vector.py): the
   * scale is the vector's, so the feed's threshold still means what it means,
   * but the ORDER is the teacher's and no column in post_scores records that.
   * Printed on every run because this is the only place it is ever visible.
   */
  score_source?: string;
};

function readCache(file: string): CachedScore[] {
  return fs.readFileSync(file, 'utf-8')
    .split('\n')
    .filter(l => l.trim())
    .map(l => JSON.parse(l));
}

async function main() {
  const args = process.argv.slice(2);
  const commit = args.includes('--commit');
  const file = args.find(a => !a.startsWith('--'));
  if (!file) throw new Error('Usage: npx tsx push_score_cache.ts <scores_loc_vector_*.jsonl> [--commit]');

  // The live DB is the Turso one behind PROD_*. BUNNY_DATABASE_URL names a
  // database the feed no longer uses, and writing the cache there would look
  // like success while changing nothing a reader sees.
  const url = process.env.PROD_DATABASE_URL;
  const authToken = process.env.PROD_DATABASE_AUTH_TOKEN;
  if (!url || !authToken) {
    throw new Error('Missing PROD_DATABASE_URL / PROD_DATABASE_AUTH_TOKEN (see bunny_feed/.env). ' +
      'BUNNY_DATABASE_URL is a different, unused database — not a fallback.');
  }

  const version = `${EMBEDDING_MODEL}:${VECTOR_VERSION}`;
  const rows = readCache(file);
  console.log(`${rows.length} cached scores in ${path.basename(file)}`);
  console.log(`target cache version: ${version}`);

  const bySource = new Map<string, number>();
  for (const r of rows) {
    const src = r.score_source ?? 'vector';
    bySource.set(src, (bySource.get(src) ?? 0) + 1);
  }
  for (const [src, n] of bySource) {
    console.log(src === 'vector'
      ? `score source: ${src} (${n}) — what the worker would have computed itself`
      : `score source: ${src} (${n}) — NOT the vector's own number. The feed will ` +
        `rank these posts by that source until the rows are pruned or overwritten.`);
  }

  // A row scored by a different vector cannot be filed under this version: the
  // fingerprint in the cache is exactly what the worker's would be, so a
  // mismatch means the artifact moved after scoring and the file needs a rerun.
  const mismatched = rows.filter(r => r.vector_version !== VECTOR_VERSION);
  const usable = rows.filter(r =>
    r.vector_version === VECTOR_VERSION && Number.isFinite(r.vector_prob));
  if (mismatched.length) {
    console.log(`skipping ${mismatched.length} rows scored by another vector ` +
      `(${[...new Set(mismatched.map(r => r.vector_version))].join(', ')}) — ` +
      `re-run score_loc_vector.py to refresh them`);
  }
  if (!usable.length) throw new Error('No rows match this vector; nothing to push.');

  const client = createClient({ url, authToken });

  // What is already there, so a dry run can say what the write would actually
  // change rather than just how many rows it would touch.
  const existing = new Map<string, number>();
  for (let i = 0; i < usable.length; i += WRITE_CHUNK) {
    const chunk = usable.slice(i, i + WRITE_CHUNK);
    const res = await client.execute({
      sql: `SELECT uri, score FROM post_scores WHERE version = ? AND uri IN (${chunk.map(() => '?').join(',')})`,
      args: [version, ...chunk.map(r => r.uri)],
    });
    for (const row of res.rows) existing.set(row.uri as string, row.score as number);
  }
  const fresh = usable.filter(r => !existing.has(r.uri));
  const changed = usable.filter(r => {
    const prev = existing.get(r.uri);
    return prev !== undefined && Math.abs(prev - r.vector_prob) > 1e-9;
  });
  console.log(`${usable.length} pushable: ${fresh.length} new, ${existing.size} already cached ` +
    `(${changed.length} with a different score)`);

  const total = await client.execute({
    sql: 'SELECT COUNT(*) AS n FROM post_scores WHERE version = ?', args: [version],
  });
  console.log(`post_scores currently holds ${total.rows[0].n} rows for this version`);

  if (!commit) {
    console.log('\nDry run — nothing written. Re-run with --commit to write.');
    process.exit(0);
  }

  // created_at is the push time, not the post's date: the prune is a range delete
  // on created_at, so backdating a row would make it instantly prunable.
  const now = Date.now();
  let written = 0;
  for (let i = 0; i < usable.length; i += WRITE_CHUNK) {
    const chunk = usable.slice(i, i + WRITE_CHUNK);
    await client.execute({
      sql: `INSERT INTO post_scores (uri, score, version, created_at) VALUES ` +
        chunk.map(() => '(?, ?, ?, ?)').join(', ') +
        // Conflict target matches the (uri, version) primary key, exactly as
        // putScores does — keyed on uri alone, another feed's vector would have
        // its score overwritten instead of sitting alongside this one.
        ` ON CONFLICT(uri, version) DO UPDATE SET score = excluded.score, created_at = excluded.created_at`,
      args: chunk.flatMap(r => [r.uri, r.vector_prob, version, now]),
    });
    written += chunk.length;
    console.log(`  wrote ${written}/${usable.length}`);
  }

  const after = await client.execute({
    sql: 'SELECT COUNT(*) AS n FROM post_scores WHERE version = ?', args: [version],
  });
  console.log(`\nDone. post_scores now holds ${after.rows[0].n} rows for ${version}.`);
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
