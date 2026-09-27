// Warm the score cache for a retrained vector BEFORE it is deployed.
//
// post_scores is keyed (uri, version), and the version carries the vector's
// fingerprint, so a retrain makes every cached row unreachable at once: the first
// hours after the deploy pay Jina for every shortlist again, and the prescore
// job's teacher scores are simply gone. This script copies the reachable part of
// the cache across to the new version while the old one is still serving:
//
//   teacher rows   every typesafe_answers post still young enough to be a
//                  candidate is re-mapped from its stored answer through the NEW
//                  prescore_calibration.json. Free: the answers do not depend on
//                  the vector, only the ladder they are mapped onto does.
//   vector rows    every other post in the old version's recent rows is fetched
//                  again from the AppView (texts are not cached), embedded through
//                  the Jina API and scored with the new vector. This costs what the
//                  cold cache would have cost, paid now instead of inside readers'
//                  requests. Posts deleted since are skipped.
//
// The new rows sit under a version the live worker does not read, so writing them
// early changes nothing until the deploy switches versions.
//
// Order, in the same commit as the retrain (CLAUDE.md: a vector export and its
// recalibration ship together):
//   1. export the new vector into src/vector.json (this script reads it)
//   2. re-score the calibration pool with the new vector and re-run
//      training/export_prescore_assets.py, so prescore_calibration.json names it
//   3. npx tsx rewarm_cache.ts            (dry run: counts only, nothing billed)
//      npx tsx rewarm_cache.ts --commit   (fetches, embeds, writes)
//   4. push; the deploy switches the feed to the warm version
//
// Run from bunny_feed/. Needs PROD_* and JINA_API_KEY in bunny_feed/.env.
import * as dotenv from 'dotenv';
import * as path from 'path';
import { createClient } from '@libsql/client';
import { resolveFeed, scoreCacheVersion } from './src/feeds';
import { getEmbeddingsBatched } from './src/jina';
import { answerKey, calibrate } from './src/prescore';
import calibration from './src/prescore_calibration.json';

dotenv.config({ path: path.resolve(__dirname, '.env') });
dotenv.config({ path: path.resolve(__dirname, '../.env') });

// A post older than GE's 72h window can never be a candidate again.
const WINDOW_MS = 72 * 60 * 60 * 1000;
const GET_POSTS_MAX = 25;          // app.bsky.feed.getPosts per call
const WRITE_CHUNK = 200;           // 200 rows x 4 columns, under 999 parameters

async function fetchTexts(uris: string[]): Promise<Map<string, string>> {
  const out = new Map<string, string>();
  for (let i = 0; i < uris.length; i += GET_POSTS_MAX) {
    const q = new URLSearchParams();
    for (const u of uris.slice(i, i + GET_POSTS_MAX)) q.append('uris', u);
    const r = await fetch(`https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?${q}`);
    if (!r.ok) { console.warn(`  getPosts ${r.status}, skipping ${GET_POSTS_MAX} posts`); continue; }
    const data: any = await r.json();
    for (const p of data.posts ?? []) {
      const text = p?.record?.text;
      if (typeof text === 'string' && text.trim()) out.set(p.uri, text);
    }
  }
  return out;
}

async function main() {
  const commit = process.argv.includes('--commit');
  const url = process.env.PROD_DATABASE_URL;
  const authToken = process.env.PROD_DATABASE_AUTH_TOKEN;
  if (!url || !authToken) throw new Error('Missing PROD_DATABASE_URL / PROD_DATABASE_AUTH_TOKEN (see bunny_feed/.env).');
  const db = createClient({ url, authToken });

  const feed = resolveFeed('feed-forward');
  const target = scoreCacheVersion(feed.scorer);
  const since = Date.now() - WINDOW_MS;
  console.log(`target version (src/vector.json): ${target}`);

  const versions = await db.execute({
    sql: 'SELECT version, COUNT(*) AS n FROM post_scores WHERE created_at > ? GROUP BY version ORDER BY n DESC',
    args: [since],
  });
  for (const v of versions.rows) console.log(`  post_scores, last 72h: ${v.version}  ${v.n} rows`);
  const sources = versions.rows.map(v => String(v.version)).filter(v => v !== target);
  if (!sources.length) {
    console.log('Nothing to rewarm: no recent rows under any other version.');
    return;
  }

  const already = new Set((await db.execute({
    sql: 'SELECT uri FROM post_scores WHERE version = ? AND created_at > ?', args: [target, since],
  })).rows.map(r => String(r.uri)));

  // Teacher rows: re-mapped from the stored answers, never re-bought.
  const answers = (await db.execute({
    sql: 'SELECT uri, answer FROM typesafe_answers WHERE COALESCE(post_created_at, created_at) > ?',
    args: [since],
  })).rows;
  const teacherUris = new Set(answers.map(r => String(r.uri)));
  const calibrationOk = calibration.vector_version === feed.scorer.version;
  const teacherRows = !calibrationOk ? [] : answers
    .filter(r => !already.has(String(r.uri)))
    .map(r => ({ uri: String(r.uri), score: calibrate(answerKey(JSON.parse(String(r.answer)))) }));
  if (!calibrationOk) {
    console.log(`\n! prescore_calibration.json is for vector ${calibration.vector_version}, not ` +
      `${feed.scorer.version}: skipping ${answers.length} teacher rows. Re-run ` +
      `training/export_prescore_assets.py first (see step 2 above).`);
  }

  // Vector rows: everything else the old versions hold, re-embedded.
  const placeholders = sources.map(() => '?').join(',');
  const old = (await db.execute({
    sql: `SELECT DISTINCT uri FROM post_scores WHERE version IN (${placeholders}) AND created_at > ?`,
    args: [...sources, since],
  })).rows.map(r => String(r.uri));
  const toEmbed = old.filter(u => !already.has(u) && !teacherUris.has(u));

  console.log(`\nalready under the target version: ${already.size}`);
  console.log(`teacher rows to re-map (free):    ${teacherRows.length}`);
  console.log(`vector rows to re-embed (Jina):   ${toEmbed.length}`);
  if (!commit) {
    console.log('\nDry run: nothing fetched, billed or written. Pass --commit.');
    return;
  }

  const jinaKey = process.env.JINA_API_KEY;
  if (toEmbed.length && !jinaKey) throw new Error('Missing JINA_API_KEY (see bunny_feed/.env).');

  const vectorRows: { uri: string; score: number }[] = [];
  if (toEmbed.length) {
    const texts = await fetchTexts(toEmbed);
    console.log(`\nfetched ${texts.size} of ${toEmbed.length} texts (the rest were deleted or empty)`);
    const uris = [...texts.keys()];
    const res = await getEmbeddingsBatched(uris.map(u => texts.get(u)!), jinaKey!);
    res.embeddings.forEach((e, i) => { if (e) vectorRows.push({ uri: uris[i], score: feed.scorer.score(e).probability }); });
    console.log(`embedded ${vectorRows.length} (${res.failedChunks} failed chunks, ${res.tokens} tokens)`);
  }

  const all = [...teacherRows, ...vectorRows];
  const now = Date.now();
  for (let i = 0; i < all.length; i += WRITE_CHUNK) {
    const chunk = all.slice(i, i + WRITE_CHUNK);
    await db.execute({
      sql: `INSERT INTO post_scores (uri, score, version, created_at) VALUES ` +
        chunk.map(() => '(?, ?, ?, ?)').join(', ') +
        ` ON CONFLICT(uri, version) DO UPDATE SET score = excluded.score, created_at = excluded.created_at`,
      args: chunk.flatMap(r => [r.uri, r.score, target, now]),
    });
  }
  console.log(`\nwrote ${all.length} rows under ${target} (${teacherRows.length} teacher, ${vectorRows.length} vector).`);
}

main().catch(e => { console.error(e); process.exit(1); });
