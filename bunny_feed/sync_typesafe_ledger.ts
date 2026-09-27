// Pull the TypeSafe answers the edge prescore job bought into the local pipeline.
//
// The job (src/prescore.ts) records every answer in the live DB's
// typesafe_answers table, which training/score_loc_typesafe.py cannot see. This
// script closes that gap in two ways:
//
//   1. Appends every answer the local ledger (training/data/typesafe_ledger.jsonl)
//      does not hold yet, in the ledger's own row format. That restores the rule
//      the ledger exists for: a post the edge bought is never bought again by an
//      offline run, and vice versa the job already checks its own table.
//   2. Rewrites training/data/corpus_prescore_<tag>.jsonl with every stored post:
//      text, the teacher's answer, and the deployed vector's own score on the same
//      text. That is a labelled corpus of live posts to distil from, and the file
//      the "where does the vector disagree with its teacher" summary below is
//      computed on.
//
// Reads the live DB (PROD_* in bunny_feed/.env), writes only local files. Append
// only for the ledger, so a re-run is safe. --dry-run reports without writing.
//
// Run from bunny_feed/:
//   npx tsx sync_typesafe_ledger.ts [--dry-run]
import * as dotenv from 'dotenv';
import * as path from 'path';
import * as fs from 'fs';
import { createClient } from '@libsql/client';
import { resolveFeed } from './src/feeds';

dotenv.config({ path: path.resolve(__dirname, '.env') });
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const LEDGER = process.env.TYPESAFE_LEDGER
  || path.resolve(__dirname, '../training/data/typesafe_ledger.jsonl');
const DATA = path.resolve(__dirname, '../training/data');
const PAGE = 1000;

// score_loc_typesafe.ANSWER_FIELDS: a row missing any of them does not count as
// paid for there, so it must not be appended here either.
const ANSWER_FIELDS = ['ts_score', 'ts_class', 'ts_class_probs', 'ts_noul'];

async function main() {
  const dryRun = process.argv.includes('--dry-run');
  const url = process.env.PROD_DATABASE_URL;
  const authToken = process.env.PROD_DATABASE_AUTH_TOKEN;
  if (!url || !authToken) {
    throw new Error('Missing PROD_DATABASE_URL / PROD_DATABASE_AUTH_TOKEN (see bunny_feed/.env).');
  }
  const db = createClient({ url, authToken });

  const local = new Set<string>();
  if (fs.existsSync(LEDGER)) {
    for (const line of fs.readFileSync(LEDGER, 'utf-8').split('\n')) {
      if (line.trim()) local.add(JSON.parse(line).uri);
    }
  }
  console.log(`local ledger: ${local.size} posts (${LEDGER})`);

  // Keyset pagination on (created_at, uri): the table only grows, and OFFSET
  // would rescan everything before each page.
  const rows: any[] = [];
  let after: [number, string] = [0, ''];
  for (;;) {
    const res = await db.execute({
      sql: `SELECT uri, text, tag, model, answer, vector_prob, vector_version, post_created_at, created_at
            FROM typesafe_answers
            WHERE created_at > ? OR (created_at = ? AND uri > ?)
            ORDER BY created_at, uri LIMIT ?`,
      args: [after[0], after[0], after[1], PAGE],
    });
    rows.push(...res.rows);
    if (res.rows.length < PAGE) break;
    const last = res.rows[res.rows.length - 1];
    after = [Number(last.created_at), String(last.uri)];
  }
  console.log(`typesafe_answers: ${rows.length} posts`);

  // 1. Ledger.
  const append: string[] = [];
  let malformed = 0;
  for (const r of rows) {
    if (local.has(String(r.uri))) continue;
    const a = JSON.parse(String(r.answer));
    if (ANSWER_FIELDS.some(k => !(k in a))) { malformed++; continue; }
    append.push(JSON.stringify({ uri: r.uri, ...a }));
  }
  console.log(`ledger: ${append.length} new rows to append` + (malformed ? `, ${malformed} malformed skipped` : ''));

  // 2. Corpus, one file per source tag.
  const byTag = new Map<string, any[]>();
  for (const r of rows) {
    const a = JSON.parse(String(r.answer));
    const tag = String(r.tag);
    if (!byTag.has(tag)) byTag.set(tag, []);
    byTag.get(tag)!.push({
      uri: r.uri, text: r.text, tag,
      post_created_at: r.post_created_at, bought_at: r.created_at,
      vector_prob: r.vector_prob, vector_version: r.vector_version,
      ts_class: a.ts_class, ts_class_probs: a.ts_class_probs, ts_score: a.ts_score,
      ts_noul: a.ts_noul, model: a.model,
    });
  }

  // Disagreement at the feed's gate, judged per vector version: a vector_prob is
  // only comparable to the threshold it was calibrated with.
  const feed = resolveFeed('feed-forward');
  const gate = feed.threshold;
  console.log(`\nvector vs teacher at the live gate ${gate} (vector ${feed.scorer.version}):`);
  for (const [tag, rs] of byTag) {
    const cur = rs.filter(r => r.vector_version === feed.scorer.version && r.vector_prob !== null);
    const missedInternal = cur.filter(r => r.ts_class === 'internal' && r.vector_prob < gate).length;
    const leakedExternal = cur.filter(r => r.ts_class === 'external' && r.vector_prob >= gate).length;
    const nInt = cur.filter(r => r.ts_class === 'internal').length;
    const nExt = cur.filter(r => r.ts_class === 'external').length;
    console.log(`  ${tag}: ${cur.length} posts with a current-vector score` +
      ` | teacher-internal the vector rejects: ${missedInternal}/${nInt}` +
      ` | teacher-external the vector admits: ${leakedExternal}/${nExt}`);
  }

  if (dryRun) {
    console.log('\n--dry-run: nothing written.');
    return;
  }
  if (append.length) {
    fs.mkdirSync(path.dirname(LEDGER), { recursive: true });
    fs.appendFileSync(LEDGER, append.join('\n') + '\n');
    console.log(`\nappended ${append.length} rows to ${LEDGER}`);
  }
  for (const [tag, rs] of byTag) {
    const out = path.join(DATA, `corpus_prescore_${tag}.jsonl`);
    fs.writeFileSync(out, rs.map(r => JSON.stringify(r)).join('\n') + '\n');
    console.log(`wrote ${rs.length} posts to ${out}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
