// Re-measure looksGerman() against the corpus the routing decision was made on.
//
// src/lang.ts is a port of the detector measured in Python for
// REPORT_minilm_vs_jina_by_lang.md, and the numbers quoted in its header and in
// feeds.ts are that measurement. A word list is easy to "improve" by hand, so the
// point of this file is that any such change has to face the same 1759 labelled
// rows, and the operating point it advertises has to still hold.
//
// Run: npx tsx test_lang.ts
//
// Reads ../dataset_annotated_bonsai_v2.jsonl (gitignored, not in this package) —
// if it is not there the corpus check is skipped and the unit cases still run.

import * as fs from 'fs';
import * as path from 'path';
import { looksGerman } from './src/lang';

let failures = 0;
function check(name: string, ok: boolean, detail = '') {
  console.log(`${ok ? '  ok  ' : '  FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures++;
}

console.log('unit cases');
const GERMAN = [
  'Ich habe das selbst in die Hand genommen und es hat funktioniert.',
  'Die Bundesländer müssen endlich größere Schritte gehen.',
  'Vor 11 Jahren sah so die "Flüchtlingskrise" aus. Menschen, die Hilfe suchten & sie bekamen',
  'Heute wieder angefangen zu laufen. Fühlt sich gut an!',
];
const ENGLISH = [
  'I took care of it myself and it worked out fine.',
  'The government must finally take bigger steps on this.',
  'Just finished a 5k. Feels good to be back at it.',
  'Reading Kafka in Köln this week, then back to the office.',  // a German name, English text
];
for (const t of GERMAN) check(`german: ${t.slice(0, 40)}…`, looksGerman(t));
for (const t of ENGLISH) check(`english: ${t.slice(0, 40)}…`, !looksGerman(t));
check('empty string is not German', !looksGerman(''));
check('null is not German', !looksGerman(null));
check('emoji-only is not German', !looksGerman('👇👇👇 🔥'));
check('bare URL is not German', !looksGerman('https://example.com/über/uns'));

const corpus = path.join(__dirname, '..', 'dataset_annotated_bonsai_v2.jsonl');
if (!fs.existsSync(corpus)) {
  console.log(`\ncorpus check skipped: ${corpus} not found`);
} else {
  console.log('\nagainst declared language tags in dataset_annotated_bonsai_v2.jsonl');
  let dePos = 0, deTot = 0, enPos = 0, enTot = 0;
  for (const line of fs.readFileSync(corpus, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (!row.text) continue;
    const tag = (row.langs?.[0] ?? '').split('-')[0];
    if (tag !== 'de' && tag !== 'en') continue;
    const flagged = looksGerman(row.text);
    if (tag === 'de') { deTot++; if (flagged) dePos++; } else { enTot++; if (flagged) enPos++; }
  }
  const recall = dePos / deTot;
  const fpr = enPos / enTot;
  const precision = dePos / (dePos + enPos);
  const acc = (dePos + (enTot - enPos)) / (deTot + enTot);
  console.log(`  n=${deTot + enTot} (${deTot} de, ${enTot} en)`);
  // Thresholds sit just under the measured values (0.955 / 0.858 / 0.908 / 0.021),
  // so drift is caught but the bootstrap-sized wobble of a reworded list is not.
  check('accuracy >= 0.94', acc >= 0.94, acc.toFixed(3));
  check('German recall >= 0.83', recall >= 0.83, recall.toFixed(3));
  check('precision >= 0.88', precision >= 0.88, precision.toFixed(3));
  check('English false-flag rate <= 0.04', fpr <= 0.04, fpr.toFixed(3));
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
