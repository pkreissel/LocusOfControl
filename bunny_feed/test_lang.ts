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
import { looksGerman, readerLangs, prefersGermanOnly, germanFirst } from './src/lang';

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

// --- reader language, as the client sends it ------------------------------
// The two header shapes that actually arrive: the Bluesky app's bare comma list,
// and the weighted form any browser sends. Both must reduce to 2-letter codes,
// because that is what the client's own filter matches on.
console.log('\nreader languages');
const eq = (a: string[], b: string[]) => a.length === b.length && a.every((x, i) => x === b[i]);
check('app form "de"', eq(readerLangs('de'), ['de']));
check('app form "de,en"', eq(readerLangs('de,en'), ['de', 'en']));
check('browser form', eq(readerLangs('de-DE,de;q=0.9,en-US;q=0.8'), ['de', 'en']));
check('absent header', eq(readerLangs(null), []));
check('empty header', eq(readerLangs(''), []));
check('wildcard is not a language', eq(readerLangs('*'), []));

// Narrow on purpose: German is the only language this worker can detect, so it is
// the only preference it acts on. Everyone else keeps today's behaviour.
check('de only -> prefer', prefersGermanOnly(['de']));
check('de,en -> no', !prefersGermanOnly(['de', 'en']));
check('en only -> no', !prefersGermanOnly(['en']));
check('fr only -> no (undetectable, not preferred)', !prefersGermanOnly(['fr']));
check('no languages -> no', !prefersGermanOnly([]));

console.log('\nGerman-first partition');
{
  const list = [
    { text: 'This is an English post about the weather today.', id: 'e1' },
    { text: 'Ich habe das selbst in die Hand genommen und es hat funktioniert.', id: 'd1' },
    { text: 'Another English one, with more of the same words in it.', id: 'e2' },
    { text: 'Die Bundesländer müssen endlich größere Schritte gehen.', id: 'd2' },
  ];
  germanFirst(list);
  check('German moves to the front', eq(list.map(x => x.id), ['d1', 'd2', 'e1', 'e2']),
    list.map(x => x.id).join(','));
}
{
  // Stable inside each group: the ranker's order is the whole point of the feed,
  // so the partition must not reorder within a language.
  const list = [
    { text: 'Die erste deutsche Nachricht mit vielen Wörtern darin.', id: 'd1' },
    { text: 'Und hier ist die zweite deutsche Nachricht für den Test.', id: 'd2' },
  ];
  germanFirst(list);
  check('all-German list is untouched', eq(list.map(x => x.id), ['d1', 'd2']));
}
{
  const list = [{ text: 'Only English here, nothing else in the list at all.', id: 'e1' }];
  germanFirst(list);
  check('all-English list is untouched', eq(list.map(x => x.id), ['e1']));
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
