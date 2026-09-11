// Run the SHIPPED language detector over a corpus and write uri -> flag.
//
// The offline evaluation needs to group posts exactly as the feed will route
// them, and the only way to guarantee that is to ask the same code. A Python
// re-implementation would be a second detector that drifts from this one silently
// — which is precisely the failure the numbers would then be blind to.
//
// Run: npx tsx dump_lang_flags.ts ../dataset_annotated_bonsai_v2.jsonl ../lang_flags_bonsai_v2.json
//      npx tsx dump_lang_flags.ts ../dataset_validation_gemini.jsonl ../lang_flags_gemini.json

import * as fs from 'fs';
import { looksGerman } from './src/lang';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: tsx dump_lang_flags.ts <corpus.jsonl> <flags.json>');
  process.exit(2);
}

const flags: Record<string, boolean> = {};
let n = 0, flagged = 0;
for (const line of fs.readFileSync(input, 'utf8').split('\n')) {
  if (!line.trim()) continue;
  const row = JSON.parse(line);
  if (!row.uri || !row.text) continue;
  const de = looksGerman(row.text);
  flags[row.uri] = de;
  n++;
  if (de) flagged++;
}
fs.writeFileSync(output, JSON.stringify(flags));
console.log(`${input}: ${n} rows, ${flagged} flagged German (${(flagged / n * 100).toFixed(1)}%) -> ${output}`);
