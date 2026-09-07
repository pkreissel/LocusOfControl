#!/bin/zsh
# Progress of the assist re-annotation run (run_assist_reannotate.sh).
# The output .jsonl files are the durable source of truth: the annotator appends
# and flushes one line per completed post, and resumes by URI, so line counts
# survive restarts and outlive any log file.
cd "$(dirname "$0")"

LOG=${LOC_LOG:-$(ls -t /private/tmp/claude-501/*/*/scratchpad/reannotate.log 2>/dev/null | head -1)}

progress () {  # name, output file, input file
  local done=0 total=0
  [ -f "$2" ] && done=$(wc -l < "$2" | tr -d ' ')
  [ -f "$3" ] && total=$(wc -l < "$3" | tr -d ' ')
  if [ "$total" -gt 0 ]; then
    printf "  %-28s %5d / %-5d (%d%%)\n" "$1" "$done" "$total" $((done * 100 / total))
  else
    printf "  %-28s %5d / ?\n" "$1" "$done"
  fi
}

echo "=== assist re-annotation ==="
if pgrep -f annotate_lmstudio_soft_labels >/dev/null; then
  echo "  status: RUNNING (pid $(pgrep -f annotate_lmstudio_soft_labels | head -1))"
else
  echo "  status: not running"
fi
progress "old-corpus requeue" dataset_annotated_assist_requeue.jsonl dataset_raw_old_requeue.jsonl
progress "firehose help sample" dataset_annotated_assist_help.jsonl dataset_raw_help.jsonl

if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  echo "\n  log: $LOG"
  echo "  failures so far: $(tr '\r' '\n' < "$LOG" | grep -c 'Giving up')"
  echo "\n--- last progress line ---"
  tr '\r' '\n' < "$LOG" | grep -E '[0-9]+/[0-9]+ \[' | tail -1
  echo "--- recent errors (if any) ---"
  tr '\r' '\n' < "$LOG" | grep -E 'attempt [0-9]+/|Giving up|Error on post' | tail -5
else
  echo "\n  (no log file found; counts above are authoritative)"
fi

echo "\n--- label counts so far ---"
for f in dataset_annotated_assist_requeue.jsonl dataset_annotated_assist_help.jsonl; do
  [ -f "$f" ] || continue
  python3 -c "
import json,collections,sys
rows=[json.loads(l) for l in open('$f')]
c=collections.Counter(r.get('loc_label') for r in rows)
n=len(rows) or 1
print(f'  $f: ' + '  '.join(f'{k}={v} ({100*v/n:.1f}%)' for k,v in c.most_common()))
"
done
