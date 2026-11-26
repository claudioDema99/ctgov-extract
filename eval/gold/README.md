# Gold annotations

This folder is intentionally empty. The gold set will be annotated by hand by the author.

Gold files follow `eval/ANNOTATION.md`, one JSONL file per batch, e.g. `eval/gold/batch-01.jsonl`.

Workflow:

1. `ctgov-extract annotate NCT01234567` writes a draft to `eval/gold/drafts/NCT01234567.draft.jsonl`.
2. Correct the draft by hand: fix, delete and add items, check every `source_text` and offset.
3. Set `"draft": false`, fill `annotator` and `annotated_at`, and move the line into a gold file here.
4. Run `ctgov-extract eval --gold eval/gold/*.jsonl --report eval/out/report.md`.

`eval` skips records with `"draft": true` unless `--include-drafts` is passed, so unreviewed
model output is never scored as if it were a reference.

Model drafts start from the model's own answers, which can bias the reference towards the model.
Note in the report how many items were changed during review.
Open question: annotating a subset from scratch, without drafts, would measure this bias.
