# Test fixtures

## `studies/`

Real public records from the ClinicalTrials.gov API v2
(`GET https://clinicaltrials.gov/api/v2/studies/{NCT_ID}`), downloaded on 2026-11-25.
To keep the repository small, `resultsSection` and `documentSection` were removed. The
`protocolSection` (which holds every field the tool reads) is unchanged.

| NCT ID | Trial | Why it is here |
|---|---|---|
| NCT03529110 | DESTINY-Breast03 (trastuzumab deruxtecan, HER2+ breast cancer) | Oncology, HER2 status, numeric thresholds (LVEF < 50%, QTc > 470 ms), numbered lists nested inside numbered lists, bullets nested under a numbered item |
| NCT03887455 | Clarity AD (lecanemab, early Alzheimer's disease) | Cognitive scores (MMSE, CDR, GDS), seven different section headers ("Core Study: Inclusion Criteria", "Key Inclusion Criteria that must be met by all participants:"...), sub-headings, markdown escapes (`\>=`), `•` and `-In` list markers |
| NCT01844505 | CheckMate 067 (nivolumab + ipilimumab, melanoma) | Free text before the first header (preamble), steroid dose threshold |
| NCT00000620 | ACCORD (type 2 diabetes) | Inclusion section only, HbA1c ranges, partial dates (`1999-09`) |

## `synthetic_gold.jsonl`

A tiny, hand-written, **synthetic** record (`"synthetic": true`, not a real trial) used only to
test the metric code. It is not a gold annotation and is skipped by `ctgov-extract eval` unless
`--include-synthetic` is passed.
