# Annotation guidelines

These rules define the reference ("gold") atomic criteria used by `ctgov-extract eval`.
They match the default extraction schema, so the model and the annotator answer the same question.

## File format

JSONL: one trial per line. Each line is a JSON object:

| Field | Type | Required | Notes |
|---|---|---|---|
| `nct_id` | string | yes | e.g. `NCT03887455` |
| `eligibility_text` | string | yes | The exact text that was annotated. Offsets refer to it. Trials get updated, so the text is stored with the labels. |
| `title` | string | no | Passed to the model as context during `eval` |
| `conditions` | list of strings | no | Same |
| `annotator` | string | yes for gold | Who reviewed the record |
| `annotated_at` | string | no | ISO date |
| `draft` | bool | no (default `false`) | `true` for unreviewed drafts; `eval` skips them |
| `synthetic` | bool | no (default `false`) | Test data only; `eval` skips it |
| `notes` | string | no | Anything worth knowing about the record |
| `items` | list | yes | Atomic criteria, see below |

Each item:

| Field | Type | Notes |
|---|---|---|
| `kind` | `inclusion` or `exclusion` | The section the criterion appears in |
| `category` | enum | See categories below |
| `concept` | string | Short normalized name, no values or units: `MMSE score`, `LVEF`, `prior anti-PD-1 therapy` |
| `operator` | enum or null | `=`, `!=`, `<`, `<=`, `>`, `>=`, `between`, `present`, `absent` |
| `value` | number, string or null | Number when the text gives one |
| `value_max` | number or null | Upper bound for `between` |
| `unit` | string or null | As written: `mg/dL`, `%`, `ms`, `years` |
| `negated` | bool | See negation below |
| `is_biomarker` | bool or null | Optional; not scored yet |
| `time_window` | string or null | Optional; as written, e.g. `within 6 months before screening`; not scored yet |
| `source_text` | string | The supporting span, copied from `eligibility_text` |
| `start`, `end` | int | Character offsets: `eligibility_text[start:end]` must equal `source_text` (after whitespace and escape normalization). The loader rejects mismatches. |
| `criterion_id` | string or null | Optional link to the split criterion (`inc-3.2`) |
| `note` | string or null | Free text for doubts |

## What is one atomic criterion

One atomic criterion tests one concept against one condition.

- Split on "and" / "or" / commas when they join different concepts.
  "History of TIA, stroke, or seizures within 12 months" is three items (TIA, stroke, seizures), each with the same time window.
- Keep a range as one item with `between`. "MMSE score of 20 to 28 (inclusive)" is one item: `between`, 20, 28.
- Split two bounds stated separately for different time points only if they are different tests.
  "MMSE \>=22 at Screening and Baseline and \<=30 at Screening and Baseline" is one `between` item.
- Two scales are two items, even in one sentence: "global CDR score of 0.5 and CDR Memory Box score of 0.5 or greater" is two.
- Exceptions and carve-outs ("with the exception of basal cell carcinoma") get their own item with the opposite polarity and a `note` saying it is an exception. (Provisional: to be confirmed after the first annotation batch.)
- Administrative criteria (consent, study partner, ability to comply) are annotated with category `consent/administrative`.
- A parent line that only introduces sub-items ("including any of the following:") is not annotated by itself.

## Categories

| Category | Use for |
|---|---|
| `demographics` | Age, sex, BMI, ethnicity, language |
| `condition/diagnosis` | The disease under study, its stage or subtype, diagnostic criteria met |
| `medical history` | Past or concurrent conditions other than the target disease |
| `medication/treatment` | Prior, current or forbidden drugs and therapies, washout periods |
| `laboratory/biomarker` | Lab values and molecular or fluid biomarkers (HbA1c, INR, HER2, amyloid, APOE4) |
| `imaging` | Findings or requirements on MRI, CT, PET, echocardiography |
| `procedure` | Surgery, transplant, ability to undergo a procedure (LP, MRI) |
| `cognitive/functional score` | MMSE, CDR, MoCA, GDS, ECOG, NYHA class, life expectancy scores |
| `pregnancy/reproductive` | Pregnancy, breastfeeding, contraception |
| `consent/administrative` | Consent, study partner, participation in other studies, site staff |
| `other` | Anything that does not fit above; add a `note` |

Biomarker vs. lab: both use `laboratory/biomarker`. Set `is_biomarker` true for molecular, genetic, imaging or fluid biomarkers, false for routine labs.
Imaging biomarkers (amyloid PET positivity) use `laboratory/biomarker` with `is_biomarker` true.
Open question: ECOG and NYHA may deserve a separate "performance status" category.

## Operators and values

- Numeric tests: `>=`, `>`, `<=`, `<`, `=`, `!=` with `value` and `unit`.
  "at least 18 years" is `>=` 18 `years`. "greater than (\>)17" is `>` 17.
- Ranges: `between` with `value` (low) and `value_max` (high). Note exclusive bounds in `note`.
- Things that must exist or must not exist: `present` / `absent`, `value` null.
  For a status word ("HER2-positive"), use `present` and put the status in the concept or `value`: concept `HER2 status`, value `positive`. (Provisional: the convention still has to be applied consistently.)
- `kind` already carries inclusion vs exclusion. Do not invert: an exclusion "History of stroke" is `kind: exclusion`, `operator: present`.

## Negation

`negated` is true only when the text itself uses negation for the concept: "no", "not", "without", "absence of", "never", "free of".

| Text | kind | operator | negated |
|---|---|---|---|
| Inclusion: "No prior systemic chemotherapy" | inclusion | absent | true |
| Exclusion: "Prior treatment with docetaxel" | exclusion | present | false |
| Exclusion: "Tumor specimen is not evaluable for PD-L1" | exclusion | present | true |

## Evidence span

- Copy the shortest span that supports the item, character for character, from `eligibility_text`.
- Keep markdown escapes as they are in the text (`\>=`). The loader normalizes escapes and whitespace when it checks offsets, but offsets must point to the raw text.
- A span can be shared by several items when the text does not allow a finer split.
- Use the `annotate` drafts as a starting point: their offsets are computed by the tool, not typed by hand.

## How predictions are matched

See the docstring of `src/ctgov_extract/evaluation.py`. In short: same `kind`, span IoU >= 0.5 when both
sides have offsets, otherwise normalized text similarity >= 0.8; greedy one-to-one matching.
Precision, recall and F1 are computed over all items; field accuracy only over matched pairs.
