# ctgov-extract

ClinicalTrials.gov exposes most of a trial as structured data: phase, status, conditions, interventions, dates.
The eligibility criteria are the exception. They are free text, and they are exactly what you need for cohort selection and patient pre-screening.
I built this tool to turn a trial (NCT ID, URL or API v2 JSON file) into a typed record plus atomic, machine-readable eligibility criteria, where every extracted item points back to the span of text it came from.

## What it extracts

| Layer | Uses an LLM | Output |
|---|---|---|
| Trial record | no | `TrialRecord`: IDs, titles, status, phases, study type, conditions, keywords, interventions, arms, enrollment, dates, lead sponsor, outcomes, age limits, sex, healthy volunteers, number of locations, raw eligibility text |
| Criteria splitting | no | Inclusion / exclusion criteria with nesting, section headers, sub-headings and character offsets |
| Atomic criteria (default schema) | yes | Per atomic criterion: category, concept, operator, value, value_max, unit, time window, negated, is_biomarker, evidence span with validation status |
| Custom schema | yes | Any structure you define in YAML or JSON, validated with Pydantic, applied to the whole trial |

## Example

```bash
ctgov-extract criteria https://clinicaltrials.gov/study/NCT03887455
```

The split criterion comes from the deterministic layer. The atomic item below is illustrative: I wrote the values by hand to show the output shape, it is not from a model run. The offsets were computed by the tool.

```json
{
  "id": "inc-9",
  "kind": "inclusion",
  "section": "Key Inclusion Criteria that must be met by all participants",
  "text": "Mini mental state examination (MMSE) score \\>=22 at Screening and Baseline and \\<=30 at Screening and Baseline",
  "start": 1256,
  "end": 1366
}
```

```json
{
  "id": "inc-9-a1",
  "criterion_id": "inc-9",
  "kind": "inclusion",
  "category": "cognitive/functional score",
  "concept": "MMSE score",
  "operator": "between",
  "value": 22.0,
  "value_max": 30.0,
  "time_window": {"text": "at Screening and Baseline"},
  "negated": false,
  "is_biomarker": false,
  "evidence": {
    "source_text": "MMSE) score >=22 at Screening and Baseline and <=30",
    "status": "normalized",
    "start": 1287,
    "end": 1340
  }
}
```

## Quick start

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
cp .env.example .env        # then edit LLM_BASE_URL / LLM_MODEL
set -a; source .env; set +a

uv run ctgov-extract fetch NCT03529110               # deterministic record, no LLM
uv run ctgov-extract split NCT03529110               # criteria with offsets, no LLM
uv run ctgov-extract criteria NCT03529110 -f csv -o destiny.csv
uv run ctgov-extract extract NCT03887455 --schema schemas/neuro_cognitive_scores.yaml
uv run ctgov-extract serve                           # web UI on http://127.0.0.1:8000
```

`fetch`, `split` and `criteria` accept an NCT ID (any case), a ClinicalTrials.gov URL (`/study/...`, `/ct2/show/...`, with or without query string) or a path to a saved API v2 JSON file.
Fetched studies are cached in `~/.cache/ctgov-extract` for 7 days.

With Docker:

```bash
LLM_BASE_URL=http://host.docker.internal:11434/v1 LLM_MODEL=qwen2.5:7b-instruct docker compose up --build
```

`docker-compose.yml` also contains a commented-out vLLM service.

## Web UI

`ctgov-extract serve` starts a FastAPI app with a single static page (no build step, no CDN).
Enter an NCT ID or URL, choose default criteria or a custom schema, and click Extract.
Results show a trial summary, then inclusion and exclusion tables. Clicking a row highlights its evidence in the original eligibility text.
Items whose evidence was not found are flagged.

JSON endpoints: `GET /api/health`, `GET /api/trial/{id}`, `GET /api/schemas`, `POST /api/criteria` (`{"trial": "..."}`), `POST /api/extract` (`{"trial": "...", "schema_text": "..."}`).
OpenAPI docs are at `/docs`.

## Using your own schema

Define fields in YAML (or JSON). Types: `string`, `number`, `integer`, `boolean`, `enum`, `list[string]`, `list[object]` (with nested `fields`). Fields are optional unless `required: true`.

```yaml
name: cognitive_scales
instructions: List each scale cut-off separately.
fields:
  - name: scales
    type: list[object]
    required: true
    fields:
      - {name: scale, type: string, required: true}
      - {name: operator, type: enum, enum: ["<", "<=", ">", ">=", "between"]}
      - {name: min_value, type: number}
      - {name: max_value, type: number}
      - {name: evidence, type: string, required: true, description: Verbatim quote}
```

The tool builds a Pydantic model with `pydantic.create_model` and puts its JSON schema in the prompt. By default that is all it sends, so it works with any chat endpoint; the answer is parsed and validated, and invalid output gets one repair round with the validation errors.
`LLM_OUTPUT_MODE=json_schema` (or `json_object`) also sends `response_format`, so servers that support constrained decoding enforce the schema. If the backend rejects it (HTTP 400, 422 or 500), the tool falls back towards plain text and remembers what works.
If validation fails, the errors go back to the model for one repair round. A second failure is reported as an error, not patched over.
Any string field named `evidence`, `source_text` or `quote` is checked against the trial text.
With one top-level `list[object]` field, `--format csv` and `--format jsonl` write one row per item.

Examples in `schemas/`: `oncology_biomarkers.yaml`, `neuro_cognitive_scores.yaml`, `simple_overview.yaml`.

## LLM backends

Any OpenAI-compatible chat completions API works. Temperature defaults to 0.

| Backend | `LLM_BASE_URL` | `LLM_MODEL` | `LLM_API_KEY` |
|---|---|---|---|
| vLLM | `http://localhost:8001/v1` | the served model, e.g. `Qwen/Qwen2.5-7B-Instruct` | empty |
| Ollama | `http://localhost:11434/v1` | e.g. `qwen2.5:7b-instruct` | empty |
| OpenAI | `https://api.openai.com/v1` | e.g. `gpt-4o-mini` | your key |

Other settings: `LLM_OUTPUT_MODE` (`text` by default, see [Using your own schema](#using-your-own-schema)), `LLM_MAX_CONCURRENCY` (default 4), `LLM_TIMEOUT`, `LLM_MAX_RETRIES`, `LLM_MAX_TOKENS`. `LLM_EXTRA_BODY` takes a JSON object merged into each request, for backend-specific options: with Qwen3 on vLLM, `{"chat_template_kwargs": {"enable_thinking": false}}` turns off the thinking block (if it is left on, `<think>` blocks are stripped before parsing). Requests retry with backoff on 429 and 5xx. Token usage is logged (`-v`) and returned in every result.

## Evidence validation

Each atomic item must quote a `source_text` from its criterion. After the model call, the tool searches for that quote:

1. exact substring: status `exact`;
2. after normalization (markdown escapes like `\>=`, whitespace, `≥`/`≤`, typographic quotes and dashes, then case): status `normalized`;
3. otherwise: status `not_found`, offsets `null`.

Offsets always point into the original text. Items with `not_found` are kept and flagged, never silently accepted.
This matters because a quote that is not in the text is a strong sign that the model paraphrased or invented the criterion. For screening, an unsupported criterion is worse than a missing one, and the span lets a reviewer check each item in seconds.

## Evaluation

`eval/ANNOTATION.md` documents the gold format (JSONL, one trial per line) and the annotation rules.

```bash
ctgov-extract annotate NCT03887455                 # writes a draft for a human to correct
ctgov-extract eval --gold eval/gold/*.jsonl --report eval/out/report.md --save-pred eval/out/pred.jsonl
ctgov-extract eval --gold eval/gold/*.jsonl --pred eval/out/pred.jsonl   # re-score without calling the model
```

Reported: criterion-level precision, recall and F1 (matching by span overlap, fuzzy text as fallback), field accuracy on matched pairs (category, operator, value, value_max, unit, negated), evidence-not-found rate, failed criteria, latency and tokens.
Drafts and synthetic records are skipped unless explicitly included.

No results yet: the gold set has not been annotated, and I'd rather report nothing than numbers on model-written references.

## Design notes and limitations

- One LLM call per criterion. It costs more calls but keeps the evidence scope small and makes failures local: one bad criterion does not sink the trial.
- The splitter is rule-based and covers the formats I found (bullets, numbered and lettered lists, nesting by indentation, inline headers, sub-headings, text before the first header). Unusual layouts can still be split wrongly; offsets stay correct either way.
- LLMs make mistakes: wrong operator, merged concepts, invented values. Evidence validation catches invented quotes, not wrong interpretations of a correct quote.
- Many criteria are ambiguous ("adequate organ function", "clinically significant"). The schema records them, it does not resolve them.
- No mapping to ontologies (SNOMED CT, LOINC, RxNorm, OMOP) yet. Concepts are normalized text only.
- No patient matching. The output is designed to feed it.

## Related work

- Criteria2Query: natural language interface for cohort definition; a hybrid ML and rule-based pipeline turns eligibility criteria into executable OMOP CDM queries. [Yuan et al., JAMIA 2019](https://doi.org/10.1093/jamia/ocy178)
- EliIE: open-source information extraction for eligibility criteria (entity and attribute recognition, negation, relations, normalization to OMOP CDM), trained on 230 Alzheimer's trials. [Kang et al., JAMIA 2017](https://doi.org/10.1093/jamia/ocx019)
- TrialGPT: zero-shot patient-to-trial matching with LLMs (retrieval, criterion-level eligibility with explanations, trial ranking). [Jin et al., Nature Communications 2024](https://doi.org/10.1038/s41467-024-53081-z)

## Development

```bash
make install   # uv sync
make check     # ruff, mypy, pytest (all tests offline)
make serve
```

Layout: `src/ctgov_extract/` (`ctgov.py` API client and cache, `mapping.py` record mapping, `criteria.py` splitter, `evidence.py` validation, `atomize.py` default schema, `user_schema.py` and `custom.py` custom schemas, `llm/` backends and structured output, `evaluation.py`, `cli.py`, `web/`).
Test fixtures are real API records, listed in `tests/fixtures/README.md`.

## License

MIT, see `LICENSE`.
