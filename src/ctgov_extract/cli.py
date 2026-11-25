"""Command line interface: ``ctgov-extract <command> ...``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ctgov_extract import __version__
from ctgov_extract.atomize import CriteriaExtractor
from ctgov_extract.config import Settings
from ctgov_extract.criteria import split_criteria
from ctgov_extract.ctgov import CtgovClient, resolve_trial
from ctgov_extract.custom import CustomExtractor
from ctgov_extract.errors import CtgovExtractError
from ctgov_extract.evaluation import (
    dump_jsonl,
    load_gold,
    load_predictions,
    render_report,
    result_to_draft,
    score_trial,
    summarize,
)
from ctgov_extract.llm.base import LLMClient
from ctgov_extract.llm.openai_compat import OpenAICompatClient
from ctgov_extract.models import CriteriaResult
from ctgov_extract.output import OutputFormat, render_criteria, render_custom, to_json
from ctgov_extract.user_schema import load_schema_file

logger = logging.getLogger("ctgov_extract")


# Factories are module-level so tests can replace them.
def build_ctgov_client(settings: Settings, *, use_cache: bool = True) -> CtgovClient:
    return CtgovClient.from_settings(settings, use_cache=use_cache)


def build_llm_client(settings: Settings) -> LLMClient:
    return OpenAICompatClient.from_settings(settings)


async def _close(obj: object) -> None:
    aclose = getattr(obj, "aclose", None)
    if aclose is not None:
        await aclose()


def _emit(text: str, out: Path | None) -> None:
    if out is None:
        sys.stdout.write(text)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


async def cmd_fetch(args: argparse.Namespace, settings: Settings) -> int:
    ctgov = build_ctgov_client(settings, use_cache=not args.no_cache)
    try:
        record = await resolve_trial(args.trial, ctgov)
    finally:
        await _close(ctgov)
    _emit(to_json(record), args.out)
    return 0


async def cmd_split(args: argparse.Namespace, settings: Settings) -> int:
    ctgov = build_ctgov_client(settings, use_cache=not args.no_cache)
    try:
        record = await resolve_trial(args.trial, ctgov)
    finally:
        await _close(ctgov)
    _emit(to_json(split_criteria(record.eligibility_text, record.nct_id)), args.out)
    return 0


async def cmd_criteria(args: argparse.Namespace, settings: Settings) -> int:
    ctgov = build_ctgov_client(settings, use_cache=not args.no_cache)
    llm = build_llm_client(settings)
    try:
        record = await resolve_trial(args.trial, ctgov)
        result = await CriteriaExtractor(llm).extract_text(
            record.eligibility_text,
            nct_id=record.nct_id,
            title=record.brief_title,
            conditions=record.conditions,
        )
    finally:
        await _close(ctgov)
        await _close(llm)
    _report_result(result)
    _emit(render_criteria(result, cast(OutputFormat, args.format)), args.out)
    return 0


def _report_result(result: CriteriaResult) -> None:
    logger.info(
        "%d criteria -> %d atomic items, %d evidence not found, %d failed, %d tokens, %.1fs",
        len(result.criteria),
        len(result.atomic),
        len(result.evidence_not_found),
        len(result.failures),
        result.usage.total_tokens,
        result.elapsed_s,
    )


async def cmd_extract(args: argparse.Namespace, settings: Settings) -> int:
    spec = load_schema_file(args.schema)
    ctgov = build_ctgov_client(settings, use_cache=not args.no_cache)
    llm = build_llm_client(settings)
    try:
        record = await resolve_trial(args.trial, ctgov)
        result = await CustomExtractor(llm).extract(record, spec)
    finally:
        await _close(ctgov)
        await _close(llm)
    _emit(render_custom(result, spec, cast(OutputFormat, args.format)), args.out)
    return 0


async def cmd_annotate(args: argparse.Namespace, settings: Settings) -> int:
    ctgov = build_ctgov_client(settings, use_cache=not args.no_cache)
    llm = build_llm_client(settings)
    try:
        record = await resolve_trial(args.trial, ctgov)
        result = await CriteriaExtractor(llm).extract_text(
            record.eligibility_text,
            nct_id=record.nct_id,
            title=record.brief_title,
            conditions=record.conditions,
        )
    finally:
        await _close(ctgov)
        await _close(llm)
    draft = result_to_draft(result, title=record.brief_title, conditions=record.conditions)
    out = args.out or Path("eval/gold/drafts") / f"{record.nct_id}.draft.jsonl"
    if out.exists() and not args.force:
        raise CtgovExtractError(f"{out} exists; use --force to overwrite.")
    dump_jsonl([draft], out)
    print(
        f"Wrote draft with {len(draft.items)} items to {out}. "
        "Review it by hand (see eval/ANNOTATION.md) before using it as gold.",
        file=sys.stderr,
    )
    return 0


async def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    gold_all = load_gold(args.gold)
    gold, skipped = [], []
    for trial in gold_all:
        if (trial.draft and not args.include_drafts) or (
            trial.synthetic and not args.include_synthetic
        ):
            skipped.append(trial.nct_id)
        else:
            gold.append(trial)
    if not gold:
        raise CtgovExtractError(
            "No gold trials to score (drafts and synthetic records are skipped by default)."
        )

    model_name: str | None
    if args.pred:
        preds = load_predictions(args.pred)
        missing = [g.nct_id for g in gold if g.nct_id not in preds]
        if missing:
            raise CtgovExtractError(f"No predictions for: {', '.join(missing)}")
        results = [preds[g.nct_id] for g in gold]
        model_name = next((r.model for r in results if r.model), None)
    else:
        llm = build_llm_client(settings)
        try:
            extractor = CriteriaExtractor(llm)
            results = []
            for g in gold:  # sequential per trial, so latency numbers are per trial
                logger.info("Extracting %s", g.nct_id)
                results.append(
                    await extractor.extract_text(
                        g.eligibility_text, nct_id=g.nct_id, title=g.title, conditions=g.conditions
                    )
                )
        finally:
            await _close(llm)
        model_name = llm.model_name
        if args.save_pred:
            dump_jsonl(results, args.save_pred)

    summary = summarize([score_trial(g, r) for g, r in zip(gold, results, strict=True)])
    report = render_report(
        summary, model=model_name, gold_files=[str(p) for p in args.gold], skipped=skipped
    )
    _emit(report, args.report)
    if args.json:
        _emit(summary.model_dump_json(indent=2) + "\n", args.json)
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from ctgov_extract.web.app import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctgov-extract",
        description="Structured eligibility criteria from ClinicalTrials.gov records.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    sub = parser.add_subparsers(dest="command", required=True)

    def trial_cmd(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("trial", help="NCT ID, ClinicalTrials.gov URL, or path to a study JSON")
        p.add_argument("--out", "-o", type=Path, help="write to this file instead of stdout")
        p.add_argument("--no-cache", action="store_true", help="bypass the on-disk study cache")
        return p

    trial_cmd("fetch", "Fetch a study and print the deterministic TrialRecord as JSON.")
    trial_cmd("split", "Split eligibility text into criteria with offsets (no LLM).")

    p = trial_cmd("criteria", "Extract atomic criteria with the default schema (uses the LLM).")
    p.add_argument("--format", "-f", choices=["json", "jsonl", "csv"], default="json")

    p = trial_cmd("extract", "Extract with a user-defined YAML/JSON schema (uses the LLM).")
    p.add_argument("--schema", "-s", type=Path, required=True, help="schema file (.yaml/.json)")
    p.add_argument("--format", "-f", choices=["json", "jsonl", "csv"], default="json")

    p = trial_cmd("annotate", "Write a draft gold annotation from model output.")
    p.add_argument("--force", action="store_true", help="overwrite an existing draft")

    p = sub.add_parser("eval", help="Score extraction against gold annotations.")
    p.add_argument("--gold", type=Path, nargs="+", required=True, help="gold JSONL file(s)")
    p.add_argument("--pred", type=Path, help="score saved predictions (JSONL) instead of running")
    p.add_argument("--save-pred", type=Path, help="save predictions made during this run")
    p.add_argument("--report", type=Path, help="markdown report path (default: stdout)")
    p.add_argument("--json", type=Path, help="also write metrics as JSON")
    p.add_argument("--include-drafts", action="store_true", help="also score draft records")
    p.add_argument(
        "--include-synthetic", action="store_true", help="also score synthetic test records"
    )

    p = sub.add_parser("serve", help="Run the web UI and JSON API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return parser


_ASYNC_COMMANDS: dict[str, Any] = {
    "fetch": cmd_fetch,
    "split": cmd_split,
    "criteria": cmd_criteria,
    "extract": cmd_extract,
    "annotate": cmd_annotate,
    "eval": cmd_eval,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()

    try:
        if args.command == "serve":
            return cmd_serve(args, settings)
        code: int = asyncio.run(_ASYNC_COMMANDS[args.command](args, settings))
    except CtgovExtractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
