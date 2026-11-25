import csv
import io
import json
from pathlib import Path

import pytest

from ctgov_extract import cli
from ctgov_extract.config import Settings
from ctgov_extract.ctgov import CtgovClient
from ctgov_extract.llm.base import LLMClient
from ctgov_extract.llm.fake import FakeLLMClient
from tests.conftest import FIXTURES, REPO_ROOT, STUDIES, echo_responder, make_ctgov_client


@pytest.fixture
def fake_backends(monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    llm = FakeLLMClient(responder=echo_responder)

    def ctgov_factory(settings: Settings, *, use_cache: bool = True) -> CtgovClient:
        return make_ctgov_client()

    def llm_factory(settings: Settings) -> LLMClient:
        return llm

    monkeypatch.setattr(cli, "build_ctgov_client", ctgov_factory)
    monkeypatch.setattr(cli, "build_llm_client", llm_factory)
    return llm


def test_fetch_stdout(fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["fetch", "https://clinicaltrials.gov/ct2/show/NCT00000620"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["nct_id"] == "NCT00000620"
    assert record["acronym"] == "ACCORD"


def test_fetch_from_file_to_out(fake_backends: FakeLLMClient, tmp_path: Path) -> None:
    out = tmp_path / "rec.json"
    assert cli.main(["fetch", str(STUDIES / "NCT01844505.json"), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["nct_id"] == "NCT01844505"


def test_split(fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["split", "NCT03529110"]) == 0
    split = json.loads(capsys.readouterr().out)
    assert len(split["criteria"]) == 22
    assert fake_backends.calls == []


def test_unknown_trial_is_an_error(
    fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["fetch", "NCT99999999"]) == 1
    assert "no study NCT99999999" in capsys.readouterr().err


def test_invalid_trial_is_an_error(
    fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["fetch", "hello"]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["json", "jsonl", "csv"])
def test_criteria_formats(
    fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str], fmt: str
) -> None:
    assert cli.main(["criteria", "NCT01844505", "--format", fmt]) == 0
    out = capsys.readouterr().out
    if fmt == "json":
        data = json.loads(out)
        assert len(data["atomic"]) == 10
        assert data["model"] == "fake-model"
    elif fmt == "jsonl":
        rows = [json.loads(line) for line in out.splitlines()]
        assert len(rows) == 10 and rows[0]["nct_id"] == "NCT01844505"
    else:
        rows = list(csv.DictReader(io.StringIO(out)))
        assert len(rows) == 10
        assert rows[0]["evidence_status"] == "exact"
        assert rows[0]["kind"] == "inclusion"


def test_extract_with_schema(
    fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_backends.responder = None
    fake_backends.responses = [
        json.dumps({"population": "Adults with melanoma", "key_exclusions": ["Ocular melanoma"]})
    ]
    schema = REPO_ROOT / "schemas" / "simple_overview.yaml"
    assert cli.main(["extract", "NCT01844505", "--schema", str(schema)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["data"]["population"] == "Adults with melanoma"
    assert data["data"]["min_age_years"] is None


def test_extract_csv_needs_one_list_field(
    fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_backends.responder = None
    fake_backends.responses = [json.dumps({"population": "x"})]
    schema = REPO_ROOT / "schemas" / "simple_overview.yaml"
    assert cli.main(["extract", "NCT01844505", "-s", str(schema), "-f", "csv"]) == 1
    assert "list[object]" in capsys.readouterr().err


def test_extract_csv_rows(fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]) -> None:
    fake_backends.responder = None
    fake_backends.responses = [
        json.dumps(
            {"biomarkers": [{"name": "HER2", "kind": "inclusion", "evidence": "HER2-positive"}]}
        )
    ]
    schema = REPO_ROOT / "schemas" / "oncology_biomarkers.yaml"
    assert cli.main(["extract", "NCT03529110", "-s", str(schema), "-f", "csv"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert rows == [
        {
            "nct_id": "NCT03529110",
            "name": "HER2",
            "kind": "inclusion",
            "biomarker_type": "",
            "requirement": "",
            "operator": "",
            "threshold": "",
            "threshold_max": "",
            "unit": "",
            "assay_or_method": "",
            "evidence": "HER2-positive",
        }
    ]


def test_annotate_writes_draft(fake_backends: FakeLLMClient, tmp_path: Path) -> None:
    out = tmp_path / "drafts" / "d.jsonl"
    assert cli.main(["annotate", "NCT00000620", "--out", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["draft"] is True
    assert record["annotator"] is None
    assert len(record["items"]) == 4
    assert cli.main(["annotate", "NCT00000620", "--out", str(out)]) == 1  # no silent overwrite
    assert cli.main(["annotate", "NCT00000620", "--out", str(out), "--force"]) == 0


def test_eval_skips_synthetic_by_default(
    fake_backends: FakeLLMClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["eval", "--gold", str(FIXTURES / "synthetic_gold.jsonl")]) == 1
    assert "No gold trials" in capsys.readouterr().err


def test_eval_runs_model_and_writes_report(fake_backends: FakeLLMClient, tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    preds = tmp_path / "preds.jsonl"
    gold = str(FIXTURES / "synthetic_gold.jsonl")
    args = ["eval", "--gold", gold, "--include-synthetic", "--report", str(report)]
    assert cli.main([*args, "--json", str(metrics), "--save-pred", str(preds)]) == 0
    text = report.read_text()
    assert "# Evaluation report" in text and "fake-model" in text
    summary = json.loads(metrics.read_text())
    assert summary["n_trials"] == 1 and summary["n_gold"] == 4
    assert summary["n_pred"] == 4  # the echo client returns one item per criterion
    # re-score the saved predictions without calling the model again
    calls_before = len(fake_backends.calls)
    assert cli.main([*args, "--pred", str(preds)]) == 0
    assert len(fake_backends.calls) == calls_before


def test_eval_pred_missing_trial(fake_backends: FakeLLMClient, tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    gold = str(FIXTURES / "synthetic_gold.jsonl")
    assert cli.main(["eval", "--gold", gold, "--include-synthetic", "--pred", str(empty)]) == 1


def test_criteria_without_llm_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key in ("LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "build_ctgov_client", lambda s, use_cache=True: make_ctgov_client())
    assert cli.main(["criteria", "NCT00000620"]) == 1
    assert "LLM_BASE_URL" in capsys.readouterr().err


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for command in ("fetch", "split", "criteria", "extract", "annotate", "eval", "serve"):
        assert command in out
