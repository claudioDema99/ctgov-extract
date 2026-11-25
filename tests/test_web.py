import json
from collections.abc import Iterator
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from ctgov_extract.config import Settings
from ctgov_extract.llm.fake import FakeLLMClient
from ctgov_extract.web.app import create_app
from tests.conftest import REPO_ROOT, STUDIES, echo_responder, make_ctgov_client


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient(responder=echo_responder)


@pytest.fixture
def client(llm: FakeLLMClient) -> Iterator[TestClient]:
    app = create_app(Settings(cache_dir=None), ctgov_client=make_ctgov_client(), llm_client=llm)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_llm() -> Iterator[TestClient]:
    app = create_app(Settings(cache_dir=None), ctgov_client=make_ctgov_client())
    with TestClient(app) as test_client:
        yield test_client


def test_index_and_static_assets(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'id="trial"' in page.text and "/static/app.js" in page.text
    assert "http://" not in page.text and "https://cdn" not in page.text  # no CDN dependencies
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "innerHTML" not in js.text  # model output is rendered with textContent only
    assert client.get("/static/styles.css").status_code == 200


def test_health(client: TestClient, client_without_llm: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["llm_configured"] is True and body["llm_model"] == "fake-model"
    assert client_without_llm.get("/api/health").json()["llm_configured"] is False


def test_trial_endpoint(client: TestClient) -> None:
    response = client.get("/api/trial/nct03887455")
    assert response.status_code == 200
    assert response.json()["nct_id"] == "NCT03887455"


def test_trial_endpoint_accepts_encoded_url(client: TestClient) -> None:
    url = quote("https://clinicaltrials.gov/ct2/show/NCT00000620?term=diabetes", safe="")
    response = client.get(f"/api/trial/{url}")
    assert response.status_code == 200
    assert response.json()["acronym"] == "ACCORD"


def test_trial_errors(client: TestClient) -> None:
    assert client.get("/api/trial/NCT99999999").status_code == 404
    bad = client.get("/api/trial/not-an-id")
    assert bad.status_code == 422 and "error" in bad.json()


def test_trial_endpoint_does_not_read_local_files(client: TestClient) -> None:
    path = str(STUDIES / "NCT00000620.json")
    response = client.post("/api/criteria", json={"trial": path})
    assert response.status_code == 422


def test_schemas_endpoint(client: TestClient) -> None:
    examples = client.get("/api/schemas").json()["examples"]
    assert "neuro_cognitive_scores.yaml" in examples
    assert examples["simple_overview.yaml"].startswith("#")


def test_criteria_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/criteria", json={"trial": "https://clinicaltrials.gov/study/NCT01844505"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["trial"]["nct_id"] == "NCT01844505"
    result = body["result"]
    assert len(result["criteria"]) == 10
    assert all(a["evidence"]["status"] == "exact" for a in result["atomic"])
    item = result["atomic"][0]
    text = body["trial"]["eligibility_text"]
    assert (
        text[item["evidence"]["start"] : item["evidence"]["end"]] == item["evidence"]["source_text"]
    )


def test_extract_endpoint(client: TestClient, llm: FakeLLMClient) -> None:
    llm.responder = None
    llm.responses = [
        json.dumps(
            {
                "scales": [
                    {
                        "scale": "MMSE",
                        "kind": "inclusion",
                        "operator": "between",
                        "min_value": 22,
                        "max_value": 30,
                        "evidence": "Mini mental state examination (MMSE) score >=22",
                    }
                ],
                "requires_study_partner": True,
            }
        )
    ]
    schema_text = (REPO_ROOT / "schemas" / "neuro_cognitive_scores.yaml").read_text()
    response = client.post(
        "/api/extract", json={"trial": "NCT03887455", "schema_text": schema_text}
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["data"]["scales"][0]["scale"] == "MMSE"
    assert result["evidence_checks"][0]["status"] == "normalized"
    assert result["evidence_checks"][0]["location"] == "eligibility"


def test_extract_bad_schema(client: TestClient) -> None:
    response = client.post("/api/extract", json={"trial": "NCT03887455", "schema_text": "name: x"})
    assert response.status_code == 422
    assert "fields" in response.json()["error"]


def test_extract_model_failure_is_502(client: TestClient, llm: FakeLLMClient) -> None:
    llm.responder = None
    llm.responses = ["nope", "still nope"]
    schema_text = (REPO_ROOT / "schemas" / "simple_overview.yaml").read_text()
    response = client.post(
        "/api/extract", json={"trial": "NCT03887455", "schema_text": schema_text}
    )
    assert response.status_code == 502
    assert response.json()["details"]


def test_llm_not_configured_is_503(client_without_llm: TestClient) -> None:
    response = client_without_llm.post("/api/criteria", json={"trial": "NCT00000620"})
    assert response.status_code == 503
    assert "LLM_BASE_URL" in response.json()["error"]
