import pytest

from ctgov_extract.errors import CtgovApiError
from ctgov_extract.mapping import study_to_record
from tests.conftest import FIXTURE_IDS, load_study


def test_destiny_breast03_fields() -> None:
    record = study_to_record(load_study("NCT03529110"))
    assert record.nct_id == "NCT03529110"
    assert record.brief_title is not None and "DS-8201a" in record.brief_title
    assert record.overall_status == "COMPLETED"
    assert record.phases == ["PHASE3"]
    assert record.study_type == "INTERVENTIONAL"
    assert record.enrollment.count == 524
    assert record.enrollment.type == "ACTUAL"
    assert record.start_date.date == "2018-08-09"
    assert record.start_date.type == "ACTUAL"
    names = [i.name for i in record.interventions]
    assert "Trastuzumab deruxtecan (T-DXd)" in names
    assert record.interventions[0].type == "DRUG"
    assert "DS-8201a" in record.interventions[0].other_names
    assert record.arms and record.arms[0].interventions == ["Drug: Trastuzumab deruxtecan (T-DXd)"]
    assert record.primary_outcomes[0].measure.startswith("Progression-Free Survival")
    assert record.primary_outcomes[0].time_frame == "Up to 33 months (data cut-off)"
    assert record.secondary_outcomes
    assert record.minimum_age == "18 Years"
    assert record.maximum_age is None
    assert record.sex == "ALL"
    assert record.healthy_volunteers is False
    assert record.num_locations == 164
    assert record.lead_sponsor is not None and record.lead_sponsor.sponsor_class == "INDUSTRY"


def test_eligibility_text_is_kept_verbatim() -> None:
    raw = load_study("NCT03887455")
    record = study_to_record(raw)
    assert (
        record.eligibility_text
        == raw["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    )
    assert "\\>=" in record.eligibility_text  # markdown escapes are not removed here
    assert record.acronym is not None
    assert record.minimum_age == "50 Years"
    assert record.maximum_age == "90 Years"


def test_partial_dates_and_missing_keywords() -> None:
    record = study_to_record(load_study("NCT00000620"))
    assert record.start_date.date == "1999-09"
    assert record.start_date.type is None
    assert record.acronym == "ACCORD"


def test_missing_modules_are_tolerated() -> None:
    record = study_to_record(
        {"protocolSection": {"identificationModule": {"nctId": "nct00000001"}}}
    )
    assert record.nct_id == "NCT00000001"
    assert record.conditions == []
    assert record.enrollment.count is None
    assert record.lead_sponsor is None
    assert record.eligibility_text == ""
    assert record.num_locations == 0


def test_malformed_values_are_ignored() -> None:
    raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002", "briefTitle": "  "},
            "designModule": {"phases": "PHASE1", "enrollmentInfo": {"count": True}},
            "armsInterventionsModule": {"interventions": [{"type": "DRUG"}, "junk"]},
            "eligibilityModule": {"healthyVolunteers": "yes"},
        }
    }
    record = study_to_record(raw)
    assert record.brief_title is None
    assert record.phases == []
    assert record.enrollment.count is None
    assert record.interventions == []
    assert record.healthy_volunteers is None


def test_missing_nct_id_raises() -> None:
    with pytest.raises(CtgovApiError):
        study_to_record({"protocolSection": {}})


@pytest.mark.parametrize("nct_id", FIXTURE_IDS)
def test_all_fixtures_map(nct_id: str) -> None:
    record = study_to_record(load_study(nct_id))
    assert record.nct_id == nct_id
    assert record.eligibility_text
    assert record.conditions
    assert record.primary_outcomes
