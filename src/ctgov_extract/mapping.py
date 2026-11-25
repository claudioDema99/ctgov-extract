"""Deterministic mapping from a ClinicalTrials.gov API v2 study to a :class:`TrialRecord`."""

from __future__ import annotations

from typing import Any

from ctgov_extract.errors import CtgovApiError
from ctgov_extract.models import (
    Arm,
    Enrollment,
    Intervention,
    Outcome,
    Sponsor,
    StudyDate,
    TrialRecord,
)

JsonDict = dict[str, Any]


def _dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: Any) -> list[str]:
    return [s for s in (_str(v) for v in _list(value)) if s]


def _date(value: Any) -> StudyDate:
    data = _dict(value)
    return StudyDate(date=_str(data.get("date")), type=_str(data.get("type")))


def _outcomes(value: Any) -> list[Outcome]:
    outcomes = []
    for item in _list(value):
        data = _dict(item)
        measure = _str(data.get("measure"))
        if measure:
            outcomes.append(
                Outcome(
                    measure=measure,
                    time_frame=_str(data.get("timeFrame")),
                    description=_str(data.get("description")),
                )
            )
    return outcomes


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def study_to_record(study: JsonDict) -> TrialRecord:
    """Map a raw API v2 study (``GET /api/v2/studies/{id}``) to a :class:`TrialRecord`.

    Missing modules or fields are tolerated and mapped to ``None`` or empty lists.

    Raises:
        CtgovApiError: if the payload has no ``protocolSection.identificationModule.nctId``.
    """
    protocol = _dict(study.get("protocolSection"))
    ident = _dict(protocol.get("identificationModule"))
    nct_id = _str(ident.get("nctId"))
    if not nct_id:
        raise CtgovApiError("Study payload has no protocolSection.identificationModule.nctId.")

    status = _dict(protocol.get("statusModule"))
    design = _dict(protocol.get("designModule"))
    conditions = _dict(protocol.get("conditionsModule"))
    arms_mod = _dict(protocol.get("armsInterventionsModule"))
    outcomes = _dict(protocol.get("outcomesModule"))
    sponsors = _dict(protocol.get("sponsorCollaboratorsModule"))
    eligibility = _dict(protocol.get("eligibilityModule"))
    contacts = _dict(protocol.get("contactsLocationsModule"))

    interventions = []
    for item in _list(arms_mod.get("interventions")):
        data = _dict(item)
        name = _str(data.get("name"))
        if name:
            interventions.append(
                Intervention(
                    type=_str(data.get("type")),
                    name=name,
                    other_names=_str_list(data.get("otherNames")),
                )
            )

    arms = []
    for item in _list(arms_mod.get("armGroups")):
        data = _dict(item)
        label = _str(data.get("label"))
        if label:
            arms.append(
                Arm(
                    label=label,
                    type=_str(data.get("type")),
                    description=_str(data.get("description")),
                    interventions=_str_list(data.get("interventionNames")),
                )
            )

    enrollment_info = _dict(design.get("enrollmentInfo"))
    lead = _dict(sponsors.get("leadSponsor"))
    lead_name = _str(lead.get("name"))
    healthy = eligibility.get("healthyVolunteers")

    return TrialRecord(
        nct_id=nct_id.upper(),
        brief_title=_str(ident.get("briefTitle")),
        official_title=_str(ident.get("officialTitle")),
        acronym=_str(ident.get("acronym")),
        overall_status=_str(status.get("overallStatus")),
        phases=_str_list(design.get("phases")),
        study_type=_str(design.get("studyType")),
        conditions=_str_list(conditions.get("conditions")),
        keywords=_str_list(conditions.get("keywords")),
        interventions=interventions,
        arms=arms,
        enrollment=Enrollment(
            count=_int(enrollment_info.get("count")),
            type=_str(enrollment_info.get("type")),
        ),
        start_date=_date(status.get("startDateStruct")),
        primary_completion_date=_date(status.get("primaryCompletionDateStruct")),
        completion_date=_date(status.get("completionDateStruct")),
        lead_sponsor=(
            Sponsor(name=lead_name, sponsor_class=_str(lead.get("class"))) if lead_name else None
        ),
        primary_outcomes=_outcomes(outcomes.get("primaryOutcomes")),
        secondary_outcomes=_outcomes(outcomes.get("secondaryOutcomes")),
        minimum_age=_str(eligibility.get("minimumAge")),
        maximum_age=_str(eligibility.get("maximumAge")),
        sex=_str(eligibility.get("sex")),
        healthy_volunteers=healthy if isinstance(healthy, bool) else None,
        num_locations=len(_list(contacts.get("locations"))),
        # Keep the text exactly as published: criterion offsets refer to it.
        eligibility_text=str(eligibility.get("eligibilityCriteria") or ""),
        last_update_posted=_date(status.get("lastUpdatePostDateStruct")).date,
    )
