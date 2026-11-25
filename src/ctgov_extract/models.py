"""Typed data models for trial records, split criteria and atomic criteria."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Deterministic trial record
# ---------------------------------------------------------------------------


class Intervention(BaseModel):
    type: str | None = None
    name: str
    other_names: list[str] = Field(default_factory=list)


class Arm(BaseModel):
    label: str
    type: str | None = None
    description: str | None = None
    interventions: list[str] = Field(default_factory=list)


class Enrollment(BaseModel):
    count: int | None = None
    type: str | None = Field(default=None, description="ACTUAL or ESTIMATED")


class StudyDate(BaseModel):
    date: str | None = Field(default=None, description="As reported: YYYY-MM or YYYY-MM-DD")
    type: str | None = Field(default=None, description="ACTUAL or ESTIMATED")


class Outcome(BaseModel):
    measure: str
    time_frame: str | None = None
    description: str | None = None


class Sponsor(BaseModel):
    name: str
    sponsor_class: str | None = None


class TrialRecord(BaseModel):
    """Clean, typed view of the fields of a ClinicalTrials.gov study that matter for screening."""

    nct_id: str
    brief_title: str | None = None
    official_title: str | None = None
    acronym: str | None = None
    overall_status: str | None = None
    phases: list[str] = Field(default_factory=list)
    study_type: str | None = None
    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)
    arms: list[Arm] = Field(default_factory=list)
    enrollment: Enrollment = Field(default_factory=Enrollment)
    start_date: StudyDate = Field(default_factory=StudyDate)
    primary_completion_date: StudyDate = Field(default_factory=StudyDate)
    completion_date: StudyDate = Field(default_factory=StudyDate)
    lead_sponsor: Sponsor | None = None
    primary_outcomes: list[Outcome] = Field(default_factory=list)
    secondary_outcomes: list[Outcome] = Field(default_factory=list)
    minimum_age: str | None = None
    maximum_age: str | None = None
    sex: str | None = None
    healthy_volunteers: bool | None = None
    num_locations: int = 0
    eligibility_text: str = ""
    last_update_posted: str | None = None


# ---------------------------------------------------------------------------
# Deterministic criteria splitting
# ---------------------------------------------------------------------------

CriterionKind = Literal["inclusion", "exclusion"]


class Criterion(BaseModel):
    """One criterion (a bullet, a numbered item or a free-standing line).

    ``start``/``end`` are character offsets into the full eligibility text, and
    ``text == eligibility_text[start:end]`` always holds. ``clean_text`` has
    markdown escapes removed and whitespace collapsed, for display and prompting.
    """

    id: str
    kind: CriterionKind
    text: str
    clean_text: str
    start: int
    end: int
    level: int = 0
    marker: str | None = None
    parent_id: str | None = None
    section: str | None = Field(default=None, description="Section header text, if any")
    group: str | None = Field(default=None, description="Sub-heading inside the section, if any")


class CriteriaSplit(BaseModel):
    nct_id: str | None = None
    criteria: list[Criterion] = Field(default_factory=list)
    preamble: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def inclusion(self) -> list[Criterion]:
        return [c for c in self.criteria if c.kind == "inclusion"]

    @property
    def exclusion(self) -> list[Criterion]:
        return [c for c in self.criteria if c.kind == "exclusion"]


# ---------------------------------------------------------------------------
# Atomic criteria (default LLM schema)
# ---------------------------------------------------------------------------


class Category(StrEnum):
    DEMOGRAPHICS = "demographics"
    CONDITION = "condition/diagnosis"
    MEDICAL_HISTORY = "medical history"
    MEDICATION = "medication/treatment"
    LABORATORY = "laboratory/biomarker"
    IMAGING = "imaging"
    PROCEDURE = "procedure"
    SCORE = "cognitive/functional score"
    REPRODUCTIVE = "pregnancy/reproductive"
    ADMINISTRATIVE = "consent/administrative"
    OTHER = "other"


class Operator(StrEnum):
    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    BETWEEN = "between"
    PRESENT = "present"
    ABSENT = "absent"


class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Time window as written, e.g. 'within 6 months before screening'")
    amount: float | None = Field(default=None, description="Numeric amount, e.g. 6")
    unit: str | None = Field(default=None, description="days, weeks, months or years")
    direction: Literal["before", "after", "during"] | None = Field(
        default=None, description="Relative to the anchor event (screening, randomization, ...)"
    )


EvidenceStatus = Literal["exact", "normalized", "not_found"]


class Evidence(BaseModel):
    source_text: str
    status: EvidenceStatus
    start: int | None = Field(default=None, description="Offset into the eligibility text")
    end: int | None = None


class AtomicCriterionDraft(BaseModel):
    """What the model is asked to return for each atomic criterion."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    concept: str = Field(description="Short normalized name of the concept, e.g. 'MMSE score'")
    operator: Operator | None = None
    value: float | str | None = None
    value_max: float | None = Field(
        default=None, description="Upper bound when operator is 'between'"
    )
    unit: str | None = None
    time_window: TimeWindow | None = None
    negated: bool = Field(
        default=False, description="True when the text states the absence of the concept"
    )
    is_biomarker: bool = False
    source_text: str = Field(description="Verbatim span copied from the criterion text")


class AtomicDraftList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AtomicCriterionDraft]


class AtomicCriterion(BaseModel):
    id: str
    criterion_id: str
    kind: CriterionKind
    category: Category
    concept: str
    operator: Operator | None = None
    value: float | str | None = None
    value_max: float | None = None
    unit: str | None = None
    time_window: TimeWindow | None = None
    negated: bool = False
    is_biomarker: bool = False
    evidence: Evidence


class Usage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0

    def add(self, other: Usage) -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.latency_s += other.latency_s


class CriterionFailure(BaseModel):
    criterion_id: str
    error: str


class CriteriaResult(BaseModel):
    nct_id: str | None = None
    model: str | None = None
    eligibility_text: str
    criteria: list[Criterion]
    atomic: list[AtomicCriterion]
    failures: list[CriterionFailure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    elapsed_s: float = 0.0

    @property
    def evidence_not_found(self) -> list[AtomicCriterion]:
        return [a for a in self.atomic if a.evidence.status == "not_found"]
