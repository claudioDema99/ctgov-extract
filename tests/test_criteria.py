import pytest

from ctgov_extract.criteria import clean_text, detect_header, split_criteria
from tests.conftest import FIXTURE_IDS, eligibility


@pytest.mark.parametrize("nct_id", FIXTURE_IDS)
def test_offsets_are_exact_for_real_trials(nct_id: str) -> None:
    text = eligibility(nct_id)
    split = split_criteria(text, nct_id)
    assert split.criteria
    for c in split.criteria:
        assert text[c.start : c.end] == c.text
        assert c.text == c.text.strip()
        assert c.clean_text
    starts = [c.start for c in split.criteria]
    assert starts == sorted(starts)
    ids = [c.id for c in split.criteria]
    assert len(ids) == len(set(ids))


def test_destiny_breast03_numbered_and_nested() -> None:
    split = split_criteria(eligibility("NCT03529110"))
    by_id = {c.id: c for c in split.criteria}
    assert len(split.inclusion) == 9
    assert len(split.exclusion) == 13
    assert by_id["inc-3"].text == "Pathologically documented breast cancer that:"
    assert [c.id for c in split.criteria if c.parent_id == "inc-3"] == [
        "inc-3.1",
        "inc-3.2",
        "inc-3.3",
    ]
    assert by_id["inc-3.2"].level == 1
    assert by_id["inc-3.2"].text.startswith("has confirmed HER2-positive expression")
    # numbered sub-items of exclusion 2, then bullet sub-items under exclusion 4
    assert by_id["exc-2.5"].clean_text == "LVEF < 50% within 28 d prior to randomization"
    assert by_id["exc-4.1"].marker == "*"
    assert by_id["exc-4.2"].parent_id == "exc-4"
    assert by_id["exc-5"].level == 0


def test_lecanemab_messy_headers_and_groups() -> None:
    split = split_criteria(eligibility("NCT03887455"))
    sections = {c.section for c in split.criteria}
    assert {
        "Core Study: Inclusion Criteria",
        "Key Inclusion Criteria that must be met by all participants",
        "Extension Phase: Inclusion Criteria",
        "Extension Phase Part B: Inclusion Criteria",
        "Exclusion Criteria",
        "Extension Phase: Exclusion Criteria",
        "Extension Phase Part B: Exclusion Criteria",
    } == sections
    first = split.criteria[0]
    assert first.kind == "inclusion"
    assert first.group is not None and first.group.startswith(
        "Diagnosis: Mild Cognitive Impairment"
    )
    mmse = next(c for c in split.criteria if c.text.startswith("Mini mental state"))
    assert "\\>=22" in mmse.text  # raw text keeps the escape
    assert "score >=22 at Screening" in mmse.clean_text
    # "\u2022" bullet and "-In countries" (dash without a space) are both list items
    assert any(c.marker == "\u2022" for c in split.criteria)
    last = split.criteria[-1]
    assert last.kind == "exclusion" and last.marker == "-"
    assert last.text.startswith("In countries where lecanemab")


def test_preamble_before_first_header() -> None:
    split = split_criteria(eligibility("NCT01844505"))
    assert split.preamble is not None and "BMSStudyConnect" in split.preamble
    assert len(split.inclusion) == 5
    assert len(split.exclusion) == 5


def test_inclusion_only_trial() -> None:
    split = split_criteria(eligibility("NCT00000620"))
    assert len(split.inclusion) == 4
    assert split.exclusion == []
    assert split.warnings == []


def test_inline_header_content_and_markdown_bold() -> None:
    text = (
        "**Inclusion Criteria:**\n- Age >= 18\n\nEXCLUSION CRITERIA: Pregnant women\n- Prior chemo"
    )
    split = split_criteria(text)
    assert [(c.kind, c.text) for c in split.criteria] == [
        ("inclusion", "Age >= 18"),
        ("exclusion", "Pregnant women"),
        ("exclusion", "Prior chemo"),
    ]
    for c in split.criteria:
        assert text[c.start : c.end] == c.text


def test_letters_nested_under_colon_item_without_indentation() -> None:
    text = "Inclusion Criteria:\n1. Disease:\na. stage IV\nb. measurable\n2. ECOG 0-1\n"
    split = split_criteria(text)
    assert [(c.id, c.parent_id) for c in split.criteria] == [
        ("inc-1", None),
        ("inc-1.1", "inc-1"),
        ("inc-1.2", "inc-1"),
        ("inc-2", None),
    ]


def test_roman_numerals_after_letters() -> None:
    text = (
        "Exclusion Criteria:\n1. Any of:\n   (i) item one\n   (ii) item two\n   (iii) item three\n"
        "   (iv) item four\n   (v) item five\n2. Next\n"
    )
    split = split_criteria(text)
    assert [(c.id, c.parent_id, c.marker) for c in split.criteria] == [
        ("exc-1", None, "1."),
        ("exc-1.1", "exc-1", "(i)"),
        ("exc-1.2", "exc-1", "(ii)"),
        ("exc-1.3", "exc-1", "(iii)"),
        ("exc-1.4", "exc-1", "(iv)"),
        ("exc-1.5", "exc-1", "(v)"),
        ("exc-2", None, "2."),
    ]


def test_continuation_lines() -> None:
    text = (
        "Inclusion Criteria:\n\n* Adequate organ function,\n  defined as below\n\n"
        "    Note: indented paragraph after a blank line\n* Second item\nlazy continuation\n"
    )
    split = split_criteria(text)
    assert len(split.criteria) == 2
    first, second = split.criteria
    assert first.clean_text == (
        "Adequate organ function, defined as below Note: indented paragraph after a blank line"
    )
    assert second.clean_text == "Second item lazy continuation"


def test_unbulleted_lines_without_headers() -> None:
    text = "Age 18 or older\nSigned informed consent\n\nNo prior chemotherapy"
    split = split_criteria(text)
    assert [c.text for c in split.criteria] == [
        "Age 18 or older",
        "Signed informed consent",
        "No prior chemotherapy",
    ]
    assert all(c.kind == "inclusion" for c in split.criteria)
    assert split.warnings


def test_not_a_bullet_and_not_a_header() -> None:
    text = (
        "Exclusion Criteria:\n\n* Item one\n\n-5 is not a bullet\n\n"
        "Patients meeting any exclusion criteria of the parent study\n"
    )
    split = split_criteria(text)
    assert [c.text for c in split.criteria] == [
        "Item one",
        "-5 is not a bullet",
        "Patients meeting any exclusion criteria of the parent study",
    ]
    assert all(c.kind == "exclusion" for c in split.criteria)


def test_crlf_and_empty_text() -> None:
    text = "Inclusion Criteria:\r\n\r\n* A\r\n* B\r\n"
    split = split_criteria(text)
    assert [c.text for c in split.criteria] == ["A", "B"]
    assert text[split.criteria[1].start : split.criteria[1].end] == "B"
    empty = split_criteria("")
    assert empty.criteria == [] and empty.warnings == []


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("Inclusion Criteria:", "inclusion"),
        ("Exclusion Criteria", "exclusion"),
        ("INCLUSION CRITERIA", "inclusion"),
        ("Key exclusion criteria for the extension:", "exclusion"),
        ("## Inclusion criteria", "inclusion"),
        ("Exclusion:", "exclusion"),
    ],
)
def test_detect_header(line: str, kind: str) -> None:
    header = detect_header(line)
    assert header is not None and header.kind == kind


@pytest.mark.parametrize(
    "line",
    [
        "* Inclusion Criteria:",  # a list item, not a header
        "Inclusion of patients with prior surgery is allowed",
        "Diagnosis: Mild Cognitive Impairment:",
    ],
)
def test_not_headers(line: str) -> None:
    assert detect_header(line) is None


def test_clean_text() -> None:
    assert clean_text("score \\>=22 and \\<= 30\n  at  screening \\[mm\\]") == (
        "score >=22 and <= 30 at screening [mm]"
    )
