import pytest

from ctgov_extract.errors import InvalidTrialIdError
from ctgov_extract.ids import is_nct_id, parse_nct_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("NCT01234567", "NCT01234567"),
        ("nct01234567", "NCT01234567"),
        ("  NcT01234567\n", "NCT01234567"),
        ("https://clinicaltrials.gov/study/NCT01234567", "NCT01234567"),
        ("https://clinicaltrials.gov/study/nct01234567?tab=results", "NCT01234567"),
        ("https://www.clinicaltrials.gov/ct2/show/NCT01234567", "NCT01234567"),
        ("https://clinicaltrials.gov/ct2/show/NCT01234567?term=alzheimer&rank=1", "NCT01234567"),
        ("http://clinicaltrials.gov/ct2/show/record/NCT01234567#wrapper", "NCT01234567"),
        ("clinicaltrials.gov/study/NCT01234567", "NCT01234567"),
        ("https://classic.clinicaltrials.gov/ct2/show/NCT01234567", "NCT01234567"),
        ("https://clinicaltrials.gov/ct2/show?id=NCT01234567", "NCT01234567"),
        ("https://clinicaltrials.gov/api/v2/studies/NCT01234567", "NCT01234567"),
    ],
)
def test_parse_valid(value: str, expected: str) -> None:
    assert parse_nct_id(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "NCT123",
        "NCT012345678",  # nine digits
        "01234567",
        "https://example.com/study/NCT01234567",  # wrong host
        "https://clinicaltrials.gov.evil.com/study/NCT01234567",
        "https://clinicaltrials.gov/search?cond=alzheimer",  # no ID
        "XNCT01234567",
    ],
)
def test_parse_invalid(value: str) -> None:
    with pytest.raises(InvalidTrialIdError):
        parse_nct_id(value)


def test_is_nct_id() -> None:
    assert is_nct_id("nct00000620")
    assert not is_nct_id("https://clinicaltrials.gov/study/NCT00000620")
