from ctgov_extract.evidence import check_evidence_fields, locate, normalize, validate_evidence


def test_exact_match_offsets_are_absolute() -> None:
    ev = validate_evidence("LVEF", "History of LVEF < 50%", base_offset=100)
    assert ev.status == "exact"
    assert (ev.start, ev.end) == (111, 115)


def test_markdown_escapes_and_whitespace_are_normalized() -> None:
    raw = "Mini mental state examination (MMSE) score \\>=22 at Screening\n  and Baseline"
    ev = validate_evidence("MMSE) score >=22 at Screening and Baseline", raw)
    assert ev.status == "normalized"
    assert ev.start is not None and ev.end is not None
    assert raw[ev.start : ev.end] == "MMSE) score \\>=22 at Screening\n  and Baseline"


def test_unicode_signs_and_case() -> None:
    raw = "Adults \u226518 y old"
    status, start, end = locate("adults >=18 Y OLD", raw)
    assert status == "normalized"
    assert raw[start:end] == raw  # type: ignore[misc]


def test_not_found_is_flagged_not_accepted() -> None:
    ev = validate_evidence("MMSE score of at least 24", "MMSE score \\>=22 at Screening")
    assert ev.status == "not_found"
    assert ev.start is None and ev.end is None
    assert validate_evidence("   ", "anything").status == "not_found"


def test_normalize() -> None:
    assert normalize("  a\u2013b \u2264 3\u00a0mg  ") == "a-b <= 3 mg"
    assert normalize("ABC", fold_case=True) == "abc"


def test_check_evidence_fields_walks_nested_data() -> None:
    eligibility = "* HER2-positive as confirmed by central laboratory\n* LVEF \\< 50%"
    other = "Trastuzumab Deruxtecan\nProgression-Free Survival"
    data = {
        "biomarkers": [
            {"name": "HER2", "evidence": "HER2-positive as confirmed by central laboratory"},
            {"name": "LVEF", "evidence": "LVEF < 50%"},
            {"name": "made up", "evidence": "PD-L1 >= 50%"},
        ],
        "outcome": {"quote": "Progression-Free Survival"},
        "unrelated": "not checked",
    }
    checks = check_evidence_fields(data, eligibility, other)
    by_path = {c.path: c for c in checks}
    assert set(by_path) == {
        "$.biomarkers[0].evidence",
        "$.biomarkers[1].evidence",
        "$.biomarkers[2].evidence",
        "$.outcome.quote",
    }
    assert by_path["$.biomarkers[0].evidence"].status == "exact"
    assert by_path["$.biomarkers[1].evidence"].status == "normalized"
    lvef = by_path["$.biomarkers[1].evidence"]
    assert eligibility[lvef.start : lvef.end] == "LVEF \\< 50%"  # type: ignore[misc]
    assert by_path["$.biomarkers[2].evidence"].status == "not_found"
    assert by_path["$.outcome.quote"].location == "other_fields"
    assert by_path["$.outcome.quote"].start is None
