import json
from datetime import timedelta

import pytest
from pydantic import ValidationError

from evidencelock.generator import generate
from evidencelock.oracle import decide
from evidencelock.schemas import EvidenceCase, parse_prediction


@pytest.fixture(scope="module")
def variants():
    cases = generate(1)
    family = cases[0].family_id
    return {case.provenance["variant"]: case for case in cases if case.family_id == family}


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("missing", "MISSING_EVIDENCE"),
        ("supported", "SUPPORTED"),
        ("conflict", "CONFLICT"),
        ("corrected", "SUPPORTED"),
        ("withdrawn", "MISSING_EVIDENCE"),
        ("unauthorized", "SUPPORTED"),
        ("stale", "MISSING_EVIDENCE"),
        ("out_of_scope", "OUT_OF_SCOPE"),
        ("irrelevant_missing", "SUPPORTED"),
        ("report_only", "SUPPORTED"),
        ("correction_withdrawn", "SUPPORTED"),
    ],
)
def test_oracle_truth_table(variants, variant, expected):
    assert decide(variants[variant]).status == expected


def test_revision_counterfactuals(variants):
    before = variants["supported"]
    withdrawn = before.model_copy(deep=True)
    withdrawn.evidence_items[-1].retracted = True
    assert decide(before).status == "SUPPORTED"
    assert decide(withdrawn).status == "MISSING_EVIDENCE"
    conflict = variants["conflict"]
    assert decide(conflict).next_step == "HOLD_FOR_REVIEW"
    assert decide(variants["corrected"]).next_step == "DRAFT_FOR_HUMAN_REVIEW"


def test_reorder_does_not_resolve_conflict(variants):
    case = variants["conflict"]
    assert decide(case) == decide(
        case.model_copy(update={"evidence_items": list(reversed(case.evidence_items))})
    )


def test_newer_unauthorized_record_does_not_override(variants):
    result = decide(variants["unauthorized"])
    assert result.supported_facts[0].evidence_ids == ["R02"]


def test_withdrawn_replacement_does_not_resurrect_old_record(variants):
    case = variants["correction_withdrawn"].model_copy(deep=True)
    case.evidence_items[1].retracted = True  # Remaining independent source withdrawn.
    assert decide(case).status == "MISSING_EVIDENCE"


def test_supported_fact_question_uses_no_action(variants):
    case = variants["supported"].model_copy(deep=True)
    case.query_policy.request_kind = "fact"
    assert decide(case).next_step == "NO_ACTION"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda row: row.update(extra_field=True),
        lambda row: row.update(oracle_status="APPROVED"),
        lambda row: row["evidence_items"].append(row["evidence_items"][0]),
        lambda row: row["evidence_items"][0].update(supersedes_id="R99"),
        lambda row: row["evidence_items"][0].update(observed_at="2026-11-01T00:00:00"),
        lambda row: row["evidence_items"][0].update(retracted="false"),
        lambda row: row["oracle_supported_facts"][0].update(evidence_ids=["R99"]),
    ],
)
def test_malformed_records_rejected(variants, corrupt):
    row = variants["supported"].model_dump(mode="json")
    corrupt(row)
    with pytest.raises(ValidationError):
        EvidenceCase.model_validate_json(json.dumps(row))


def test_future_assertion_rejected(variants):
    row = variants["supported"].model_dump(mode="json")
    row["evidence_items"][0]["asserted_at"] = (
        variants["supported"].as_of + timedelta(hours=1)
    ).isoformat()
    with pytest.raises(ValidationError):
        EvidenceCase.model_validate_json(json.dumps(row))


@pytest.mark.parametrize("raw", ["not json", "```json\n{}\n```", '{"status":"SUPPORTED"}'])
def test_prediction_not_silently_repaired(variants, raw):
    with pytest.raises(ValueError):
        parse_prediction(raw, variants["supported"])


def test_invalid_prediction_citation_rejected(variants):
    raw = decide(variants["supported"]).model_dump()
    raw["supported_facts"][0]["evidence_ids"] = ["R99"]
    with pytest.raises(ValueError):
        parse_prediction(json.dumps(raw), variants["supported"])
