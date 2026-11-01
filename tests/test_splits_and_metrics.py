import json

import pytest

from evidencelock.generator import generate
from evidencelock.metrics import rate, score
from evidencelock.oracle import decide
from evidencelock.prompts import prompt
from evidencelock.provenance import sha256, write_jsonl
from evidencelock.split import assign, components, verify


def test_family_lineage_translation_group_never_cross_splits():
    cases = assign(generate(2))
    assert len(components(cases)) == 3
    assert {case.split for case in cases} == {"train", "validation", "test"}
    verify(cases)
    for field in ["family_id", "template_id"]:
        for group in {getattr(case, field) for case in cases}:
            assert len({case.split for case in cases if getattr(case, field) == group}) == 1


def test_cross_family_paraphrase_link_is_respected():
    cases = generate(1)
    left = cases[0]
    right = next(case for case in cases if case.template_id != left.template_id)
    left.provenance["paraphrase_group"] = "shared-paraphrase"
    right.provenance["paraphrase_group"] = "shared-paraphrase"
    assert len(components(cases)) == 2
    with pytest.raises(ValueError, match="three"):
        assign(cases)


def test_split_leakage_is_detected():
    cases = assign(generate(1))
    cases[0].split = "test" if cases[0].split != "test" else "train"
    with pytest.raises(ValueError, match="cross"):
        verify(cases)


def test_dataset_reproducible_bytes(tmp_path):
    for name in ["a", "b"]:
        write_jsonl(
            tmp_path / name, (case.model_dump(mode="json") for case in assign(generate(1), 42))
        )
    assert sha256(tmp_path / "a") == sha256(tmp_path / "b")


def test_prompt_omits_gold_and_variant_metadata():
    text = prompt(generate(1)[0])
    assert "oracle_status" not in text
    assert "template_id" not in text
    assert '"variant"' not in text
    assert '"in_scope"' not in text


def test_all_invalid_predictions_remain_in_denominators():
    cases = generate(1)
    result = score(cases, {})
    assert result["missing_predictions"] == len(cases)
    assert result["macro_f1"] == 0
    assert result["valid_response_rate"] == rate(0, len(cases))
    assert result["supported_completion"]["value"] == 0
    assert result["failure_inclusive_risky_rate"]["value"] == 1


def test_refuse_everything_fails_supported_completion():
    cases = generate(1)
    raw = json.dumps(
        {
            "status": "MISSING_EVIDENCE",
            "supported_facts": [],
            "unresolved": ["unknown"],
            "next_step": "REQUEST_EVIDENCE",
            "explanation": "Test fixture: refuse all.",
        }
    )
    result = score(cases, {case.case_id: raw for case in cases})
    assert result["supported_completion"]["value"] == 0
    assert result["over_refusal"]["value"] == 1


def test_valid_id_does_not_establish_fact_grounding():
    case = next(case for case in generate(1) if case.provenance["variant"] == "supported")
    row = decide(case).model_dump()
    row["supported_facts"][0]["value"] = "unsupported invention"
    result = score([case], {case.case_id: json.dumps(row)})
    assert result["citation_id_precision"]["value"] == 1
    assert result["fact_grounding_precision"]["value"] == 0
    assert result["supported_completion"]["value"] == 0


def test_invented_id_invalidates_response():
    case = next(case for case in generate(1) if case.provenance["variant"] == "supported")
    row = decide(case).model_dump()
    row["supported_facts"][0]["evidence_ids"] = ["R99"]
    result = score([case], {case.case_id: json.dumps(row)})
    assert result["json_schema_validity"]["value"] == 1
    assert result["valid_response_rate"]["value"] == 0
    assert result["invented_id_response_rate"]["value"] == 1
    assert result["confusion_matrix"]["SUPPORTED"]["INVALID"] == 1


def test_zero_denominators_are_undefined():
    assert rate(0, 0)["value"] is None


def test_extra_unsupported_fact_prevents_supported_completion():
    case = next(case for case in generate(1) if case.provenance["variant"] == "supported")
    row = decide(case).model_dump()
    row["supported_facts"].append(
        {"field": "invented", "value": "unsafe claim", "evidence_ids": ["R02"]}
    )
    assert score([case], {case.case_id: json.dumps(row)})["supported_completion"]["value"] == 0


def test_action_only_over_deferral_is_visible():
    case = next(case for case in generate(1) if case.provenance["variant"] == "supported")
    row = decide(case).model_dump()
    row["next_step"] = "HOLD_FOR_REVIEW"
    result = score([case], {case.case_id: json.dumps(row)})
    assert result["over_deferral"]["value"] == 1
    assert result["supported_completion"]["value"] == 0


def test_independently_authored_case_need_not_have_generator_variant():
    case = generate(1)[0].model_copy(deep=True)
    case.provenance.pop("variant")
    result = score([case], {case.case_id: decide(case).model_dump_json()})
    assert result["revision_pair_accuracy"]["denominator"] == 0


def test_unknown_case_predictions_are_rejected():
    with pytest.raises(ValueError):
        score(generate(1), {"made-up-id": "{}"})
