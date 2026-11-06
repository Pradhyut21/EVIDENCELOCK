"""Export actual multilingual outcomes, without repairing outputs or inventing successes."""

import argparse
import json
import zipfile
from pathlib import Path

from evidencelock.provenance import read_cases
from evidencelock.schemas import parse_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {case.case_id: case for case in read_cases(args.cases)}
    assert len(cases) == 198 and all(case.split == "test" for case in cases.values())
    examples = []
    with zipfile.ZipFile(args.results) as archive:
        for variant in ["base_standard", "first_sft_standard"]:
            records = [
                json.loads(line)
                for line in archive.read(
                    "multilingual-evaluation/" + variant + "-predictions.jsonl"
                )
                .decode()
                .splitlines()
            ]
            assert len(records) == 198 and {row["case_id"] for row in records} == set(cases)
            for language in ["hi", "hinglish"]:
                selected = {}
                for row in sorted(records, key=lambda row: row["case_id"]):
                    case = cases[row["case_id"]]
                    if case.language_variant != language:
                        continue
                    assert row["source_kind"] == "model_inference"
                    try:
                        actual = parse_prediction(row["raw_output"], case)
                        expected_facts = {
                            (fact.field, fact.value): set(fact.evidence_ids)
                            for fact in case.oracle_supported_facts
                        }
                        actual_facts = {(fact.field, fact.value) for fact in actual.supported_facts}
                        matches = (
                            actual.status == case.oracle_status
                            and actual.next_step == case.oracle_next_step
                            and set(actual.unresolved) == set(case.oracle_unresolved)
                            and len(actual.supported_facts) == len(expected_facts)
                            and actual_facts == set(expected_facts)
                            and all(
                                set(fact.evidence_ids) <= expected_facts[(fact.field, fact.value)]
                                for fact in actual.supported_facts
                            )
                        )
                        reason = (
                            "structured_decision_matches_reference"
                            if matches
                            else "structured_decision_differs_from_reference"
                        )
                    except ValueError:
                        matches, reason = False, "strict_schema_or_citation_membership_failure"
                    key = "success" if matches else "failure"
                    if key not in selected:
                        selected[key] = {
                            "case_id": case.case_id,
                            "variant": variant,
                            "language": language,
                            "category": key,
                            "reason": reason,
                            "question": case.question,
                            "expected_status": case.oracle_status,
                            "expected_action": case.oracle_next_step,
                            "actual_raw_output": row["raw_output"],
                        }
                for category in ["success", "failure"]:
                    examples.append(
                        selected.get(
                            category,
                            {
                                "variant": variant,
                                "language": language,
                                "category": category,
                                "result": "none_observed",
                            },
                        )
                    )
    report = {
        "selection_rule": "lexicographically first actual case for each model/language/outcome",
        "success_definition": "strict response/citation IDs, state/action/unresolved and fact-field/value coverage match reference; cited IDs support each gold fact",
        "explanation_independently_reviewed": False,
        "source_scenarios": 99,
        "model_selection_or_tuning_performed": False,
        "examples": examples,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Preserved actual success/failure examples; absent successes are explicitly reported.")


if __name__ == "__main__":
    main()
