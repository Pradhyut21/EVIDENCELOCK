"""AuditKIT consumes preserved predictions; custom metrics remain explicitly custom."""

import json
from importlib.metadata import version
from pathlib import Path

from .metrics import score
from .prompts import prompt
from .provenance import sha256, write_json
from .schemas import parse_prediction


def evaluate(cases, prediction_file: Path, output: Path):
    import auditkit as ak

    records = [
        json.loads(line) for line in prediction_file.read_text(encoding="utf-8").splitlines()
    ]
    if len({row["case_id"] for row in records}) != len(records):
        raise ValueError("Duplicate prediction IDs")
    kinds = {row["source_kind"] for row in records}
    if len(kinds) != 1 or not kinds <= {"model_inference", "oracle_reference", "test_fixture"}:
        raise ValueError("A run must declare exactly one supported prediction origin")
    predictions = {row["case_id"]: row["raw_output"] for row in records}
    case_map = {case.case_id: case for case in cases}
    if set(predictions) - set(case_map):
        raise ValueError("Unknown prediction IDs")
    samples = [
        ak.Sample(
            id=case.case_id,
            input=prompt(case),
            target=case.oracle_status,
            actual_output=predictions.get(case.case_id, ""),
            metadata={"source_kind": next(iter(kinds))},
        )
        for case in cases
    ]

    def exact_status_action(sample, raw):
        case = case_map[sample.id]
        try:
            parsed = parse_prediction(raw, case)
            return float(
                parsed.status == case.oracle_status and parsed.next_step == case.oracle_next_step
            )
        except ValueError:
            return 0.0

    scorer = ak.scorer(direction=ak.Direction.MAXIMIZE)(exact_status_action)
    run = ak.evaluate(samples, model="precomputed", scorers=scorer)
    result = score(cases, predictions)
    result.update(
        source_kind=next(iter(kinds)),
        model_improvement_measured=False,
        toolkit={
            "name": "auditkit",
            "version": version("auditkit"),
            "api": "evaluate(Sample(actual_output=...), model='precomputed', scorers=custom)",
            "fingerprint": run.fingerprint,
        },
        predictions_sha256=sha256(prediction_file),
    )
    write_json(output / "metrics.json", result)
    run.save(str(output / "auditkit-run.json"))
    return result
