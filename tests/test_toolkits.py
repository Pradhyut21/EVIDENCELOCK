import json

from curatorkit import Curator, CuratorConfig

from evidencelock.curation import curate
from evidencelock.evaluation import evaluate
from evidencelock.generator import generate
from evidencelock.oracle import decide
from evidencelock.provenance import write_jsonl
from evidencelock.split import assign


def test_real_curatorkit_dedup_and_rejection(tmp_path):
    row = {
        "instruction": "Answer this synthetic testing question using its context.",
        "input": "A synthetic record has the verified value 42.",
        "output": "The value is 42.",
    }
    source = tmp_path / "input.jsonl"
    write_jsonl(source, [row, row, {**row, "output": ""}])
    result = Curator(
        CuratorConfig(
            dataset=str(source),
            format="alpaca",
            min_tokens=1,
            dedup="exact",
            output_dir=str(tmp_path / "curated"),
            export_formats=["messages"],
        )
    ).run()
    assert len(result.passed) == 1
    assert len(result.rejected) == 1  # The toolkit reports quality rejection separately.
    dedup = result.stage_counts["ExactDeduplicator"]
    assert dedup["input_count"] - dedup["output_count"] == 1
    assert (tmp_path / "curated" / "manifest.json").is_file()


def test_real_curatorkit_preserves_training_targets_and_evidence(tmp_path):
    cases = assign(generate(1))
    manifests = curate(cases, tmp_path)
    assert sum(item["generated"] for item in manifests.values()) == len(cases)
    assert all(
        item["generated"] == item["accepted"] + item["rejected"] for item in manifests.values()
    )
    assert all(item["accepted"] == item["exported"] for item in manifests.values())


def test_real_auditkit_on_explicit_test_fixtures(tmp_path):
    cases = generate(1)[:4]
    source = tmp_path / "predictions.jsonl"
    write_jsonl(
        source,
        (
            {
                "case_id": case.case_id,
                "source_kind": "test_fixture",
                "raw_output": decide(case).model_dump_json(),
            }
            for case in cases
        ),
    )
    result = evaluate(cases, source, tmp_path / "audit")
    assert result["source_kind"] == "test_fixture"
    assert result["model_improvement_measured"] is False
    assert result["toolkit"]["fingerprint"]
    assert json.loads((tmp_path / "audit" / "auditkit-run.json").read_text())
