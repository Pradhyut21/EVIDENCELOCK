"""Real CuratorKIT curation, with independent semantic/citation preservation checks."""

from importlib.metadata import version
from pathlib import Path

from .oracle import decide
from .prompts import prompt
from .provenance import sha256, write_json, write_jsonl
from .schemas import parse_prediction
from .split import verify


def curate(cases, output: Path):
    from curatorkit import Curator, CuratorConfig

    verify(cases)
    manifests = {}
    for split in ["train", "validation", "test"]:
        subset = [case for case in cases if case.split == split]
        if not subset:
            raise ValueError(f"Empty {split} split")
        directory = output / split
        source = directory / "input.jsonl"
        rows = [
            {
                "case_id": case.case_id,
                "instruction": prompt(case),
                "input": "",
                "output": decide(case).model_dump_json(),
            }
            for case in subset
        ]
        write_jsonl(source, rows)
        result = Curator(
            CuratorConfig(
                dataset=str(source),
                format="alpaca",
                min_tokens=1,
                max_tokens=10000,
                dedup="exact",
                clean=True,
                output_dir=str(directory / "curatorkit"),
                export_formats=["messages", "alpaca"],
                name=f"evidencelock-{split}",
            )
        ).run()
        by_id = {case.case_id: case for case in subset}
        accepted_ids = []
        for sample in result.passed:
            case = by_id[sample.metadata["case_id"]]
            if sample.instruction != prompt(case) or sample.input:
                raise ValueError("Curator changed the evidence prompt; semantic review required")
            actual = parse_prediction(sample.output, case)
            if actual.model_dump() != decide(case).model_dump():
                raise ValueError("Curator changed the supervised target")
            accepted_ids.append(case.case_id)
        export = directory / "curatorkit" / "sft_messages.jsonl"
        import json

        exported = [json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()]
        dedup = result.stage_counts.get("ExactDeduplicator", {})
        deduplicated = dedup.get("input_count", 0) - dedup.get("output_count", 0)
        if len(rows) != len(result.passed) + len(result.rejected) + deduplicated or len(
            exported
        ) != len(accepted_ids):
            raise ValueError("Curation counts do not reconcile")
        quality_ids = {sample.metadata["case_id"] for sample in result.rejected}
        duplicate_ids = set(by_id) - set(accepted_ids) - quality_ids
        if len(duplicate_ids) != deduplicated:
            raise ValueError("Could not trace every deduplicated case")
        write_jsonl(
            directory / "rejected-with-lineage.jsonl",
            [
                {
                    "case_id": sample.metadata["case_id"],
                    "reason": sample.rejection_reason,
                    "stage": sample.rejecting_step,
                }
                for sample in result.rejected
            ]
            + [
                {"case_id": case_id, "reason": "duplicate_removed", "stage": "ExactDeduplicator"}
                for case_id in sorted(duplicate_ids)
            ],
        )
        for row, case_id in zip(exported, accepted_ids, strict=True):
            messages = row["messages"]
            case = next(case for case in subset if case.case_id == case_id)
            if messages[0]["content"] != prompt(case):
                raise ValueError("Trainer export changed the evidence prompt")
            if parse_prediction(messages[-1]["content"], case) != decide(case):
                raise ValueError("Trainer export changed the decision")
        manifest = {
            "toolkit": "curatorkit",
            "version": version("curatorkit"),
            "api": "Curator(CuratorConfig(...)).run",
            "split": split,
            "generated": len(rows),
            "accepted": len(result.passed),
            "rejected": len(result.rejected) + deduplicated,
            "quality_rejected": len(result.rejected),
            "deduplicated": deduplicated,
            "exported": len(exported),
            "accepted_case_ids": accepted_ids,
            "stage_counts": result.stage_counts,
            "hallucination_judge_used": False,
            "input_sha256": sha256(source),
            "export_sha256": sha256(export),
        }
        write_json(directory / "lineage.json", manifest)
        manifests[split] = manifest
    write_json(output / "manifest.json", manifests)
    return manifests
