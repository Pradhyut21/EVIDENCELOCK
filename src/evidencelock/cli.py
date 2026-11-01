"""Small, explicit research CLI. No command reports a skipped stage as success."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .curation import curate
from .evaluation import evaluate
from .generator import generate
from .oracle import decide
from .provenance import read_cases, sha256, write_json, write_jsonl
from .runtime import ExecutionBlocked, doctor
from .split import assign, verify


def main():
    parser = argparse.ArgumentParser(prog="evidencelock")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    gen = sub.add_parser("generate")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--families-per-lineage", type=int, default=3)
    for name in ["split", "curate", "verify", "reference", "evaluate", "baseline"]:
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        if name != "verify":
            command.add_argument("--output", type=Path, required=True)
        if name == "split":
            command.add_argument("--seed", type=int, default=42)
        if name == "evaluate":
            command.add_argument("--predictions", type=Path, required=True)
        if name == "baseline":
            command.add_argument("--adapter", type=Path)
        if name in ["reference", "evaluate", "baseline"]:
            command.add_argument("--subset", choices=["train", "validation", "test"])
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "generate":
            cases = generate(args.families_per_lineage)
            if args.output.exists():
                raise ValueError(
                    "Refusing to overwrite an existing dataset; use a new version path"
                )
            write_jsonl(args.output, (case.model_dump(mode="json") for case in cases))
            result = {
                "generated": len(cases),
                "sha256": sha256(args.output),
                "output": str(args.output),
            }
        else:
            cases = read_cases(args.input)
            for case in cases:
                truth = decide(case)
                if (
                    case.oracle_status,
                    case.oracle_supported_facts,
                    case.oracle_unresolved,
                    case.oracle_next_step,
                ) != (truth.status, truth.supported_facts, truth.unresolved, truth.next_step):
                    raise ValueError(f"Oracle fields drifted for {case.case_id}")
            if getattr(args, "subset", None):
                cases = [case for case in cases if case.split == args.subset]
                if not cases:
                    raise ValueError("Selected split is empty")
            if args.command == "split":
                if args.output.exists():
                    raise ValueError("Refusing to overwrite a split; create a new version")
                cases = assign(cases, args.seed)
                write_jsonl(args.output, (case.model_dump(mode="json") for case in cases))
                result = {
                    "counts": dict(Counter(case.split for case in cases)),
                    "seed": args.seed,
                    "input_sha256": sha256(args.input),
                    "sha256": sha256(args.output),
                    "families": {case.family_id: case.split for case in cases},
                    "template_lineages": {case.template_id: case.split for case in cases},
                }
                write_json(args.output.with_suffix(".manifest.json"), result)
            elif args.command == "verify":
                verify(cases)
                result = {
                    "verified": len(cases),
                    "split_counts": dict(Counter(case.split for case in cases)),
                    "sha256": sha256(args.input),
                }
            elif args.command == "curate":
                result = curate(cases, args.output)
            elif args.command == "reference":
                write_jsonl(
                    args.output,
                    (
                        {
                            "case_id": case.case_id,
                            "source_kind": "oracle_reference",
                            "raw_output": decide(case).model_dump_json(),
                        }
                        for case in cases
                    ),
                )
                result = {"reference_predictions": len(cases), "is_model_inference": False}
            elif args.command == "evaluate":
                result = evaluate(cases, args.predictions, args.output)
            else:
                from .inference import infer

                result = infer(cases, args.output, args.adapter)
        print(
            json.dumps(
                {"status": "PASS", "command": args.command, "result": result},
                ensure_ascii=False,
                default=str,
            )
        )
        return 0
    except ExecutionBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "command": args.command, "reason": str(exc)}))
        return 2
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "FAIL", "command": args.command, "reason": str(exc)}))
        return 1
