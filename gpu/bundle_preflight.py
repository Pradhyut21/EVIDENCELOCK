"""Validate the complete state-training bundle using only Python's standard library.

Runs locally before submission and remotely before dependency/model downloads.
Historical reports establish prior feasibility, not a new run or quality result.
"""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

MODEL = "CohereLabs/tiny-aya-fire"
REVISION = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"
SOURCE = "evidencelock/"
STATE = SOURCE + "data/state-only-v1/curated/train/"
PILOT = SOURCE + "data/pilot-v1/curated/"
REQUIRED = (
    "model-smoke/result.json",
    "align-smoke/latest-result.json",
    "historical-evidence-provenance.json",
    SOURCE + "pyproject.toml",
    SOURCE + "gpu/prepare_colab.py",
    SOURCE + "gpu/inspect_training_env.py",
    SOURCE + "gpu/requirements-linux-py312.txt",
    SOURCE + "gpu/download_recovery_model.py",
    SOURCE + "gpu/train_state_only.py",
    SOURCE + "gpu/score_compact_states.py",
    SOURCE + "gpu/smoke_checks.py",
    SOURCE + "gpu/decision_protocol.py",
    SOURCE + "gpu/classification_protocol.py",
    SOURCE + "gpu/compact_ledger.py",
    SOURCE + "gpu/experiment-validation-cases.jsonl",
    SOURCE + "src/evidencelock/__init__.py",
    SOURCE + "src/evidencelock/provenance.py",
    SOURCE + "src/evidencelock/schemas.py",
    STATE + "lineage.json",
    STATE + "source-lineage.json",
    STATE + "tokenization-preflight.json",
    STATE + "curatorkit/sft_messages.jsonl",
    PILOT + "train/lineage.json",
    PILOT + "train/curatorkit/sft_messages.jsonl",
    PILOT + "validation/lineage.json",
    PILOT + "validation/curatorkit/sft_messages.jsonl",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def validate_bundle(root):
    root = Path(root).resolve()
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    require(not missing, "Missing required bundle files: " + ", ".join(missing))

    def data(name):
        return (root / name).read_bytes()

    def document(name):
        return json.loads(data(name))

    def rows(name):
        return [json.loads(line) for line in data(name).decode().splitlines() if line.strip()]

    provenance = document("historical-evidence-provenance.json")
    for member in ("align-smoke/latest-result.json", "model-smoke/result.json"):
        origin = provenance[member]
        require(
            origin["member_sha256"] == digest(data(member)),
            "Historical report hash mismatch: " + member,
        )
        require(len(origin["source_archive_sha256"]) == 64, "Missing historical archive provenance")
        report = document(member)
        require(report.get("status") == "PASS", "Prior feasibility gate failed: " + member)
        require(
            report.get("model_id") == MODEL and report.get("model_revision") == REVISION,
            "Historical model revision mismatch: " + member,
        )
    baseline = document("model-smoke/result.json")
    require(
        baseline.get("model_loaded") is True and baseline.get("baseline_predictions", 0) > 0,
        "No verified historical model load/baseline inference",
    )
    smoke = document("align-smoke/latest-result.json")
    require(
        all(
            smoke.get(key) is True
            for key in ("adapter_trained", "adapter_reloaded", "lora_backward_verified")
        ),
        "Historical AlignTune training/reload/backward gate is incomplete",
    )
    require(
        smoke.get("actual_optimizer_steps", 0) > 0 and smoke.get("global_steps", 0) > 0,
        "Historical smoke has no actual optimizer update",
    )
    require(
        smoke.get("gradient_checks", {}).get("finite_nonzero_events", 0) > 0
        and smoke["gradient_checks"].get("nonfinite_events") == 0,
        "Historical smoke gradient verification failed",
    )

    lineage = document(STATE + "lineage.json")
    training = rows(STATE + "curatorkit/sft_messages.jsonl")
    sources = document(STATE + "source-lineage.json")
    require(lineage.get("curatorkit_executed") is True, "CuratorKIT execution is unverified")
    require(
        lineage["export_sha256"] == digest(data(STATE + "curatorkit/sft_messages.jsonl")),
        "State training export hash mismatch",
    )
    tokenization = document(STATE + "tokenization-preflight.json")
    require(
        tokenization.get("status") == "PASS"
        and tokenization.get("training_export_sha256") == lineage["export_sha256"]
        and tokenization.get("exposures_checked") == 864
        and tokenization.get("class_token_present_in_all_targets") is True
        and tokenization.get("max_input_tokens", 1537) <= 1536,
        "Offline tokenizer preflight is missing or refers to a different export",
    )
    require(
        len(training) == len(sources) == len(lineage["accepted_case_ids"]) == 864,
        "Expected 864 aligned training exposures",
    )
    require(
        [row["sample_id"] for row in sources] == lineage["accepted_case_ids"],
        "Curated export lineage order changed",
    )
    require(len(set(lineage["accepted_case_ids"])) == 864, "Duplicate exposure IDs")
    require(
        len({row["source_case_id"] for row in sources}) == 99, "Expected 99 unique source cases"
    )
    require(
        Counter(row["state"] for row in sources)
        == Counter(
            {key: 216 for key in ("SUPPORTED", "MISSING_EVIDENCE", "CONFLICT", "OUT_OF_SCOPE")}
        ),
        "State exposures are not balanced",
    )
    require(
        Counter(row["letter"] for row in sources) == Counter(dict.fromkeys("ABCD", 216)),
        "Letter exposures are not balanced",
    )
    require(
        all(row["source_split"] == "train" and row["rotation"] in range(4) for row in sources),
        "Non-training source or invalid rotation",
    )
    prompts = {}
    for row, origin in zip(training, sources, strict=True):
        messages = row["messages"]
        require(
            len(messages) == 2 and [m["role"] for m in messages] == ["user", "assistant"],
            "Unexpected state conversation shape",
        )
        prompt, letter = messages[0]["content"], messages[1]["content"]
        require(letter == origin["letter"] and letter in "ABCD", "State label/lineage mismatch")
        require(
            '"in_scope":' not in prompt and '"oracle_status":' not in prompt,
            "Hidden annotation in model input",
        )
        require(
            prompt not in prompts or prompts[prompt] == letter,
            "Contradictory labels for the same prompt",
        )
        prompts[prompt] = letter
    require(len(prompts) == 396, "Expected 396 distinct case/rotation prompts")
    original_train = document(PILOT + "train/lineage.json")
    train_ids = set(original_train["accepted_case_ids"])
    require({row["source_case_id"] for row in sources} == train_ids, "State source IDs changed")
    require(
        original_train["export_sha256"]
        == digest(data(PILOT + "train/curatorkit/sft_messages.jsonl")),
        "Original training export hash mismatch",
    )
    require(
        smoke["train_export_sha256"] == original_train["export_sha256"],
        "Smoke evidence refers to a different original training export",
    )

    validation = rows(SOURCE + "gpu/experiment-validation-cases.jsonl")
    val_ids = {row["case_id"] for row in validation}
    require(len(validation) == len(val_ids) == 99, "Expected 99 full validation cases")
    require(
        all(
            row.get("split") == "validation"
            and "question" in row
            and "evidence_items" in row
            and "oracle_status" in row
            for row in validation
        ),
        "Expected full validation cases, not Alpaca exports",
    )
    require(not train_ids & val_ids, "Training/validation source overlap")
    val_lineage = document(PILOT + "validation/lineage.json")
    require(set(val_lineage["accepted_case_ids"]) == val_ids, "Validation lineage mismatch")
    require(
        val_lineage["export_sha256"]
        == baseline["dataset_sha256"]
        == digest(data(PILOT + "validation/curatorkit/sft_messages.jsonl")),
        "Validation export differs from historical baseline evidence",
    )

    parsed = 0
    for path in (root / SOURCE).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        parsed += 1
        if path.parent == root / SOURCE / "gpu":
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in {
                    "protocol",
                    "decision_protocol",
                    "classification_protocol",
                    "compact_ledger",
                    "smoke_checks",
                }:
                    require(
                        (path.parent / (node.module + ".py")).is_file(),
                        "Missing local import " + node.module + " in " + path.name,
                    )
    lock = data(SOURCE + "gpu/requirements-linux-py312.txt").decode()
    require(
        all(
            pin in lock
            for pin in (
                "torch==2.11.0",
                "transformers==5.22.1",
                "aligntune==0.1.28",
                "auditkit==1.1.3",
            )
        ),
        "Pinned remote dependency lock changed",
    )
    require(
        not any("/curated/test/" in path.as_posix() for path in root.rglob("*")),
        "Frozen test files must not be included",
    )
    return {
        "status": "PASS",
        "required_files_verified": len(REQUIRED),
        "python_files_parsed": parsed,
        "training_exposures": 864,
        "unique_source_cases": 99,
        "unique_prompts": 396,
        "validation_cases": 99,
        "historical_evidence_reused": True,
        "new_training_executed": False,
        "full_task_completion_claim": False,
        "required_file_sha256": {name: digest(data(name)) for name in REQUIRED},
    }


if __name__ == "__main__":
    import sys

    result = validate_bundle(Path(sys.argv[1]))
    (Path(sys.argv[1]) / "fresh-bundle-preflight.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
