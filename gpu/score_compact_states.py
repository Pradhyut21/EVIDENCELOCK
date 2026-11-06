"""Classification diagnostic, not JSON drafting and not end-to-end task performance."""

import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

from classification_protocol import (
    LABELS,
    STATUSES,
    select_decision,
)
from classification_protocol import (
    compact_decision_protocol as decision_protocol,
)


def main():
    root = Path("/content/evidencelock-gpu")
    source = root / "evidencelock"
    output = root / "compact-state-diagnostic"
    output.mkdir(exist_ok=True)
    report = {
        "status": "RUNNING",
        "training_executed": False,
        "frozen_test_used": False,
        "task": "forced four-choice state classification only; not drafting/citation performance",
        "full_task_completion_claim": False,
        "rotations": 4,
    }

    def save():
        (output / "latest-result.json").write_text(json.dumps(report, indent=2))

    save()
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from evidencelock.provenance import read_cases

        cases = read_cases(source / "gpu/experiment-validation-cases.jsonl")
        assert len(cases) == 99 and all(c.split == "validation" for c in cases)
        revision = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"
        snapshot = Path(
            hf_hub_download(
                "CohereLabs/tiny-aya-fire",
                "config.json",
                revision=revision,
                local_files_only=True,
            )
        ).parent
        tok = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        import os

        original = json.loads(Path(os.environ["STATE_MODEL_REPORT"]).read_text())
        adapter = Path(original["adapter_directory"])
        weight = adapter / "adapter_model.safetensors"
        assert (
            hashlib.sha256(weight.read_bytes()).hexdigest()
            == original["adapter_sha256"]
        )
        report.update(
            model_revision=revision,
            adapter_sha256=original["adapter_sha256"],
            precision="FP16 NF4 double quantization for both base and adapter",
            validation_cases_sha256=hashlib.sha256(
                (source / "gpu/experiment-validation-cases.jsonl").read_bytes()
            ).hexdigest(),
        )
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="eager",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        summaries = {}
        for variant in ["base", "state_sft"]:
            if variant != "base":
                model = PeftModel.from_pretrained(
                    model, adapter, is_trainable=False, local_files_only=True
                )
            model.eval()
            counts = Counter()
            matrix = {s: {p: 0 for p in (*STATUSES, "ABSTAIN")} for s in STATUSES}
            mass_sum = 0.0
            top_in_choices = 0
            letter_counts = Counter()
            path = output / (variant + "-scores.jsonl")
            with path.open("w") as stream:
                for index, case in enumerate(cases):
                    votes = Counter()
                    for rotation in range(4):
                        protocol = decision_protocol(case, rotation=rotation)
                        text = tok.apply_chat_template(
                            [{"role": "user", "content": protocol["prompt"]}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        prefix_ids = tok(text, add_special_tokens=False)["input_ids"]
                        if len(prefix_ids) > 2048:
                            raise ValueError(
                                "Classification input exceeds explicit budget; no truncation"
                            )
                        label_ids = {}
                        for label in LABELS:
                            extended = tok(text + label, add_special_tokens=False)[
                                "input_ids"
                            ]
                            if (
                                extended[:-1] != prefix_ids
                                or len(extended) != len(prefix_ids) + 1
                            ):
                                raise ValueError(
                                    "Choice must be exactly one continuation token with stable prefix"
                                )
                            label_ids[label] = extended[-1]
                        if len(set(label_ids.values())) != 4:
                            raise ValueError("Choice tokens must be distinct")
                        inputs = tok(
                            text, add_special_tokens=False, return_tensors="pt"
                        ).to("cuda")
                        with torch.inference_mode():
                            result = model(**inputs, use_cache=False)
                        logits = result.logits[0, -1].float()
                        normalizer = torch.logsumexp(logits, dim=-1)
                        scores = {
                            label: float(logits[token].item())
                            for label, token in label_ids.items()
                        }
                        decision = select_decision(scores, protocol["options"])
                        top_id = int(logits.argmax().item())
                        mass = float(
                            sum(
                                torch.exp(logits[token] - normalizer).item()
                                for token in label_ids.values()
                            )
                        )
                        mass_sum += mass
                        top_in_choices += top_id in label_ids.values()
                        votes[decision["status"]] += 1
                        letter_counts[decision["selected_option"]] += 1
                        row = {
                            "case_id": case.case_id,
                            "variant": variant,
                            "rotation": rotation,
                            "actual_prompt": protocol["prompt"],
                            "observable_sha256": protocol["observable_sha256"],
                            "options": protocol["options"],
                            "label_token_ids": label_ids,
                            "actual_next_token_logits": scores,
                            **decision,
                            "choice_probability_mass_in_full_vocabulary": mass,
                            "full_vocabulary_top_token_id": top_id,
                            "full_vocabulary_top_token": tok.decode([top_id]),
                            "full_vocabulary_top_is_option": top_id
                            in label_ids.values(),
                        }
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        del result, logits, inputs
                    # Require a strict majority across rotations; ties are explicit abstentions.
                    state, n = votes.most_common(1)[0]
                    predicted = state if n >= 3 else "ABSTAIN"
                    matrix[case.oracle_status][predicted] += 1
                    counts["correct"] += predicted == case.oracle_status
                    counts["unanimous"] += n == 4
                    counts["abstain"] += predicted == "ABSTAIN"
                    print(
                        variant,
                        index + 1,
                        "/99 scored at all four option rotations",
                        flush=True,
                    )
                    gc.collect()
            import auditkit as ak

            records = [json.loads(line) for line in path.read_text().splitlines()]
            truth = {case.case_id: case.oracle_status for case in cases}
            samples = [
                ak.Sample(
                    id=row["case_id"] + ":rotation" + str(row["rotation"]),
                    input=row["actual_prompt"],
                    target=truth[row["case_id"]],
                    actual_output=row["status"],
                    metadata={
                        "source_kind": "model_option_scoring",
                        "task": "classification_only",
                        "rotation": row["rotation"],
                        "full_vocabulary_option_mass": row[
                            "choice_probability_mass_in_full_vocabulary"
                        ],
                    },
                )
                for row in records
            ]

            def exact_state(sample, actual):
                return float(actual == sample.target)

            native_scorer = ak.scorer(direction=ak.Direction.MAXIMIZE)(exact_state)
            run = ak.evaluate(samples, model="precomputed", scorers=native_scorer)
            audit_dir = output / ("audit-" + variant)
            audit_dir.mkdir(exist_ok=True)
            run.save(str(audit_dir / "auditkit-run.json"))
            summaries[variant] = {
                "case_count": 99,
                "case_rotation_scores": 396,
                "correct_strict_majority_state": counts["correct"],
                "abstentions": counts["abstain"],
                "rotation_unanimous_cases": counts["unanimous"],
                "confusion_matrix": matrix,
                "selected_option_counts": dict(letter_counts),
                "always_one_letter_rotation_accuracy": 0.25,
                "mean_choice_mass_in_full_vocabulary": mass_sum / 396,
                "full_vocabulary_top_is_option_count": top_in_choices,
                "scores_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "native_auditkit_fingerprint": run.fingerprint,
            }
            report["classification_only_results"] = summaries
            save()
        report["status"] = "PASS"
    except Exception as exc:
        report.update(
            status="FAIL", error_type=type(exc).__name__, error_message=str(exc)[:3000]
        )
    save()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
