"""Controlled one-epoch AlignTune experiment with paired validation evaluation.

Consumes all 99 real CuratorKIT training exports. Never uses the frozen test set.
Runs offline against the pinned model weights already cached by model_smoke.py.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from smoke_checks import validate_cached_weights

MODEL = "CohereLabs/tiny-aya-fire"
REVISION = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_ADAPTER_SHA = "78c8018037f7c33c728353f5795a6ed7d1b01affa1f29ec8243d2bd5216a5c52"


def main():
    source = Path(__file__).resolve().parents[1]
    root = source.parent
    original = json.loads((root / "controlled-experiment/latest-result.json").read_text())
    attempt = Path(original["attempt_dir"])
    if not attempt.resolve().is_relative_to((root / "controlled-experiment").resolve()):
        raise ValueError("Invalid original attempt path")
    parent = root / "recovered-validation"
    parent.mkdir(exist_ok=True)
    output = parent / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir()
    adapter = attempt / "adapter"
    report = {
        "status": "RUNNING",
        "phase": "verify_saved_adapter",
        "attempt_dir": str(output),
        "original_training_result": original,
        "training_quality_gate": "FAIL_NONFINITE_GRADIENT",
        "frozen_test_used": False,
        "model_improvement_measured": False,
        "additional_training_executed": False,
    }

    def save():
        text = json.dumps(report, indent=2, default=str)
        (output / "result.json").write_text(text)
        (parent / "latest-result.json").write_text(text)

    save()
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from evidencelock.evaluation import evaluate
        from evidencelock.provenance import read_cases

        weights = adapter / "adapter_model.safetensors"
        if digest(weights) != EXPECTED_ADAPTER_SHA:
            raise ValueError("Saved adapter differs from the inspected archive")
        saved = load_file(str(weights), device="cpu")
        if not saved or any(not bool(torch.isfinite(t).all()) for t in saved.values()):
            raise ValueError("Saved adapter has nonfinite tensor values")
        report.update(adapter_sha256=digest(weights), all_saved_adapter_values_finite=True)
        del saved
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        snapshot = Path(
            hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True)
        ).parent
        report["cache_validation"] = validate_cached_weights(snapshot)
        tok = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        baseline_report = json.loads((root / "model-smoke/result.json").read_text())
        lineage = json.loads((source / "data/pilot-v1/curated/train/lineage.json").read_text())
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            snapshot,
            trust_remote_code=False,
            local_files_only=True,
            device_map={"": 0},
            dtype=torch.float16,
            quantization_config=quantization,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        cases = read_cases(source / "gpu/experiment-validation-cases.jsonl")
        validation = source / "data/pilot-v1/curated/validation/curatorkit/sft_messages.jsonl"
        val_lineage = json.loads((validation.parents[1] / "lineage.json").read_text())
        if (
            digest(validation) != val_lineage["export_sha256"]
            or digest(validation) != baseline_report["dataset_sha256"]
        ):
            raise ValueError("Validation export checksum mismatch")
        rows = [json.loads(line) for line in validation.read_text().splitlines()]
        case_map = {case.case_id: case for case in cases}
        ids = val_lineage["accepted_case_ids"]
        if len(rows) != 99 or len(ids) != 99 or set(ids) != set(case_map):
            raise ValueError("Expected exactly the 99 validation cases")
        if set(ids) & set(lineage["accepted_case_ids"]):
            raise ValueError("Training/validation overlap")
        report.update(
            validation_export_sha256=digest(validation),
            validation_cases=99,
            decoding={"max_new_tokens": 256, "do_sample": False},
        )

        def generate_validation(model, path, adapter_path):
            model.eval()
            model.config.use_cache = True
            with path.open("w", encoding="utf-8") as stream:
                for index, (case_id, row) in enumerate(zip(ids, rows, strict=True)):
                    inputs = tok.apply_chat_template(
                        row["messages"][:-1],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")
                    input_tokens = inputs["input_ids"].shape[-1]
                    if input_tokens + 256 > 2048:
                        raise ValueError("Validation context budget exceeded; truncation forbidden")
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id
                            if tok.pad_token_id is not None
                            else tok.eos_token_id,
                        )
                    tokens = generated[0, input_tokens:]
                    if not len(tokens):
                        raise ValueError("Empty validation generation")
                    stream.write(
                        json.dumps(
                            {
                                "case_id": case_id,
                                "source_kind": "model_inference",
                                "model_id": MODEL,
                                "model_revision": REVISION,
                                "adapter": adapter_path,
                                "raw_output": tok.decode(tokens, skip_special_tokens=True),
                                "input_tokens": input_tokens,
                                "generated_tokens": len(tokens),
                                "hit_generation_limit": len(tokens) == 256,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                    print(f"{path.stem}: {index + 1}/99 saved", flush=True)

        report["phase"] = "full_baseline_validation"
        save()
        baseline_predictions = output / "baseline-predictions.jsonl"
        generate_validation(base, baseline_predictions, None)
        reloaded = PeftModel.from_pretrained(
            base, adapter, is_trainable=False, local_files_only=True
        )
        report.update(adapter_reloaded=True, phase="full_adapter_validation")
        save()
        predictions = output / "reloaded-predictions.jsonl"
        generate_validation(reloaded, predictions, str(adapter))
        report["phase"] = "actual_auditkit_comparison"
        save()
        baseline = evaluate(cases, baseline_predictions, output / "audit-base")
        tuned = evaluate(cases, predictions, output / "audit-adapter")
        comparison = {}
        for key in [
            "json_schema_validity",
            "valid_response_rate",
            "macro_f1",
            "citation_id_recall",
            "conflict_recall",
            "supported_completion",
            "revision_pair_accuracy",
        ]:

            def scalar(value):
                return value.get("value") if isinstance(value, dict) else value

            before_value, after_value = scalar(baseline[key]), scalar(tuned[key])
            comparison[key] = {
                "baseline": before_value,
                "adapter": after_value,
                "delta": None
                if before_value is None or after_value is None
                else after_value - before_value,
            }
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
        report["validation_comparison"] = comparison
        report["paired_validation_measured"] = True
        report.update(
            status="EVALUATED_WITH_TRAINING_WARNING",
            phase="recovered_validation_complete",
            baseline_cases=baseline["count"],
            adapter_cases=tuned["count"],
            baseline_audit_fingerprint=baseline["toolkit"]["fingerprint"],
            adapter_audit_fingerprint=tuned["toolkit"]["fingerprint"],
            warning="One seed and one epoch on synthetic data; validation comparison is not frozen-test evidence.",
        )
    except Exception as exc:
        message = re.sub(r"https?://\S+", "[URL redacted]", str(exc))
        message = re.sub(r"hf_[A-Za-z0-9_]+", "[credential redacted]", message)
        report.update(status="FAIL", error_type=type(exc).__name__, error_message=message[:5000])
    save()
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["status"] == "EVALUATED_WITH_TRAINING_WARNING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
