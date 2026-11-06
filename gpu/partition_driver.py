"""One frozen evaluation: preserve poor outputs as results, never tune against them."""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path("/content/evidencelock-gpu")


def verify_freeze(root):
    freeze = json.loads((root / "language-partition/freeze.json").read_text())
    if freeze["test_used_for_selection"] or freeze["training_enabled"]:
        raise ValueError("Frozen evaluation cannot train or select by test results")
    for name, expected in freeze["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Missing or unsafe frozen input: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input changed: " + name)
    return freeze


def main():
    output = ROOT / "language-partition"
    output.mkdir(exist_ok=True)
    result = {
        "status": "RUNNING",
        "phase": "frozen_input_preflight",
        "training_executed": False,
        "test_used_for_selection": False,
        "frozen_test_used": False,
        "full_project_completion_claim": False,
    }

    def save():
        (output / "latest-result.json").write_text(json.dumps(result, indent=2))

    save()
    try:
        freeze = verify_freeze(ROOT)
        import torch
        from final_prompt import strong_prompt
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from smoke_checks import validate_cached_weights
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from evidencelock.evaluation import evaluate
        from evidencelock.oracle import decide
        from evidencelock.prompts import prompt
        from evidencelock.provenance import read_cases, write_jsonl

        cases = read_cases(ROOT / "evidencelock/gpu/multilingual-test-cases.jsonl")
        if len(cases) != 198 or any(case.split != "test" for case in cases):
            raise ValueError("Expected exactly 198 paired frozen Hindi/Hinglish question cases")
        language = freeze["execution_language"]
        if language not in ["hi", "hinglish"]:
            raise ValueError("Unknown execution partition")
        cases = [case for case in cases if case.language_variant == language]
        if len(cases) != 99:
            raise ValueError("Each language partition must contain all99 cases")
        result["execution_language"] = language
        result.update(
            freeze_sha256=hashlib.sha256((output / "freeze.json").read_bytes()).hexdigest(),
            model_revision=freeze["model_revision"],
            case_count=len(cases),
        )
        snapshot = Path(
            hf_hub_download(
                freeze["model_id"],
                "config.json",
                revision=freeze["model_revision"],
                local_files_only=True,
            )
        ).parent
        validate_cached_weights(snapshot)
        for name, expected in freeze["tokenizer_files_sha256"].items():
            if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Pinned tokenizer/config content mismatch: " + name)
        tok = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        torch.manual_seed(freeze["seed"])
        torch.cuda.manual_seed_all(freeze["seed"])
        base = AutoModelForCausalLM.from_pretrained(
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
        audits = {}
        model = base
        for variant in ["base_standard", "first_sft_standard"]:
            if variant == "first_sft_standard":
                model = PeftModel.from_pretrained(
                    base,
                    output / "first-adapter",
                    is_trainable=False,
                    local_files_only=True,
                )
            model.eval()
            path = output / (variant + "-predictions.jsonl")
            if path.exists():
                raise ValueError("Refusing to overwrite existing frozen-test outputs")
            result.update(phase=variant, frozen_test_used=True)
            save()
            with path.open("x", encoding="utf-8") as stream:
                for index, case in enumerate(cases):
                    actual_prompt = (
                        strong_prompt(case) if variant == "base_strong_prompt" else prompt(case)
                    )
                    inputs = tok.apply_chat_template(
                        [{"role": "user", "content": actual_prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")
                    size = inputs["input_ids"].shape[-1]
                    if size + 256 > 2048:
                        raise ValueError("Frozen context budget exceeded; truncation is forbidden")
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id
                            if tok.pad_token_id is not None
                            else tok.eos_token_id,
                        )
                    tokens = generated[0, size:]
                    row = {
                        "case_id": case.case_id,
                        "source_kind": "model_inference",
                        "variant": variant,
                        "actual_prompt": actual_prompt,
                        "model_id": freeze["model_id"],
                        "model_revision": freeze["model_revision"],
                        "adapter": str(output / "first-adapter")
                        if variant == "first_sft_standard"
                        else None,
                        "raw_output": tok.decode(tokens, skip_special_tokens=True),
                        "input_tokens": size,
                        "generated_tokens": len(tokens),
                        "hit_generation_limit": len(tokens) == 256,
                    }
                    if variant == "first_sft_standard":
                        row["adapter_sha256"] = freeze["first_adapter_sha256"]
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(variant, index + 1, "/99 raw outputs saved", flush=True)
                    del inputs, generated, tokens
            audits[variant] = evaluate(cases, path, output / ("audit-" + variant))
            result["full_response_metrics"] = audits
            save()
        reference = output / "oracle-reference.jsonl"
        write_jsonl(
            reference,
            (
                {
                    "case_id": case.case_id,
                    "source_kind": "oracle_reference",
                    "raw_output": decide(case).model_dump_json(),
                }
                for case in cases
            ),
        )
        result["oracle_reference_metrics"] = evaluate(
            cases, reference, output / "audit-oracle-reference"
        )
        result.update(
            status="PASS",
            phase="frozen_language_partition_complete",
            scope="question-only Hindi/Hinglish supplement; structured evidence stays English",
            interpretation="Same selected model/config and original test families; no broad language-generalization or post-test optimization claim.",
        )
    except Exception as exc:
        message = re.sub(r"hf_[A-Za-z0-9_]+|https?://\S+", "[redacted]", str(exc))
        result.update(status="FAIL", error_type=type(exc).__name__, error_message=message[:4000])
    save()
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-only":
        verify_freeze(Path(sys.argv[2]))
        print("PASS: frozen evaluation inputs verified; no model loaded")
        raise SystemExit(0)
    raise SystemExit(main())
