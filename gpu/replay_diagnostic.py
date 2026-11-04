"""Small train/validation replay diagnostic; no training or frozen-test use."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from evidencelock.prompts import prompt
    from evidencelock.provenance import read_cases
    from evidencelock.schemas import parse_prediction

    root = Path("/content/evidencelock-gpu")
    source = root / "evidencelock"
    output = root / "replay-diagnostic"
    output.mkdir(exist_ok=True)
    report = {"status": "RUNNING", "training_executed": False, "frozen_test_used": False}
    try:
        cases = read_cases(source / "gpu/replay-cases.jsonl")
        assert len(cases) == 16 and all(c.split != "test" for c in cases)
        original = json.loads((root / "balanced-experiment/latest-result.json").read_text())
        adapter = Path(original["attempt_dir"]) / "adapter"
        assert (
            hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
            == original["adapter_sha256"]
        )
        snapshot = Path(
            hf_hub_download(
                "CohereLabs/tiny-aya-fire",
                "config.json",
                revision="73500c80fcb6c8bea3e830c21d5d2920249a0e1a",
                local_files_only=True,
            )
        ).parent
        tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        saved_tok = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
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
        groups = defaultdict(
            lambda: {"cases": 0, "valid": 0, "state_action_correct": 0, "evidence_copy_prefix": 0}
        )
        drift = []
        with (output / "predictions.jsonl").open("w") as stream:
            for variant in ["base", "balanced"]:
                if variant == "balanced":
                    model = PeftModel.from_pretrained(
                        model, adapter, is_trainable=False, local_files_only=True
                    )
                model.eval()
                for index, case in enumerate(cases):
                    messages = [{"role": "user", "content": prompt(case)}]
                    ids = tok.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_dict=True
                    )["input_ids"]
                    saved_ids = saved_tok.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_dict=True
                    )["input_ids"]
                    if not isinstance(ids, list) or len(ids) < 100:
                        raise ValueError("Expected a full token ID sequence, not mapping keys")
                    if variant == "base":
                        drift.append(
                            {
                                "case_id": case.case_id,
                                "tokenizers_identical": ids == saved_ids,
                                "base_tokens": len(ids),
                                "saved_tokens": len(saved_ids),
                            }
                        )
                    inputs = tok.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id
                            if tok.pad_token_id is not None
                            else tok.eos_token_id,
                        )
                    input_length = inputs["input_ids"].shape[-1]
                    if input_length != len(ids):
                        raise ValueError("Prompt token counts disagree")
                    raw = tok.decode(generated[0, input_length:], skip_special_tokens=True)
                    group = groups[variant + "/" + case.split]
                    group["cases"] += 1
                    group["evidence_copy_prefix"] += raw.startswith('{"as_of"')
                    try:
                        decision = parse_prediction(raw, case)
                        group["valid"] += 1
                        group["state_action_correct"] += (
                            decision.status == case.oracle_status
                            and decision.next_step == case.oracle_next_step
                        )
                    except ValueError:
                        pass
                    stream.write(
                        json.dumps(
                            {
                                "variant": variant,
                                "split": case.split,
                                "case_id": case.case_id,
                                "raw_output": raw,
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                    print(variant, index + 1, "/16", flush=True)
        report.update(
            status="PASS",
            groups=dict(groups),
            tokenizer_checks=drift,
            interpretation="Diagnostic subset only; training replay is not held-out performance.",
        )
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc)[:2000])
    (output / "latest-result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
