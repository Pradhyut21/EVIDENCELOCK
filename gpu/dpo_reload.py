"""Reload-only recovery of the completed DPO policy; no further training."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("/content/evidencelock-gpu")
    output = root / "dpo-reload"
    output.mkdir(exist_ok=True)
    report = {
        "status": "RUNNING",
        "training_executed": False,
        "frozen_test_used": False,
        "model_improvement_measured": False,
    }
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        original = json.loads((root / "dpo-smoke/latest-result.json").read_text())
        assert original["optimizer_steps"] == 4 and original["gradient_checks"]["nonfinite"] == 0
        adapter = root / "dpo-smoke/adapter"
        weights = adapter / "adapter_model.safetensors"
        tensors = load_file(str(weights))
        assert tensors and all(torch.isfinite(t).all() for t in tensors.values())
        report.update(
            original_optimizer_steps=4,
            policy_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
            policy_tensor_values_finite=True,
        )
        snapshot = Path(
            hf_hub_download(
                "CohereLabs/tiny-aya-fire",
                "config.json",
                revision="73500c80fcb6c8bea3e830c21d5d2920249a0e1a",
                local_files_only=True,
            )
        ).parent
        base = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            device_map={"": 0},
            dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        pair = json.loads((root / "dpo-smoke/pairs.jsonl").read_text().splitlines()[0])
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": pair["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        model.eval()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id
            )
        tokens = generated[0, inputs["input_ids"].shape[-1] :]
        assert len(tokens) > 0
        (output / "reload-generation.json").write_text(
            json.dumps(
                {
                    "source_kind": "model_inference",
                    "selection": "first training preference pair; reload feasibility only",
                    "raw_output": tok.decode(tokens, skip_special_tokens=True),
                }
            )
        )
        report.update(
            status="PASS",
            adapter_reloaded=True,
            generation_tokens=len(tokens),
            warning="Reload feasibility only, not a model-quality experiment",
        )
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc)[:3000])
    (output / "latest-result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
