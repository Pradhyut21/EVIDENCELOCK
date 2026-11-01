"""Pinned Tiny Aya load + real validation baseline. No training or publication."""

import gc
import hashlib
import json
import re
import time
from importlib.metadata import version
from pathlib import Path

MODEL = "CohereLabs/tiny-aya-fire"
REVISION = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"


def main():
    source = Path(__file__).resolve().parents[1]
    root = source.parent
    output = root / "model-smoke"
    output.mkdir(exist_ok=True)
    report = {
        "status": "RUNNING",
        "phase": "authentication",
        "model_id": MODEL,
        "model_revision": REVISION,
        "adapter_trained": False,
        "lora_backward_verified": False,
        "model_improvement_measured": False,
    }

    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        from huggingface_hub import HfApi, get_token

        token = get_token()
        if not token:
            raise RuntimeError(
                "No Colab-local HF credential; use the private prompt in the runner cell"
            )
        HfApi().whoami(token=token)
        info = HfApi().model_info(MODEL, revision=REVISION, token=token, timeout=30)
        if info.sha != REVISION:
            raise RuntimeError("Pinned model revision mismatch")
        report["authentication"] = "PASS"
        print("Authenticated. Loading the pinned tokenizer and quantized model...", flush=True)

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        from evidencelock.schemas import Decision

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the isolated environment")
        report.update(
            phase="model_load",
            gpu=torch.cuda.get_device_name(0),
            versions={
                name: version(name) for name in ["torch", "transformers", "peft", "bitsandbytes"]
            },
        )
        save()
        torch.cuda.reset_peak_memory_stats()
        cfg = AutoConfig.from_pretrained(
            MODEL, revision=REVISION, token=token, trust_remote_code=False
        )
        if cfg.model_type != "cohere2":
            raise RuntimeError("Unexpected model architecture")
        tok = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, token=token, trust_remote_code=False
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            token=token,
            trust_remote_code=False,
            device_map={"": 0},
            dtype=torch.float16,
            quantization_config=quantization,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.eval()
        report.update(
            model_loaded=True,
            model_class=type(model).__name__,
            load_seconds=time.perf_counter() - started,
            quantization="NF4, double quantization, FP16 compute",
            attention="eager",
        )
        save()

        directory = source / "data/pilot-v1/curated/validation"
        dataset = directory / "curatorkit/sft_messages.jsonl"
        lineage = json.loads((directory / "lineage.json").read_text())
        digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
        if digest != lineage["export_sha256"]:
            raise RuntimeError("Curated validation export checksum mismatch")
        rows = [json.loads(line) for line in dataset.read_text().splitlines()]
        if len(rows) != len(lineage["accepted_case_ids"]):
            raise RuntimeError("Validation lineage count mismatch")
        report.update(
            phase="baseline_inference",
            dataset_sha256=digest,
            selection="first two accepted validation rows; feasibility only",
            decoding={"do_sample": False, "max_new_tokens": 256},
            seed=42,
        )
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        predictions = []
        with (output / "baseline-predictions.jsonl").open("w", encoding="utf-8") as stream:
            for index, row in enumerate(rows[:2]):
                messages = row["messages"][:-1]  # The model never sees the gold assistant target.
                inputs = tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
                input_tokens = inputs["input_ids"].shape[-1]
                if input_tokens + 256 > 2048:
                    raise RuntimeError(
                        "Smoke context exceeds its explicit budget; no truncation allowed"
                    )
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id
                        if tok.pad_token_id is not None
                        else tok.eos_token_id,
                    )
                torch.cuda.synchronize()
                continuation = generated[0, input_tokens:]
                raw = tok.decode(continuation, skip_special_tokens=True)
                item = {
                    "case_id": lineage["accepted_case_ids"][index],
                    "source_kind": "model_inference",
                    "model_id": MODEL,
                    "model_revision": REVISION,
                    "adapter": None,
                    "raw_output": raw,
                    "input_tokens": input_tokens,
                    "generated_tokens": len(continuation),
                    "latency_seconds": time.perf_counter() - started,
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                valid = False
                try:
                    decision = Decision.model_validate_json(raw)
                    payload = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
                    known = {item["evidence_id"] for item in payload["evidence_items"]}
                    valid = all(
                        eid in known
                        for fact in decision.supported_facts
                        for eid in fact.evidence_ids
                    )
                except ValueError:
                    pass  # Invalid responses remain saved and included in the count.
                predictions.append(
                    {
                        "case_id": item["case_id"],
                        "generated_tokens": len(continuation),
                        "schema_and_citation_ids_valid": valid,
                    }
                )
                print(f"Saved actual baseline response {index + 1}/2", flush=True)
        if not predictions or any(item["generated_tokens"] == 0 for item in predictions):
            raise RuntimeError("Missing model generation")
        report.update(
            status="PASS",
            phase="baseline_complete",
            baseline_predictions=len(predictions),
            response_checks=predictions,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            free_total_vram_bytes=torch.cuda.mem_get_info(),
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        message = re.sub(r"https?://\S+", "[URL redacted]", str(exc))
        message = re.sub(r"hf_[A-Za-z0-9_]+", "[credential redacted]", message)
        report.update(
            status="FAIL",
            error_type=type(exc).__name__,
            error_message=message[:4000],
            http_status=getattr(getattr(exc, "response", None), "status_code", None),
        )
    save()
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
