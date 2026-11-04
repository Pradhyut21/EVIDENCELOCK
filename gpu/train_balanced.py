"""Controlled one-epoch AlignTune experiment with paired validation evaluation.

Consumes all 99 real CuratorKIT training exports. Never uses the frozen test set.
Runs offline against the pinned model weights already cached by model_smoke.py.
"""

import gc
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from smoke_checks import validate_cached_weights, verify_supervision

MODEL = "CohereLabs/tiny-aya-fire"
REVISION = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    source = Path(__file__).resolve().parents[1]
    root = source.parent
    parent = root / "balanced-experiment"
    parent.mkdir(exist_ok=True)
    output = parent / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir()
    report = {
        "status": "RUNNING",
        "phase": "preflight",
        "attempt_dir": str(output),
        "model_id": MODEL,
        "model_revision": REVISION,
        "model_improvement_measured": False,
        "adapter_trained": False,
        "adapter_reloaded": False,
        "lora_backward_verified": False,
        "frozen_test_used": False,
        "training_steps_requested": 216,
    }

    def save():
        text = json.dumps(report, indent=2, default=str)
        (output / "result.json").write_text(text)
        (parent / "latest-result.json").write_text(text)

    save()
    try:
        baseline_report = json.loads((root / "model-smoke/result.json").read_text())
        if baseline_report["status"] != "PASS" or baseline_report["model_revision"] != REVISION:
            raise ValueError("The matching real baseline/model-load gate must pass first")
        gate = json.loads((root / "align-smoke/latest-result.json").read_text())
        if (
            not all(
                gate.get(key)
                for key in ["adapter_trained", "adapter_reloaded", "lora_backward_verified"]
            )
            or gate.get("status") != "PASS"
        ):
            raise ValueError("A successful real AlignTune smoke is required")
        report["successful_smoke_adapter_sha256"] = gate["adapter_sha256"]
        from huggingface_hub import hf_hub_download

        snapshot = Path(
            hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True)
        ).parent
        report["cache_validation"] = validate_cached_weights(snapshot)
        import torch
        from aligntune import create_sft_trainer
        from datasets import Dataset
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback

        from evidencelock.evaluation import evaluate
        from evidencelock.provenance import read_cases

        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        report.update(
            versions={
                name: version(name)
                for name in [
                    "torch",
                    "transformers",
                    "peft",
                    "trl",
                    "bitsandbytes",
                    "aligntune",
                    "curatorkit",
                    "auditkit",
                ]
            },
            gpu=torch.cuda.get_device_name(0),
        )
        directory = source / "data/balanced-v1/curated/train"
        export = directory / "curatorkit/sft_messages.jsonl"
        lineage = json.loads((directory / "lineage.json").read_text())
        if digest(export) != lineage["export_sha256"]:
            raise ValueError("Curated training export hash mismatch")
        originals = [json.loads(line) for line in export.read_text().splitlines()]
        if len(originals) != 216:
            raise ValueError("Exactly 216 balanced training exposures are required")
        smoke_file = output / "curated-smoke-input.jsonl"
        smoke_file.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in originals)
        )
        report.update(
            phase="aligntune_model_setup",
            train_export_sha256=digest(export),
            smoke_input_sha256=digest(smoke_file),
            train_case_ids=lineage["accepted_case_ids"],
            seed=42,
            training_exposures=216,
            unique_training_cases=99,
            sampling_manifest=lineage,
            max_seq_length=1536,
            lora_rank=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            precision="fp16",
            attention="eager",
            quantization="nf4-double",
        )
        save()
        print(
            "Creating the genuine AlignTune TRL SFT trainer from cached model weights...",
            flush=True,
        )
        outer = create_sft_trainer(
            model_name=str(snapshot),
            dataset_name=str(smoke_file),
            backend="trl",
            output_dir=str(output / "adapter"),
            num_epochs=1,
            batch_size=1,
            learning_rate=1e-4,
            max_seq_length=1536,
            max_samples=216,
            seed=42,
            use_peft=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "v_proj"],
            precision="fp16",
            fp16=True,
            bf16=False,
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "float16",
            },
            device_map={"": 0},
            trust_remote_code=False,
            attn_implementation="eager",
            use_gradient_checkpointing=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_steps=216,
            gradient_accumulation_steps=1,
            warmup_ratio=0.0,
            optimizer="adamw_torch",
            completion_only_loss=True,
            assistant_only_loss=False,
            packing=False,
            padding_free=False,
            group_by_length=False,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            load_best_model_at_end=False,
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            loggers=[],
            logging_steps=1,
            compute_perplexity=False,
            compute_rouge=False,
            compute_bleu=False,
            compute_bertscore=False,
            compute_semantic_similarity=False,
            curator_clean=False,
            curator_dedup="none",
        )
        report["trainer_class"] = type(outer).__module__ + "." + type(outer).__name__
        if ".backends.trl." not in type(outer).__module__:
            raise ValueError("AlignTune did not select the requested TRL backend")
        outer.setup_model()
        outer.model.config.use_cache = False
        outer.setup_data()  # Actual AlignTune DataManager/CuratorKIT path, not a mocked load.
        tok = outer.tokenizer
        by_messages = {json.dumps(row["messages"], sort_keys=True): row for row in originals}
        prepared, expected = [], {}
        for row in outer.train_dataset:
            messages = row["messages"]
            if json.dumps(messages, sort_keys=True) not in by_messages:
                raise ValueError("AlignTune data conversion changed a curated conversation")
            prefix = tok.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if not full.startswith(prefix):
                raise ValueError("Chat template prefix mismatch")
            completion = full[len(prefix) :]
            for specials in [False, True]:
                ids = tok(full, add_special_tokens=specials)["input_ids"]
                prefix_ids = tok(prefix, add_special_tokens=specials)["input_ids"]
                if ids[: len(prefix_ids)] != prefix_ids or len(ids) > 1536:
                    raise ValueError("Empty supervision, prefix drift or context overflow")
                expected[tuple(ids)] = len(prefix_ids)
            prepared.append({"prompt": prefix, "completion": completion})
        if len(prepared) != 216:
            raise ValueError("AlignTune dropped curated smoke examples")
        # Explicit task-specific format adaptation after AlignTune's real normalization:
        # Tiny Aya's template has no assistant generation blocks, so use TRL's
        # prompt/completion boundary rather than an unsupported assistant-only mask.
        outer.train_dataset = Dataset.from_list(prepared)
        outer.dataset = outer.train_dataset
        outer.eval_dataset = None
        trainable = {
            name: param for name, param in outer.model.named_parameters() if param.requires_grad
        }
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("Expected only nonempty LoRA trainable parameters")
        before = {name: param.detach().cpu().clone() for name, param in trainable.items()}
        report["trainable_parameters"] = sum(param.numel() for param in trainable.values())
        gradients = {"finite_nonzero_events": 0, "nonfinite_events": 0}

        def observe_gradient(grad):
            if not bool(torch.isfinite(grad).all().item()):
                gradients["nonfinite_events"] += 1
            elif bool(torch.count_nonzero(grad).item()):
                gradients["finite_nonzero_events"] += 1
            return grad

        handles = [param.register_hook(observe_gradient) for param in trainable.values()]
        losses = []

        class AuditCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                checks = []
                native = outer.trainer
                # TRL 1.7.1 unconditionally casts quantized-model adapters to BF16
                # during construction, even with FP16 arguments. Restore FP32
                # master adapter weights before the first forward/optimizer step.
                if not args.fp16 or args.bf16:
                    raise ValueError("Expected FP16 mixed precision with BF16 disabled")
                # The previous full run overflowed at iteration 2 with the
                # default loss scale. Use the public GradScaler state API to
                # start lower; retain the strict nonfinite-gradient gate.
                scaler = native.accelerator.scaler
                if scaler is None:
                    raise ValueError("FP16 gradient scaler is required")
                scaler_state = scaler.state_dict()
                report["initial_loss_scale_before"] = scaler_state["scale"]
                scaler_state["scale"] = 1024.0
                scaler.load_state_dict(scaler_state)
                report["initial_loss_scale_for_training"] = scaler.get_scale()
                if scaler.get_scale() != 1024.0:
                    raise ValueError("Gradient scaler configuration did not take effect")
                params = dict(native.model.named_parameters())
                active = {name: p for name, p in params.items() if p.requires_grad}
                if not active or any("lora_" not in name for name in active):
                    raise ValueError("Precision repair is restricted to LoRA adapters")
                report["adapter_dtypes_before_precision_fix"] = sorted(
                    {str(p.dtype) for p in active.values()}
                )
                with torch.no_grad():
                    for param in active.values():
                        if param.grad is not None:
                            raise ValueError("Precision repair must precede backward")
                        param.data = param.data.to(dtype=torch.float32)
                if any(p.dtype != torch.float32 for p in active.values()):
                    raise ValueError("FP32 adapter master weight check failed")
                report["adapter_dtypes_for_training"] = ["torch.float32"]
                report["mixed_precision_for_training"] = "fp16"
                for index in range(len(native.train_dataset)):
                    batch = native.data_collator([native.train_dataset[index]])
                    mask = batch["attention_mask"][0].bool()
                    ids = batch["input_ids"][0][mask].tolist()
                    labels = batch["labels"][0][mask].tolist()
                    checks.append(verify_supervision(ids, labels, expected))
                report.update(supervision_checks=checks, phase="actual_aligntune_training")
                save()

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    losses.append({"step": state.global_step, "loss": float(logs["loss"])})

        original_callbacks = outer.get_hf_callbacks
        outer.get_hf_callbacks = lambda: [*original_callbacks(), AuditCallback()]
        report["phase"] = "trainer_preparation_and_training"
        save()
        started = time.perf_counter()
        train_result = outer.train()  # The genuine AlignTune pathway creates/runs its SFTTrainer.
        report["training_seconds"] = time.perf_counter() - started
        changed = sum(
            not torch.equal(param.detach().cpu(), before[name]) for name, param in trainable.items()
        )
        optimizer = outer.trainer.optimizer
        optimizer = getattr(optimizer, "optimizer", optimizer)
        step_values = [
            float(value["step"].item()) if hasattr(value["step"], "item") else float(value["step"])
            for value in optimizer.state.values()
            if "step" in value
        ]
        optimizer_steps = max(step_values, default=0)
        global_steps = outer.trainer.state.global_step
        loss = float(train_result["final_loss"])
        # Preserve diagnostics even when the numerical gate rejects this run.
        report.update(
            changed_trainable_tensors=changed,
            actual_optimizer_steps=optimizer_steps,
            global_steps=global_steps,
            loss=loss,
            loss_log=losses,
            gradient_checks=gradients,
        )
        save()
        if (
            not changed
            or not optimizer_steps
            or global_steps != 216
            or gradients["finite_nonzero_events"] == 0
        ):
            raise ValueError("No verified backward/optimizer/parameter update")
        if not math.isfinite(loss) or loss <= 0 or gradients["nonfinite_events"]:
            raise ValueError("Nonfinite/zero training loss or nonfinite gradients")
        adapter = output / "adapter"
        weights = adapter / "adapter_model.safetensors"
        if not weights.is_file():
            raise ValueError("AlignTune did not save adapter weight tensors")
        saved = load_file(str(weights), device="cpu")
        if not any(
            "lora_B" in key and bool(torch.count_nonzero(value).item())
            for key, value in saved.items()
        ):
            raise ValueError("Saved adapter does not contain updated LoRA B weights")
        report.update(
            adapter_trained=True,
            lora_backward_verified=True,
            changed_trainable_tensors=changed,
            actual_optimizer_steps=optimizer_steps,
            global_steps=global_steps,
            loss=loss,
            loss_log=losses,
            gradient_checks=gradients,
            adapter_sha256=digest(weights),
            adapter_bytes=weights.stat().st_size,
            peak_training_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            phase="adapter_reload",
        )
        save()
        for handle in handles:
            handle.remove()
        trainable.clear()
        outer = optimizer = original_callbacks = None
        gc.collect()
        torch.cuda.empty_cache()
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
            status="PASS",
            phase="controlled_experiment_complete",
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
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
