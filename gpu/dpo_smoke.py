"""Four-step genuine AlignTune preference smoke; no quality claim or test data."""

import gc
import hashlib
import json
from pathlib import Path


def main():
    root = Path("/content/evidencelock-gpu")
    source = root / "evidencelock"
    output = root / "dpo-smoke"
    output.mkdir(exist_ok=True)
    report = {
        "status": "RUNNING",
        "training_steps_requested": 4,
        "frozen_test_used": False,
        "model_improvement_measured": False,
        "reference": "adapter-disabled base model",
    }

    def save():
        (output / "latest-result.json").write_text(json.dumps(report, indent=2, default=str))

    save()
    try:
        import torch
        from aligntune import create_rl_trainer
        from datasets import Dataset
        from huggingface_hub import hf_hub_download
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback

        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        test = torch.ones(8, device="cuda", dtype=torch.bfloat16)
        assert torch.isfinite(test @ test)
        model_id = "CohereLabs/tiny-aya-fire"
        revision = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"
        snapshot = Path(
            hf_hub_download(model_id, "config.json", revision=revision, local_files_only=True)
        ).parent
        pairs = [
            json.loads(line)
            for line in (source / "gpu/dpo-smoke-pairs.jsonl").read_text().splitlines()
        ]
        assert len(pairs) == 4
        data = output / "pairs.jsonl"
        data.write_text("".join(json.dumps(p) + "\n" for p in pairs))
        outer = create_rl_trainer(
            model_name=str(snapshot),
            dataset_name=str(data),
            algorithm="dpo",
            backend="trl",
            output_dir=str(output / "adapter"),
            num_epochs=1,
            max_steps=4,
            batch_size=1,
            max_samples=4,
            learning_rate=1e-5,
            max_seq_length=1536,
            seed=42,
            use_peft=True,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=["q_proj", "v_proj"],
            precision="bf16",
            bf16=True,
            fp16=False,
            device_map={"": 0},
            quantization={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "bfloat16",
            },
            attn_implementation="eager",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=1,
            warmup_ratio=None,
            warmup_steps=0,
            beta=0.1,
            loss_type="sigmoid",
            max_length=1536,
            max_prompt_length=1200,
            truncation_mode="keep_start",
            eval_strategy="no",
            save_strategy="no",
            report_to="none",
            loggers=[],
            logging_steps=1,
            dataloader_num_workers=0,
            curator_dedup="none",
            curator_clean=False,
        )
        report["trainer_class"] = type(outer).__module__ + "." + type(outer).__name__
        if ".trl." not in report["trainer_class"]:
            raise ValueError("Expected genuine TRL backend")
        before, params, hooks = {}, {}, []
        gradients = {"finite_nonzero": 0, "nonfinite": 0}
        native_setup = outer.setup_model

        def observe(grad):
            gradients["nonfinite"] += int(not bool(torch.isfinite(grad).all()))
            gradients["finite_nonzero"] += int(
                bool(torch.isfinite(grad).all()) and bool(torch.count_nonzero(grad))
            )
            return grad

        def setup_model():
            native_setup()
            outer.model.config.use_cache = False
            for name, param in outer.model.named_parameters():
                if param.requires_grad:
                    if "lora_" not in name:
                        raise ValueError("Only LoRA training is allowed")
                    params[name] = param
                    before[name] = param.detach().cpu().float().clone()
                    hooks.append(param.register_hook(observe))
            assert params

        outer.setup_model = setup_model
        native_data = outer.setup_data

        def setup_data():
            native_data()
            expected = {(p["prompt"], p["chosen"], p["rejected"]) for p in pairs}
            actual = {(p["prompt"], p["chosen"], p["rejected"]) for p in outer.train_dataset}
            if actual != expected:
                raise ValueError("AlignTune changed preference pairs")
            formatted = []
            for p in pairs:
                messages = [{"role": "user", "content": p["prompt"]}]
                prefix = outer.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                item = {"prompt": prefix}
                for side in ["chosen", "rejected"]:
                    full = outer.tokenizer.apply_chat_template(
                        messages + [{"role": "assistant", "content": p[side]}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    assert full.startswith(prefix)
                    assert len(outer.tokenizer(full, add_special_tokens=False)["input_ids"]) <= 1536
                    item[side] = full[len(prefix) :]
                assert len(outer.tokenizer(prefix, add_special_tokens=False)["input_ids"]) <= 1200
                formatted.append(item)
            outer.train_dataset = Dataset.from_list(formatted)
            outer.eval_dataset = None
            report["genuine_data_path_verified"] = True
            save()

        outer.setup_data = setup_data

        class PrecisionCheck(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if not args.bf16 or args.fp16:
                    raise ValueError("Expected BF16 autocast on supported GPU")
                with torch.no_grad():
                    for p in params.values():
                        p.data = p.data.float()

        native_callbacks = outer.get_hf_callbacks
        outer.get_hf_callbacks = lambda: [*native_callbacks(), PrecisionCheck()]
        result = outer.train()
        optimizer = getattr(outer.trainer.optimizer, "optimizer", outer.trainer.optimizer)
        steps = max((float(s["step"]) for s in optimizer.state.values() if "step" in s), default=0)
        changed = sum(
            not torch.equal(p.detach().cpu().float(), before[n]) for n, p in params.items()
        )
        report.update(
            optimizer_steps=steps,
            global_steps=outer.trainer.state.global_step,
            changed_tensors=changed,
            gradient_checks=gradients,
            train_result=result,
        )
        save()
        assert steps == 4 and outer.trainer.state.global_step == 4 and changed > 0
        assert gradients["finite_nonzero"] > 0 and gradients["nonfinite"] == 0
        weights = list(output.rglob("adapter_model.safetensors"))
        assert len(weights) == 1
        tensors = load_file(str(weights[0]))
        assert all(torch.isfinite(t).all() for t in tensors.values())
        report["adapter_sha256"] = hashlib.sha256(weights[0].read_bytes()).hexdigest()
        for h in hooks:
            h.remove()
        params.clear()
        outer = native_callbacks = native_setup = native_data = optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
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
        loaded = PeftModel.from_pretrained(
            base, weights[0].parent, is_trainable=False, local_files_only=True
        )
        assert loaded.peft_config
        report.update(status="PASS", adapter_reloaded=True, backward_verified=True)
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc)[:4000])
    save()
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
