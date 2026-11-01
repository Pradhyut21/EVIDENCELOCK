"""Real-model inference pathway; unexecuted until the external GPU gate passes."""

from time import perf_counter

from .prompts import prompt
from .provenance import write_jsonl
from .runtime import ExecutionBlocked, require_gpu

MODEL_ID = "CohereLabs/tiny-aya-fire"
MODEL_REVISION = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"


def infer(cases, output, adapter=None):
    torch = require_gpu()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ExecutionBlocked("Install the reviewed, isolated GPU dependency lock first.") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        ),
    )
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    rows = []
    for case in cases:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(case)}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        if inputs["input_ids"].shape[-1] + 512 > 8192:
            raise ValueError("Input exceeds the model budget; no silent truncation is permitted")
        started = perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
        rows.append(
            {
                "case_id": case.case_id,
                "source_kind": "model_inference",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "adapter": str(adapter) if adapter else None,
                "raw_output": tokenizer.decode(new_tokens, skip_special_tokens=True),
                "latency_seconds": perf_counter() - started,
                "generated_tokens": len(new_tokens),
            }
        )
        write_jsonl(output, rows)  # Preserve completed raw responses on interruption.
    return {"predictions": len(rows), "path": str(output)}
