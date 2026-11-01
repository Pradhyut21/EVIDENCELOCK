"""Verify the isolated GPU stack; no model loading or training is attempted."""

import inspect
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path


def main():
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "isolated": sys.prefix != sys.base_prefix,
        "model_loaded": False,
        "lora_backward_verified": False,
    }
    try:
        report["versions"] = {
            name: version(name)
            for name in [
                "torch",
                "transformers",
                "peft",
                "trl",
                "accelerate",
                "bitsandbytes",
                "curatorkit",
                "aligntune",
                "auditkit",
            ]
        }
        import torch

        report.update(
            torch_runtime=torch.__version__,
            cuda_runtime=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
        )
        if not report["isolated"] or not torch.cuda.is_available():
            raise RuntimeError("An isolated CUDA-enabled runtime is required")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["capability"] = torch.cuda.get_device_capability(0)
        report["free_total_vram_bytes"] = torch.cuda.mem_get_info()
        x = torch.ones((8, 8), device="cuda", dtype=torch.float16)
        report["fp16_tensor_check"] = (x @ x).sum().item() == 512
        if not report["fp16_tensor_check"]:
            raise RuntimeError("FP16 CUDA arithmetic failed")
        from bitsandbytes.functional import dequantize_4bit, quantize_4bit

        data = torch.randn(256, device="cuda", dtype=torch.float16)
        quantized, state = quantize_4bit(data, quant_type="nf4")
        restored = dequantize_4bit(quantized, state)
        report["small_nf4_kernel_check"] = bool(torch.isfinite(restored).all().item())
        if not report["small_nf4_kernel_check"]:
            raise RuntimeError("Small NF4 kernel check failed")
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        report["cohere2_config_class"] = CONFIG_MAPPING["cohere2"].__name__
        report["transformers_runtime"] = transformers.__version__
        from aligntune import create_sft_trainer

        report["aligntune_factory_signature"] = str(inspect.signature(create_sft_trainer))
        report["aligntune_factory_file"] = inspect.getfile(create_sft_trainer)
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error_message=str(exc))
    Path("gpu-environment-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
