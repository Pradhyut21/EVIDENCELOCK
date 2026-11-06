"""Authenticated pinned model download; token is supplied privately in process env."""

import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL = "CohereLabs/tiny-aya-fire"
REV = "73500c80fcb6c8bea3e830c21d5d2920249a0e1a"
root = Path("/content/evidencelock-gpu")
report = {"status": "RUNNING", "model_id": MODEL, "model_revision": REV}
try:
    token = os.environ.get("HF_TOKEN")
    info = HfApi().model_info(MODEL, revision=REV, token=token)
    if info.sha != REV:
        raise ValueError("Resolved revision mismatch")
    path = snapshot_download(
        MODEL,
        revision=REV,
        token=token,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "chat_template*",
            "model*.safetensors",
            "model*.safetensors.index.json",
        ],
    )
    report.update(
        status="PASS",
        resolved_revision=info.sha,
        snapshot=path,
        authenticated_access_tested=bool(token),
    )
except Exception as exc:
    message = re.sub(r"https?://\S+", "[URL redacted]", str(exc))
    message = re.sub(r"hf_[A-Za-z0-9_]+", "[credential redacted]", message)
    report.update(status="FAIL", error_type=type(exc).__name__, error_message=message[:2000])
finally:
    os.environ.pop("HF_TOKEN", None)
    (root / "fresh-model-access-result.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["status"] == "PASS" else 1)
