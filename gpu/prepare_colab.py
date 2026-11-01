"""Install the complete reviewed dependency graph into a separate Python 3.12 venv.

Intended for the user-selected Colab GPU runtime. No model weights, paid services,
account login, training or publication are requested. Failures are preserved.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This candidate lock targets Linux x86_64")
    libc, release = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, release.split(".")[:2])) < (2, 35):
        raise RuntimeError("glibc >= 2.35 is required by this resolved lock target")
    source = Path(__file__).resolve().parents[1]
    root = source.parent
    log_path = root / "environment-install.log"
    env = os.environ.copy()
    env.update(
        UV_CACHE_DIR=str(root / "uv-cache"),
        HF_HOME=str(root / "hf-home"),
        PURE_TRL_MODE="1",
        TRL_ONLY_MODE="1",
        DISABLE_UNSLOTH_FOR_TRL="1",
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
    )
    # The earlier tokenizer-only flags must not disable frameworks in this process.
    for flag in ["USE_TORCH", "USE_TF", "USE_FLAX", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        env.pop(flag, None)
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"

    def run(command):
        print("Running: " + " ".join(map(str, command)), flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nCOMMAND: " + " ".join(map(str, command)) + "\n")
            process = subprocess.Popen(
                list(map(str, command)),
                env=env,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        if returncode:
            print(
                "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-50:])
            )
            raise RuntimeError(f"Command failed with exit {returncode}; see {log_path}")

    run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"])
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "build-essential", "pkg-config", "libicu-dev"])
    bootstrap = root / ".venv-bootstrap"
    if not (bootstrap / "bin/python").exists():
        run([sys.executable, "-m", "venv", bootstrap])
    run([bootstrap / "bin/python", "-m", "pip", "install", "uv==0.11.3"])
    uv = bootstrap / "bin/uv"
    target = root / ".venv-training"
    if not (target / "bin/python").exists():
        run([uv, "venv", "--python", "3.12", target])
    python = target / "bin/python"
    lock = source / "gpu/requirements-linux-py312.txt"
    run([uv, "pip", "sync", "--python", python, "--require-hashes", lock])
    run([uv, "pip", "install", "--python", python, "-c", lock, "-e", source])
    run([uv, "pip", "check", "--python", python])
    run([python, source / "gpu/inspect_training_env.py"])
    result_path = root / "gpu-environment-result.json"
    report = json.loads(result_path.read_text())
    print(json.dumps(report, indent=2), flush=True)
    print("Environment checks finished. No model load or LoRA backward pass has run.", flush=True)


if __name__ == "__main__":
    main()
