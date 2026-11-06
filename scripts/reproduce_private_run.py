"""Restore a pinned private input bundle; execute only on explicitly selected Linux GPU.

This script never submits paid jobs. Authenticate locally using the HF CLI first.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Private artifact repository commit")
    parser.add_argument("--path", required=True, help="Versioned private payload path")
    parser.add_argument("--sha256", required=True, help="Expected outer ZIP hash")
    parser.add_argument("--output", type=Path, required=True, help="New extraction directory")
    parser.add_argument(
        "--execute", action="store_true", help="Run on this Linux GPU after verification"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory; existing files are never overwritten")
    if args.execute:
        if platform.system() != "Linux":
            raise ValueError("Execution requires Linux; the local Windows MX330 is excluded")
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        available_devices = [int(line.strip()) for line in gpu.stdout.splitlines()]
        if not available_devices or available_devices[0] < 12 * 1024:
            raise ValueError("This verified run requires at least 12 GiB on its first GPU")
    from huggingface_hub import hf_hub_download

    archive_path = hf_hub_download(
        "pradhyut/evidencelock-private-runs",
        args.path,
        revision=args.revision,
        repo_type="dataset",
    )
    actual = hashlib.sha256(Path(archive_path).read_bytes()).hexdigest()
    if actual != args.sha256:
        raise ValueError("Private input bundle SHA256 mismatch")
    destination = args.output.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if sorted(names) != ["recovery-payload.zip", "run_remote.py"]:
            raise ValueError("Unexpected outer bundle members")
        for name in names:
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError("Unsafe ZIP member")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    record = {
        "artifact_revision": args.revision,
        "artifact_path": args.path,
        "artifact_sha256": actual,
        "executed": False,
        "paid_job_submitted": False,
        "exact_gpu_determinism_claimed": False,
    }
    (destination / "reproduction-input.json").write_text(json.dumps(record, indent=2))
    if args.execute:
        completed = subprocess.run(
            [sys.executable, "-u", str(destination / "run_remote.py")], cwd=destination, check=False
        )
        record.update(executed=True, returncode=completed.returncode)
        (destination / "reproduction-input.json").write_text(json.dumps(record, indent=2))
        return completed.returncode
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
