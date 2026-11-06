# Build 1 execution and readiness evidence

The training environment is **READY** on the inspected remote Linux L4. Build 1's research pipeline has actually executed: CuratorKIT curation, AlignTune training, adapter save/reload, baseline inference and AuditKIT evaluation. Model reliability fails the quality gate; production readiness and full project completion are not claimed.

1. **Actual machine — PASS.** Hugging Face Linux GPU jobs; training job `6aedd1d50e0338d1bb8153d7`. The local Windows MX330 machine was excluded from full-model execution.
2. **GPU/VRAM — PASS.** NVIDIA L4, 23,659,151,360 total VRAM bytes and 23,455,727,616 free at environment inspection; driver 580.178.04. Final language jobs each used a separate L4; memory was not pooled.
3. **Python/CUDA — PASS.** Isolated Linux Python 3.12.16; PyTorch 2.11.0+cu130, CUDA runtime 13.0. Actual CUDA/NF4 calculations and real backward/optimizer updates succeeded.
4. **Dependencies — PASS.** torch 2.11.0, Transformers 5.22.1, PEFT 0.21.2, TRL 1.7.1, Accelerate 1.15.0, bitsandbytes 0.50.2, CuratorKIT 0.1.22, AlignTune 0.1.28, AuditKIT 1.1.3. The complete Linux lock was installed in isolation and its declared dependencies checked.
5. **Hugging Face access — PASS.** Authenticated access to `CohereLabs/tiny-aya-fire` was tested. Resolved revision: `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`.
6. **Model loading/training — PASS.** Cohere2 model and tokenizer loaded; actual baseline inference ran. The state-only retry completed 864 optimizer updates, with 124,344 finite nonzero gradient observations and zero nonfinite observations. Saved adapter SHA256: `cad9f72b601182473a88633ad5e68d703eb5f93a21360639276be7871a4b0fba`; independently reloaded. The separate full-response adapter is identified in the frozen evaluation records.
7. **Executed commands.** Remote: `nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv`; `python evidencelock/gpu/bundle_preflight.py /content/evidencelock-gpu`; `python -m uv pip sync --python .venv-training/bin/python --require-hashes evidencelock/gpu/requirements-linux-py312.txt`; `python -m uv pip check --python .venv-training/bin/python`; `.venv-training/bin/python evidencelock/gpu/train_state_only.py`; `.venv-training/bin/python evidencelock/gpu/final_driver.py`; `.venv-training/bin/python evidencelock/gpu/partition_driver.py`. Full command evidence is in `EVIDENCELOCK-Nov6-readiness-evidence.json`.
8. **Results/status.** Frozen English evaluation and both complete Hindi/Hinglish partitions: PASS execution. All 693 full-response outputs and 792 auxiliary classification score records are preserved and verified. Model quality: FAIL. Independent native-speaker validation and public release: NOT_EXECUTED.
9. **Compatibility and study limits.** Native TRL's BF16 adapter cast required verified FP32 master weights under FP16; initial gradient scale 1024 was retained. A missing parent directory, snapshot preflight and bundle provenance failures were corrected and preserved. Combined-language revision pairs now group by family and language; native per-language metrics are unchanged. This is a small synthetic study of 99 underlying test scenarios; only questions were translated, with English structured evidence. The approved training-derived review was model-assisted, not independent native-speaker validation. Original policy ambiguity and repeated development-validation reuse remain disclosed.
10. **Single most important remaining blocker.** Full-response reliability: the models still mishandle missing evidence and conflicting records. These negative findings complete the measured pilot, but prevent a safe-use claim.

| Question language | Full-response model | Valid schema | Macro F1 | Conflict recall | Supported completion |
|---|---|---:|---:|---:|---:|
| en | base_standard | 0/99 | 0.0000 | 0/9 | 0/54 |
| en | first_sft_standard | 24/99 | 0.3042 | 0/9 | 15/54 |
| hi | base_standard | 0/99 | 0.0000 | 0/9 | 0/54 |
| hi | first_sft_standard | 7/99 | 0.2188 | 0/9 | 0/54 |
| hinglish | base_standard | 0/99 | 0.0000 | 0/9 | 0/54 |
| hinglish | first_sft_standard | 6/99 | 0.0250 | 0/9 | 3/54 |

The separate English state classifier scores 77/99 against 39/99 for its base comparator, including 9/9 conflicts. It incorrectly labels six missing-evidence cases SUPPORTED and abstains on one; classification improvement does not establish drafting or citation reliability. The stronger-prompt English base comparator is preserved separately (99/99 valid schema, macro F1 0.2083, 0/9 conflicts, 0/54 supported completions).

The cumulative configured compute ceiling is **$19.99 of $20**, including canceled and failed attempts; actual billing is unverified. No further GPU jobs or automatic retries are authorized under this reservation. No test-driven tuning was performed.

Reproduction uses the supplied source, pinned Linux dependency lock, dataset/adapter manifests and raw result archives. `python scripts/reproduce_private_run.py` restores the exact private training bundle only; it provisions no compute. Audit outputs can be regenerated on the CPU using the documented evaluation command. All artifacts remain private.
