# EVIDENCELOCK

**Revision-Aware Evidence Grounding for a Small Multilingual Model**

When records disagree, the model must not improvise.

Implementation began November 1, 2026. This is a research pipeline under construction, not a completed fine-tune or a deployed safety system. See [PREEXISTING.md](PREEXISTING.md) for the preparation history.

## Current implementation

- Strict Pydantic case, evidence and response contracts, including source/revision chronology and citation validation.
- Independent evidence oracle, original synthetic English generator and connected-component family/template/translation grouping.
- Real CuratorKIT ingestion, schema gate, exact deduplication, cleaning and trainer exports with label/evidence preservation checks.
- Real AuditKIT precomputed-prediction adapter and custom metrics. Oracle/fixture runs are explicitly identified and are not model benchmarks.
- GPU-guarded real Tiny Aya inference code. No successful model inference, AlignTune training or adapter reload is claimed.

Only English pilot data is currently generated. Hindi/Hinglish and an independently authored challenge set remain pending actual review. Three template lineages give a coarse split: target 600/100/150 sizes are not claimed. CuratorKIT's optional hallucination judge is not used.

## Install and verify

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The CPU environment does not install the full GPU stack.

```powershell
uv sync --extra dev --locked
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests
.venv\Scripts\ruff.exe format --check src tests
.venv\Scripts\python.exe -m evidencelock doctor
```

On Linux replace `.venv\Scripts\python.exe` with `.venv/bin/python` and Ruff with `.venv/bin/ruff`. The same locked CPU project is intended to run on Linux, but this checkout was initially exercised on Windows; Linux execution remains unverified.

## CPU pipeline

```powershell
.venv\Scripts\python.exe -m evidencelock generate --families-per-lineage 3 --output data/pilot-v1/generated.jsonl
.venv\Scripts\python.exe -m evidencelock split --input data/pilot-v1/generated.jsonl --output data/pilot-v1/split.jsonl --seed 42
.venv\Scripts\python.exe -m evidencelock verify --input data/pilot-v1/split.jsonl
.venv\Scripts\python.exe -m evidencelock curate --input data/pilot-v1/split.jsonl --output data/pilot-v1/curated
```

Generator/split commands refuse to overwrite existing dataset versions. Keep raw and curated checksums, group membership and input/export lineage. Never tune against held-out test outputs. The generator exposes structured synthetic claims and explicit field-authority policy; this is a narrow benchmark, not general document understanding.

Audit wiring can be exercised with the **deterministic reference**, not a learned model:

```powershell
.venv\Scripts\python.exe -m evidencelock reference --input data/pilot-v1/split.jsonl --subset validation --output artifacts/reference/predictions.jsonl
.venv\Scripts\python.exe -m evidencelock evaluate --input data/pilot-v1/split.jsonl --subset validation --predictions artifacts/reference/predictions.jsonl --output artifacts/reference/audit
```

`source_kind=oracle_reference` and `model_improvement_measured=false` are preserved. Invalid/missing outputs stay in metric denominators. See [evaluation policy](docs/EVALUATION-POLICY.md).

## GPU stage — blocked on available hardware

Model: `CohereLabs/tiny-aya-fire`, pinned revision `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`. Gated access and offline tokenizer preparation were verified before implementation. No tokens or model weights belong in Git.

The observed local MX330/2 GiB GPU is excluded from full-model inference/training. The CPU environment intentionally lacks CUDA PyTorch and AlignTune. A separate reviewed Linux training lock exists in the preparation kit; it is not yet a runtime-tested training environment. No paid compute is authorized.

```powershell
.venv\Scripts\python.exe -m evidencelock baseline --input data/pilot-v1/split.jsonl --subset validation --output artifacts/base/predictions.jsonl
```

In this CPU checkout that command must report BLOCKED (exit 2). On a verified GPU environment its code performs actual pinned-model 4-bit inference; that pathway has not been executed or verified yet. A decoded output is preserved before schema scoring; no output repair is applied. Train-smoke/train commands will be added only after the actual installed AlignTune interface can be exercised and validated; there are no dummy training commands.

Build 1 remains incomplete until real CuratorKIT output is consumed by AlignTune, a learned adapter is saved/reloaded and AuditKIT evaluates actual base/adapter predictions. CPU test counts or oracle scores do not establish model improvement.

## Attribution and publication

Inspired by the engineering and evidence-handling principles in Pradhyut21/celebrate-ai-event-planner, jalrakshak, pantrytrace and suryaledger. None of their existing data or benchmark results is reused.

Original EVIDENCELOCK code uses MIT. CuratorKIT and AuditKIT retain their source-available, noncommercial LSAL licenses; they are dependencies, not relicensed under MIT. Tiny Aya has its own gated CC-BY-NC/AUP conditions. See [third-party notices](docs/THIRD-PARTY.md). Source, dataset, adapter and Hugging Face publication remain pending verification and explicit approval.
