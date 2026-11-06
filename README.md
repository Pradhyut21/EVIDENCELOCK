# EVIDENCELOCK

**Revision-Aware Evidence Grounding for a Small Multilingual Model**

When records disagree, the model must not improvise.

Implementation began November 1, 2026. The English pilot has executed curation, actual model training, adapter reloading and evaluation. The wider multilingual study remains incomplete, and the measured models have serious reliability failures. See [PREEXISTING.md](PREEXISTING.md) for the preparation history.

## Current implementation

- Strict Pydantic case, evidence and response contracts, including source/revision chronology and citation validation.
- Independent evidence oracle, original synthetic English generator and connected-component family/template/translation grouping.
- Real CuratorKIT ingestion, schema gate, exact deduplication, cleaning and trainer exports with label/evidence preservation checks.
- Real AuditKIT evaluation of preserved baseline/adapter generations, with custom reliability metrics and actual prompt provenance. Oracle/fixture runs remain explicitly identified.
- Real pinned Tiny Aya inference, NF4 LoRA backward passes, native AlignTune TRL SFT/DPO training, adapter saving and independent reloading on remote NVIDIA GPUs.
- An auxiliary state-only classification experiment using lossless observable ledgers and four option rotations. This measures state selection separately from drafting and citation correctness.

The original synthetic English pilot contains 297 cases: 99 each in training, development validation and frozen test, separated by three template lineages. This is a coarse split with limited independent support. Hindi/Hinglish translations and a separate challenge draft exist but remain unreviewed. A 16-case training-derived language probe has executed and is explicitly diagnostic. CuratorKIT's optional hallucination judge is not used.

The first 99-update SFT adapter produced 72/99 valid validation responses and 35/54 supported completions, but detected 0/9 conflicts and 0/27 missing-evidence cases. Later DPO ablations improved some development metrics while retaining serious safety failures. These are reused development-validation results on unreviewed synthetic examples, not frozen-test evidence or production readiness. Negative results are valid research findings; candidate promotion thresholds do not redefine the original research success criteria.

## Install and verify

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The CPU environment does not install the full GPU stack.

```powershell
uv sync --extra dev --locked
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src tests
.venv\Scripts\ruff.exe format --check src tests
.venv\Scripts\python.exe -m evidencelock doctor
```

On Linux replace `.venv\Scripts\python.exe` with `.venv/bin/python` and Ruff with `.venv/bin/ruff`. The isolated Linux training stack has now executed on Tesla T4 and L4 runtimes. The Windows CPU environment stays separate.

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

## Verified remote GPU execution

Model: `CohereLabs/tiny-aya-fire`, pinned revision `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`. Gated access and offline tokenizer preparation were verified before implementation. No tokens or model weights belong in Git.

The observed local MX330/2 GiB GPU is excluded from full-model inference/training. Remote jobs use an isolated Linux Python 3.12 environment with torch 2.11.0+cu130, Transformers 5.22.1, PEFT 0.21.2, TRL 1.7.1, Accelerate 1.15.0, bitsandbytes 0.50.2, CuratorKIT 0.1.22, AlignTune 0.1.28 and AuditKIT 1.1.3. The complete dependency graph is pinned and checked in `gpu/requirements-linux-py312.txt`; declared dependencies are retained.

FP16 NF4 training uses FP32 adapter master weights after native TRL setup and an initial gradient scale of 1024. Native batch checks verify the actual completion mask; model quality is evaluated separately from successful training. The user authorized at most $20 of existing Hugging Face credits. Each submitted L4 run has a bounded timeout; duplicate submission guards and cumulative compute ceilings are recorded privately. Actual billing totals are not inferred from configured ceilings.

```powershell
.venv\Scripts\python.exe -m evidencelock baseline --input data/pilot-v1/split.jsonl --subset validation --output artifacts/base/predictions.jsonl
```

In this CPU checkout that command reports BLOCKED (exit 2). Remote training entry points are `gpu/train_experiment.py` for the original pilot and `gpu/train_state_only.py` for the auxiliary classifier. They require the prepared input bundle and real historical feasibility gates. Raw decoded outputs are preserved before schema scoring; no output repair is applied. The remote bundle validator runs before dependency installation or model download.

Actual CuratorKIT output has been consumed by AlignTune, learned adapters have been saved/reloaded, and AuditKIT has evaluated real paired model predictions. Full project completion still requires the documented review and final evaluation requirements. CPU test counts and oracle scores do not establish model improvement. The latest state-only retry is documented in [its execution record](docs/STATE-TRAINING-RETRY.md).

Private immutable bundles preserve code, curated inputs, historical feasibility reports and dependency locks. Restore the current classifier retry without running it:

```bash
python scripts/reproduce_private_run.py \
  --revision 886de99bb318f56d01295a6284fc06187a3179ee \
  --path input/state-only-training-v2/payload.zip \
  --sha256 c23c06ad17cc8771c1270c7cad7ec981ef819beb6e430f5f5a0482a8a5d90b48 \
  --output /tmp/evidencelock-reproduction
```

Authenticate locally with Hugging Face before restoring private inputs. Add `--execute` only on a selected Linux GPU with at least 12 GiB VRAM; the wrapper never provisions compute. `/content/evidencelock-gpu` must be unused or contain byte-identical restored inputs. Successful execution and exact GPU determinism are separate claims.

## Attribution and publication

Inspired by the engineering and evidence-handling principles in Pradhyut21/celebrate-ai-event-planner, jalrakshak, pantrytrace and suryaledger. None of their existing data or benchmark results is reused.

Original EVIDENCELOCK code uses MIT. CuratorKIT and AuditKIT retain their source-available, noncommercial LSAL licenses; they are dependencies, not relicensed under MIT. Tiny Aya has its own gated CC-BY-NC/AUP conditions. See [third-party notices](docs/THIRD-PARTY.md). Source, dataset, adapter and Hugging Face publication remain pending verification and explicit approval.
