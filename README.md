# EVIDENCELOCK

**Revision-Aware Evidence Grounding for a Small Multilingual Model**

When records disagree, the model must account for evidence, authority and revisions before choosing an action.

EVIDENCELOCK provides an original synthetic benchmark, an independent evidence oracle and an executed training/evaluation pipeline for `CohereLabs/tiny-aya-fire`. **CuratorKIT** processes the data, **AlignTune** trains real LoRA adapters, and **AuditKIT** evaluates preserved baseline and adapter outputs. The frozen study covers English, Hindi and Hinglish questions and measures state selection, response structure, citations and permitted actions separately.

Implementation began November 1, 2026. See [the execution record](docs/BUILD1-EXECUTION-REPORT.md) for runtime evidence and [PREEXISTING.md](PREEXISTING.md) for the preparation history.

## Public artifacts

The submission is organized into source, reusable adapters, benchmark data and research records:

| Artifact | Repository |
|---|---|
| Source, tests and reproduction instructions | [Pradhyut21/EVIDENCELOCK](https://github.com/Pradhyut21/EVIDENCELOCK) |
| Auxiliary state-classification adapter | [pradhyut/evidencelock-state-lora](https://huggingface.co/pradhyut/evidencelock-state-lora) |
| Retained complete-response adapter | [pradhyut/evidencelock-response-lora](https://huggingface.co/pradhyut/evidencelock-response-lora) |
| Historical research ablations | [pradhyut/evidencelock-research-ablations](https://huggingface.co/pradhyut/evidencelock-research-ablations) |
| Synthetic benchmark and curated splits | [pradhyut/evidencelock-benchmark](https://huggingface.co/datasets/pradhyut/evidencelock-benchmark) |
| Logs, traces, predictions and evaluation records | [pradhyut/evidencelock-research-records](https://huggingface.co/datasets/pradhyut/evidencelock-research-records) |

Adapter repositories identify the pinned base model and their measured task. Historical ablations are retained as research artifacts rather than recommended replacements for the retained adapters.

## Current implementation

- Strict Pydantic case, evidence and response contracts, including source/revision chronology and citation validation.
- Independent evidence oracle, original synthetic English generator and connected-component family/template/translation grouping.
- CuratorKIT ingestion, schema checks, exact deduplication, cleaning and trainer exports with label/evidence preservation checks.
- AuditKIT evaluation of preserved baseline/adapter generations, with custom reliability metrics and actual prompt provenance. Oracle/fixture runs remain explicitly identified.
- Pinned Tiny Aya inference, NF4 LoRA backward passes, native AlignTune TRL SFT/DPO training, adapter saving and independent reloading on remote NVIDIA GPUs.
- A verified auxiliary state-only classifier using lossless observable ledgers and four option rotations. It completed 864 genuine optimizer updates and independently reloaded its saved adapter. State selection is measured separately from drafting and citation correctness.

## Benchmark and measured results

The original synthetic English pilot contains 297 cases: 99 each in training, development validation and frozen test, separated by three template lineages. This is a coarse split with limited independent support. The frozen language supplement contains 198 question translations of the same 99 test scenarios; evidence stays in English. Test translations and a separate challenge draft remain unreviewed. The owner approved a small training-derived translation subset with model assistance; independent native-speaker validation is outside the measured pilot. A 16-case training-derived language probe is explicitly diagnostic. CuratorKIT's optional hallucination judge is not used.

The state-only classifier scores **77/99 versus 39/99** for its base comparator on the frozen English test and recognizes **9/9 conflicts**. It also incorrectly calls six missing-evidence cases supported. State classification and complete-response generation are therefore reported separately; classification accuracy does not establish drafting or citation reliability.

The retained complete-response SFT produces 24/99 schema-valid English responses and 15/54 supported completions, with zero conflict/missing recall. It produces 7/99 valid Hindi responses and 6/99 valid Hinglish responses, with zero conflict recall in both languages. A stronger-prompt English base comparator produces 99/99 schema-valid responses while detecting 0/9 conflicts and completing 0/54 supported cases. These measurements show why output structure, decision correctness and evidence handling need separate evaluation. See [the frozen English results](docs/FINAL-PILOT-RESULTS.md) and [the multilingual results](docs/MULTILINGUAL-PILOT-RESULTS.md).

All 396 language-supplement outputs and 297 English comparator outputs are preserved. A documented correction to combined-language revision-pair aggregation leaves every native per-language result unchanged. The results describe this small synthetic pilot and do not establish deployment safety or broad multilingual generalization.

For development context, the first 99-update SFT adapter produced 72/99 valid validation responses and 35/54 supported completions, but detected 0/9 conflicts and 0/27 missing-evidence cases. Later DPO ablations improved some development metrics while retaining evidence-handling errors. These figures use repeatedly reused synthetic validation examples and remain distinct from frozen-test results.

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

Actual CuratorKIT output has been consumed by AlignTune, learned adapters have been saved/reloaded, and AuditKIT has evaluated real paired model predictions. The frozen research pilot has completed execution. Independent linguistic and challenge validation remain separate research work. CPU test counts and oracle scores do not establish model improvement. The state-only retry is documented in [its execution record](docs/STATE-TRAINING-RETRY.md). The held-out question-language supplement is documented in [its protocol](docs/MULTILINGUAL-EVALUATION-PROTOCOL.md), [historical runtime recovery](docs/RUNTIME-RECOVERY.md) and [final language partitions](docs/LANGUAGE-PARTITIONS-AND-METRIC-ERRATUM.md).

Historical private immutable bundles preserve the original run's code, curated inputs, feasibility reports and dependency locks. Account-authorized contributors can restore the classifier retry without running it:

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

Original EVIDENCELOCK code uses MIT. CuratorKIT and AuditKIT retain their source-available, noncommercial LSAL licenses; they are dependencies, not relicensed under MIT. Tiny Aya has its own gated CC-BY-NC/AUP conditions. See [third-party notices](docs/THIRD-PARTY.md). Each public artifact's card records its scope, provenance and applicable distribution terms. The gated base model weights are not redistributed in this repository.
