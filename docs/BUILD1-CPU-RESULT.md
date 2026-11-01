# November 1, 2026 — CPU implementation checkpoint

**CPU foundation executed and tested. Full Build 1 remains incomplete.**

## Implemented and genuinely executed

- Strict evidence/case/output contracts, independent oracle, original generator, connected-group splitter, raw/export provenance and custom metrics.
- CuratorKIT 0.1.22: real Curator/CuratorConfig ingestion, SchemaGate, ExactDeduplicator, TextCleaner and messages/Alpaca export.
- AuditKIT 1.1.3: real Sample(actual_output=...), explicit-direction custom scorer and evaluate(model="precomputed"). Current recorded run uses the deterministic oracle reference, not an LLM.
- A real-model inference pathway is implemented but unexecuted; its local runtime check correctly returned BLOCKED (exit 2). No fallback model or fake inference was substituted.

Source repository: `D:\EVIDENCELOCK`, branch `codex/build-1`. Initial implementation commit: `42af77751e181168b75d022e061e1d0201bc5c3f`. Subsequent tokenizer/metric hardening is recorded in actual Git history. No backdating and no GitHub/HF publication occurred.

## Pilot dataset actually produced

297 newly generated synthetic English cases, 27 incident families and only **three independent template lineages**. Entire related lineages remain in one split, so the coarse pilot is **99 train / 99 validation / 99 test**. No row-random split was used. Group membership, seed 42 and checksums are saved before any model tuning.

CuratorKIT accepted/exported all 297 and rejected none in this pilot. A separate integration control genuinely exercised one duplicate removal and one schema rejection; those are test-fixture counts, not extra research dataset records. Every actual export's labels, source IDs and prompt were checked. An LLM hallucination judge was not used. All 297 cases remain **human-unreviewed**.

Status counts: SUPPORTED 162; MISSING_EVIDENCE 81; CONFLICT 27; OUT_OF_SCOPE 27. This is a pilot, not the target-size or independent multilingual challenge benchmark. Hindi/Hinglish review and independent challenge authoring remain pending.

Generated SHA-256: `685ace3420acff132476cd8e3efde7292696774d769016d0d69923555f095110`.

Split SHA-256: `163545625dd01c1b585643239f48ed7a7680451d2c48edb538781bf4d7e54ded`.

See data/pilot-v1/split.manifest.json, curated/*/lineage.json and artifacts/run-manifest.json for exact artifact provenance. Dataset and run outputs are included in the local delivery archive but excluded from Git; they are reproducible with the CLI.

## Verification actually executed

- Final reviewed suite: **45 passed**, including actual toolkit calls, revision truth tables, malformed/invalid citations, leakage relationships, reproducible bytes and refusal/grounding failure controls.
- Ruff lint and format checks: passed.
- Isolated Python 3.12.13 CPU environment: dependency check passed; pinned Pydantic 2.14.2, CuratorKIT 0.1.22, AuditKIT 1.1.3, pytest 9.0.2 and Ruff 0.14.3.
- Main generate → split → verify → curate commands: exit 0.
- AuditKIT on 99 validation **oracle-reference** outputs: exit 0; fingerprint `2f41d3909c4531e2`. This is software integration evidence, not measured base/tuned performance.
- Local model baseline command: BLOCKED, exit 2, because this isolated environment intentionally lacks CUDA PyTorch. The observed GPU is still an MX330 with 2,048 MiB total/1,969 MiB free in the captured doctor report.

The first toolkit test run found and preserved three failures. CuratorKIT's SFT dedup key omits Alpaca input context: moving the complete evidence question into instruction preserved semantically different revision cases. Duplicate removals are tracked separately from the toolkit's quality rejection list and reconciled into an explicit lineage sidecar. AuditKIT's installed FunctionScorer requires a scoring direction; its documented explicit-direction decorator corrected the integration. All affected tests were rerun.

## Tokenization feasibility finding

Real offline Tiny Aya tokenization of the 99 curated training messages: **786–1,195 total tokens**, **733–1,122 prompt tokens**, **41–73 assistant-span tokens**; zero prefix-alignment failures. A right-truncated 512-token context would leave all 99 without assistant tokens; at 1,024, 18 would have no assistant span. All fit the tested 2,048-token budget. These are actual token counts, not a verified trainer loss mask or backward pass.

This CPU profile used existing global Python 3.13.5/Transformers 4.57.6/tokenizers 0.22.2 with model frameworks disabled, without changing global packages. The eventual isolated GPU environment must rerun the check using its actual pinned tokenizer/trainer versions. No model weights were loaded.

## Remaining blocker and next action

An authenticated, suitable GPU runtime is not available through this session. Colab's public page can be read, but no connected tool controls the user's signed-in Google account or verifies an allocation. Opening the browser automatically was blocked by the local execution policy. No paid compute was requested.

Open https://colab.research.google.com/#create=true manually, sign in if needed, choose Runtime → Change runtime type → a free GPU option if offered → Save, and run `!nvidia-smi`. Alternatively upload notebooks/check_free_gpu.ipynb and run its one check cell. Paste the GPU name/VRAM here; no passwords or tokens are needed in chat. GPU availability is not guaranteed: https://research.google.com/colaboratory/faq.html.

Then verify the actual GPU/driver, full isolated Lexsi dependency installation, authenticated model access, weight load, baseline generation, AlignTune supervised backward/optimizer steps, real adapter export/reload and AuditKIT before/after. Until those execute, there is no trained adapter, no model improvement measurement, no final release and no Build 1 completion claim.
