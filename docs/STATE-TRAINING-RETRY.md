# State-only training retry — 2026-11-06

The previous attempt (`6aedcebc0e0338d1bb814cdb`) failed before training because the remote package omitted `align-smoke/latest-result.json`. No trained classifier resulted from that attempt.

The corrected package restores the unmodified real report from `EVIDENCELOCK-verified-smoke-results.zip`, with archive/member SHA256 provenance. That historical report records four genuine AlignTune updates, nonzero finite gradients, adapter saving and reloading on Tiny Aya Fire revision `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`. It is historical feasibility evidence, not a new result or model quality evidence.

`gpu/bundle_preflight.py` now validates all required code, historical reports, revision, data lineage/hash, train/validation separation and tokenizer evidence before installation or model download. Eleven local regression checks exercise missing files, invalid historical gates, export corruption, Python syntax errors and forbidden frozen-test data. Every check passed against the actual retry archive.

The observable ledger now uses canonical JSON ordering. Raw dictionary and Pydantic inputs yield the same prompt at all four option rotations. Real CuratorKIT was rerun after this change: 864 accepted exposures from 99 unique training cases, 396 unique case/rotation prompts, 216 exposures per state and per letter. All 864 tokenizer boundary checks pass, with 824–1111 input tokens. Each target supervises the class-bearing letter plus two chat delimiter tokens. Native training batch checks remain required on the GPU.

New training export SHA256: `9995250011ccea3e7d6d6d89dbba662db8d7834fa655061a5418cb12e02ce92a`.

The retry uses a fresh base model and native AlignTune TRL SFT for 864 bounded iterations, followed by actual base/adapter next-token state scoring and AuditKIT. This is classification-only research; drafting, citation correctness, challenge performance and the final untouched test remain separate requirements. No frozen-test cases were packaged.

Retry job: https://huggingface.co/jobs/pradhyut/6aedd1d50e0338d1bb8153d7

This retry has a one-hour timeout and a configured compute ceiling of $0.80. The cumulative configured compute ceiling is $14.40 under the user's $20 authorization. Actual billing remains unverified. No automatic resubmission is enabled. Results are pending actual remote execution; full project completion is not claimed.
