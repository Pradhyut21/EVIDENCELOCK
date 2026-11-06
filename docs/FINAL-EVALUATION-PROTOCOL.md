# Final evaluation protocol — 2026-11-06

The frozen comparison selects the retained original99-update SFT full-response adapter before test generation. Its SHA256 is `0d4a3e99deb3dbb27c6af2f1c3f334c268e51b51557372e7b40da1cb725754ac`. The new state-only adapter (`cad9f72b601182473a88633ad5e68d703eb5f93a21360639276be7871a4b0fba`) remains auxiliary and cannot replace drafting/citation evaluation.

The original99 English test cases remain unchanged. The protocol pins case bytes, original split manifests, all runtime source files, strict parser/metrics, dependency lock, adapter weights/config, the model revision and the tokenizer/config file contents. Protocol commitment precedes test materialization; no test predictions were available for selection. Full input freeze SHA256: `d7474eacdcf0c5a029fae8fa0195ba90795d69e69550b3fef671ef702d2e498f`.

Full responses: base standard prompt, base strong-prompt-only, original SFT standard prompt. Each uses FP16 NF4 double quantization, eager attention, seed42, greedy decoding,256 new tokens maximum and2048 total context maximum. All198 distinct standard/strong case-prompt boundaries were checked offline with the actual pinned tokenizer: standard732–1119 tokens; strong1055–1442 tokens. There is no truncation or repair. Raw outputs are written before AuditKIT scoring. Deterministic reference outputs are labeled oracle_reference and evaluated separately.

The auxiliary four-choice state classifier runs separately across four rotations on the same frozen cases, with actual next-token scores and its own AuditKIT records. It does not report JSON, drafting or citation success. No post-test tuning is permitted. Technical failures are distinct from negative quality outcomes.

The user approved the12-row training-derived translation subset with model assistance and explicitly did not claim independent native-speaker review. The original factual-wording/request_kind=draft ambiguity remains unchanged and disclosed.

Submitted job: https://huggingface.co/jobs/pradhyut/6aedf8eb0e0338d1bb81a57d

Artifact revision: `3d314531c431ec28d3cf0d1b4cb04a42695f7630`.

This run has a one-hour timeout, configured compute ceiling$0.80 and cumulative configured compute ceiling$15.20. Actual billing remains unverified. No automatic resubmission is enabled.

A legacy fresh-recovery manifest inside the remote wrapper carries inherited adapter/historical-report restoration flags. Those flags are not evidence of this payload's contents. The immutable freeze file and per-file hashes identify the actual inputs; the final payload includes both selected adapters and does not restore historical smoke reports. No training is performed by this job.

Results are pending actual execution. Successful execution does not imply good model performance or completed publication.
