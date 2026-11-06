# Direct evidence-state classification — measured development result

The genuine AlignTune run completed864 optimizer updates on864 balanced exposures from99 distinct training cases, using396 distinct case/option-order prompts. Each of the four states and four answer letters has216 exposures. Repetition is sampling, not additional independent data. Real CuratorKIT accepted every exposure without changing its prompt or label. Native TRL collator checks verified the first supervised token is the intended class letter for all864 rows; each completion also includes two delimiter tokens.

Model: CohereLabs/tiny-aya-fire at73500c80fcb6c8bea3e830c21d5d2920249a0e1a. FP16 NF4 double quantization; FP32 LoRA master weights; q_proj/v_proj rank4, alpha8; learning rate1e-4; seed42; batch1; gradient accumulation1;1536 training-token limit. There were124344 finite nonzero gradient events and zero nonfinite events.864 actual optimizer steps and changed saved weights were verified. Training took921.59 seconds on an NVIDIA L4. The saved3,852,744-byte adapter was independently reloaded. Adapter SHA256: `cad9f72b601182473a88633ad5e68d703eb5f93a21360639276be7871a4b0fba`.

The ledger is a lossless canonical representation of observable questions, evidence, scope and policy. It does not filter evidence using the oracle, leak the in_scope annotation or include gold labels. Dict/Pydantic prompt equivalence is tested. Four option rotations expose answer-position bias; an aggregate prediction needs at least3 matching state votes. Otherwise it is ABSTAIN. Scores are actual model next-token logits, not calibrated probabilities.

| Reused development validation | Base | State-only SFT |
|---|---:|---:|
| Correct strict-majority state |54/99|75/99|
| Supported-state recall |54/54|36/54|
| Missing-evidence recall |0/27|21/27|
| Conflict recall |0/9|9/9|
| Out-of-scope recall |0/9|9/9|
| Abstentions |8/99|9/99|
| Unanimous across option rotations |15/99|86/99|
| Full-vocabulary top token is an option |396/396|396/396|

The adapter improves missing-evidence/conflict/scope detection while regressing on supported cases:6 supported cases predict missing evidence,9 predict conflict and3 abstain. Of27 missing-evidence cases,21 are correct and6 abstain. These tradeoffs must remain visible. The prespecified classifier promotion thresholds are not all met:75.76% accuracy misses80%, and77.78% missing-evidence recall misses80%. Research findings remain useful without model promotion; no further automatic optimization is scheduled.

Real AuditKIT evaluated396 per-rotation outputs per variant. Its fingerprints are base `dce7f8ae3f60a9c9`, adapter `138a936b19c95c33`. A separate CPU verification recomputed every decision and majority from792 preserved score records, checked all observable payloads and confirmed every aggregate count. This verifies analysis consistency rather than independent experimental replication.

These are reused synthetic development cases with a coarse template-lineage split. They are classification-only measurements: no claim about complete JSON, next-step actions, drafting, citation grounding, production safety, multilingual generalization or final-test performance follows from this table. Raw scores, exact prompts, matrices, adapter weights and environment reports are retained privately. The final frozen comparison is a separate run, with no post-test selection or tuning.
