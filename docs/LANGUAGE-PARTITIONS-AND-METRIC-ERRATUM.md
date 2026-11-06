# Operational language partitions and scoring erratum

Measured combined-run decoding exceeded its runtime budget; an A100 did not provide enough margin. The final execution therefore runs all99 Hindi cases and all99 Hinglish cases in two independent L4 jobs. Both produced real outputs before the A100 copy was canceled. No quality metrics from these attempts were used to select the execution. The same198 case bytes, model revision, retained adapter, tokenizer, prompts, precision and decoding settings are preserved. Only execution grouping and durable private raw-output checkpoints changed. Partition manifests retain the original model/data protocol and hash the operational driver separately. The GPUs do not share memory.

Each partition has a55-minute timeout and a configured compute ceiling$0.74. Cumulative configured compute ceiling, including both canceled combined attempts, is$19.99 under the user's$20 authorization. No additional GPU submissions or automatic retries are permitted under this reservation. Actual billing remains unverified.

- hi: https://huggingface.co/jobs/pradhyut/6aee0f37c78863f9c634dc82
- hinglish: https://huggingface.co/jobs/pradhyut/6aee0f4ec78863f9c634dcb7

The native per-language raw outputs and AuditKIT reports are retained independently. The union is validated for disjoint IDs, exactly99 cases per language, exact actual prompts, model revision, adapter SHA and recorded prediction checksums before actual CPU AuditKIT evaluation.

## Metric correction

A code audit found that the original revision-pair grouping keyed only by family_id. When both translations share a family and variant, the second language overwrites the first. Within-language results are unaffected; the combined revision-pair denominator would incorrectly be45 instead of90.

The corrected scorer groups by(family_id,language_variant). Independent synthetic fixtures with perfect Hindi reference outputs and deliberately invalid Hinglish fixtures must yield5/10 correct revision pairs, regardless of row order. Both tests pass. These fixtures are software controls, not model results.

Original frozen native reports are preserved. Corrected combined analysis is explicitly versioned with both metric-source hashes and must reproduce every native per-language metric exactly. No gold labels, model outputs, prompts or model parameters are changed, repaired or selected after test outcomes. All inference outputs remain raw model predictions; deterministic reference outputs are separately labeled oracle_reference.

The translations remain unreviewed test rows. The owner approved a small training-derived subset with model assistance, without an independent native-speaker claim. The resulting study remains a narrow synthetic question-language evaluation with English structured evidence, not broad multilingual document validation or deployment evidence.
