# Matched question-language supplement

The same retained99-update full-response adapter and base model from the frozen English comparison are evaluated on198 question translations:99 Hindi and99 Hinglish, paired to the exact99 original English test cases. All evidence records, declared policies, dates, group/lineage membership and oracle decisions remain unchanged. These are99 source scenarios, not198 independent scenarios. No related translation enters training or development validation.

The initial CPU preparation rejected unmapped held-out question wording before any supplemental GPU execution. The held-out lineage uses different question templates from the reviewed training subset. New deterministic model-authored translations were recorded under an explicit second protocol commitment before generation; the initial failed CPU commitment remains preserved. No model predictions were used to choose translations or change model/prompt settings. The resulting198 cases passed Pydantic schema, oracle invariance and evidence/policy/lineage identity checks. Actual pinned-tokenizer inputs span740–1139 tokens;256 generation tokens fit the frozen2048-token budget.

The test questions remain unreviewed. The project owner's approval of12 training-derived translations was explicitly model assisted, not an independent native-speaker check, and is not extended to these held-out question wordings. The supplement measures a narrow question-language effect with English structured evidence/output schema. It cannot establish broad multilingual document understanding.

Model revision, selected adapter SHA, dependency lock, tokenizer/config file hashes, strict parser/metrics and all payload files are pinned in the supplemental freeze. Both models use the same standard prompt, FP16 NF4 double quantization, eager attention, seed42 and greedy256-token generation. There is no truncation, JSON repair, training or post-test selection. Raw model outputs precede actual AuditKIT scoring, including per-language reliability metrics. Deterministic oracle-reference records remain separate.

Job: https://huggingface.co/jobs/pradhyut/6aedfde00e0338d1bb81aef5

Artifact revision: `da36b221a9ad1ecaf19c0d5ab170b0b23d7c20fc`.

Freeze SHA256: `d01b4594474b3167c245a49d2f0f3ccff0303578ce3950319505f91383b69f1b`.

This job's configured compute ceiling is$0.80; cumulative configured compute ceiling for all submitted jobs is$16.00 under the authorized$20. Actual billing remains unverified. No automatic further GPU submissions are scheduled. Results are pending execution and independent verification.
