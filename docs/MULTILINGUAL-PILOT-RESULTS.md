# Paired Hindi/Hinglish question evaluation — actual partitioned execution

Two separate L4 jobs each evaluated all99 unchanged cases for one language with the frozen base and retained SFT models. All396 raw model outputs were verified by case ID, prompt, model revision and source-run checksum, then evaluated by real AuditKIT on the CPU. The198 language rows represent99 source scenarios.

| Language | Model | Valid responses | Macro F1 | Conflict recall | Supported completion |
|---|---|---:|---:|---:|---:|
|hi|base_standard|0/99|0.0000|0/9|0/54|
|hinglish|base_standard|0/99|0.0000|0/9|0/54|
|hi|first_sft_standard|7/99|0.2188|0/9|0/54|
|hinglish|first_sft_standard|6/99|0.0250|0/9|3/54|

The combined-language revision-pair scorer had a family-key collision: the second language replaced the first in its internal map. This was corrected transparently using independent fixtures, giving90 pairs across both languages. All frozen native per-language metrics remain exactly unchanged. Original native reports and outputs are preserved alongside corrected combined analysis. No labels, model weights, prompts or decoding settings were tuned or repaired.

Earlier combined L4 and A100 attempts were canceled for runtime risk. The selected final partitions were chosen after real outputs, without quality-based selection, and preserve the original model/data protocol. Checkpoint uploads preserve partial raw records during execution.

Only questions are translated; evidence and schema remain English. Test translations remain unreviewed. The owner approved12 training-lineage translations with model assistance, explicitly without an independent native-speaker claim. Coarse synthetic template separation, factual-wording/draft-policy ambiguity and the small review subset limit generalization.

Cumulative configured compute ceiling:$19.99 under the authorized$20; actual billing remains unverified. No further GPU submission or automatic retry is enabled. Public release and production safety are not claimed.