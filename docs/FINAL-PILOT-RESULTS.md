# Frozen pilot evaluation — actual results

Execution: **PASS**. Job `6aedf8eb0e0338d1bb81a57d`.

The original99 test cases were evaluated once under the recorded frozen protocol. No model, prompt or label changes follow from these outcomes. Classification is reported separately from complete responses.

| Full response comparator | Valid responses | Macro F1 | Conflict recall | Supported completion |
|---|---:|---:|---:|---:|
|base_standard|0/99|0.0000|0/9|0/54|
|base_strong_prompt|99/99|0.2083|0/9|0/54|
|first_sft_standard|24/99|0.3042|0/9|15/54|

| Auxiliary state classification | Correct | Abstentions | Rotation unanimity |
|---|---:|---:|---:|
|base|39/99|29/99|10/99|
|state_sft|77/99|1/99|94/99|

Every one of297 raw model response records and792 classification-score records was checked against its recorded checksum and case membership. AuditKIT runs, raw prompts/outputs, per-case failures, oracle-reference records, environment and adapter evidence are preserved in the private result archive. Oracle-reference outputs are never counted as learned-model predictions.

The dataset is synthetic with coarse template-lineage separation. Human review was a small question-only subset approved with model assistance, not an independent native-speaker check. Development ablations repeatedly reused validation data. The original factual-question/draft-policy ambiguity stays disclosed. No broad multilingual, independent challenge, production safety or public-release claim follows from this pilot.

Result archive SHA256: `a2136b319f2cf8110c27d0142d7b46dde61fa776218a50153ba0e9550c224713`.

Cumulative configured GPU compute ceiling:$16.00 under the authorized$20. Actual billing remains unverified. No automatic further training or GPU submission is enabled.