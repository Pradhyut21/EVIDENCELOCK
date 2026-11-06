# Supplemental evaluation runtime recovery — historical attempt

The original L4 supplement's observed SFT decoding rate was about14.75 seconds per case, projecting beyond its one-hour timeout. The documented Jobs API exposes no running-job timeout update. This was a runtime-budget error, not a model-quality trigger. No supplemental quality metrics had been inspected for runtime selection.

A single recovery reused the identical private artifact revision and frozen protocol on a100-large. Actual NVIDIA A100-SXM4-80GB, Python3.12.16, CUDA13.0, model loading and15 real baseline outputs were observed before the L4 original was canceled. The original submission record is retained separately; partial original prediction files are not claimed as preserved. The recovery is not an additional training experiment.

Historical recovery, now canceled: https://huggingface.co/jobs/pradhyut/6aee0851c78863f9c634ce4c

Artifact revision: `da36b221a9ad1ecaf19c0d5ab170b0b23d7c20fc`.

Freeze SHA256: `d01b4594474b3167c245a49d2f0f3ccff0303578ce3950319505f91383b69f1b`.

The A100 attempt was also canceled after its measured decoding rate did not provide enough timeout margin. Its partial outputs are not treated as final multilingual results. Both final language partitions use independent L4 GPUs, preserving the original model, adapter, data, prompts and decoding settings. See [the final execution protocol](LANGUAGE-PARTITIONS-AND-METRIC-ERRATUM.md). No model-quality scores were used for runtime selection.

The historical recovery had a one-hour timeout and a configured compute ceiling of $2.51, rounded conservatively from the live hardware API price. Its full reservation remains counted despite cancellation. Including the final two language partitions, the cumulative configured compute ceiling is $19.99 under the user's $20 authorization. Actual billing remains unverified. No further GPU submissions or automatic retries are permitted under this reservation.

API reference checked: https://huggingface.co/docs/hub/jobs-reference (official OpenAPI at https://huggingface.co/.well-known/openapi.json).
