# Supplemental evaluation runtime recovery

The original L4 supplement's observed SFT decoding rate was about14.75 seconds per case, projecting beyond its one-hour timeout. The documented Jobs API exposes no running-job timeout update. This was a runtime-budget error, not a model-quality trigger. No supplemental quality metrics had been inspected for runtime selection.

A single recovery reused the identical private artifact revision and frozen protocol on a100-large. Actual NVIDIA A100-SXM4-80GB, Python3.12.16, CUDA13.0, model loading and15 real baseline outputs were observed before the L4 original was canceled. The original submission record is retained separately; partial original prediction files are not claimed as preserved. The recovery is not an additional training experiment.

Selected recovery: https://huggingface.co/jobs/pradhyut/6aee0851c78863f9c634ce4c

Artifact revision: `da36b221a9ad1ecaf19c0d5ab170b0b23d7c20fc`.

Freeze SHA256: `d01b4594474b3167c245a49d2f0f3ccff0303578ce3950319505f91383b69f1b`.

Both model comparators in the supplement run on the A100. English results used L4. Numerical hardware differences therefore confound a causal attribution of cross-language differences solely to question language. Report the measurements with this limitation; do not select candidates, tune prompts or relabel cases after outcomes.

The recovery has a one-hour timeout and configured compute ceiling$2.51, rounded conservatively from the live hardware API price. Cumulative configured compute ceiling for all submitted jobs, including the canceled original, is$18.51 under the user's$20 authorization. Actual billing remains unverified. No new automatic GPU submissions are enabled.

API reference checked: https://huggingface.co/docs/hub/jobs-reference (official OpenAPI at https://huggingface.co/.well-known/openapi.json).
