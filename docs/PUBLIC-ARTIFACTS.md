# Public artifacts

Publication date: November 7, 2026. Repository and collection visibility is independently checked without authentication. All EVIDENCELOCK repositories in the submission collections are public and ungated. The upstream gated base model is cited in adapter cards rather than included as a collection member.

| Artifact | Public location |
|---|---|
| Model collection | [https://huggingface.co/collections/pradhyut/evidencelock-models-and-adapters-6aee1db0365774ae339c75a1](https://huggingface.co/collections/pradhyut/evidencelock-models-and-adapters-6aee1db0365774ae339c75a1) |
| Dataset collection | [https://huggingface.co/collections/pradhyut/evidencelock-datasets-and-research-records-6aee1db1fe3d3d356aa1b918](https://huggingface.co/collections/pradhyut/evidencelock-datasets-and-research-records-6aee1db1fe3d3d356aa1b918) |
| State adapter | https://huggingface.co/pradhyut/evidencelock-state-lora |
| Full-response adapter | https://huggingface.co/pradhyut/evidencelock-response-lora |
| Ten historical adapter artifacts | https://huggingface.co/pradhyut/evidencelock-research-ablations |
| Benchmark and curated splits | https://huggingface.co/datasets/pradhyut/evidencelock-benchmark |
| Logs, traces, outputs and immutable inputs | https://huggingface.co/datasets/pradhyut/evidencelock-research-records |
| Source | https://github.com/Pradhyut21/EVIDENCELOCK |
| Hugging Face article | https://huggingface.co/blog/pradhyut/evidencelock |

All 12 unique available adapter weight files retain their original SHA256 values. Standalone public adapter configurations change only the base identifier and pinned revision to portable values; originals and metadata-change records are preserved. Historical reference/smoke artifacts are labeled accurately.

The original 99 English test scenarios have 198 question translations; they are not independent new scenarios. All 693 full-response outputs and 792 auxiliary option-score records are preserved with separate oracle references. Nine actual dataset configs load successfully in an isolated CPU environment.

The [Hugging Face article](https://huggingface.co/blog/pradhyut/evidencelock) is public. Its source manuscript is available in [HUGGINGFACE-ARTICLE.md](HUGGINGFACE-ARTICLE.md). The official submission receipt is tracked separately; this index does not claim a completed form submission. The official form includes a human CAPTCHA. Personal registration details remain outside the public repositories.

Original code: MIT. Original benchmark: CC BY 4.0. Adapter derivatives and generated outputs retain applicable Cohere CC BY-NC/AUP conditions; historical tokenizer assets retain upstream terms. See [third-party notices](THIRD-PARTY.md).
