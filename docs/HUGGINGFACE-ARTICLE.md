# EVIDENCELOCK: What Changes When the Evidence Changes?

Reliable AI assistance requires more than a plausible answer. A model must recognize when records are incomplete, when authorized sources disagree, and when a later correction changes what can be concluded. It must also keep the proposed action within the evidence's scope.

**EVIDENCELOCK** is a reproducible research prototype for studying those decisions with a small multilingual model. It brings together a versioned evidence benchmark, an independent decision oracle, genuine model training, and evaluation that preserves the original model outputs. The project asks a practical question: **which parts of evidence handling improve through training, and which still require stronger safeguards?**

## A contract for evidence-based decisions

Each case contains a structured ledger of evidence records. Records identify their source, assertion and observation times, authorized fields, correction relationships, and retraction status. A case also specifies the time of assessment and its permitted scope.

The model must choose one of four states:

- **SUPPORTED:** the required conclusion has sufficient authorized evidence.
- **MISSING_EVIDENCE:** the required evidence is absent or insufficient.
- **CONFLICT:** relevant evidence contains an unresolved disagreement.
- **OUT_OF_SCOPE:** the requested conclusion exceeds the case's declared scope.

The complete response contract includes supported facts, evidence citations, unresolved issues, an explanation, and a permitted next action. This makes several questions independently measurable: Did the model choose the right state? Do its citations belong to the case? Does the proposed action match the evidence? Did it account for a correction?

An independent oracle supplies the reference decisions. Its deterministic rules operate on the synthetic ledger's explicit evidence policy. These references are labeled separately from learned-model predictions throughout the project.

## A small benchmark with traceable lineage

The original benchmark contains **297 synthetic English cases**, divided into 99 training, 99 development-validation, and 99 frozen-test cases. Connected evidence families and template lineages determine the partitions. This prevents related variants from being scattered across ordinary random row splits, while retaining a clearly documented limitation: the pilot has only three template lineages.

The language supplement adds **99 Hindi and 99 Hinglish questions** for the same frozen English scenarios. Evidence records, reference decisions, and split membership remain unchanged. These are 198 translated question rows derived from 99 underlying scenarios, rather than 198 independent evidence problems.

A small training-derived translation subset received owner approval with model assistance. The held-out translations have not received independent native-speaker validation. Accordingly, the language results describe this specific question-translation experiment with English evidence.

## An executed curation, training, and evaluation pipeline

EVIDENCELOCK uses **CuratorKIT** for actual dataset ingestion, schema checks, deduplication, cleaning, and trainer exports. Lineage checks verify that curation preserves labels and evidence.

**AlignTune** drives native training of [CohereLabs/tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire), pinned to revision `73500c80fcb6c8bea3e830c21d5d2920249a0e1a`. The pipeline includes real NF4 LoRA execution, gradient checks, optimizer updates, adapter saving, and independent reloading. A separate state-only experiment completed **864 optimizer updates**.

**AuditKIT** evaluates preserved model generations with custom reliability metrics and actual prompt provenance. Invalid responses remain in the denominators. No output repair is applied before scoring. Baseline, adapter, and deterministic-reference records remain distinguishable.

The frozen evaluation preserves **693 complete-response generations** across English, Hindi, and Hinglish comparisons, plus **792 auxiliary classification score records**. Model revisions, dataset checksums, adapters, inference settings, and dependency locks support replay and inspection.

## The central finding: three capabilities need separate measurement

The auxiliary English state classifier makes **77/99 correct decisions**, compared with **39/99** for its base comparator. It recognizes all **9/9 conflict cases**. The remaining errors matter: six missing-evidence cases are incorrectly classified as supported, and one missing-evidence case receives an abstention. This classifier measures state selection; it does not establish drafting or citation reliability.

Complete-response evaluation exposes additional weaknesses:

| Frozen comparator | Schema-valid responses | Conflict recall | Supported completion |
|---|---:|---:|---:|
| English base, standard prompt | 0/99 | 0/9 | 0/54 |
| English base, stronger prompt | 99/99 | 0/9 | 0/54 |
| English retained SFT adapter | 24/99 | 0/9 | 15/54 |
| Hindi retained SFT adapter | 7/99 | 0/9 | 0/54 |
| Hinglish retained SFT adapter | 6/99 | 0/9 | 3/54 |

The stronger-prompt baseline is especially informative: every English response satisfies the schema, yet none of the conflict cases is detected. A successful training run, a correct output shape, and a correct evidence decision therefore require separate evaluation.

The reports preserve real examples of successful structured decisions and failed generations. Where no successful example exists for a comparator, that absence is stated explicitly. The study does not establish the root cause of every generation error.

## Reproducibility and interpretation

Frozen outcomes were not used for subsequent prompt, label, or model tuning. Development experiments repeatedly reused validation data, so their gains remain development findings. An original factual-question versus drafting-policy ambiguity is disclosed in the dataset documentation.

A code audit also found a combined-language revision-pair aggregation error: grouping only by family allowed one translation to overwrite another. The corrected scorer groups by family and language. Independent software fixtures verify the correction; every original per-language metric remains unchanged, and native reports are preserved. CPU replay reproduces the combined SFT metrics and AuditKIT fingerprint.

EVIDENCELOCK contributes an inspectable benchmark and an executed safety research workflow. Its results identify a useful classification improvement alongside a clear remaining problem in complete evidence-grounded responses. The pilot supports further controlled research; it does not establish deployment safety or broad multilingual generalization.

Explore the [source and reproduction instructions](https://github.com/Pradhyut21/EVIDENCELOCK), [model and adapter collection](https://huggingface.co/collections/pradhyut/evidencelock-models-and-adapters-6aee1db0365774ae339c75a1), and [datasets, traces, and evaluation artifacts](https://huggingface.co/collections/pradhyut/evidencelock-datasets-and-research-records-6aee1db1fe3d3d356aa1b918).
