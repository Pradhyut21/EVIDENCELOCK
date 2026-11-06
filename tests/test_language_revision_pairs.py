"""Fixtures expose mixed-language pair loss without using any real test predictions."""

from evidencelock.generator import generate
from evidencelock.metrics import score
from evidencelock.oracle import decide


def bilingual_fixture():
    generated = generate(1)
    family = generated[0].family_id
    source = [case for case in generated if case.family_id == family]
    cases, predictions = [], {}
    for language in ["hi", "hinglish"]:
        for case in source:
            translated = case.model_copy(
                update={"case_id": case.case_id + "-" + language, "language_variant": language},
                deep=True,
            )
            cases.append(translated)
            # Reference fixture versus deliberate invalid fixture, never model results.
            predictions[translated.case_id] = (
                decide(translated).model_dump_json() if language == "hi" else "{}"
            )
    return cases, predictions


def test_combined_revision_pairs_include_each_language():
    cases, predictions = bilingual_fixture()
    result = score(cases, predictions)
    assert result["revision_pair_accuracy"] == {"numerator": 5, "denominator": 10, "value": 0.5}
    assert result["by_language"]["hi"]["revision_pair_accuracy"]["value"] == 1.0
    assert result["by_language"]["hinglish"]["revision_pair_accuracy"]["value"] == 0.0


def test_row_order_cannot_choose_which_language_counts():
    cases, predictions = bilingual_fixture()
    forward = score(cases, predictions)["revision_pair_accuracy"]
    reverse = score(list(reversed(cases)), predictions)["revision_pair_accuracy"]
    assert forward == reverse == {"numerator": 5, "denominator": 10, "value": 0.5}
