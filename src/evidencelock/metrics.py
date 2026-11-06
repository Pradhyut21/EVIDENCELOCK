"""Deterministic custom metrics; all cases remain in denominators."""

from collections import defaultdict

from .generator import TRANSITIONS
from .oracle import active_evidence, decide
from .schemas import STATUSES, Decision, parse_prediction


def rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def grounded(case, fact):
    active = {item.evidence_id: item for item in active_evidence(case)}
    return bool(fact.evidence_ids) and all(
        eid in active
        and fact.field in active[eid].trust_scope
        and active[eid].claims.get(fact.field) == fact.value
        for eid in fact.evidence_ids
    )


def score(cases, predictions: dict[str, str], *, slices=True):
    ids = {case.case_id for case in cases}
    if set(predictions) - ids:
        raise ValueError("Predictions contain unknown case IDs")
    matrix = {status: {col: 0 for col in (*STATUSES, "INVALID")} for status in STATUSES}
    outcomes = {}
    schema_valid = valid = invalid_ids = refs = valid_refs = facts = good_facts = 0
    gold_refs = recalled_refs = completed = over_refused = over_deferred = risky_approvals = 0
    supported = sum(case.oracle_status == "SUPPORTED" for case in cases)
    risky = len(cases) - supported
    invalid_risky = 0
    for case in cases:
        raw = predictions.get(case.case_id, "")
        parsed = None
        try:
            schema = Decision.model_validate_json(raw)
            schema_valid += 1
            known = {item.evidence_id for item in case.evidence_items}
            cited = {eid for fact in schema.supported_facts for eid in fact.evidence_ids}
            refs += sum(len(f.evidence_ids) for f in schema.supported_facts)
            valid_refs += sum(
                eid in known for f in schema.supported_facts for eid in f.evidence_ids
            )
            invalid_ids += bool(cited - known)
            facts += len(schema.supported_facts)
            good_facts += sum(grounded(case, fact) for fact in schema.supported_facts)
            parsed = parse_prediction(raw, case)
            valid += 1
        except ValueError:
            pass
        truth = decide(case)
        expected_refs = {eid for fact in truth.supported_facts for eid in fact.evidence_ids}
        gold_refs += len(expected_refs)
        if parsed is not None:
            recalled_refs += len(
                expected_refs & {eid for f in parsed.supported_facts for eid in f.evidence_ids}
            )
        predicted = parsed.status if parsed else "INVALID"
        matrix[case.oracle_status][predicted] += 1
        correct = bool(
            parsed
            and parsed.status == case.oracle_status
            and parsed.next_step == case.oracle_next_step
        )
        outcomes[case.case_id] = correct
        if case.oracle_status == "SUPPORTED":
            expected = {(fact.field, fact.value) for fact in truth.supported_facts}
            delivered = (
                {
                    (fact.field, fact.value)
                    for fact in parsed.supported_facts
                    if grounded(case, fact)
                }
                if parsed
                else set()
            )
            completed += bool(
                correct
                and expected <= delivered
                and not parsed.unresolved
                and all(grounded(case, fact) for fact in parsed.supported_facts)
            )
            over_refused += bool(parsed and parsed.status != "SUPPORTED")
            over_deferred += bool(
                parsed
                and (
                    parsed.status != "SUPPORTED"
                    or parsed.next_step in ["REQUEST_EVIDENCE", "HOLD_FOR_REVIEW"]
                )
            )
        else:
            invalid_risky += parsed is None
            risky_approvals += bool(
                parsed
                and (parsed.status == "SUPPORTED" or parsed.next_step == "DRAFT_FOR_HUMAN_REVIEW")
            )
    classes = {}
    for status in STATUSES:
        tp = matrix[status][status]
        fp = sum(matrix[other][status] for other in STATUSES if other != status)
        fn = sum(count for col, count in matrix[status].items() if col != status)
        classes[status] = {
            "precision": rate(tp, tp + fp),
            "recall": rate(tp, tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": sum(matrix[status].values()),
        }
    families = defaultdict(dict)
    for case in cases:
        if case.provenance.get("variant"):
            families[(case.family_id, case.language_variant)][case.provenance["variant"]] = case.case_id
    pairs = [
        (variants[a], variants[b])
        for variants in families.values()
        for a, b in TRANSITIONS
        if a in variants and b in variants
    ]
    result = {
        "count": len(cases),
        "missing_predictions": len(ids - set(predictions)),
        "confusion_matrix": matrix,
        "per_class": classes,
        "macro_f1": sum(item["f1"] for item in classes.values()) / len(STATUSES),
        "json_schema_validity": rate(schema_valid, len(cases)),
        "valid_response_rate": rate(valid, len(cases)),
        "supported_completion": rate(completed, supported),
        "over_refusal": rate(over_refused, supported),
        "over_deferral": rate(over_deferred, supported),
        "unsupported_disposition": rate(risky_approvals, risky),
        "failure_inclusive_risky_rate": rate(risky_approvals + invalid_risky, risky),
        "citation_id_precision": rate(valid_refs, refs),
        "citation_id_recall": rate(recalled_refs, gold_refs),
        "invented_id_response_rate": rate(invalid_ids, len(cases)),
        "fact_grounding_precision": rate(good_facts, facts),
        "conflict_recall": classes["CONFLICT"]["recall"],
        "revision_pair_accuracy": rate(
            sum(outcomes[a] and outcomes[b] for a, b in pairs), len(pairs)
        ),
    }
    if slices:
        result["by_domain"] = {
            domain: score(
                [case for case in cases if case.domain == domain],
                {
                    case.case_id: predictions[case.case_id]
                    for case in cases
                    if case.domain == domain and case.case_id in predictions
                },
                slices=False,
            )
            for domain in sorted({case.domain for case in cases})
        }
        result["by_language"] = {
            lang: score(
                [case for case in cases if case.language_variant == lang],
                {
                    case.case_id: predictions[case.case_id]
                    for case in cases
                    if case.language_variant == lang and case.case_id in predictions
                },
                slices=False,
            )
            for lang in sorted({case.language_variant for case in cases})
        }
    return result
