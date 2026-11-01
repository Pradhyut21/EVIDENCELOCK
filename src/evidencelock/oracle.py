"""Independent deterministic policy over structured claims, never model outputs."""

from collections import defaultdict
from datetime import timedelta

from .schemas import CaseInput, Decision, SupportedFact


def active_evidence(case: CaseInput):
    ledger = {item.evidence_id: item for item in case.evidence_items}
    superseded = set()
    unauthorized = set()
    for item in case.evidence_items:
        if item.supersedes_id:
            target = ledger[item.supersedes_id]
            if set(target.claims) <= set(item.correction_scope):
                # Historical supersession persists if the replacement is later withdrawn.
                # Withdrawal does not silently resurrect an obsolete record.
                superseded.add(target.evidence_id)
            else:
                unauthorized.add(item.evidence_id)
    return [
        item
        for item in case.evidence_items
        if item.evidence_id not in superseded | unauthorized
        and not item.retracted
        and case.as_of - item.observed_at <= timedelta(hours=case.query_policy.freshness_hours)
    ]


def decide(case: CaseInput) -> Decision:
    if not case.query_policy.in_scope:
        return Decision(
            status="OUT_OF_SCOPE",
            supported_facts=[],
            unresolved=[],
            next_step="NO_ACTION",
            explanation="The question is outside this record's scope.",
        )
    facts, missing, conflicts = [], [], []
    for field in sorted(case.query_policy.required_fields):
        values = defaultdict(list)
        for item in active_evidence(case):
            if field in item.trust_scope and field in item.claims:
                values[item.claims[field]].append(item.evidence_id)
        if not values:
            missing.append(field)
        elif len(values) > 1:
            conflicts.append(field)
        else:
            value, ids = next(iter(values.items()))
            facts.append(SupportedFact(field=field, value=value, evidence_ids=sorted(ids)))
    status = "CONFLICT" if conflicts else "MISSING_EVIDENCE" if missing else "SUPPORTED"
    action = {
        "CONFLICT": "HOLD_FOR_REVIEW",
        "MISSING_EVIDENCE": "REQUEST_EVIDENCE",
        "SUPPORTED": "DRAFT_FOR_HUMAN_REVIEW"
        if case.query_policy.request_kind == "draft"
        else "NO_ACTION",
    }[status]
    return Decision(
        status=status,
        supported_facts=facts,
        unresolved=sorted(missing + conflicts),
        next_step=action,
        explanation={
            "CONFLICT": "Admissible active sources disagree on required facts.",
            "MISSING_EVIDENCE": "Required current authorized support is absent.",
            "SUPPORTED": "The requested facts have consistent current authorized support.",
        }[status],
    )
