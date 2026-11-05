"""Evidence-linked training explanations; never used to supply inference answers."""

import json
from collections import defaultdict

from .oracle import active_evidence, decide
from .schemas import parse_prediction


def explanation_first_target(case):
    gold = decide(case)
    observations = []
    if gold.status == "OUT_OF_SCOPE":
        observations.append("The request concerns a different site from the authorized record.")
    else:
        for field in sorted(case.query_policy.required_fields):
            values = defaultdict(list)
            for record in active_evidence(case):
                if field in record.trust_scope and field in record.claims:
                    values[record.claims[field]].append(record.evidence_id)
            if not values:
                observations.append(f"{field}: no current authorized supporting record.")
            elif len(values) > 1:
                witnesses = "; ".join(
                    f"{','.join(sorted(ids))}={value}" for value, ids in sorted(values.items())
                )
                observations.append(f"{field}: conflicting active authorized values ({witnesses}).")
            else:
                value, ids = next(iter(values.items()))
                observations.append(
                    f"{field}: {','.join(sorted(ids))} consistently supports {value}."
                )
    payload = {"explanation": " ".join(observations), **gold.model_dump(exclude={"explanation"})}
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    actual = parse_prediction(text, case)
    if actual.model_dump(exclude={"explanation"}) != gold.model_dump(exclude={"explanation"}):
        raise ValueError("Target redesign changed a decision or its evidence citations")
    return text
