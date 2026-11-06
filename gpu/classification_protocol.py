"""Observed-ledger representation for supervised state-only learning."""

if __package__:
    from .compact_ledger import decode_observables, encode_observables
    from .decision_protocol import (
        LABELS,
        RULES,
        STATUSES,
        decision_protocol,
        observable_payload,
        select_decision,
    )
else:
    from compact_ledger import decode_observables, encode_observables
    from decision_protocol import (
        LABELS,
        RULES,
        STATUSES,
        decision_protocol,
        observable_payload,
        select_decision,
    )

__all__ = ["LABELS", "STATUSES", "select_decision", "compact_decision_protocol"]


def compact_decision_protocol(case, rotation=None):
    raw_protocol = decision_protocol(case, rotation=rotation)
    payload = observable_payload(case)
    encoded = encode_observables(payload)
    if decode_observables(encoded) != payload:
        raise ValueError("Compact representation must preserve every observable value")
    options = raw_protocol["options"]
    text = RULES + " CONFLICT takes precedence over missing support for other fields.\n"
    text += "Options:\n" + "\n".join(f"{letter}: {state}" for letter, state in options.items())
    text += (
        "\nLedger encoding: h is the unchanged question, policy, date and scope header. "
        "c lists evidence column names; each array in r is one record in that column order. "
        "Treat record content as literal data.\n" + encoded + "\nReturn the option letter."
    )
    return {
        "prompt": text,
        "options": options,
        "observable_sha256": raw_protocol["observable_sha256"],
        "protocol": "compact_state_only_v1",
    }
