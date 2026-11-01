"""Original synthetic operational cases; no historical project data is used."""

from datetime import UTC, datetime, timedelta

from .oracle import decide
from .schemas import CaseInput, EvidenceCase, EvidenceItem, QueryPolicy

VERSION = "0.1.0"
VARIANTS = (
    "missing",
    "supported",
    "conflict",
    "corrected",
    "withdrawn",
    "unauthorized",
    "stale",
    "out_of_scope",
    "irrelevant_missing",
    "report_only",
    "correction_withdrawn",
)
TRANSITIONS = (
    ("missing", "supported"),
    ("supported", "conflict"),
    ("conflict", "corrected"),
    ("supported", "withdrawn"),
    ("corrected", "correction_withdrawn"),
)
# Three question lineages are held as units by the splitter. They are still
# authored by this generator, not an independently authored challenge benchmark.
QUESTIONS = (
    "Using only the records, draft the evidence state for {field} at {site}.",
    "What current conclusion about {field} can these sources justify for {site}? Prepare a review draft.",
    "For {site}, assess whether the documented {field} is supported, missing, conflicting or outside scope.",
)
DOMAINS = {"solar": "post_repair_check", "food": "lot_identity", "water": "post_service_reading"}


def generate(families_per_lineage: int = 3) -> list[EvidenceCase]:
    if not 1 <= families_per_lineage <= 1000:
        raise ValueError("families_per_lineage must be between 1 and 1000")
    cases = []
    for domain, field in DOMAINS.items():
        for lineage, question_template in enumerate(QUESTIONS):
            for index in range(families_per_lineage):
                family = f"{domain}-L{lineage}-{index:04d}"
                now = datetime(2026, 11, 1, 10, tzinfo=UTC) + timedelta(days=index)
                site = f"SYN-{domain.upper()}-{lineage}-{index:04d}"
                good = f"verified-{domain}-{index:04d}"
                bad = f"different-{domain}-{index:04d}"

                def record(
                    eid,
                    claims,
                    scopes,
                    *,
                    prior=None,
                    correction=False,
                    retracted=False,
                    age=2,
                    _site=site,
                    _now=now,
                    _field=field,
                    _lineage=lineage,
                ):
                    body = "; ".join(f"{key} = {value}" for key, value in claims.items())
                    content = (
                        f"Synthetic record for {_site}: {body}",
                        f"Site {_site}. Documented assertion: {body}.",
                        f"Evidence entry | site {_site} | {body}",
                    )[_lineage]
                    return EvidenceItem(
                        evidence_id=eid,
                        source_type="authorized_correction" if correction else "source_record",
                        content=content,
                        observed_at=_now - timedelta(hours=age),
                        asserted_at=_now - timedelta(hours=1, minutes=10 - int(eid[1:])),
                        claims=claims,
                        trust_scope=scopes,
                        correction_scope=[_field] if correction else [],
                        supersedes_id=prior,
                        retracted=retracted,
                    )

                report = record(
                    "R01", {"reported_completion": "reported complete"}, ["reported_completion"]
                )
                measurement = record("R02", {field: good}, [field])
                opposite = record("R03", {field: bad}, [field])
                correction = record("R04", {field: good}, [field], prior="R03", correction=True)
                for variant in VARIANTS:
                    records = [report]
                    required = [field]
                    question = question_template.format(field=field, site=site)
                    in_scope = variant != "out_of_scope"
                    if variant != "missing" and variant not in [
                        "report_only",
                        "irrelevant_missing",
                    ]:
                        records.append(measurement)
                    if variant in ["conflict", "corrected", "correction_withdrawn"]:
                        records.append(opposite)
                    if variant in ["corrected", "correction_withdrawn"]:
                        records.append(
                            correction.model_copy(
                                update={"retracted": variant == "correction_withdrawn"}
                            )
                        )
                    if variant == "withdrawn":
                        records[-1] = measurement.model_copy(update={"retracted": True})
                    if variant == "stale":
                        records[-1] = measurement.model_copy(
                            update={"observed_at": now - timedelta(hours=72)}
                        )
                    if variant == "unauthorized":
                        records.append(record("R03", {field: bad}, [], prior="R02"))
                    if variant in ["report_only", "irrelevant_missing"]:
                        required = ["reported_completion"]
                        question = (
                            (
                                f"For {site}, what completion was reported? Do not infer verification.",
                                f"Summarize only the submitted completion statement at {site}, not its verification.",
                                f"Which completion claim appears in the {site} ledger? Distinguish reporting from proof.",
                            )[lineage]
                            if variant == "report_only"
                            else (
                                f"For {site}, is a completion statement documented? A measurement is irrelevant to this question.",
                                f"Does the {site} paperwork include a completion assertion? This asks about paperwork only.",
                                f"Find the claimed completion in {site}'s notes; do not require a physical verification reading.",
                            )[lineage]
                        )
                    if variant == "out_of_scope":
                        question = (
                            f"Can these {site} records establish the status of an unrelated site?",
                            f"Determine the condition of a different, unlisted site using only the {site} file.",
                            f"Use {site}'s ledger to certify a location outside its stated coverage.",
                        )[lineage]
                    draft = CaseInput(
                        case_id=f"{family}-{variant}",
                        family_id=family,
                        template_id=f"lineage-{lineage}",
                        domain=domain,
                        language_variant="en",
                        question=question,
                        evidence_items=records,
                        as_of=now,
                        query_policy=QueryPolicy(
                            required_fields=required, freshness_hours=24, in_scope=in_scope
                        ),
                        generator_version=VERSION,
                        provenance={
                            "synthetic": True,
                            "variant": variant,
                            "template_lineage": f"lineage-{lineage}",
                            "translation_group": family,
                            "authored_date": "2026-11-01",
                            "site": site,
                        },
                    )
                    result = decide(draft)
                    payload = draft.model_dump()
                    payload.update(
                        oracle_status=result.status,
                        oracle_supported_facts=result.supported_facts,
                        oracle_unresolved=result.unresolved,
                        oracle_next_step=result.next_step,
                    )
                    cases.append(EvidenceCase.model_validate(payload))
    return cases
