"""Strict, versioned case and response contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["SUPPORTED", "MISSING_EVIDENCE", "CONFLICT", "OUT_OF_SCOPE"]
NextStep = Literal["DRAFT_FOR_HUMAN_REVIEW", "REQUEST_EVIDENCE", "HOLD_FOR_REVIEW", "NO_ACTION"]
Split = Literal["unassigned", "train", "validation", "test"]
STATUSES = ("SUPPORTED", "MISSING_EVIDENCE", "CONFLICT", "OUT_OF_SCOPE")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)


class SupportedFact(StrictModel):
    field: str = Field(min_length=1)
    value: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def unique_ids(cls, value):
        if len(value) != len(set(value)) or any(not item for item in value):
            raise ValueError("Fact citations must be nonempty and unique")
        return value


class Decision(StrictModel):
    status: Status
    supported_facts: list[SupportedFact]
    unresolved: list[str]
    next_step: NextStep
    explanation: str = Field(min_length=1)


class EvidenceItem(StrictModel):
    evidence_id: str = Field(pattern=r"^R[0-9]{2,}$")
    source_type: str = Field(min_length=1)
    content: str = Field(min_length=1)
    observed_at: datetime
    asserted_at: datetime
    claims: dict[str, str] = Field(min_length=1)
    trust_scope: list[str]
    correction_scope: list[str] = Field(default_factory=list)
    supersedes_id: str | None = None
    retracted: bool = False

    @field_validator("observed_at", "asserted_at")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamps must be timezone-aware")
        return value

    @field_validator("claims")
    @classmethod
    def nonempty_claims(cls, value):
        if any(not key.strip() or not val.strip() for key, val in value.items()):
            raise ValueError("Claim fields/values cannot be blank")
        return value

    @model_validator(mode="after")
    def chronology(self):
        if self.observed_at > self.asserted_at:
            raise ValueError("Observation cannot follow assertion")
        return self


class QueryPolicy(StrictModel):
    required_fields: list[str] = Field(min_length=1)
    freshness_hours: int = Field(gt=0)
    in_scope: bool = True
    request_kind: Literal["draft", "fact"] = "draft"

    @field_validator("required_fields")
    @classmethod
    def unique_fields(cls, value):
        if len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("Required fields must be nonempty and unique")
        return value


class CaseInput(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    case_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    domain: Literal["solar", "food", "water"]
    language_variant: Literal["en", "hi", "hinglish"]
    question: str = Field(min_length=1)
    evidence_items: list[EvidenceItem]
    as_of: datetime
    query_policy: QueryPolicy
    split: Split = "unassigned"
    generator_version: str
    provenance: dict
    human_review_status: Literal["unreviewed", "reviewed", "rejected"] = "unreviewed"

    @field_validator("as_of")
    @classmethod
    def aware(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_ledger(self):
        ids = [item.evidence_id for item in self.evidence_items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence IDs")
        ledger = {item.evidence_id: item for item in self.evidence_items}
        for item in self.evidence_items:
            if item.asserted_at > self.as_of:
                raise ValueError("Evidence asserted after the query time")
            if item.supersedes_id is not None:
                previous = ledger.get(item.supersedes_id)
                if previous is None or previous.asserted_at >= item.asserted_at:
                    raise ValueError("Revision must reference a strictly earlier existing record")
                if set(previous.claims) != set(item.claims):
                    raise ValueError("Record-level corrections must cover identical claim fields")
        if self.provenance.get("synthetic") is not True:
            raise ValueError("This dataset accepts explicitly synthetic cases only")
        return self


class EvidenceCase(CaseInput):
    oracle_status: Status
    oracle_supported_facts: list[SupportedFact]
    oracle_unresolved: list[str]
    oracle_next_step: NextStep

    @model_validator(mode="after")
    def valid_gold_citations(self):
        validate_citations(self, self.oracle_supported_facts)
        return self


def validate_citations(case: CaseInput, facts: list[SupportedFact]) -> None:
    ids = {item.evidence_id for item in case.evidence_items}
    for fact in facts:
        if not set(fact.evidence_ids) <= ids:
            raise ValueError("Cited evidence ID does not exist in this case")


def parse_prediction(raw: str, case: CaseInput) -> Decision:
    """No fence stripping, repair, status coercion or invented-ID forgiveness."""
    result = Decision.model_validate_json(raw)
    validate_citations(case, result.supported_facts)
    return result
