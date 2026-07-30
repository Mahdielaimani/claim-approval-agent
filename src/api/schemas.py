"""Request and response contracts for the API boundary."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class Persona(str, Enum):
    """Audience an explanation is written for."""

    CUSTOMER = "customer"
    ADJUSTER = "adjuster"
    AUDITOR = "auditor"


class ClaimType(str, Enum):
    """Loss types present in the training data."""

    ACCIDENTAL_DAMAGE = "Accidental Damage"
    THEFT = "Theft"
    LIQUID_DAMAGE = "Liquid Damage"


class Country(str, Enum):
    """Markets the model was trained on."""

    NL = "NL"
    SE = "SE"
    FI = "FI"


class Coverage(str, Enum):
    """Policy coverage levels."""

    ADLD = "ADLD"
    ADLD_THEFT = "ADLD/THEFT"


class PolicyStatus(str, Enum):
    """Policy states at the time of the claim."""

    ACTIVE = "Active"
    GRACE_PERIOD = "Grace Period"
    INACTIVE = "InActive"


class Channel(str, Enum):
    """Intake channels. Diagnostics are missing not-at-random by channel."""

    ONLINE_PORTAL = "Online Portal"
    EMAIL = "Email"
    PHONE_CALL = "Phone Call"
    WHATSAPP = "Whatsapp"
    FACEBOOK = "Facebook"


# Absent means the check was never run, which the preprocessor encodes as -1, not 0.
Diagnostic = Annotated[float | None, Field(ge=0, le=1)]


class ClaimRequest(BaseModel):
    """One claim, in the shape the source system produces."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "claim_id": "CLM-000123",
                "rrp": 1449.0,
                "excessFee": 59.0,
                "balanceRRP": 1391.04,
                "coverage": "ADLD",
                "productName": "NL_ADLD_12M_MONTHLY_SMARTPHONE",
                "productCode": "NLADLD12M",
                "policyStartDate": "2025-03-01 00:00:00",
                "policyEndDate": "2026-03-01 00:00:00",
                "purchaseDate": "2025-03-01 00:00:00",
                "policyStatus": "Active",
                "retailerName": "WUAWEI eStore",
                "deviceType": "SMARTPHONES",
                "model": "WUAWEI-P40",
                "channel": "Online Portal",
                "claimType": "Accidental Damage",
                "country": "NL",
                "turnOnOff": 1.0,
                "touchScreen": 0.0,
                "frontCamera": 1.0,
                "backCamera": 1.0,
                "audio": 1.0,
                "mic": 1.0,
                "buttons": 0.0,
                "connection": 0.0,
                "charging": 0.0,
                "issueDesc": "Het toestel viel uit mijn hand en het scherm is gebarsten.",
            }
        },
    )

    claim_id: str | None = Field(
        default=None, description="Caller's reference. Generated when absent."
    )

    # In the currency the policy was written in; SEK to EUR conversion is preprocessing.
    rrp: float = Field(gt=0, description="Recommended retail price of the device.")
    balanceRRP: float = Field(ge=0, description="Outstanding balance on the device.")
    excessFee: float | None = Field(default=None, ge=0, description="Policy excess.")
    oldBalanceRRP: float | None = Field(default=None, ge=0)

    coverage: Coverage
    productName: str = Field(
        min_length=1,
        description="Composite product code. Parsed for device type, contract length, "
        "payment mode and policy class — three of which exist in no other field.",
    )
    productDesc: str | None = None
    productCode: str | None = None

    # Strings: the source emits two date formats, and the preprocessor owns that parsing.
    policyStartDate: str
    policyEndDate: str
    purchaseDate: str

    policyStatus: PolicyStatus
    retailerName: str | None = None
    deviceType: str | None = Field(
        default=None, description="Recovered from productName when absent."
    )
    make: str | None = None
    model: str | None = None
    deviceCost: float | None = None
    relationship: str | None = None

    channel: Channel
    claimType: ClaimType
    country: Country

    turnOnOff: Diagnostic = None
    touchScreen: Diagnostic = None
    frontCamera: Diagnostic = None
    backCamera: Diagnostic = None
    audio: Diagnostic = None
    mic: Diagnostic = None
    buttons: Diagnostic = None
    connection: Diagnostic = None
    charging: Diagnostic = None
    smashed: Diagnostic = None
    frontOrBackCamera: Diagnostic = None

    issueDesc: str = Field(
        min_length=1,
        description="Customer's account of the loss. Never a model feature — it is "
        "multilingual, PII-masked prose routed to the explanation layer. Treated as "
        "untrusted input throughout.",
    )
    other: str | None = None

    def to_claim(self) -> dict[str, Any]:
        """Flatten to the dict the pipeline expects."""
        # exclude_none stays off: the preprocessor needs every column to exist.
        return self.model_dump(exclude_none=False, exclude=self._non_claim_fields())

    @staticmethod
    def _non_claim_fields() -> set[str]:
        """Request metadata that is not part of the claim record."""
        return set()


class ExplainRequest(ClaimRequest):
    """A claim plus the audience the explanation is for."""

    persona: Persona = Field(
        default=Persona.CUSTOMER,
        description="Determines which fields reach the prompt and the response.",
    )

    @staticmethod
    def _non_claim_fields() -> set[str]:
        """Routing metadata, excluded so it never reaches the feature matrix."""
        return {"persona"}


class PredictionResponse(BaseModel):
    """Routing decision for one claim. No LLM involved on this path."""

    claim_id: str
    request_id: str
    decline_risk_score: float = Field(description="P(Declined). Not a decision.")
    decision: str
    requires_human_review: bool
    human_review_reason: str | None
    risk_flags: list[str] = Field(description="Conditions forcing review regardless of the score.")
    decision_threshold: float
    model_version: str
    model_algorithm: str
    latency_ms: int


class GenerationMeta(BaseModel):
    """How the explanation text was produced."""

    provider: str
    model: str
    prompt_version: str
    used_fallback_provider: bool
    used_template_fallback: bool
    retried: bool
    llm_latency_ms: int


class CustomerExplanationResponse(BaseModel):
    """Customer view. Carries no score, threshold, attribution or model identity."""

    # A client renders what it receives, so the score here would undo the prompt filtering.
    claim_id: str
    request_id: str
    persona: Persona
    status: str
    under_review: bool
    explanation: dict[str, Any]
    factors: list[dict[str, Any]]
    total_latency_ms: int


class TechnicalExplanationResponse(PredictionResponse):
    """Adjuster and auditor view. Full attribution, provenance and validation record."""

    persona: Persona
    explanation: dict[str, Any]
    factors: list[dict[str, Any]]
    generation: GenerationMeta
    validation: dict[str, Any]
    total_latency_ms: int


class HealthResponse(BaseModel):
    """What is loaded and serving."""

    status: str
    model_version: str
    model_algorithm: str
    decision_threshold: float
    n_features: int
    prompt_version: str
    llm_mode: str
    llm_chain: list[str]
    trained_at: str
    git_commit: str


class ErrorResponse(BaseModel):
    """Error envelope carrying the request id for correlation with the logs."""

    detail: str
    request_id: str | None = None
