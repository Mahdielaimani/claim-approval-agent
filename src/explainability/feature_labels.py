"""Human-readable names for the model's features, per audience."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureLabel:
    """How one feature is named for a technical reader and for a customer."""

    technical: str
    label: str
    plain: str
    unit: str = ""


# Customer wording is a boundary, not a style: "productCode_freq" tells them nothing.
_LABELS: tuple[FeatureLabel, ...] = (
    FeatureLabel("rrp", "Device value (RRP, EUR)", "the value of your device", "EUR"),
    FeatureLabel("excessFee", "Excess fee (EUR)", "the excess on your policy", "EUR"),
    FeatureLabel(
        "balanceRRP",
        "Outstanding device balance (EUR)",
        "the amount still owed on your device",
        "EUR",
    ),
    FeatureLabel(
        "fee_to_rrp_ratio",
        "Excess as a share of device value",
        "how large your excess is compared with the device's value",
    ),
    FeatureLabel(
        "balance_ratio",
        "Outstanding balance as a share of device value",
        "how much of your device's value is still outstanding",
    ),
    FeatureLabel(
        "remaining_balance_ratio",
        "Share of device value already paid",
        "how much of your device you have already paid off",
    ),
    FeatureLabel("policyStatus", "Policy status", "the status of your policy"),
    FeatureLabel("coverage", "Coverage type", "the cover included in your policy"),
    FeatureLabel(
        "policy_class",
        "Policy class (mandatory / voluntary / discount)",
        "the type of policy you hold",
    ),
    FeatureLabel("is_mandatory_policy", "Mandatory policy", "the type of policy you hold"),
    FeatureLabel("payment_mode", "Payment mode (monthly / upfront)", "how your policy is paid"),
    FeatureLabel(
        "contract_duration_months", "Contract length", "the length of your cover", "months"
    ),
    FeatureLabel("policy_duration_days", "Policy duration", "how long your policy runs", "days"),
    FeatureLabel(
        "days_purchase_to_policy",
        "Days between device purchase and policy start",
        "how long after buying the device the cover started",
        "days",
    ),
    FeatureLabel("policy_start_month", "Policy start month", "when your cover began"),
    FeatureLabel("claimType", "Claim type", "the type of damage reported"),
    FeatureLabel("channel", "Submission channel", "how the claim was submitted"),
    FeatureLabel("country", "Country", "the country your policy is held in"),
    FeatureLabel("deviceType", "Device category", "the kind of device covered"),
    FeatureLabel("turnOnOff", "Power-on diagnostic", "whether the device powers on"),
    FeatureLabel("touchScreen", "Touchscreen diagnostic", "the condition of the screen"),
    FeatureLabel("frontCamera", "Front camera diagnostic", "the condition of the front camera"),
    FeatureLabel("backCamera", "Rear camera diagnostic", "the condition of the rear camera"),
    FeatureLabel("audio", "Audio diagnostic", "whether sound works"),
    FeatureLabel("mic", "Microphone diagnostic", "whether the microphone works"),
    FeatureLabel("buttons", "Buttons diagnostic", "the condition of the buttons"),
    FeatureLabel("connection", "Connectivity diagnostic", "whether the device connects"),
    FeatureLabel("charging", "Charging diagnostic", "whether the device charges"),
    FeatureLabel(
        "diag_completeness",
        "Number of diagnostic checks performed",
        "how many checks were carried out on your device",
    ),
    FeatureLabel(
        "diag_ones", "Diagnostic checks returning 1", "the results of the checks on your device"
    ),
    FeatureLabel(
        "diag_zeros", "Diagnostic checks returning 0", "the results of the checks on your device"
    ),
    FeatureLabel(
        "issue_desc_length",
        "Length of the reported description",
        "the description you provided",
        "characters",
    ),
    FeatureLabel(
        "issue_desc_word_count",
        "Word count of the reported description",
        "the description you provided",
        "words",
    ),
    FeatureLabel("has_other_notes", "Additional notes provided", "the extra notes you provided"),
    FeatureLabel(
        "model_freq", "Device model frequency in the portfolio", "how common your device model is"
    ),
    FeatureLabel(
        "productCode_freq",
        "Product code frequency in the portfolio",
        "how common your policy type is",
    ),
    FeatureLabel(
        "retailerName_freq", "Retailer frequency in the portfolio", "where the device was purchased"
    ),
)

FEATURE_LABELS: dict[str, FeatureLabel] = {f.technical: f for f in _LABELS}


def label_for(technical: str, *, plain: bool = False) -> str:
    """Readable name for a feature; plain=True gives the customer-facing wording."""
    entry = FEATURE_LABELS.get(technical)
    if entry is None:
        # A newly added feature must not take the explanation endpoint down.
        return technical.replace("_", " ")
    return entry.plain if plain else entry.label


def unit_for(technical: str) -> str:
    """Unit of a feature, or an empty string when it has none."""
    entry = FEATURE_LABELS.get(technical)
    return entry.unit if entry else ""


def unlabelled(feature_names: list[str]) -> list[str]:
    """Features with no label, so a test can fail when one is added and not described."""
    return [f for f in feature_names if f not in FEATURE_LABELS]
