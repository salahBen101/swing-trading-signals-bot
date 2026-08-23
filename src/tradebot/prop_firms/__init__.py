"""Versioned prop-firm rule profiles.

Firm rules are data, never strategy logic.  The public surface is intentionally small so
the risk engine and simulator consume the same immutable objects.
"""

from .loader import PropProfileError, load_prop_profile
from .models import (
    AccountPhase,
    AccountRuleSet,
    BreachKind,
    ComplianceRules,
    ConsistencyRule,
    ContractLimits,
    ContractScaleTier,
    DrawdownMethod,
    DrawdownRule,
    PayoutRules,
    PropFirmProfile,
    RuleSource,
    TradingWindow,
)
from .verification import (
    RuleVerificationResult,
    SourceBaseline,
    SourceCheck,
    VerificationStatus,
    append_verification_record,
    content_sha256,
    fetch_official_page,
    load_source_baselines,
    verify_profile_sources,
    write_change_proposal,
)

__all__ = [
    "AccountPhase",
    "AccountRuleSet",
    "BreachKind",
    "ComplianceRules",
    "ConsistencyRule",
    "ContractLimits",
    "ContractScaleTier",
    "DrawdownMethod",
    "DrawdownRule",
    "PayoutRules",
    "PropFirmProfile",
    "PropProfileError",
    "RuleSource",
    "TradingWindow",
    "load_prop_profile",
    "RuleVerificationResult",
    "SourceBaseline",
    "SourceCheck",
    "VerificationStatus",
    "append_verification_record",
    "content_sha256",
    "fetch_official_page",
    "load_source_baselines",
    "verify_profile_sources",
    "write_change_proposal",
]
