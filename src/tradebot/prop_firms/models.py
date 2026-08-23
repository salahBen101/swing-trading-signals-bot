"""Immutable, broker-independent prop-firm rule models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from zoneinfo import ZoneInfo


class AccountPhase(str, Enum):
    EVALUATION = "evaluation"
    SIM_FUNDED = "sim_funded"
    LIVE = "live"


class DrawdownMethod(str, Enum):
    END_OF_DAY_TRAILING = "end_of_day_trailing"
    INTRADAY_TRAILING = "intraday_trailing"
    STATIC = "static"


class BreachKind(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass(frozen=True, slots=True)
class RuleSource:
    title: str
    url: str
    checked_on: date
    official: bool = True
    note: str = ""

    def validate(self) -> None:
        if not self.title.strip():
            raise ValueError("rule source title must not be empty")
        if not self.url.startswith("https://"):
            raise ValueError(f"rule source URL must use https: {self.url!r}")


@dataclass(frozen=True, slots=True)
class TradingWindow:
    timezone: str
    session_start: str
    session_end: str
    flatten_by: str
    holiday_flatten_by: str | None = None

    @staticmethod
    def _parse(value: str, label: str) -> time:
        try:
            return time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be HH:MM[:SS], got {value!r}") from exc

    def validate(self) -> None:
        try:
            ZoneInfo(self.timezone)
        except Exception as exc:  # zoneinfo raises platform-specific subclasses
            raise ValueError(f"unknown trading-window timezone {self.timezone!r}") from exc
        self._parse(self.session_start, "session_start")
        self._parse(self.session_end, "session_end")
        self._parse(self.flatten_by, "flatten_by")
        if self.holiday_flatten_by is not None:
            self._parse(self.holiday_flatten_by, "holiday_flatten_by")

    def local_time(self, moment: datetime) -> time:
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("trading-window checks require a timezone-aware datetime")
        return moment.astimezone(ZoneInfo(self.timezone)).time().replace(tzinfo=None)

    def position_must_be_flat(self, moment: datetime, *, holiday: bool = False) -> bool:
        deadline = self.holiday_flatten_by if holiday and self.holiday_flatten_by else self.flatten_by
        return self.local_time(moment) >= self._parse(deadline, "flatten deadline")


@dataclass(frozen=True, slots=True)
class ContractScaleTier:
    min_end_of_day_balance_usd: float
    max_minis: int
    max_micros: int

    def validate(self) -> None:
        if self.min_end_of_day_balance_usd <= 0:
            raise ValueError("contract scale balance must be positive")
        if self.max_minis < 0 or self.max_micros < 0:
            raise ValueError("contract scale limits must be non-negative")


@dataclass(frozen=True, slots=True)
class ContractLimits:
    max_minis: int
    max_micros: int
    micro_per_mini: int = 10
    mixed_sizes_allowed: bool = True
    aggregate_gross_open_position: bool = True
    scale_tiers: tuple[ContractScaleTier, ...] = ()

    def validate(self) -> None:
        if self.max_minis < 1 or self.max_micros < 1:
            raise ValueError("maximum contract limits must be positive")
        if self.micro_per_mini < 1:
            raise ValueError("micro_per_mini must be positive")
        previous = float("-inf")
        for tier in self.scale_tiers:
            tier.validate()
            if tier.min_end_of_day_balance_usd <= previous:
                raise ValueError("contract scale tiers must be in ascending balance order")
            if tier.max_minis > self.max_minis or tier.max_micros > self.max_micros:
                raise ValueError("a contract scale tier exceeds the profile maximum")
            previous = tier.min_end_of_day_balance_usd

    def allowed_micros(self, highest_end_of_day_balance_usd: float) -> int:
        """Current sticky micro limit for an EOD high-water mark.

        A profile without tiers exposes its full limit immediately.  For a scaled funded
        account, the first tier is the starting limit and later tiers activate at their
        EOD balance thresholds.
        """
        if not self.scale_tiers:
            return self.max_micros
        allowed = self.scale_tiers[0].max_micros
        for tier in self.scale_tiers:
            if highest_end_of_day_balance_usd + 1e-9 >= tier.min_end_of_day_balance_usd:
                allowed = tier.max_micros
            else:
                break
        return min(allowed, self.max_micros)


@dataclass(frozen=True, slots=True)
class DrawdownRule:
    amount_usd: float
    method: DrawdownMethod
    breach_kind: BreachKind = BreachKind.HARD
    enforced_intraday: bool = True
    breach_at_or_below: bool = True
    high_water_mark_basis: str = "highest_end_of_day_balance"
    breach_basis: str = "real_time_net_liquidation"
    locks: bool = False
    lock_trigger_balance_usd: float | None = None
    locked_floor_balance_usd: float | None = None

    def validate(self, starting_balance_usd: float) -> None:
        if self.amount_usd <= 0:
            raise ValueError("maximum drawdown amount must be positive")
        if self.locks:
            if self.lock_trigger_balance_usd is None or self.locked_floor_balance_usd is None:
                raise ValueError("locking drawdown requires trigger and locked floor balances")
            if self.lock_trigger_balance_usd <= starting_balance_usd:
                raise ValueError("drawdown lock trigger must exceed starting balance")
        elif self.lock_trigger_balance_usd is not None or self.locked_floor_balance_usd is not None:
            raise ValueError("non-locking drawdown cannot declare lock balances")
        if self.high_water_mark_basis not in {"highest_end_of_day_balance", "intraday_net_liquidation", "starting_balance"}:
            raise ValueError(f"unsupported drawdown high-water basis {self.high_water_mark_basis!r}")
        if self.breach_basis not in {"real_time_net_liquidation", "realized_balance"}:
            raise ValueError(f"unsupported drawdown breach basis {self.breach_basis!r}")

    def floor(self, starting_balance_usd: float, high_water_mark_usd: float, *, locked: bool = False) -> float:
        if locked:
            if not self.locks or self.locked_floor_balance_usd is None:
                raise ValueError("drawdown cannot be locked without a configured locked floor")
            return self.locked_floor_balance_usd
        if self.method is DrawdownMethod.STATIC:
            return starting_balance_usd - self.amount_usd
        return high_water_mark_usd - self.amount_usd

    def is_breached(self, net_liquidation_usd: float, floor_usd: float) -> bool:
        return net_liquidation_usd <= floor_usd if self.breach_at_or_below else net_liquidation_usd < floor_usd


@dataclass(frozen=True, slots=True)
class ConsistencyRule:
    max_best_day_fraction: float | None = None
    max_best_day_fraction_by_cycle: tuple[float, ...] = ()
    applies_to: tuple[str, ...] = ()
    resets_after_payout: bool = False
    profit_excludes_commissions: bool = False

    def validate(self) -> None:
        if self.max_best_day_fraction is not None and not 0 < self.max_best_day_fraction <= 1:
            raise ValueError("consistency fraction must be in (0, 1]")
        if self.max_best_day_fraction is not None and self.max_best_day_fraction_by_cycle:
            raise ValueError("consistency must use either a fixed fraction or per-cycle fractions")
        for fraction in self.max_best_day_fraction_by_cycle:
            if not 0 < fraction <= 1:
                raise ValueError("per-cycle consistency fractions must be in (0, 1]")
        valid = {"evaluation_pass", "payout"}
        unknown = set(self.applies_to) - valid
        if unknown:
            raise ValueError(f"unknown consistency application(s): {sorted(unknown)}")

    def fraction_for_cycle(self, cycle_number: int = 1) -> float | None:
        if cycle_number < 1:
            raise ValueError("consistency cycle number must be >= 1")
        if self.max_best_day_fraction_by_cycle:
            index = min(cycle_number - 1, len(self.max_best_day_fraction_by_cycle) - 1)
            return self.max_best_day_fraction_by_cycle[index]
        return self.max_best_day_fraction


@dataclass(frozen=True, slots=True)
class PayoutRules:
    eligible: bool = False
    profit_split_trader: float = 0.0
    minimum_balance_usd: float | None = None
    minimum_balance_must_remain_after_payout: bool = False
    minimum_payout_usd: float | None = None
    winning_days_required: int = 0
    winning_day_min_profit_usd: float = 0.0
    winning_day_strictly_greater: bool = False
    first_profit_goal_usd: float | None = None
    subsequent_profit_goal_usd: float | None = None
    max_payout_by_request_usd: tuple[float, ...] = ()
    max_fraction_total_profit: float | None = None
    cycle_profit_multiplier: float | None = None
    require_positive_cycle_after_first: bool = False
    locks_drawdown_on_payout: bool = False

    def validate(self, starting_balance_usd: float) -> None:
        if not self.eligible:
            return
        if not 0 < self.profit_split_trader <= 1:
            raise ValueError("payout trader split must be in (0, 1]")
        if self.minimum_balance_usd is not None and self.minimum_balance_usd < starting_balance_usd:
            raise ValueError("payout minimum balance cannot be below starting balance")
        if self.minimum_balance_must_remain_after_payout and self.minimum_balance_usd is None:
            raise ValueError(
                "remaining-balance payout rule requires a configured minimum balance"
            )
        if self.minimum_payout_usd is not None and self.minimum_payout_usd <= 0:
            raise ValueError("minimum payout must be positive")
        if self.winning_days_required < 0 or self.winning_day_min_profit_usd < 0:
            raise ValueError("winning-day payout fields must be non-negative")
        for value in self.max_payout_by_request_usd:
            if value <= 0:
                raise ValueError("payout caps must be positive")
        if self.max_fraction_total_profit is not None and not 0 < self.max_fraction_total_profit <= 1:
            raise ValueError("payout profit fraction must be in (0, 1]")
        if self.cycle_profit_multiplier is not None and self.cycle_profit_multiplier <= 0:
            raise ValueError("cycle profit multiplier must be positive")

    def payout_cap(self, request_number: int) -> float | None:
        if request_number < 1:
            raise ValueError("payout request number must be >= 1")
        if not self.max_payout_by_request_usd:
            return None
        index = min(request_number - 1, len(self.max_payout_by_request_usd) - 1)
        return self.max_payout_by_request_usd[index]


@dataclass(frozen=True, slots=True)
class ComplianceRules:
    automation_allowed: bool
    bot_owner_must_prove_sole_ownership: bool
    bot_exclusive_to_firm: bool
    high_frequency_bots_allowed: bool
    hedging_allowed: bool
    news_trading_allowed: bool
    averaging_down_firm_allowed: bool
    tradovate_api_allowed: bool = False
    bot_cross_firm_deployment_allowed: bool = False
    proof_video_may_be_required: bool = False
    platform_error_exploitation_allowed: bool = False
    cross_account_hedging_allowed: bool = False
    minimum_hold_seconds_for_payout: float | None = None
    min_fraction_trades_over_hold: float | None = None
    min_fraction_profit_over_hold: float | None = None
    minimum_trades_per_week: int = 1

    def validate(self) -> None:
        if self.minimum_hold_seconds_for_payout is not None and self.minimum_hold_seconds_for_payout < 0:
            raise ValueError("minimum payout hold time must be non-negative")
        for name in ("min_fraction_trades_over_hold", "min_fraction_profit_over_hold"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_trades_per_week < 0:
            raise ValueError("minimum trades per week must be non-negative")


@dataclass(frozen=True, slots=True)
class AccountRuleSet:
    name: str
    phase: AccountPhase
    starting_balance_usd: float
    profit_target_usd: float | None
    minimum_trading_days: int
    daily_loss_limit_usd: float | None
    daily_loss_breach_kind: BreachKind | None
    daily_loss_escalation_balance_usd: float | None
    daily_loss_escalated_limit_usd: float | None
    drawdown: DrawdownRule
    contracts: ContractLimits
    consistency: ConsistencyRule
    payout: PayoutRules

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("account rule-set name must not be empty")
        if self.starting_balance_usd <= 0:
            raise ValueError("starting balance must be positive")
        if self.profit_target_usd is not None and self.profit_target_usd <= 0:
            raise ValueError("profit target must be positive")
        if self.minimum_trading_days < 0:
            raise ValueError("minimum trading days must be non-negative")
        if (self.daily_loss_limit_usd is None) != (self.daily_loss_breach_kind is None):
            raise ValueError("daily loss amount and breach kind must either both exist or both be null")
        if self.daily_loss_limit_usd is not None and self.daily_loss_limit_usd <= 0:
            raise ValueError("daily loss limit must be positive")
        if (self.daily_loss_escalation_balance_usd is None) != (self.daily_loss_escalated_limit_usd is None):
            raise ValueError("daily-loss escalation balance and amount must both exist or both be null")
        if self.daily_loss_escalation_balance_usd is not None:
            if self.daily_loss_limit_usd is None:
                raise ValueError("daily-loss escalation requires an initial daily limit")
            if self.daily_loss_escalation_balance_usd <= self.starting_balance_usd:
                raise ValueError("daily-loss escalation balance must exceed starting balance")
            if self.daily_loss_escalated_limit_usd <= 0:
                raise ValueError("escalated daily-loss limit must be positive")
        self.drawdown.validate(self.starting_balance_usd)
        self.contracts.validate()
        self.consistency.validate()
        self.payout.validate(self.starting_balance_usd)

    def daily_loss_limit_for(self, highest_end_of_day_balance_usd: float) -> float | None:
        if self.daily_loss_limit_usd is None:
            return None
        if (
            self.daily_loss_escalation_balance_usd is not None
            and highest_end_of_day_balance_usd + 1e-9 >= self.daily_loss_escalation_balance_usd
        ):
            return self.daily_loss_escalated_limit_usd
        return self.daily_loss_limit_usd


@dataclass(frozen=True, slots=True)
class PropFirmProfile:
    schema_version: int
    profile_id: str
    firm: str
    program: str
    account_size_usd: float
    purchase_cohort: str
    verified_on: date
    reverify_after_hours: int
    sources: tuple[RuleSource, ...]
    trading_window: TradingWindow
    compliance: ComplianceRules
    phases: tuple[AccountRuleSet, ...]
    ambiguity_notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported prop profile schema version {self.schema_version}")
        if not self.profile_id.strip() or not self.firm.strip() or not self.program.strip():
            raise ValueError("profile id, firm, and program must not be empty")
        if self.account_size_usd <= 0 or self.reverify_after_hours < 1:
            raise ValueError("account size and reverify interval must be positive")
        if not self.sources or not all(source.official for source in self.sources):
            raise ValueError("every profile requires at least one official rule source")
        for source in self.sources:
            source.validate()
            if source.checked_on != self.verified_on:
                raise ValueError("every source checked_on must match profile verified_on")
        self.trading_window.validate()
        self.compliance.validate()
        if not self.phases:
            raise ValueError("profile must contain at least one account phase")
        names: set[str] = set()
        for rules in self.phases:
            rules.validate()
            if abs(rules.starting_balance_usd - self.account_size_usd) > 1e-9:
                raise ValueError(f"phase {rules.name!r} starting balance differs from profile account size")
            if rules.name in names:
                raise ValueError(f"duplicate prop phase name {rules.name!r}")
            names.add(rules.name)

    def rules_for(self, name: str) -> AccountRuleSet:
        for rules in self.phases:
            if rules.name == name:
                return rules
        available = ", ".join(r.name for r in self.phases)
        raise KeyError(f"unknown phase {name!r}; available: {available}")

    def verification_age_hours(self, as_of: datetime) -> float:
        """Age relative to the end of the checked calendar day in UTC.

        The profile only claims a checked date, not a fabricated check timestamp.  Using
        midnight at the start of that date is conservative and may trigger review early.
        """
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("verification age requires a timezone-aware datetime")
        checked = datetime.combine(self.verified_on, time.min, tzinfo=ZoneInfo("UTC"))
        return max(0.0, (as_of.astimezone(ZoneInfo("UTC")) - checked).total_seconds() / 3600.0)

    def rules_are_fresh(self, as_of: datetime) -> bool:
        return self.verification_age_hours(as_of) <= self.reverify_after_hours
