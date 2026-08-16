"""Tradeable universe for the RSI(2) scanner.

Chosen on ex-ante criteria, deliberately NOT on backtest results:

  * Mega/large-cap US names with deep, continuously quoted options chains.
  * Broad index and sector ETFs, which are the cleanest expression of the strategy - index
    mean reversion is where the effect is best established and least contaminated by
    single-name news.
  * Long enough listed history to validate against.

Picking constituents by how well they backtested would recreate exactly the selection bias
that inflates the Mag 7 numbers: those seven look excellent partly because they are the seven
that went up. This list is defined by liquidity and then validated as a whole, winners and
losers together.

Anything here still has to pass the strategy's own 200-day trend filter each day, which is
what keeps it out of names that are genuinely breaking down.
"""

from __future__ import annotations

# Broad market and sector ETFs. The strategy's cleanest habitat: diversified, news-resistant,
# tight spreads, and mean reversion driven by flow rather than by company-specific repricing.
ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC",
    "SMH", "XBI", "KRE", "ITB",
]

MEGA_CAP_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "ORCL", "CRM", "AMD", "ADBE", "CSCO", "ACN", "INTU",
    "QCOM", "TXN", "IBM", "NOW", "AMAT", "MU", "LRCX", "ADI", "PANW", "SNPS",
]

FINANCIALS = [
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI",
    "BLK", "SCHW", "C", "PGR", "CB",
]

HEALTHCARE = [
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE",
    "AMGN", "ISRG", "BMY", "GILD", "CVS", "MDT",
]

CONSUMER_INDUSTRIAL = [
    "WMT", "COST", "PG", "HD", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "LMT", "DE", "UNP", "MMM",
]

ENERGY_MATERIALS_UTILITIES = [
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    "LIN", "SHW", "FCX", "NEM",
    "NEE", "DUK", "SO", "D",
]

COMMUNICATION_OTHER = [
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "UBER", "ABNB", "BKNG", "PYPL",
]

MAG7 = MEGA_CAP_TECH[:7]

# A small, pre-specified core for paper-trading alerts. The scanner can still observe all 120
# liquid names, but the broader current-stock universe has survivorship bias and its later
# historical period informed the retrospective labels below.
CORE_ETFS = ["SPY", "QQQ", "IWM", "DIA", "MDY"]

# Commodity-linked names are retained for observation but marked as historically unsuitable
# for this rule. That observation came from the same historical research, so it is a caution,
# not a clean out-of-sample filter.
COMMODITY_LINKED = {
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "XLE",
    "FCX", "NEM", "LIN", "SHW", "XLB",
}

ALL_STOCKS = (
    MEGA_CAP_TECH
    + FINANCIALS
    + HEALTHCARE
    + CONSUMER_INDUSTRIAL
    + ENERGY_MATERIALS_UTILITIES
    + COMMUNICATION_OTHER
)

FULL_UNIVERSE = ETFS + ALL_STOCKS

# --------------------------------------------------------------------------------------
# Retrospective research labels.
#
# These were derived after examining the 2019-2026 history, so that period is no longer an
# independent holdout. They are useful descriptive context for a scan, never proof that a
# name is currently tradeable. Default alerts are restricted to CORE_ETFS for this reason.
# --------------------------------------------------------------------------------------

TIER_RETROSPECTIVE = "retrospective"
TIER_UNCONFIRMED = "unconfirmed"
TIER_NO_EDGE = "no edge"    # no measurable edge

# Backward-compatible names for existing callers and the legacy --tickers strong preset.
TIER_STRONG = TIER_RETROSPECTIVE
TIER_WEAK = TIER_UNCONFIRMED

_TIER_MAP: dict[str, str] = {}
for _t in set(MEGA_CAP_TECH) | set(ETFS) | set(FINANCIALS) | {"NFLX", "PYPL", "UBER", "ABNB", "BKNG"}:
    _TIER_MAP[_t] = TIER_STRONG
for _t in HEALTHCARE + CONSUMER_INDUSTRIAL + ["DIS", "CMCSA", "T", "VZ", "TMUS"]:
    _TIER_MAP.setdefault(_t, TIER_WEAK)
for _t in ENERGY_MATERIALS_UTILITIES:
    _TIER_MAP[_t] = TIER_NO_EDGE
for _t in COMMODITY_LINKED:
    _TIER_MAP[_t] = TIER_NO_EDGE


def tier_of(ticker: str) -> str:
    return _TIER_MAP.get(ticker, TIER_WEAK)


def is_core(ticker: str) -> bool:
    """Whether a symbol belongs to the pre-specified diversified ETF paper-trading core."""
    return ticker.upper() in CORE_ETFS


# --------------------------------------------------------------------------------------
# Sector labels. Worth showing next to every reading because the evidence tiers are largely
# sector-driven - seeing "energy" explains a NO EDGE tag without having to look it up, and
# several signals firing in one sector is different information from several across the board.
# Sector ETFs get their own label rather than a generic "ETF": XLF behaves like financials,
# and lumping it with SPY would hide that.
# --------------------------------------------------------------------------------------
_SECTOR_MAP: dict[str, str] = {}

_SECTOR_GROUPS = {
    "tech": MEGA_CAP_TECH,
    "financials": FINANCIALS,
    "healthcare": HEALTHCARE,
    "consumer/indl": CONSUMER_INDUSTRIAL,
    "energy/matls": ENERGY_MATERIALS_UTILITIES,
    "comms/other": COMMUNICATION_OTHER,
}
for _sector, _members in _SECTOR_GROUPS.items():
    for _t in _members:
        _SECTOR_MAP.setdefault(_t, _sector)

# ETFs, labelled by what they actually track.
_SECTOR_MAP.update(
    {
        "SPY": "broad mkt", "QQQ": "broad mkt", "IWM": "broad mkt",
        "DIA": "broad mkt", "MDY": "broad mkt",
        "XLK": "tech ETF", "SMH": "tech ETF",
        "XLF": "finl ETF", "KRE": "finl ETF",
        "XLV": "hlth ETF", "XBI": "hlth ETF",
        "XLY": "cons ETF", "XLP": "cons ETF",
        "XLI": "indl ETF", "ITB": "indl ETF",
        "XLE": "enrgy ETF", "XLB": "matls ETF",
        "XLU": "util ETF", "XLRE": "reit ETF", "XLC": "comms ETF",
    }
)


def sector_of(ticker: str) -> str:
    return _SECTOR_MAP.get(ticker, "other")


STRONG_TIER = [t for t in FULL_UNIVERSE if tier_of(t) == TIER_STRONG]

PRESETS = {
    "core": CORE_ETFS,
    "mag7": MAG7,
    "etfs": ETFS,
    "stocks": ALL_STOCKS,
    "all": FULL_UNIVERSE,
    "tech": MEGA_CAP_TECH,
    "index": ["SPY", "QQQ", "IWM", "DIA"],
    # Legacy research subset. It was selected using the historical period being described;
    # use it only for research, not as an independently validated trading universe.
    "strong": STRONG_TIER,
}


def resolve(name_or_tickers) -> list[str]:
    """Accepts a preset name or an explicit ticker list."""
    if isinstance(name_or_tickers, str):
        key = name_or_tickers.lower()
        if key in PRESETS:
            return list(PRESETS[key])
        return [name_or_tickers.upper()]
    resolved: list[str] = []
    for item in name_or_tickers:
        key = str(item).lower()
        resolved.extend(PRESETS[key] if key in PRESETS else [str(item).upper()])
    return list(dict.fromkeys(resolved))
