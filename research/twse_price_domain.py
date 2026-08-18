"""v16 Stage B2 — TWSE legal price domain for ordinary listed stocks.

Rules per reports/continuous_research/v16_next_session_execution/
twse_price_domain_rules.md (verified vs official TWSE material
2026-08-18): the daily +/-10% limit anchors to the day's market-opening
AUCTION REFERENCE price (normally the previous close on an ordinary
session), and final limit prices are direction-aware projections onto
the ordinary-stock tick ladder. This repo cannot detect ex-right /
ex-dividend / IPO / resumption days, so the domain status is at best
NORMAL_DAY_ASSUMPTION and never CONFIRMED_STANDARD_LIMIT; special-day
references are never inferred from price movement. Non-stock instrument
types (ETF etc.) are NOT given a domain (ladder/limit unverified).

No orders, no broker APIs. Anchor regression: reference 40.60 ->
legal limit up 44.65, legal limit down 36.55.
"""

from dataclasses import dataclass

import numpy as np

# Ordinary-stock ladder only. Do NOT apply to ETFs/warrants/ETNs/bonds.
_STOCK_TICKS = ((10.0, 0.01), (50.0, 0.05), (100.0, 0.10),
                (500.0, 0.50), (1000.0, 1.00), (float("inf"), 5.00))
LIMIT_PCT = 0.10
_EPS = 1e-9


class PriceDomainValidationError(ValueError):
    """An emitted price violated the legal domain or the tick grid."""


def stock_tick(price):
    for lim, tick in _STOCK_TICKS:
        if price < lim - _EPS:
            return tick
    return 5.0


def is_legal_tick(price):
    if price is None or not np.isfinite(price) or price <= 0:
        return False
    t = stock_tick(price)
    return abs(price / t - round(price / t)) < 1e-6


def legal_floor(price):
    """Greatest legal quote price <= price (direction-aware, never
    produces an illegal tick, band-boundary safe)."""
    if not np.isfinite(price) or price <= 0:
        return np.nan
    t = stock_tick(price)
    v = np.floor(price / t + _EPS) * t
    return round(v, 2)


def legal_ceil(price):
    """Smallest legal quote price >= price."""
    if not np.isfinite(price) or price <= 0:
        return np.nan
    t = stock_tick(price)
    v = np.ceil(price / t - _EPS) * t
    # ceiling may land exactly on a band boundary (legal) or above it —
    # a boundary value like 100.0 is legal in both bands.
    return round(v, 2)


def legal_nearest(price):
    """Nearest legal tick, ties DOWN (frozen display rule)."""
    if not np.isfinite(price) or price <= 0:
        return np.nan
    lo, hi = legal_floor(price), legal_ceil(price)
    return lo if (price - lo) <= (hi - price) + 1e-9 else hi


@dataclass
class TWSEPriceDomain:
    symbol: str
    security_type: str            # 'stock' | 'etf' | ...
    reference_price: float        # NaN when unknown
    reference_source: str         # PREVIOUS_CLOSE | SPECIAL_REFERENCE | UNKNOWN
    reference_confidence: str     # MEDIUM | LOW
    standard_limit_applicable: bool
    raw_limit_up: float
    raw_limit_down: float
    legal_limit_up: float
    legal_limit_down: float
    tick_size_at_reference: float
    price_domain_status: str
    # CONFIRMED_STANDARD_LIMIT (never emitted by this repo) |
    # NORMAL_DAY_ASSUMPTION | SPECIAL_REFERENCE_REQUIRED |
    # NO_STANDARD_LIMIT | UNKNOWN

    def known(self):
        return (self.price_domain_status == "NORMAL_DAY_ASSUMPTION"
                and np.isfinite(self.legal_limit_up)
                and np.isfinite(self.legal_limit_down))

    def clamp(self, price, side):
        """Clamp into the legal domain, then project to a legal tick
        (side='buy' -> floor, 'sell' -> ceil, 'nearest'). Returns the
        input untouched when the domain is unknown."""
        if price is None or not np.isfinite(price):
            return np.nan
        if not self.known():
            return price
        p = min(max(price, self.legal_limit_down), self.legal_limit_up)
        p = (legal_floor(p) if side == "buy" else
             legal_ceil(p) if side == "sell" else legal_nearest(p))
        # projection may step outside after clamping to a non-tick bound
        p = min(max(p, self.legal_limit_down), self.legal_limit_up)
        return p

    def validate(self, price, context=""):
        """Hard gate: raise PriceDomainValidationError on any emitted
        price outside the known domain or off the tick grid."""
        if price is None or (isinstance(price, float) and np.isnan(price)):
            return
        if not is_legal_tick(price):
            raise PriceDomainValidationError(
                f"{self.symbol} {context}: {price} is not a legal TWSE "
                "tick")
        if self.known() and not (self.legal_limit_down - 1e-9 <= price
                                 <= self.legal_limit_up + 1e-9):
            raise PriceDomainValidationError(
                f"{self.symbol} {context}: {price} outside legal domain "
                f"[{self.legal_limit_down}, {self.legal_limit_up}]")

    def position_of(self, price):
        """legal_domain_position in [0,1]; execution context only."""
        if not self.known() or price is None or not np.isfinite(price):
            return np.nan
        rng = self.legal_limit_up - self.legal_limit_down
        return float((price - self.legal_limit_down) / rng) if rng > 0 \
            else np.nan


def build_domain(symbol, prev_close, security_type="stock"):
    """Conservative domain construction. prev_close = the latest cached
    close at/before the signal date (the normal-day auction-reference
    assumption); NaN/None -> UNKNOWN. Non-stock types -> UNKNOWN (their
    tick ladders/limits are unverified in this repo)."""
    nan = float("nan")
    if security_type != "stock":
        return TWSEPriceDomain(symbol, security_type, nan, "UNKNOWN",
                               "LOW", False, nan, nan, nan, nan, nan,
                               "UNKNOWN")
    if prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return TWSEPriceDomain(symbol, security_type, nan, "UNKNOWN",
                               "LOW", False, nan, nan, nan, nan, nan,
                               "UNKNOWN")
    ref = float(prev_close)
    raw_up, raw_dn = ref * (1 + LIMIT_PCT), ref * (1 - LIMIT_PCT)
    return TWSEPriceDomain(
        symbol=symbol, security_type=security_type, reference_price=ref,
        reference_source="PREVIOUS_CLOSE", reference_confidence="MEDIUM",
        standard_limit_applicable=True,
        raw_limit_up=raw_up, raw_limit_down=raw_dn,
        legal_limit_up=legal_floor(raw_up),
        legal_limit_down=legal_ceil(raw_dn),
        tick_size_at_reference=stock_tick(ref),
        price_domain_status="NORMAL_DAY_ASSUMPTION")
