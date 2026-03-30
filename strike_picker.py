"""Live 0DTE SPX options pricing via yfinance."""

from __future__ import annotations

import logging
import math
import time as _time
from datetime import datetime
from typing import Optional

import pytz

PT_TZ = pytz.timezone("US/Pacific")
log = logging.getLogger(__name__)

# Module-level cache
_chain_cache = None
_cache_ts: float = 0
_CACHE_TTL = 300  # 5 minutes


def _get_chain():
    """Fetch today's SPX options chain, cached for 5 minutes."""
    global _chain_cache, _cache_ts
    now = _time.time()
    if _chain_cache is not None and (now - _cache_ts) < _CACHE_TTL:
        return _chain_cache

    try:
        import yfinance as yf
        ticker = yf.Ticker("^SPX")
        # Get today's expiration (0DTE)
        expirations = ticker.options
        if not expirations:
            log.warning("strike_picker: No options expirations available")
            return None

        today_str = datetime.now(PT_TZ).strftime("%Y-%m-%d")
        # Find today's expiration or closest
        exp = None
        for e in expirations:
            if e == today_str:
                exp = e
                break
        if exp is None:
            # Take the nearest expiration
            exp = expirations[0]
            log.info(f"strike_picker: No 0DTE found, using nearest: {exp}")

        chain = ticker.option_chain(exp)
        _chain_cache = chain
        _cache_ts = now
        return chain
    except Exception as e:
        log.warning(f"strike_picker: yfinance chain fetch failed: {e}")
        return _chain_cache  # return stale cache if available


def get_0dte_play(
    direction: str,
    price: float,
    target: float,
    stop: float,
) -> dict:
    """Get a 0DTE options play suggestion.

    Args:
        direction: "CALL" or "PUT"
        price: Current SPX price
        target: Target price for the trade
        stop: Stop loss price

    Returns:
        dict with keys: symbol, bid, ask, target_premium, stop_note
    """
    # Round to $5 increments
    if direction == "CALL":
        strike = math.ceil(price / 5) * 5  # ATM or 1-strike OTM
    else:
        strike = math.floor(price / 5) * 5

    suffix = "C" if direction == "CALL" else "P"
    symbol = f"SPXW {strike}{suffix}"

    chain = _get_chain()
    bid, ask = None, None

    if chain is not None:
        try:
            if direction == "CALL":
                df = chain.calls
            else:
                df = chain.puts

            # Find the strike in the chain
            row = df[df["strike"] == strike]
            if row.empty:
                # Try ATM
                atm_strike = round(price / 5) * 5
                row = df[df["strike"] == atm_strike]
                if not row.empty:
                    strike = atm_strike
                    symbol = f"SPXW {strike}{suffix}"

            if not row.empty:
                bid = float(row.iloc[0]["bid"])
                ask = float(row.iloc[0]["ask"])
        except Exception as e:
            log.debug(f"strike_picker: chain lookup failed: {e}")

    # Estimate target/stop premiums
    target_move = abs(target - price)
    entry_to_strike = max(1, abs(price - strike) + 5)  # rough delta proxy

    if bid and bid > 0:
        mid = (bid + ask) / 2
        # Rough delta approximation for premium targets
        target_prem_low = round(mid * (1 + target_move / entry_to_strike * 0.5), 0)
        target_prem_high = round(mid * (1 + target_move / entry_to_strike * 0.8), 0)
        target_premium = f"${target_prem_low:.0f}-{target_prem_high:.0f}"
    else:
        # Fallback estimates when no chain data
        bid_est = max(0.50, (15 - abs(price - strike)) * 0.25)
        ask_est = bid_est + 0.50
        bid = round(bid_est, 2)
        ask = round(ask_est, 2)
        target_premium = f"${round(bid * 2)}-{round(bid * 3)}"

    stop_note = f"cut below {stop:.0f}" if direction == "CALL" else f"cut above {stop:.0f}"

    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "target_premium": target_premium,
        "stop_note": stop_note,
    }
