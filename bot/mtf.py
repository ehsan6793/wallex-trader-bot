"""Multi-timeframe context (1m, 5m, 15m, 30m, 1h, 4h) for every market.

The 15-minute rebound rule is the only one that passed the multi-timeframe backtest (research/run4.py):
the same rule on 5m/30m/1h/4h was weaker or negative, and higher-timeframe trend filters did not improve
the 15m results. So the other timeframes are used as CONTEXT: shown on the dashboard, logged with every
decision and handed to the LLM supervisor — they never open trades on their own.
Cost: two extra API calls per market per 15 minutes (1-minute bars for the last 20h, native 4h bars).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

TFS = ["1m", "5m", "15m", "30m", "1h", "4h"]


def _state(df: pd.DataFrame) -> dict:
    c = df["close"].dropna()
    if len(c) < 22:
        return {"trend": None}
    ema = c.ewm(span=20, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float(100 - 100 / (1 + up.iloc[-1] / dn.iloc[-1])) if dn.iloc[-1] else 100.0
    return {"trend": "up" if c.iloc[-1] > ema.iloc[-1] else "down", "rsi": round(rsi), "chg": round(float(c.iloc[-1] / c.iloc[-3] - 1), 4)}


def snapshot(api, symbol: str) -> dict:
    now = int(time.time())
    out = {}
    try:
        m1 = api.candles(symbol, "1", now - 72000, now)
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        out["1m"] = _state(m1.iloc[-120:])
        for tf, rule in (("5m", "5min"), ("15m", "15min"), ("30m", "30min")):
            out[tf] = _state(m1.resample(rule).agg(agg).dropna())
    except Exception:
        pass
    try:
        out["1h"] = _state(api.candles(symbol, "60", now - 86400 * 4, now))
    except Exception:
        pass
    try:
        out["4h"] = _state(api.candles(symbol, "240", now - 86400 * 12, now))
    except Exception:
        pass
    return out


def compact(m: dict) -> str:
    """'1m↑ 5m↓ ...' — tiny text for the LLM prompt (token-cheap)."""
    return " ".join(f"{tf}{'↑' if (m.get(tf) or {}).get('trend') == 'up' else '↓' if (m.get(tf) or {}).get('trend') == 'down' else '?'}" for tf in TFS)
