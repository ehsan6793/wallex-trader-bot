"""Features, triple-barrier labels and the exit levels — shared by backtest, shadow and live."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "r1", "r3", "r6", "r12", "r24", "r48",   # log returns over 5m..4h
    "rsi14", "z_ema20", "ema_slope", "trend_gap",
    "atr_pct", "vol_ratio", "vol_z", "bb_pos", "range_pos",
    "upper_wick", "hour_sin", "hour_cos",
]


def _rsi(close: pd.Series, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """df: OHLCV indexed by UTC bar time. Returns features (no look-ahead: everything uses data up to bar close)."""
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    lc = np.log(c)
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24, 48):
        f[f"r{n}"] = lc.diff(n)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    f["atr_pct"] = atr / c
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema60 = c.ewm(span=60, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    f["rsi14"] = _rsi(c) / 100.0
    f["z_ema20"] = (c - ema20) / atr.replace(0, np.nan)
    f["ema_slope"] = np.log(ema60).diff(12)
    f["trend_gap"] = np.log(ema60 / ema200)
    f["vol_ratio"] = (atr / c) / (atr / c).rolling(288, min_periods=48).mean()
    qv = (v * c)
    f["vol_z"] = (np.log1p(qv) - np.log1p(qv).rolling(96, min_periods=24).mean()) / np.log1p(qv).rolling(96, min_periods=24).std()
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - m20) / (2 * s20.replace(0, np.nan))
    hh, ll = h.rolling(48).max(), l.rolling(48).min()
    f["range_pos"] = (c - ll) / (hh - ll).replace(0, np.nan)
    f["upper_wick"] = (h - np.maximum(o, c)) / atr.replace(0, np.nan)
    hrs = df.index.hour + df.index.minute / 60.0  # UTC; Tehran = +3.5h, the model only needs the cycle
    f["hour_sin"] = np.sin(2 * np.pi * hrs / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hrs / 24)
    return f.replace([np.inf, -np.inf], np.nan)


def exit_levels(atr_pct: float, s) -> tuple[float, float]:
    """Take-profit and stop-loss as fractions of entry price."""
    tp = float(np.clip(s.tp_atr * atr_pct, s.tp_min, s.tp_max))
    sl = float(np.clip(s.sl_atr * atr_pct, s.sl_min, s.sl_max))
    return tp, sl


def label(df: pd.DataFrame, feats: pd.DataFrame, s, with_return: bool = False):
    """Triple-barrier label: 1 if TP is touched before SL within `horizon_bars` (entry at next bar open).

    Uses exactly the same TP/SL rule as the trading engine, so the model learns the trade we actually take.
    With `with_return`, also returns the realised gross return of that trade (TP, SL, or the close at the
    time limit) — used to judge thresholds on what the strategy would really have earned.
    """
    o, h, l, c = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    atr = feats["atr_pct"].to_numpy()
    n, H = len(df), s.horizon_bars
    y = np.full(n, np.nan)
    r = np.full(n, np.nan)
    for i in range(n - H - 1):
        if np.isnan(atr[i]):
            continue
        tp, sl = exit_levels(atr[i], s)
        entry = o[i + 1] * (1 + s.slippage)
        up, dn = entry * (1 + tp), entry * (1 - sl)
        res, ret = 0.0, c[i + H] / entry - 1
        for j in range(i + 1, i + 1 + H):
            if l[j] <= dn:        # conservative: stop first when both are inside one bar
                res, ret = 0.0, -sl
                break
            if h[j] >= up:
                res, ret = 1.0, tp
                break
        y[i] = res
        r[i] = ret
    if with_return:
        return pd.Series(y, index=df.index), pd.Series(r, index=df.index)
    return pd.Series(y, index=df.index)
