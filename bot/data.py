"""Candle download with on-disk cache + dataset builder for training."""
from __future__ import annotations

import time

import pandas as pd

from . import features as F


def fetch(api, symbol: str, s, days: float, end: int | None = None, use_cache=True) -> pd.DataFrame:
    end = end or int(time.time())
    start = end - int(days * 86400)
    path = s.data_dir / f"{symbol}_{s.resolution}.csv"
    cached = None
    if use_cache and path.exists():
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        if cached.index.tz is None:
            cached.index = cached.index.tz_localize("UTC")
        have_to = int(cached.index.max().timestamp()) if len(cached) else start
        frm = max(start, have_to - 2 * s.bar_seconds) if int(cached.index.min().timestamp()) <= start + s.bar_seconds else start
    else:
        frm = start
    new = api.candles(symbol, s.resolution, frm, end)
    df = pd.concat([cached, new]) if cached is not None else new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    df = df[df.index >= pd.Timestamp(start, unit="s", tz="UTC")]
    # drop the still-forming bar
    return df[df.index + pd.Timedelta(seconds=s.bar_seconds) <= pd.Timestamp(end, unit="s", tz="UTC")]


def build_dataset(frames: dict, s, until=None) -> pd.DataFrame:
    """Pool features+labels of all symbols. `until`: only rows whose label is fully known before this time."""
    rows = []
    for sym, df in frames.items():
        if len(df) < 300:
            continue
        f = F.compute(df)
        y, r = F.label(df, f, s, with_return=True)
        d = f.copy()
        d["y"] = y
        d["r"] = r
        tpsl = d["atr_pct"].apply(lambda a: F.exit_levels(a, s) if pd.notna(a) else (float("nan"), float("nan")))
        d["tp"] = [t[0] for t in tpsl]
        d["sl"] = [t[1] for t in tpsl]
        d["symbol"] = sym
        if until is not None:
            d = d[d.index <= until - pd.Timedelta(seconds=s.bar_seconds * (s.horizon_bars + 1))]
        rows.append(d)
    return pd.concat(rows).sort_index() if rows else pd.DataFrame()
