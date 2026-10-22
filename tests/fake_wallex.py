"""In-memory fake of the Wallex API (same method signatures) for offline tests."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def synth(symbol: str, days: float, seed: int, start_price: float, end: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(days * 288)
    end = end or int(time.time()) // 300 * 300
    t = pd.to_datetime(np.arange(end - n * 300, end, 300), unit="s", utc=True)
    # AR(1) momentum in returns + volatility clustering -> a learnable but noisy signal
    r = np.zeros(n)
    vol = 0.003
    for i in range(1, n):
        vol = 0.97 * vol + 0.03 * 0.003 + 0.02 * abs(r[i - 1])
        r[i] = 0.12 * r[i - 1] + rng.normal(0, vol)
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0015, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0015, n)))
    vol_ = np.abs(rng.normal(10, 3, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol_}, index=t)


class FakeWallex:
    def __init__(self, days=20, symbols=("BTCTMN", "ETHTMN", "USDTTMN", "SOLTMN")):
        prices = {"BTCTMN": 9_000_000_000, "ETHTMN": 350_000_000, "USDTTMN": 105_000, "SOLTMN": 15_000_000}
        self.data = {s: synth(s, days, i + 1, prices.get(s, 1_000_000)) for i, s in enumerate(symbols)}
        self.bal = {"TMN": 5_000_000.0}
        self.orders = {}
        self.calls = []

    def markets(self, refresh=False):
        out = {}
        for s in self.data:
            out[s] = {"symbol": s, "base": s[:-3], "quote": "TMN", "qty_dec": 8 if s != "USDTTMN" else 2, "price_dec": 0,
                      "min_qty": 0.0, "min_notional": 50_000, "stats": {"24h_tmnVolume": str(1e9 * (len(s)))}}
        return out

    def _last(self, s):
        return float(self.data[s]["close"].iloc[-1])

    def depth(self, s):
        p = self._last(s)
        return {"asks": [(p * 1.0005, 1e6)], "bids": [(p * 0.9995, 1e6)]}

    def candles(self, s, res, frm, to):
        df = self.data[s]
        d = df[(df.index >= pd.Timestamp(frm, unit="s", tz="UTC")) & (df.index <= pd.Timestamp(to, unit="s", tz="UTC"))]
        minutes = int(res) if str(res).isdigit() else 1440
        if minutes > 5:
            d = d.resample(f"{minutes}min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        return d.copy()

    def balances(self):
        return {a: {"total": v, "locked": 0.0, "free": v} for a, v in self.bal.items()}

    def fees(self):
        return {"BTCTMN": {"makerFeeRate": "0.002", "takerFeeRate": "0.0025"}}

    def place_order(self, symbol, side, price, quantity, client_id, type_="LIMIT"):
        price, q = float(price), float(quantity)
        base = symbol[:-3]
        fee = 0.0025
        if side == "BUY":
            cost = price * q
            assert self.bal["TMN"] >= cost - 1e-6, "insufficient TMN"
            self.bal["TMN"] -= cost
            self.bal[base] = self.bal.get(base, 0) + q * (1 - fee)
        else:
            assert self.bal.get(base, 0) >= q - 1e-12, "insufficient base"
            self.bal[base] -= q
            self.bal["TMN"] += price * q * (1 - fee)
        o = {"symbol": symbol, "side": side, "price": str(price), "origQty": str(q), "executedQty": str(q), "executedSum": str(price * q),
             "executedPrice": str(price), "executedPercent": 100, "status": "FILLED", "active": False, "clientOrderId": client_id}
        self.orders[client_id] = o
        self.calls.append(("order", side, symbol, q, price))
        return o

    def get_order(self, cid):
        return self.orders[cid]

    def cancel_order(self, cid):
        return {}

    def open_orders(self, symbol=None):
        return []
