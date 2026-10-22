"""Thin, defensive Wallex REST client (https://api-docs.wallex.ir)."""
from __future__ import annotations

import math
import time
from decimal import ROUND_DOWN, Decimal

import httpx
import pandas as pd


class WallexError(RuntimeError):
    pass


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def decimals_of(step) -> int:
    """Wallex `stepSize`/`tickSize` are documented as *number of decimals*. Accept a step value (0.001) too."""
    s = _num(step, 0)
    if s >= 1 and float(s).is_integer():
        return int(s)
    if 0 < s < 1:
        return max(0, -int(math.floor(math.log10(s) + 1e-9)))
    return 0


def round_down(x: float, decimals: int) -> float:
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_DOWN))


class WallexAPI:
    def __init__(self, api_key: str = "", base_url: str = "https://api.wallex.ir", timeout: float = 12, min_interval: float = 0.4):
        self.key = api_key
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=timeout, headers={"Content-Type": "application/json", "User-Agent": "starkoch-wallex-bot/1.0"})
        self.min_interval = min_interval
        self._last = 0.0
        self._markets = None

    # ------------------------------------------------------------------ core
    def _req(self, method: str, path: str, params=None, body=None, auth=False, retries=3):
        headers = {}
        if auth:
            if not self.key:
                raise WallexError("WALLEX_API_KEY تنظیم نشده است")
            headers["X-API-Key"] = self.key
        err = None
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.http.request(method, self.base + path, params=params, json=body, headers=headers)
            except httpx.HTTPError as e:
                err = f"network: {e}"
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:  # Wallex answers bursts with 500 -> back off
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(5.0 * (attempt + 1))
                continue
            try:
                data = r.json()
            except ValueError:
                raise WallexError(f"HTTP {r.status_code}: non-JSON {r.text[:200]}")
            if r.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
                raise WallexError(f"HTTP {r.status_code}: {data.get('message') if isinstance(data, dict) else data} {str(data)[:300]}")
            return data
        raise WallexError(err or "request failed")

    @staticmethod
    def _result(data):
        return data.get("result", data) if isinstance(data, dict) else data

    # ------------------------------------------------------------------ public
    def markets(self, refresh=False) -> dict:
        if self._markets is None or refresh:
            res = self._result(self._req("GET", "/v1/markets"))
            syms = res.get("symbols", res)
            out = {}
            for sym, m in syms.items():
                out[sym] = {
                    "symbol": m.get("symbol", sym), "base": m.get("baseAsset"), "quote": m.get("quoteAsset"),
                    "qty_dec": decimals_of(m.get("stepSize")), "price_dec": decimals_of(m.get("tickSize")),
                    "min_qty": _num(m.get("minQty")), "min_notional": _num(m.get("minNotional")),
                    "stats": m.get("stats") or {},
                }
            self._markets = out
        return self._markets

    def depth(self, symbol: str) -> dict:
        res = self._result(self._req("GET", "/v1/depth", params={"symbol": symbol}))
        asks = sorted(((_num(a["price"]), _num(a["quantity"])) for a in res.get("ask", [])), key=lambda x: x[0])
        bids = sorted(((_num(b["price"]), _num(b["quantity"])) for b in res.get("bid", [])), key=lambda x: -x[0])
        return {"asks": asks, "bids": bids}

    def candles(self, symbol: str, resolution: str, frm: int, to: int) -> pd.DataFrame:
        """OHLCV bars of `resolution` minutes. Index = bar open time (UTC).

        Observed Wallex /v1/udf/history behaviour (2026-10): valid resolutions are "1", "60", "240", "1D";
        "1" only answers for ranges below ~1 day (bursts get HTTP 500). So minute bars are fetched day by day and resampled here.
        """
        minutes = int(resolution) if resolution.isdigit() else 1440
        native = {60: "60", 240: "240", 1440: "1D"}.get(minutes, "1")
        span = 72000 if native == "1" else 86400 * 30  # 20h chunks: larger 1m ranges intermittently return HTTP 500
        frames = []
        t0 = frm
        while t0 < to:
            t1 = min(to, t0 + span)
            d = self._result(self._req("GET", "/v1/udf/history", params={"symbol": symbol, "resolution": native, "from": t0, "to": t1}))
            if isinstance(d, dict) and d.get("s") == "ok" and d.get("t"):
                frames.append(pd.DataFrame({"t": d["t"], "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"], "volume": d["v"]}))
            t0 = t1
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)
        df = pd.concat(frames).drop_duplicates("t").sort_values("t")
        df.index = pd.to_datetime(df.pop("t").astype("int64"), unit="s", utc=True)
        df = df.astype(float)
        if native == "1" and minutes > 1:
            df = df.resample(f"{minutes}min", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
        return df

    # ------------------------------------------------------------------ private
    def balances(self) -> dict:
        res = self._result(self._req("GET", "/v1/account/balances", auth=True))
        out = {}
        for asset, b in (res.get("balances") or {}).items():
            total, locked = _num(b.get("value")), _num(b.get("locked"))
            out[asset] = {"total": total, "locked": locked, "free": max(0.0, total - locked)}
        return out

    def fees(self) -> dict:
        return self._result(self._req("GET", "/v1/account/fee", auth=True))

    def place_order(self, symbol, side, price, quantity, client_id, type_="LIMIT") -> dict:
        body = {"symbol": symbol, "type": type_, "side": side, "price": str(price), "quantity": str(quantity), "client_id": client_id}
        return self._result(self._req("POST", "/v1/account/orders", body=body, auth=True, retries=1))

    def get_order(self, client_id) -> dict:
        return self._result(self._req("GET", f"/v1/account/orders/{client_id}", auth=True))

    def cancel_order(self, client_id) -> dict:
        return self._result(self._req("DELETE", "/v1/account/orders", params={"clientOrderId": client_id}, auth=True, retries=1))

    def open_orders(self, symbol=None) -> list:
        res = self._result(self._req("GET", "/v1/account/openOrders", params={"symbol": symbol} if symbol else None, auth=True))
        return res.get("orders", []) if isinstance(res, dict) else res

    def my_trades(self, symbol=None) -> list:
        res = self._result(self._req("GET", "/v1/account/trades", params={"symbol": symbol} if symbol else None, auth=True))
        return res.get("AccountLatestTrades", []) if isinstance(res, dict) else res
