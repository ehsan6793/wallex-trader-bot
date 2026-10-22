"""Execution layer. PaperBroker (shadow test, no real orders) and LiveBroker share one interface."""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass

from .api import WallexAPI, WallexError, round_down


@dataclass
class Fill:
    qty: float           # base asset received (buy) / sold (sell)
    price: float         # average price
    toman: float         # toman spent (buy, incl. fee) / received (sell, after fee)
    fee: float
    order_id: str


def _round_up(x, dec):
    m = 10 ** dec
    return math.ceil(x * m) / m


class BaseBroker:
    def __init__(self, api: WallexAPI, s, journal):
        self.api, self.s, self.j = api, s, journal
        self.mk = api.markets()

    def top(self, symbol):
        d = self.api.depth(symbol)
        if not d["asks"] or not d["bids"]:
            raise WallexError(f"{symbol}: دفتر سفارش خالی است")
        ask, bid = d["asks"][0][0], d["bids"][0][0]
        return bid, ask, (ask - bid) / ask, d

    def market(self, symbol):
        m = self.mk.get(symbol)
        if not m:
            raise WallexError(f"بازار {symbol} وجود ندارد")
        return m

    def _cid(self, symbol, side):
        return f"sk{side[0]}{symbol[:6]}{uuid.uuid4().hex[:10]}"


class PaperBroker(BaseBroker):
    """Shadow test: real prices and order book, simulated fills (walks the book), no orders sent."""

    def __init__(self, api, s, journal, cash=None, holdings=None):
        super().__init__(api, s, journal)
        self.cash = cash if cash is not None else s.start_capital
        self.holdings = holdings or {}

    def _walk(self, levels, qty):
        got, cost = 0.0, 0.0
        for p, q in levels:
            take = min(q, qty - got)
            got += take
            cost += take * p
            if got >= qty - 1e-12:
                break
        return got, (cost / got if got else 0.0)

    def buy(self, symbol, toman, ref_ask=None) -> Fill | None:
        m = self.market(symbol)
        bid, ask, spread, d = self.top(symbol)
        toman = min(toman, self.cash)
        qty = round_down(toman / (ask * (1 + self.s.slippage)), m["qty_dec"])
        if qty <= 0 or qty < m["min_qty"] or qty * ask < m["min_notional"]:
            return None
        got, avg = self._walk(d["asks"], qty)
        if got <= 0:
            return None
        spent = got * avg
        fee = spent * self.s.taker_fee
        self.cash -= spent + fee
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + got
        return Fill(got, avg, spent + fee, fee, self._cid(symbol, "BUY"))

    def sell(self, symbol, qty) -> Fill | None:
        m = self.market(symbol)
        bid, ask, spread, d = self.top(symbol)
        qty = round_down(min(qty, self.holdings.get(symbol, 0.0)), m["qty_dec"])
        if qty <= 0:
            return None
        got, avg = self._walk(d["bids"], qty)
        gross = got * avg
        fee = gross * self.s.taker_fee
        self.cash += gross - fee
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) - got
        return Fill(got, avg, gross - fee, fee, self._cid(symbol, "SELL"))

    def balances(self):
        out = {"TMN": {"total": self.cash, "free": self.cash, "locked": 0}}
        for sym, q in self.holdings.items():
            base = self.mk[sym]["base"]
            out[base] = {"total": q, "free": q, "locked": 0}
        return out

    def equity(self, prices: dict) -> tuple[float, float]:
        return self.cash + sum(q * prices.get(sym, 0) for sym, q in self.holdings.items()), self.cash


class LiveBroker(BaseBroker):
    """Real orders: aggressive LIMIT (price-capped), poll, cancel the remainder. Never market orders without a cap."""

    POLL = 1.5
    WAIT = 15

    def _execute(self, symbol, side, price, qty):
        cid = self._cid(symbol, side)
        self.j.event(f"ORDER {side} {symbol} qty={qty} price={price} cid={cid}")
        o = self.api.place_order(symbol, side, price, qty, cid)
        t0 = time.time()
        while time.time() - t0 < self.WAIT:
            if not o.get("active", True) or float(o.get("executedPercent") or 0) >= 99.999:
                break
            time.sleep(self.POLL)
            try:
                o = self.api.get_order(cid)
            except WallexError as e:
                self.j.event(f"poll error {e}")
        if o.get("active", False) and float(o.get("executedPercent") or 0) < 99.999:
            try:
                self.api.cancel_order(cid)
                time.sleep(1)
                o = self.api.get_order(cid)
            except WallexError as e:
                self.j.event(f"cancel error {e}")
        q = float(o.get("executedQty") or 0)
        ssum = float(o.get("executedSum") or 0)
        px = ssum / q if q else float(o.get("executedPrice") or 0)
        return q, px, ssum, cid, o

    def buy(self, symbol, toman, ref_ask=None) -> Fill | None:
        m = self.market(symbol)
        bid, ask, spread, _ = self.top(symbol)
        tmn_free = self.api.balances().get("TMN", {}).get("free", 0.0)
        toman = min(toman, tmn_free * 0.995)
        price = _round_up(ask * (1 + self.s.slippage), m["price_dec"])
        qty = round_down(toman / price / (1 + self.s.taker_fee), m["qty_dec"])
        if qty <= 0 or qty < m["min_qty"] or qty * price < m["min_notional"]:
            self.j.event(f"skip BUY {symbol}: below exchange minimum (qty={qty})")
            return None
        base = m["base"]
        before = self.api.balances().get(base, {}).get("total", 0.0)
        q, px, ssum, cid, o = self._execute(symbol, "BUY", price, qty)
        if q <= 0:
            return None
        after = self.api.balances().get(base, {}).get("total", 0.0)
        received = after - before if after - before > 0 else q * (1 - self.s.taker_fee)
        fee = max(0.0, (q - received) * px) if received < q else ssum * self.s.taker_fee
        return Fill(received, px, ssum + (fee if received >= q else 0), fee, cid)

    def sell(self, symbol, qty) -> Fill | None:
        m = self.market(symbol)
        base = m["base"]
        free = self.api.balances().get(base, {}).get("free", 0.0)
        qty = round_down(min(qty, free), m["qty_dec"])
        total_q, total_sum, cid_last = 0.0, 0.0, ""
        for attempt in range(3):  # retry the unfilled remainder with a fresh price
            if qty <= 0 or qty < m["min_qty"]:
                break
            bid, ask, spread, _ = self.top(symbol)
            price = round_down(bid * (1 - self.s.slippage * (attempt + 1)), m["price_dec"])
            if qty * price < m["min_notional"]:
                break
            q, px, ssum, cid, o = self._execute(symbol, "SELL", price, qty)
            total_q += q
            total_sum += ssum
            cid_last = cid
            qty = round_down(qty - q, m["qty_dec"])
        if total_q <= 0:
            return None
        fee = total_sum * self.s.taker_fee
        return Fill(total_q, total_sum / total_q, total_sum - fee, fee, cid_last)

    def balances(self):
        return self.api.balances()

    def equity(self, prices: dict) -> tuple[float, float]:
        b = self.api.balances()
        cash = b.get("TMN", {}).get("total", 0.0)
        eq = cash
        for sym, m in self.mk.items():
            if m["quote"] == "TMN" and m["base"] in b and sym in prices:
                eq += b[m["base"]]["total"] * prices[sym]
        return eq, cash
