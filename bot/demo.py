"""Demo round-trip on 1-minute bars: ONE real (or simulated) BUY followed by an algorithmic SELL, then exit.

Purpose: prove the whole execution chain on the exchange (signal -> LIMIT order -> fill -> exit -> Toman) with a
small fixed amount. A loss from fees/spread is acceptable; the point is a complete, logged, fully algorithmic trade.

Rules (all deterministic, no manual input):
  entry  : 1-minute bars. Micro-dip rebound = fall of >= DEMO_DROP over the last 15 closed bars AND the last closed bar is green
           AND spread is small. The market with the biggest dip wins.
  forced : if no market qualifies within DEMO_WAIT_MIN minutes, the market with the tightest spread (ties: biggest 24h volume)
           is bought anyway. The reason is written to the log as "forced demo entry" so nothing is hidden from the judges.
  exit   : take-profit DEMO_TP, stop-loss DEMO_SL, or DEMO_HOLD_SEC seconds, whichever comes first (or challenge end guard).
  limits : exactly one round trip, fixed DEMO_AMOUNT toman, only coins this run bought are ever sold, Toman markets only.
Logs go to logs/*_demo.* (separate from the main live bot).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd

from .api import WallexError


def _f(name, default):
    v = os.environ.get(name)
    return type(default)(v) if v not in (None, "") else default


class Demo:
    def __init__(self, s, api, broker, journal, amount=None, wait_min=None, loop_seconds=5.0, hold_sec=None):
        self.s, self.api, self.b, self.j = s, api, broker, journal
        self.amount = float(amount if amount is not None else _f("DEMO_AMOUNT", 500_000.0))
        self.wait_sec = 60 * float(wait_min if wait_min is not None else _f("DEMO_WAIT_MIN", 5.0))
        self.hold_sec = float(hold_sec if hold_sec is not None else _f("DEMO_HOLD_SEC", 240.0))
        self.drop = _f("DEMO_DROP", -0.0025)       # fall over the last 15 one-minute bars
        self.lookback = _f("DEMO_LOOKBACK_BARS", 15)
        self.tp = _f("DEMO_TP", 0.0035)
        self.sl = _f("DEMO_SL", 0.0050)
        self.max_run_sec = 60 * _f("DEMO_MAX_MIN", 30.0)   # hard stop: never loop forever without a position
        self.loop_seconds = loop_seconds
        self.pos: dict | None = None
        self.done = False
        self.result: dict | None = None

    # ------------------------------------------------------------ helpers
    def minutes_to_end(self):
        e = self.s.end_dt()
        return None if e is None else (e - datetime.now(timezone.utc)).total_seconds() / 60

    def universe(self):
        mk = self.api.markets()
        avail = [x for x in self.s.symbols if x in mk and mk[x]["quote"] == self.s.quote]
        vols = {}
        for sym in avail:
            st = mk[sym].get("stats") or {}
            try:
                vols[sym] = float(st.get("24h_tmnVolume") or st.get("24h_quoteVolume") or 0)
            except (TypeError, ValueError):
                vols[sym] = 0.0
        return [x for x in sorted(avail, key=lambda y: -vols[y])], vols

    def scan(self, syms, vols):
        """One row per market: dip over the last N one-minute bars, last bar colour, spread."""
        rows = []
        now = int(time.time())
        for sym in syms:
            try:
                bid, ask, spread, _ = self.b.top(sym)
                df = self.api.candles(sym, "1", now - 3600 * 2, now)
                df = df[df.index + pd.Timedelta(seconds=60) <= pd.Timestamp(now, unit="s", tz="UTC")]  # drop the forming bar
                if len(df) <= self.lookback + 1:
                    continue
                c = df["close"]
                drop = float(c.iloc[-1] / c.iloc[-1 - int(self.lookback)] - 1)
                green = bool(c.iloc[-1] > df["open"].iloc[-1])
                rows.append({"symbol": sym, "drop": drop, "green": green, "spread": spread, "bid": bid, "ask": ask, "vol": vols.get(sym, 0.0)})
            except WallexError as e:
                self.j.event(f"scan {sym}: {e}")
        return rows

    # ------------------------------------------------------------ steps
    def choose(self, rows, waited):
        ok = [r for r in rows if r["drop"] <= self.drop and r["green"] and r["spread"] <= self.s.max_spread]
        if ok:
            r = min(ok, key=lambda x: x["drop"])
            return r, (f"[demo-1m] برگشت از افت کوتاه: {r['drop']:+.2%} در {int(self.lookback)} کندل ۱ دقیقه‌ای (ماشه ≤ {self.drop:+.2%}) "
                       f"و کندل آخر سبز؛ اسپرد {r['spread']:.3%}")
        if waited >= self.wait_sec and rows:
            tight = [r for r in rows if r["spread"] <= self.s.max_spread] or rows
            r = min(tight, key=lambda x: (round(x["spread"], 4), -x["vol"]))
            return r, (f"[demo-1m] ورود نمایشی اجباری (forced demo entry): در {self.wait_sec / 60:g} دقیقه سیگنال طبیعی پیدا نشد؛ "
                       f"بازار با کمترین اسپرد ({r['spread']:.3%}) برای اثبات اجرای کامل عملیات خریداری شد")
        return None, ""

    def enter(self, r, reason):
        f = self.b.buy(r["symbol"], self.amount)
        if not f:
            self.j.event(f"خرید {r['symbol']} انجام نشد (کمتر از حداقل سفارش یا پر نشد)")
            return False
        self.pos = {"symbol": r["symbol"], "qty": f.qty, "entry": f.price, "cost": f.toman, "t": time.time(), "reason": reason, "buy_id": f.order_id}
        self.j.trade(symbol=r["symbol"], side="BUY", qty=f.qty, price=f.price, toman=round(f.toman), fee_est=round(f.fee), order_id=f.order_id, reason=reason)
        self.j.decision({"type": "demo_entry", "symbol": r["symbol"], "reason": reason, "tp": self.tp, "sl": self.sl, "hold_sec": self.hold_sec,
                         "amount": self.amount, "dip": r["drop"], "spread": r["spread"]})
        self.j.event(f"BUY {r['symbol']} {f.qty} @ {f.price:.0f} ({f.toman:,.0f} تومان) | {reason}")
        return True

    def exit_reason(self, bid):
        p = self.pos
        ret = bid / p["entry"] - 1
        if ret >= self.tp:
            return f"حد سود {ret:+.2%}"
        if ret <= -self.sl:
            return f"حد ضرر {ret:+.2%}"
        if time.time() - p["t"] >= self.hold_sec:
            return f"پایان زمان نگهداری ({self.hold_sec:g} ثانیه) بازده {ret:+.2%}"
        m = self.minutes_to_end()
        if m is not None and m <= self.s.flatten_before_end_min:
            return "تسویهٔ پایان چالش"
        return None

    def close(self, why):
        p = self.pos
        f = self.b.sell(p["symbol"], p["qty"])
        if not f:
            self.j.event(f"فروش {p['symbol']} انجام نشد؛ دوباره تلاش می‌شود ({why})")
            return False
        pnl = f.toman - p["cost"] * (f.qty / p["qty"] if p["qty"] else 1)
        self.j.trade(symbol=p["symbol"], side="SELL", qty=f.qty, price=f.price, toman=round(f.toman), fee_est=round(f.fee), order_id=f.order_id,
                     reason=why, pnl=round(pnl), position_ret=round(pnl / p["cost"], 5) if p["cost"] else "")
        self.j.event(f"SELL {p['symbol']} {f.qty} @ {f.price:.0f} → {why} | PnL {pnl:+,.0f} تومان")
        left = p["qty"] - f.qty
        p["qty"], p["cost"] = left, p["cost"] * left / (left + f.qty) if left > 0 else 0.0
        mn = self.b.mk[p["symbol"]]["min_notional"]
        bid = self.b.top(p["symbol"])[0]
        if left * bid < max(mn, 1):
            self.result = {"symbol": p["symbol"], "pnl": round(pnl)}
            self.pos = None
            self.done = True
        return True

    def step(self, syms, vols, t_start):
        if self.pos is None:
            rows = self.scan(syms, vols)
            r, why = self.choose(rows, time.time() - t_start)
            if r:
                self.enter(r, why)
            else:
                best = min(rows, key=lambda x: x["drop"]) if rows else None
                if best:
                    self.j.event(f"منتظر سیگنال… بیشترین افت: {best['symbol']} {best['drop']:+.2%}")
            return
        bid = self.b.top(self.pos["symbol"])[0]
        why = self.exit_reason(bid)
        if why:
            self.close(why)

    def run(self):
        m = self.minutes_to_end()
        if m is not None and m <= self.s.flatten_before_end_min + 5:
            self.j.event("به پایان چالش کمتر از ۲۵ دقیقه مانده؛ معاملهٔ نمایشی شروع نمی‌شود")
            return None
        syms, vols = self.universe()
        if not syms:
            self.j.event("هیچ بازار تومانی معتبری پیدا نشد")
            return None
        if hasattr(self.b, "balances"):
            free = self.b.balances().get("TMN", {}).get("free", 0.0)
            if free < self.amount * 1.01:
                self.j.event(f"موجودی تومان ({free:,.0f}) برای {self.amount:,.0f} تومان کافی نیست")
                return None
        self.j.event(f"شروع معاملهٔ نمایشی ۱ دقیقه‌ای | مبلغ {self.amount:,.0f} تومان | حداکثر انتظار برای سیگنال {self.wait_sec / 60:g} دقیقه | "
                     f"TP {self.tp:.2%} SL {self.sl:.2%} نگهداری ≤ {self.hold_sec:g} ثانیه | بازارها: {', '.join(syms)}")
        t_start = time.time()
        while not self.done:
            if self.pos is None and time.time() - t_start > self.max_run_sec:
                self.j.event("زمان مجاز معاملهٔ نمایشی تمام شد و خریدی انجام نشد (بازار/موجودی/اسپرد را بررسی کنید)")
                return None
            try:
                self.step(syms, vols, t_start)
            except KeyboardInterrupt:
                if self.pos:
                    self.j.event(f"توقف دستی؛ پوزیشن {self.pos['symbol']} باز است. برای فروش: python -m bot flatten --yes")
                return self.result
            except Exception as e:
                self.j.event(f"خطا: {type(e).__name__}: {e}")
                time.sleep(5)
            if not self.done:
                time.sleep(self.loop_seconds)
        self.j.event(f"معاملهٔ نمایشی کامل شد: {self.result}")
        return self.result
