"""Live / shadow trading loop."""
from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import features as F
from .api import WallexError
from .data import build_dataset, fetch
from .llm import Supervisor
from .model import Model, train
from .strategy import Decision, Position, check_exit, evaluate, position_size, trail
from . import strategies as STR
from . import mtf as MTF


CONTROL_DEFAULT = {"paused": False, "flatten": False, "pending": [], "decisions": {}}


class Engine:
    """Live/shadow loop. A small control file (written by the dashboard) lets the operator pause new entries,
    request an emergency flatten, or (APPROVAL_MODE=1, outside the challenge) approve each signal."""

    def control(self) -> dict:
        p = self.s.log_dir / f"control_{self.mode}.json"
        try:
            import json as _j
            return {**CONTROL_DEFAULT, **_j.loads(p.read_text(encoding="utf-8"))}
        except Exception:
            return dict(CONTROL_DEFAULT)

    def write_control(self, c: dict):
        import json as _j
        p = self.s.log_dir / f"control_{self.mode}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(_j.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)

    def __init__(self, s, api, broker, journal, mode: str):
        self.s, self.api, self.b, self.j, self.mode = s, api, broker, journal, mode
        self.sup = Supervisor(s, journal)
        self.positions: dict[str, Position] = {}
        self.cooldown: dict[str, float] = {}
        self.model: Model | None = None
        self.last_train = 0.0
        self.last_bar = None
        self.last_llm = 0.0
        self.peak = None
        self.start_equity = None
        self.halted = False
        self.universe: list[str] = []
        self.prices: dict[str, float] = {}
        self.frames: dict[str, pd.DataFrame] = {}
        self.operator_paused = False
        self.mtf: dict = {}

    # ------------------------------------------------------------ setup
    def select_universe(self):
        mk = self.api.markets()
        avail = [s for s in self.s.symbols if s in mk and mk[s]["quote"] == self.s.quote]
        missing = [s for s in self.s.symbols if s not in mk]
        if missing:
            self.j.event(f"نمادهای ناموجود حذف شدند: {missing}")
        # rank by last-24h toman volume (from /v1/markets stats; one request)
        vols = {}
        for sym in avail:
            st = mk[sym].get("stats") or {}
            try:
                vols[sym] = float(st.get("24h_tmnVolume") or st.get("24h_quoteVolume") or 0)
            except (TypeError, ValueError):
                vols[sym] = 0.0
        self.universe = [s for s in sorted(avail, key=lambda x: -vols[x]) if vols[s] > 0][: self.s.max_symbols]
        self.j.event("بازارهای انتخابی (بر اساس حجم تومانی ۲۴ساعت): " + ", ".join(f"{s}:{vols[s]/1e9:.1f}B" for s in self.universe))

    def restore(self):
        st = self.j.load_state()
        for sym, p in (st.get("positions") or {}).items():
            self.positions[sym] = Position(**p)
        self.peak = st.get("peak")
        self.start_equity = st.get("start_equity")
        self.halted = st.get("halted", False)
        if self.positions:
            self.j.event(f"بازیابی {len(self.positions)} پوزیشن از state")
        if self.mode == "live":  # adopt coins we hold but don't track (e.g. after a crash) so they get managed/flattened
            bals = self.b.balances()
            for sym in self.universe:
                base = self.b.mk[sym]["base"]
                q = bals.get(base, {}).get("total", 0.0)
                if q > 0 and sym not in self.positions:
                    try:
                        bid, ask, _, _ = self.b.top(sym)
                    except WallexError:
                        continue
                    if q * bid >= max(self.b.mk[sym]["min_notional"], 50_000):
                        self.positions[sym] = Position(sym, q, bid, datetime.now(timezone.utc).isoformat(), bid * 1.01, bid * 0.99,
                                                       q * bid, reason="دارایی موجود در کیف پول (بازیابی‌شده)")
                        self.j.event(f"دارایی {base}={q} به‌عنوان پوزیشن مدیریت می‌شود")

    def save(self):
        self.j.save_state({"positions": {k: asdict(v) for k, v in self.positions.items()}, "peak": self.peak,
                           "start_equity": self.start_equity, "halted": self.halted,
                           "model_meta": self.model.meta if self.model else None,
                           "threshold": self.model.threshold if self.model else None, "settings": self.s.public()})

    # ------------------------------------------------------------ data/model
    def refresh_frames(self, days):
        for sym in self.universe:
            try:
                self.frames[sym] = fetch(self.api, sym, self.s, days)
            except WallexError as e:
                self.j.event(f"candles {sym}: {e}")

    def retrain(self):
        self.refresh_frames(self.s.train_days)
        ds = build_dataset(self.frames, self.s, until=pd.Timestamp.now(tz="UTC"))
        m = train(ds, self.s)
        self.model = m
        self.last_train = time.time()
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        (self.s.log_dir / f"model_{self.mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M}.json").write_text(m.to_json(), encoding="utf-8")
        self.j.event(f"مدل آموزش دید: {m.meta} آستانه={m.threshold:.2f}")
        self.j.decision({"type": "model_trained", "meta": m.meta, "threshold": m.threshold})

    # ------------------------------------------------------------ helpers
    def end_dt(self):
        return self.s.end_dt()

    def minutes_to_end(self):
        e = self.end_dt()
        return None if e is None else (e - datetime.now(timezone.utc)).total_seconds() / 60

    def update_prices(self):
        tops = {}
        for sym in set(self.universe) | set(self.positions):
            try:
                bid, ask, spread, _ = self.b.top(sym)
                self.prices[sym] = bid
                tops[sym] = (bid, ask, spread)
            except WallexError as e:
                self.j.event(f"depth {sym}: {e}")
        return tops

    def close(self, sym, why):
        p = self.positions[sym]
        f = self.b.sell(sym, p.qty)
        if not f:
            self.j.event(f"فروش {sym} انجام نشد ({why})")
            return False
        pnl = f.toman - p.cost * (f.qty / p.qty if p.qty else 1)
        self.j.trade(symbol=sym, side="SELL", qty=f.qty, price=f.price, toman=round(f.toman), fee_est=round(f.fee),
                     order_id=f.order_id, reason=why, pnl=round(pnl), position_ret=round(pnl / p.cost, 5) if p.cost else "")
        self.j.event(f"SELL {sym} {f.qty} @ {f.price:.0f} → {why} | PnL {pnl:+,.0f} تومان")
        if pnl < 0:
            self.cooldown[sym] = time.time() + self.s.cooldown_bars * self.s.bar_seconds
        left = p.qty - f.qty
        min_n = self.b.mk[sym]["min_notional"]
        if left * self.prices.get(sym, p.entry) < max(min_n, 1):
            del self.positions[sym]
        else:
            p.qty, p.cost = left, p.cost * left / p.qty
        return True

    def flatten(self, why):
        for sym in list(self.positions):
            self.close(sym, why)
        self.save()

    # ------------------------------------------------------------ main step
    def step(self):
        now = time.time()
        mins_left = self.minutes_to_end()
        tops = self.update_prices()
        equity, cash = self.b.equity(self.prices)
        if self.start_equity is None:
            self.start_equity = equity
        self.peak = max(self.peak or equity, equity)
        dd = 1 - equity / self.peak if self.peak else 0.0
        self.j.equity_point(equity, cash, len(self.positions), dd)

        # 1) end of challenge: everything back to Toman
        if mins_left is not None and mins_left <= self.s.flatten_before_end_min:
            if self.positions:
                self.flatten("تسویه پایان چالش (همه دارایی به تومان)")
            return "ended"

        # 1b) operator controls from the dashboard
        ctl = self.control()
        if ctl.get("flatten"):
            self.j.event("درخواست فروش اضطراری از داشبورد: همه پوزیشن‌ها به تومان")
            self.flatten("فروش اضطراری با دستور اپراتور")
            ctl["flatten"] = False
            self.write_control(ctl)
        self.operator_paused = bool(ctl.get("paused"))

        # 2) kill switch
        if dd >= self.s.dd_halt and not self.halted:
            self.halted = True
            self.j.event(f"⛔ افت {dd:.2%} ≥ {self.s.dd_halt:.0%}: همه پوزیشن‌ها بسته و ربات متوقف شد")
            self.flatten("توقف اضطراری: حد افت سرمایه")

        # 3) exits (checked every loop with live bid)
        for sym, p in list(self.positions.items()):
            if sym not in self.prices:
                continue
            bid = self.prices[sym]
            prob = None
            if self.model is not None and sym in self.frames and len(self.frames[sym]) > 0:
                x = F.compute(self.frames[sym]).iloc[-1][F.FEATURES].to_numpy(dtype=float)
                prob = float(self.model.proba(x.reshape(1, -1))[0])
            ex = check_exit(p, bid, bid, bid, self.s, prob, self.model.threshold if self.model else None)
            if ex:
                self.close(sym, ex[1])
            else:
                trail(p, bid, self.s)

        # 4) retrain on schedule
        due = self.model is None or now - self.last_train > self.s.retrain_hours * 3600
        if due and now - getattr(self, "last_train_try", 0) > 600:
            self.last_train_try = now
            try:
                self.retrain()
                if self.model.meta.get("no_edge") and self.s.trade_mode != "rebound":
                    self.j.event("⚠ مدل در اعتبارسنجی خارج از نمونه لبهٔ سودآور پیدا نکرد؛ تا آموزش بعدی ورود جدید انجام نمی‌شود.")
            except (ValueError, WallexError) as e:
                self.j.event(f"آموزش مدل ناموفق: {e}")
        if self.model is None:
            self.save()
            return "no-model"

        # 4b) approved signals (APPROVAL_MODE) are executed as soon as the operator approves them (valid 15 min)
        if self.s.approval_mode:
            ctl = self.control()
            changed = False
            for p_ in list(ctl.get("pending", [])):
                verdict = ctl.get("decisions", {}).get(p_["id"])
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(p_["time"])).total_seconds()
                if verdict in ("approve", "reject") or age > 900:
                    ctl["pending"].remove(p_)
                    changed = True
                    if verdict == "approve" and age <= 900 and p_["symbol"] not in self.positions:
                        f = self.b.buy(p_["symbol"], p_["amount"])
                        if f:
                            self.positions[p_["symbol"]] = Position(p_["symbol"], f.qty, f.price, datetime.now(timezone.utc).isoformat(),
                                                                   f.price * (1 + p_["tp"]), f.price * (1 - p_["sl"]), f.toman,
                                                                   reason="تأیید اپراتور | " + " | ".join(p_["reasons"]), order_id=f.order_id)
                            self.j.trade(symbol=p_["symbol"], side="BUY", qty=f.qty, price=f.price, toman=round(f.toman), fee_est=round(f.fee),
                                         order_id=f.order_id, reason="تأیید اپراتور | " + " | ".join(p_["reasons"]))
            if changed:
                self.write_control(ctl)

        # 5) on each new closed bar: features -> decisions -> entries
        bar = int(now // self.s.bar_seconds)
        if bar == self.last_bar:
            self.save()
            return "wait"
        self.last_bar = bar
        for p in self.positions.values():
            p.bars += 1
        self.refresh_frames(max(2.5, 320 * self.s.bar_seconds / 86400))  # >= 320 bars for the indicators
        rows, feats_last = {}, {}
        for sym in self.universe:
            df = self.frames.get(sym)
            if df is None or len(df) < 150:
                continue
            f = F.compute(df).iloc[-1]
            rows[sym] = f[F.FEATURES].to_numpy(dtype=float)
            feats_last[sym] = f

        paused = dd >= self.s.dd_pause or (mins_left is not None and mins_left <= self.s.no_entry_before_end_min) or self.operator_paused
        cands = []
        for sym, x in rows.items():
            dfc = self.frames[sym]["close"]
            L = self.s.drop_lookback_bars
            drop = float(dfc.iloc[-1] / dfc.iloc[-1 - L] - 1) if len(dfc) > L else float("nan")
            ctx = {"drop": drop, "halted": self.halted, "paused": paused, "in_position": sym in self.positions,
                   "cooldown": self.cooldown.get(sym, 0) > now, "spread": tops.get(sym, (0, 0, None))[2], "llm": {}}
            d = evaluate(sym, x, self.model, self.s, ctx)
            scan = STR.scan(self.frames[sym], self.s)
            mt = MTF.snapshot(self.api, sym)
            self.mtf[sym] = mt
            self.j.decision({"type": "scan", "symbol": sym, "results": scan, "mtf": mt})
            if d.action != "BUY":
                # other ENABLED strategies (ENABLED_STRATEGIES) may enter when the generic risk checks pass
                hard = ctx["halted"] or ctx["paused"] or ctx["in_position"] or ctx["cooldown"] \
                    or (ctx["spread"] is not None and ctx["spread"] > self.s.max_spread)
                for r in scan:
                    if r["strategy"] != "rebound" and r["enabled"] and r["signal"] and not hard:
                        d = Decision(sym, "BUY", prob=d.prob, threshold=d.threshold, tp=r["tp"], sl=r["sl"], edge=d.edge,
                                     reasons=[f"استراتژی {r['name']}: {r['why']}", f"بک‌تست: {r['backtest']}"],
                                     drivers=d.drivers, size_mult=0.7, strategy=r["strategy"])
                        break
            if d.action == "BUY":
                cands.append(d)
            else:
                self.j.decision({"type": "signal", **d.to_dict()})

        # LLM supervisor: only when there is something to decide (token-efficient), or a rare heartbeat with open positions
        llm = {"veto": [], "risk_multiplier": 1.0}
        def snap(syms):
            out = {}
            for sy in syms:
                c = self.frames[sy]["close"]
                L = self.s.drop_lookback_bars
                out[sy] = {"drop2h": round(float(c.iloc[-1] / c.iloc[-1 - L] - 1), 4), "r4h": round(float(feats_last[sy]["r48"]), 4),
                           "atr": round(float(feats_last[sy]["atr_pct"]), 4), "spr": round(float(tops.get(sy, (0, 0, 0))[2] or 0), 4),
                           "p": round(next(c_.prob for c_ in cands if c_.symbol == sy), 2), "tf": MTF.compact(self.mtf.get(sy, {}))}
            return {"eq": round(equity), "dd": round(dd, 4), "pos": list(self.positions), "cand": out}
        if cands and self.sup.enabled:
            llm = self.sup.review(snap([c.symbol for c in cands]), "بررسی سیگنال خرید")
        elif self.sup.enabled and self.positions and now - self.last_llm > self.s.llm_every_min * 60:
            self.sup.review({"eq": round(equity), "dd": round(dd, 4), "pos": list(self.positions), "cand": {}}, "پایش دوره‌ای")
            self.last_llm = now
        kept = []
        for d in cands:
            if d.symbol in (llm.get("veto") or []):
                d.action = "HOLD"
                d.reasons = ["وتوی ناظر ریسک LLM: " + (llm.get("rationale") or "")] + d.reasons
            else:
                if self.sup.enabled:
                    d.reasons.append(f"ناظر LLM: تأیید (ضریب ریسک {llm.get('risk_multiplier', 1.0)})")
                kept.append(d)
            self.j.decision({"type": "signal", **d.to_dict()})
        cands = kept
        cands.sort(key=lambda d: -d.edge)
        free_slots = self.s.max_open_positions - len(self.positions)
        for d in cands[:max(0, free_slots)]:
            equity, cash = self.b.equity(self.prices)
            amount = position_size(equity, cash, d.sl, self.s, llm.get("risk_multiplier", 1.0) * d.size_mult)
            if amount < 200_000:
                self.j.event(f"{d.symbol}: حجم پیشنهادی {amount:,.0f} کمتر از حداقل")
                continue
            if self.s.approval_mode:  # human-in-the-loop (not allowed for the challenge account)
                ctl = self.control()
                key = f"{d.symbol}-{bar}"
                verdict = ctl.get("decisions", {}).get(key)
                if verdict != "approve":
                    if key not in [p["id"] for p in ctl.get("pending", [])]:
                        ctl.setdefault("pending", []).append({"id": key, "symbol": d.symbol, "amount": round(amount), "prob": round(d.prob, 3),
                                                              "tp": d.tp, "sl": d.sl, "reasons": d.reasons, "time": datetime.now(timezone.utc).isoformat()})
                        self.write_control(ctl)
                        self.j.event(f"سیگنال {d.symbol} منتظر تأیید اپراتور است")
                    continue
            f = self.b.buy(d.symbol, amount)
            if not f:
                continue
            entry = f.price
            self.positions[d.symbol] = Position(d.symbol, f.qty, entry, datetime.now(timezone.utc).isoformat(),
                                                entry * (1 + d.tp), entry * (1 - d.sl), f.toman, reason=f"[{d.strategy}] " + " | ".join(d.reasons), order_id=f.order_id)
            self.j.trade(symbol=d.symbol, side="BUY", qty=f.qty, price=entry, toman=round(f.toman), fee_est=round(f.fee),
                         order_id=f.order_id, reason=" | ".join(d.reasons))
            self.j.event(f"BUY {d.symbol} {f.qty} @ {entry:.0f} ({f.toman:,.0f} تومان) p={d.prob:.2f} edge={d.edge:+.3%} TP={d.tp:.2%} SL={d.sl:.2%}")
        self.save()
        return "ok"

    def run(self):
        self.select_universe()
        self.restore()
        self.j.event(f"شروع حالت {self.mode} | پایان چالش: {self.s.end_time or 'تعیین نشده'} | LLM: {'فعال' if self.sup.enabled else 'غیرفعال'}")
        while True:
            try:
                st = self.step()
                if st == "ended":
                    self.j.event("چالش به پایان رسید؛ همه دارایی‌ها تومان است. خروج.")
                    self.save()
                    return
            except KeyboardInterrupt:
                self.j.event("توقف دستی (پوزیشن‌ها باز ماندند؛ برای تسویه: python -m bot flatten)")
                self.save()
                return
            except Exception as e:  # keep running; errors are logged
                self.j.event(f"خطا: {type(e).__name__}: {e}")
                time.sleep(10)
            time.sleep(self.s.loop_seconds)
