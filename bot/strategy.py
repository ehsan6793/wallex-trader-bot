"""Decision logic shared by backtest / shadow / live: entry rules, sizing, exit rules.

Every function returns a reason so each trade can be explained (challenge requirement: decision log).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .features import FEATURES, exit_levels
from .model import Model, expected_edge


@dataclass
class Decision:
    symbol: str
    action: str                 # BUY | HOLD
    prob: float = 0.0
    threshold: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    edge: float = 0.0
    reasons: list = field(default_factory=list)
    drivers: list = field(default_factory=list)   # top model feature contributions
    size_mult: float = 1.0
    strategy: str = "ml"

    def to_dict(self):
        return asdict(self)


def evaluate(symbol: str, x: np.ndarray, model: Model, s, ctx: dict) -> Decision:
    """x: feature vector (FEATURES order) at the last *closed* bar. ctx: spread, llm, cooldown, risk state."""
    d = Decision(symbol, "HOLD", threshold=model.threshold)
    if np.isnan(x).sum() > 2:
        d.reasons.append("داده ناکافی برای اندیکاتورها")
        return d
    fx = dict(zip(FEATURES, x))
    atr = float(fx["atr_pct"])
    d.tp, d.sl = exit_levels(atr, s)
    d.prob = float(model.proba(x.reshape(1, -1))[0])
    d.edge = expected_edge(d.prob, d.tp, d.sl, s.round_trip_cost(), model.meta.get("loss_frac", 1.0))
    d.drivers = model.explain(x)

    blocks = []
    if ctx.get("halted"):
        blocks.append("ربات متوقف است (حد افت سرمایه)")
    if ctx.get("paused"):
        blocks.append("ورود جدید متوقف (افت سرمایه بالای حد هشدار یا نزدیک پایان چالش)")
    if ctx.get("in_position"):
        blocks.append("پوزیشن باز روی همین نماد وجود دارد")
    if ctx.get("cooldown"):
        blocks.append("دوره استراحت بعد از ضرر روی همین نماد")
    rebound = s.trade_mode == "rebound"
    if atr > (0.06 if rebound else 0.03):
        blocks.append(f"نوسان بیش از حد (ATR={atr:.2%})")
    if rebound:
        drop = ctx.get("drop")
        if drop is None or not np.isfinite(drop) or drop > s.drop_trigger:
            hrs = s.drop_lookback_bars * s.bar_seconds / 3600
            blocks.append(f"ماشه فعال نیست: افت {hrs:g} ساعت اخیر {(drop or 0):+.1%} (لازم ≤ {s.drop_trigger:.0%})")
    elif fx["r12"] < -0.03:
        blocks.append("ریزش شدید در یک ساعت اخیر (چاقوی در حال سقوط)")
    spread = ctx.get("spread")
    if spread is not None and spread > s.max_spread:
        blocks.append(f"اسپرد زیاد ({spread:.2%})")
    llm = ctx.get("llm") or {}
    if symbol in llm.get("veto", []):
        blocks.append("وتوی ناظر ریسک LLM: " + llm.get("rationale", "")[:120])
    active = model.meta.get("mode") == "active"
    if not rebound:
        if d.prob < model.threshold:
            blocks.append(f"احتمال موفقیت {d.prob:.0%} کمتر از آستانه {model.threshold:.0%}")
        if not active and d.edge < s.min_edge:
            blocks.append(f"امید ریاضی بعد از کارمزد {d.edge:+.2%} کافی نیست")
    if blocks:
        d.reasons = blocks
        return d
    d.action = "BUY"
    if rebound:
        base = model.meta.get("pos_rate", 0.3)
        d.size_mult = float(np.clip(d.prob / max(base, 1e-3), 0.7, 1.3))
        hrs = s.drop_lookback_bars * s.bar_seconds / 3600
        d.strategy = "rebound"
        d.reasons = [f"استراتژی برگشت: افت {ctx.get('drop'):+.1%} در {hrs:g} ساعت اخیر (ماشه ≤ {s.drop_trigger:.0%})",
                     f"مدل ML احتمال رسیدن به +{d.tp:.0%} پیش از −{d.sl:.0%} را {d.prob:.0%} می‌بیند (نرخ پایه {base:.0%}) → ضریب حجم {d.size_mult:.2f}",
                     "عوامل اصلی مدل: " + "، ".join(f"{c['feature']}({c['logit']:+.2f})" for c in d.drivers[:3])]
        return d
    d.size_mult = s.active_size_mult if active else 1.0
    d.reasons = [("حالت فعال: سیگنال در دهک بالای احتمال مدل (بازده ناخالص اعتبارسنجی مثبت) با حجم کاهش‌یافته" if active else "حالت سخت‌گیرانه: لبهٔ خالص بعد از کارمزد در اعتبارسنجی تأیید شده"),f"مدل: احتمال رسیدن به سود {d.tp:.2%} قبل از ضرر {d.sl:.2%} = {d.prob:.0%} (آستانه {model.threshold:.0%})",
                 f"امید ریاضی بعد از کارمزد و لغزش: {d.edge:+.2%}",
                 "عوامل اصلی: " + "، ".join(f"{c['feature']}({c['logit']:+.2f})" for c in d.drivers[:3])]
    return d


def position_size(equity: float, cash: float, sl: float, s, risk_mult: float = 1.0) -> float:
    """Toman amount to spend: risk a fixed fraction of equity if the stop is hit, with hard caps."""
    by_risk = equity * s.risk_per_trade * risk_mult / max(sl, 1e-4)
    cap = equity * s.max_position_frac
    spendable = cash - equity * s.min_cash_frac
    return max(0.0, min(by_risk, cap, spendable))


@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    entry_time: str
    tp_price: float
    sl_price: float
    cost: float                 # toman spent incl. fee
    bars: int = 0
    be_moved: bool = False
    reason: str = ""
    order_id: str = ""


def check_exit(p: Position, high: float, low: float, last: float, s, prob_now: float | None = None, threshold: float | None = None):
    """Returns (exit_price, reason) or None. In backtest high/low are the bar's; live passes the current bid for all three."""
    if low <= p.sl_price:
        return p.sl_price, ("حد ضرر" if not p.be_moved else "حد سود ثابت‌شده (break-even)")
    if high >= p.tp_price:
        return p.tp_price, "حد سود"
    if p.bars >= s.time_stop_bars:
        return last, "پایان زمان نگهداری (time stop)"
    if s.signal_exit and prob_now is not None and threshold is not None and prob_now < 0.40 and last > p.entry * (1 + s.round_trip_cost()):
        return last, "افت سیگنال مدل؛ خروج با سود"
    return None


def trail(p: Position, high: float, s):
    """Move stop to break-even (+costs) once price ran +breakeven_after."""
    if not p.be_moved and high >= p.entry * (1 + s.breakeven_after):
        p.sl_price = max(p.sl_price, p.entry * (1 + s.round_trip_cost() / 2 + 0.001))
        p.be_moved = True
