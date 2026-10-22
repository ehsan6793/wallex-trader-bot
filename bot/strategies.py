"""Strategy scanner: every closed bar, every market is checked against several classic strategies.

Only strategies that passed the walk-forward backtests on real Wallex data (after the real 0.30% fee) are
allowed to trade; the others are still computed and shown on the dashboard ("monitor only") so the operator
and the judges can see what each one says and why it is not trading.

Results on 21 days of real data, 15-minute bars, majors, fee 0.30%/side (research/):
  rebound   : positive in all 3 weeks, +0.86% avg/trade, 81% win      -> TRADES
  bollinger : -0.60% avg/trade, 18% win (in uptrend filter)           -> monitor only
  ema_trend : -0.95% avg/trade, 6% win                                -> monitor only
  ichimoku  : -0.92% avg/trade, 8% win                                -> monitor only
Round 5 (research/run5.py) - four extra proposals, all negative in every week after fees -> monitor only:
  pullback (trend pullback 1h/4h) -0.78%, retest (breakout & retest) -0.98%, range -0.60% (4 trades),
  volume momentum -0.67% (15m) / -0.82% (5m).
Spot Toman markets cannot be shorted, so every strategy is long-only.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BACKTEST = {
    "rebound": "سودده در هر ۳ هفته (میانگین +۰٫۸۶٪، برد ۸۱٪)",
    "bollinger": "زیان‌ده در بک‌تست (میانگین −۰٫۶۰٪ هر معامله)",
    "ema_trend": "زیان‌ده در بک‌تست (میانگین −۰٫۹۵٪ هر معامله)",
    "ichimoku": "زیان‌ده در بک‌تست (میانگین −۰٫۹۲٪ هر معامله)",
    "pullback": "زیان‌ده در هر ۳ هفته (میانگین −۰٫۷۸٪، برد ۲۹٪)",
    "retest": "زیان‌ده در هر ۳ هفته (میانگین −۰٫۹۸٪، برد ۲۰٪)",
    "range": "زیان‌ده و بسیار کم‌سیگنال (۴ معامله در ۳ هفته، −۰٫۶۰٪)",
    "volume": "زیان‌ده در هر ۳ هفته (میانگین −۰٫۶۷٪، برد ۳۴٪)",
}
NAMES = {"rebound": "برگشت بعد از افت", "bollinger": "باند بولینگر", "ema_trend": "روند EMA", "ichimoku": "ایچیموکو",
         "pullback": "اصلاح روند صعودی", "retest": "شکست و پولبک", "range": "معامله در محدوده", "volume": "مومنتوم حجم"}
SHORT = {"rebound": "برگشت", "bollinger": "بولینگر", "ema_trend": "EMA", "ichimoku": "ایچیموکو",
         "pullback": "پولبک", "retest": "شکست", "range": "رنج", "volume": "حجم"}
RULES = {
    "rebound": "خرید وقتی قیمت در ۲ ساعت ≥۳٪ افت کرده؛ حد سود ۳٪، حد ضرر ۴٪، حداکثر ۸ ساعت",
    "bollinger": "خرید زیر باند پایین (‎−2σ‎) فقط در روند صعودی؛ هدف: میانگین",
    "ema_trend": "خرید وقتی قیمت EMA50 را رو به بالا قطع کند",
    "ichimoku": "تقاطع تنکان بالای کیجون و قیمت بالای ابر",
    "pullback": "روند ۴ساعته صعودی، برگشت قیمت ۱ساعته به EMA20 و کندل تأیید؛ هدف ۲ برابر ریسک",
    "retest": "بسته‌شدن بالای مقاومت دوبار تست‌شده با حجم بالا، سپس پولبک و حفظ سطح؛ هدف ۲R",
    "range": "محدودهٔ افقی ۲–۸٪، کندل برگشتی نزدیک کف؛ هدف نزدیک سقف (حداقل ۲R پس از کارمزد)",
    "volume": "حجم ≥۲ برابر میانگین ۲۰ کندل و شکست سقف ۲۰ کندل؛ هدف ۲R",
}


def enabled() -> set:
    return {x.strip() for x in os.environ.get("ENABLED_STRATEGIES", "rebound").split(",") if x.strip()}


def _rebound(df, s):
    c = df["close"]
    L = s.drop_lookback_bars
    drop = float(c.iloc[-1] / c.iloc[-1 - L] - 1) if len(c) > L else float("nan")
    ok = bool(np.isfinite(drop) and drop <= s.drop_trigger)
    hrs = L * s.bar_seconds / 3600
    return ok, f"افت {hrs:g} ساعته {drop:+.1%} (ماشه ≤ {s.drop_trigger:.0%})", s.tp_min, s.sl_min


def _bollinger(df, s, n=20):
    c = df["close"]
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    trend_up = c.ewm(span=4 * n).mean().diff(n).iloc[-1] > 0
    z = float((c.iloc[-1] - m.iloc[-1]) / sd.iloc[-1]) if sd.iloc[-1] else 0.0
    ok = bool(z < -2 and trend_up)
    tp = float(max(0.006, m.iloc[-1] / c.iloc[-1] - 1))
    return ok, f"فاصله از میانگین {z:+.1f}σ، روند {'صعودی' if trend_up else 'نزولی'}", tp, 0.03


def _ema_trend(df, s, n=50):
    c = df["close"]
    e = c.ewm(span=n).mean()
    ok = bool(c.iloc[-1] > e.iloc[-1] and c.iloc[-2] <= e.iloc[-2] and e.diff(5).iloc[-1] > 0)
    return ok, f"قیمت {'بالای' if c.iloc[-1] > e.iloc[-1] else 'زیر'} EMA{n}", 0.03, 0.04


def _ichimoku(df, s, a=9, b=26, c_=52):
    hi, lo, c = df["high"], df["low"], df["close"]
    tenkan = (hi.rolling(a).max() + lo.rolling(a).min()) / 2
    kijun = (hi.rolling(b).max() + lo.rolling(b).min()) / 2
    spa = ((tenkan + kijun) / 2).shift(b)
    spb = ((hi.rolling(c_).max() + lo.rolling(c_).min()) / 2).shift(b)
    top = float(np.nanmax([spa.iloc[-1], spb.iloc[-1]]))
    cross = tenkan.iloc[-1] > kijun.iloc[-1] and tenkan.iloc[-2] <= kijun.iloc[-2]
    ok = bool(cross and c.iloc[-1] > top)
    where = "بالای ابر" if c.iloc[-1] > top else "زیر/داخل ابر"
    return ok, f"تنکان {'>' if tenkan.iloc[-1] > kijun.iloc[-1] else '<'} کیجون، قیمت {where}", 0.03, 0.04


def _ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def _hourly(df):
    return df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def _pullback(df, s):
    h = _hourly(df)
    c = h["close"]
    e20, e50 = _ema(c, 20), _ema(c, 50)
    trend = bool(e20.iloc[-1] > e50.iloc[-1] and e50.diff(4).iloc[-1] > 0)
    touch = bool(h["low"].iloc[-3:].min() <= e20.iloc[-1] * 1.003)
    trig = bool(c.iloc[-1] > h["high"].iloc[-2] and c.iloc[-1] > h["open"].iloc[-1])
    sl = float(h["low"].iloc[-6:].min() * 0.997)
    risk = c.iloc[-1] / sl - 1
    ok = trend and touch and trig and 0.004 < risk < 0.04
    return ok, f"روند {'صعودی' if trend else 'نه صعودی'}، فاصله تا EMA20 {c.iloc[-1] / e20.iloc[-1] - 1:+.1%}", 2 * risk, risk


def _retest(df, s, N=48):
    c, hi, lo, v = df["close"], df["high"], df["low"], df["volume"]
    R = float(hi.iloc[-N - 13:-13].max())
    tests = int((hi.iloc[-N - 13:-13] >= R * 0.997).sum())
    broke = bool((c.iloc[-13:-1] > R).any() and (v.iloc[-13:-1] > v.rolling(20).mean().iloc[-13:-1]).any())
    hold = bool(lo.iloc[-1] <= R * 1.003 and c.iloc[-1] > R and c.iloc[-1] > df["open"].iloc[-1])
    sl = min(float(lo.iloc[-4:].min()), R * 0.99) * 0.998
    risk = c.iloc[-1] / sl - 1
    ok = tests >= 2 and broke and hold and 0.004 < risk < 0.04
    return ok, f"مقاومت ۱۲ساعته {c.iloc[-1] / R - 1:+.1%} فاصله، {'شکسته' if broke else 'نشکسته'}", 2 * risk, risk


def _range(df, s, N=48):
    c, hi, lo = df["close"], df["high"], df["low"]
    top, bot = float(hi.iloc[-N - 1:-1].max()), float(lo.iloc[-N - 1:-1].min())
    width = top / bot - 1
    flat = abs(float(_ema(c, N).pct_change(N).iloc[-1])) < 0.01
    pos = (c.iloc[-1] - bot) / (top - bot) if top > bot else 0.5
    rev = bool(c.iloc[-1] > df["open"].iloc[-1] and c.iloc[-1] > hi.iloc[-2])
    tp = (top - 0.1 * (top - bot)) / c.iloc[-1] - 1
    sl = 1 - bot * 0.995 / c.iloc[-1]
    ok = flat and 0.02 < width < 0.08 and pos <= 0.15 and rev and tp - 0.0074 >= 2 * (sl + 0.0074)
    return ok, f"{'رنج' if flat else 'روند‌دار'} {width:.1%}، جای قیمت در محدوده {pos:.0%}", max(tp, 0.0), sl


def _volume(df, s):
    c, hi, lo, v = df["close"], df["high"], df["low"], df["volume"]
    ratio = float(v.iloc[-1] / v.iloc[-21:-1].mean()) if v.iloc[-21:-1].mean() else 0.0
    brk = bool(c.iloc[-1] > hi.iloc[-21:-1].max())
    body = bool(c.iloc[-1] - df["open"].iloc[-1] > 0.5 * (hi.iloc[-1] - lo.iloc[-1]))
    sl = 1 - min(lo.iloc[-1], lo.iloc[-2]) * 0.998 / c.iloc[-1]
    ok = ratio >= 2 and brk and body and 0.003 < sl < 0.03
    return ok, f"حجم {ratio:.1f}× میانگین، {'شکست سقف' if brk else 'زیر سقف ۲۰ کندل'}", 2 * sl, sl


FUNCS = {"rebound": _rebound, "bollinger": _bollinger, "ema_trend": _ema_trend, "ichimoku": _ichimoku,
         "pullback": _pullback, "retest": _retest, "range": _range, "volume": _volume}


def scan(df: pd.DataFrame, s) -> list[dict]:
    """Evaluate every strategy on the last closed bar of one market."""
    on = enabled()
    out = []
    for key, fn in FUNCS.items():
        try:
            ok, why, tp, sl = fn(df, s)
        except Exception as e:  # not enough history etc.
            ok, why, tp, sl = False, f"داده ناکافی ({type(e).__name__})", 0.0, 0.0
        out.append({"strategy": key, "name": NAMES[key], "short": SHORT[key], "rule": RULES[key], "signal": ok, "why": why, "tp": tp, "sl": sl,
                    "enabled": key in on, "backtest": BACKTEST[key]})
    return out
