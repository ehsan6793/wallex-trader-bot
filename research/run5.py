"""Round 5: four user-proposed strategies + faster rebound variants. Real Wallex data, fee 0.30%/side + slippage.

Each trade has its own stop/target (structure based), so a small dynamic simulator is used.
Judged per week (3 separate weeks) - a strategy must be positive in every week to be allowed to trade.
"""
import numpy as np, pandas as pd
from lab import load, simulate, report, COST, MAJORS

def sim_dyn(df, sig, sl_px, tp_px, max_bars):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    es = sig.reindex(df.index).fillna(False).to_numpy()
    sl_a, tp_a = sl_px.reindex(df.index).to_numpy(), tp_px.reindex(df.index).to_numpy()
    n, i, out = len(df), 0, []
    while i < n - 2:
        if not es[i] or not np.isfinite(sl_a[i]) or not np.isfinite(tp_a[i]):
            i += 1; continue
        e, sl, tp = o[i + 1], sl_a[i], tp_a[i]
        if not (sl < e < tp):
            i += 1; continue
        j, ret = i + 1, None
        while j < n:
            if l[j] <= sl: ret = sl / e - 1; break
            if h[j] >= tp: ret = tp / e - 1; break
            if j - i >= max_bars: ret = c[j] / e - 1; break
            j += 1
        if ret is None: break
        out.append((df.index[i + 1], ret - COST)); i = j + 1
    return out

def rep(name, fr, fn, max_bars):
    allt = []
    for s, df in fr.items():
        sig, sl, tp = fn(df)
        allt += [(t, r) for t, r in sim_dyn(df, sig, sl, tp, max_bars)]
    if not allt:
        print(f"{name:52s} n=0"); return
    t = pd.DataFrame(allt, columns=["t", "r"])
    end = max(d.index.max() for d in fr.values())
    wk = lambda a, b: t[(t.t > end - pd.Timedelta(days=a)) & (t.t <= end - pd.Timedelta(days=b))].r
    f = lambda x: f"{x.sum()*100:+6.2f}%({len(x):2d})"
    print(f"{name:52s} n={len(t):3d} avg={t.r.mean()*100:+.2f}% win={(t.r>0).mean():.0%} | "
          f"wk1 {f(wk(21,14))} wk2 {f(wk(14,7))} wk3 {f(wk(7,0))}")

ema = lambda x, n: x.ewm(span=n, adjust=False).mean()

# 1) Trend pullback: 4h trend up (EMA20>EMA50 on 4h; EMA50/200 needs 33 days of 4h data we do not have),
#    1h price touches EMA20/50, bullish candle closing above previous high. SL under pullback low, TP = 2R.
def trend_pullback(df, rr=2.0):
    h4 = df.resample("4h").agg({"close": "last"}).dropna()
    up4 = (ema(h4.close, 20) > ema(h4.close, 50)).shift(1).reindex(df.index, method="ffill").fillna(False)
    c, lo, hi, op = df.close, df.low, df.high, df.open
    e20, e50 = ema(c, 20), ema(c, 50)
    touch = (lo.rolling(3).min() <= e20 * 1.003)
    trig = (c > hi.shift(1)) & (c > op) & (c > e50)
    not_stretched = (c / e20 - 1) < 0.02
    sl = lo.rolling(6).min() * 0.997
    risk = c - sl
    ok = up4 & touch & trig & not_stretched & (risk / c > 0.004) & (risk / c < 0.04)
    return ok, sl, c + rr * risk

# 2) Breakout & retest: close above a resistance tested twice, volume > avg20, then a pullback that holds the level.
def breakout_retest(df, N=48, wait=12, rr=2.0):
    c, hi, lo, v = df.close, df.high, df.low, df.volume
    R = hi.shift(1).rolling(N).max()
    tests = (hi.shift(1).rolling(N).apply(lambda a: (a >= a.max() * 0.997).sum(), raw=True) >= 2)
    brk = (c > R) & tests & (v > v.rolling(20).mean())
    level = R.where(brk).ffill(limit=wait)
    since = brk.astype(int).replace(0, np.nan).ffill(limit=wait).notna()
    retest = since & ~brk & (lo <= level * 1.003) & (c > level) & (c > df.open)
    sl = np.minimum(lo.rolling(4).min(), level * 0.99) * 0.998
    risk = c - sl
    return retest & (risk / c > 0.004) & (risk / c < 0.04), sl, c + rr * risk

# 3) Range trading: flat range (low slope), width 2-8%, bullish reversal candle near support, target near resistance
def range_trade(df, N=48):
    c, hi, lo, op = df.close, df.high, df.low, df.open
    top, bot = hi.shift(1).rolling(N).max(), lo.shift(1).rolling(N).min()
    width = top / bot - 1
    flat = (ema(c, N).pct_change(N).abs() < 0.01)
    near = (lo <= bot + 0.15 * (top - bot))
    rev = (c > op) & (c > hi.shift(1))
    sl = bot * 0.995
    tp = top - 0.1 * (top - bot)
    rr_ok = (tp / c - 1 - COST) >= 2 * (1 - sl / c + COST)
    return flat & near & rev & (width > 0.02) & (width < 0.08) & rr_ok, sl, tp

# 4) Volume momentum: volume >= 2x avg20, close above 20-bar high, strong body, not over-extended
def vol_momentum(df, rr=2.0):
    c, hi, lo, op, v = df.close, df.high, df.low, df.open, df.volume
    brk = c > hi.shift(1).rolling(20).max()
    vol = v >= 2 * v.shift(1).rolling(20).mean()
    body = (c - op) > 0.5 * (hi - lo)
    ext = (c / ema(c, 20) - 1) < 0.025
    sl = np.minimum(lo, lo.shift(1)) * 0.998
    risk = c - sl
    return brk & vol & body & ext & (risk / c > 0.003) & (risk / c < 0.03), sl, c + rr * risk

if __name__ == "__main__":
    f5, f15, f1h = load("5min"), load("15min"), load("1h")
    print("=== four proposed strategies (majors, fee 0.30%) ===")
    rep("1 trend pullback 1h (4h trend) RR2", f1h, trend_pullback, 24)
    rep("1 trend pullback 1h RR1.5", f1h, lambda d: trend_pullback(d, 1.5), 24)
    rep("1 trend pullback 15m (4h trend) RR2", f15, trend_pullback, 48)
    rep("2 breakout+retest 15m RR2", f15, breakout_retest, 48)
    rep("2 breakout+retest 1h RR2", f1h, lambda d: breakout_retest(d, 24, 6), 24)
    rep("3 range 15m", f15, range_trade, 48)
    rep("3 range 1h", f1h, lambda d: range_trade(d, 24), 24)
    rep("4 volume momentum 5m RR2", f5, vol_momentum, 48)
    rep("4 volume momentum 15m RR2", f15, vol_momentum, 32)
    rep("4 volume momentum 15m RR1.5", f15, lambda d: vol_momentum(d, 1.5), 32)

    print("\n=== rebound: faster variants ===")
    def rb(L, trig):
        return lambda d: ((d.close / d.close.shift(L) - 1) <= trig, None)
    for name, fr, L, trig, tp, sl, mb in [
        ("rebound 15m 2h drop>=3% hold<=8h (LIVE)", f15, 8, -0.03, .03, .04, 32),
        ("rebound 15m 2h drop>=3% hold<=4h", f15, 8, -0.03, .03, .04, 16),
        ("rebound 15m 2h drop>=3% TP2% hold<=4h", f15, 8, -0.03, .02, .04, 16),
        ("rebound 15m 1h drop>=2.5%", f15, 4, -0.025, .03, .04, 32),
        ("rebound 15m 1h drop>=2%", f15, 4, -0.02, .03, .04, 32),
        ("rebound 15m 2h drop>=2.5%", f15, 8, -0.025, .03, .04, 32),
        ("rebound 15m 2h drop>=2%", f15, 8, -0.02, .03, .04, 32),
        ("rebound 5m 1h drop>=3%", f5, 12, -0.03, .03, .04, 96),
        ("rebound 5m 1h drop>=2.5%", f5, 12, -0.025, .03, .04, 96),
        ("rebound 5m 30m drop>=2%", f5, 6, -0.02, .025, .04, 96),
        ("rebound 5m 2h drop>=3% hold<=4h", f5, 24, -0.03, .03, .04, 48),
    ]:
        report(name, fr, rb(L, trig), tp=tp, sl=sl, max_bars=mb)
