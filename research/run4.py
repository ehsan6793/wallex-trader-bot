from lab import *
# 1) the same rebound rule on every timeframe we can rebuild from 5m data (1m is not in the snapshot)
for tf, bh in (("5min",12),("15min",4),("30min",2),("1h",1),("4h",0.25)):
    fr = load(tf)
    L = max(1, int(2*bh)) if tf != "4h" else 1
    H = max(1, int(8*bh)) if tf != "4h" else 2
    report(f"rebound {tf}: drop>=3% in {L} bars, tp3 sl4 hold {H} bars", fr,
           lambda df, L=L: (np.log(df.close).diff(L) <= -0.03, None), tp=0.03, sl=0.04, max_bars=H)
# 2) 15m rebound + higher-timeframe context filters
fr = load("15min")
def htf(df, rule, fn):
    h = df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    return fn(h).shift(1).reindex(df.index, method="ffill")   # only CLOSED higher-TF bars (no look-ahead)
base = lambda df: np.log(df.close).diff(8) <= -0.03
tests = {
 "15m rebound (baseline)": lambda df: base(df),
 "+ 4h trend up (close>EMA20 4h)": lambda df: base(df) & htf(df,"4h",lambda h: h.close>h.close.ewm(span=20).mean()).astype(bool),
 "+ 4h trend down": lambda df: base(df) & ~htf(df,"4h",lambda h: h.close>h.close.ewm(span=20).mean()).astype(bool),
 "+ 1h RSI<35 (deep oversold)": lambda df: base(df) & htf(df,"1h",lambda h: (lambda d: 100-100/(1+d.clip(lower=0).ewm(alpha=1/14).mean()/(-d.clip(upper=0)).ewm(alpha=1/14).mean()))(h.close.diff())<35).astype(bool),
 "+ 1h drop not >6% (no crash)": lambda df: base(df) & htf(df,"1h",lambda h: np.log(h.close).diff(4)>-0.06).astype(bool),
}
for k, fn in tests.items():
    report(k, fr, lambda df, fn=fn: (fn(df), None), tp=0.03, sl=0.04, max_bars=32)
