from lab import *
for tf, bh in (("15min", 4), ("1h", 1)):
    fr = load(tf)
    print("=====", tf)
    def ichi(df, a=9, b=26, c=52):
        hi, lo = df.high, df.low
        tenkan = (hi.rolling(a).max() + lo.rolling(a).min()) / 2
        kijun = (hi.rolling(b).max() + lo.rolling(b).min()) / 2
        spa = ((tenkan + kijun) / 2).shift(b); spb = ((hi.rolling(c).max() + lo.rolling(c).min()) / 2).shift(b)
        top = np.maximum(spa, spb)
        cross = (tenkan > kijun) & (tenkan.shift() <= kijun.shift())
        entry = cross & (df.close > top)
        exit_ = (tenkan < kijun) | (df.close < kijun)
        return entry, exit_
    report("Ichimoku TK-cross above cloud, exit TK/kijun", fr, ichi, sl=0.04)
    def ichi_break(df, a=9, b=26, c=52):
        hi, lo = df.high, df.low
        tenkan = (hi.rolling(a).max() + lo.rolling(a).min()) / 2; kijun = (hi.rolling(b).max() + lo.rolling(b).min()) / 2
        spa = ((tenkan + kijun) / 2).shift(b); spb = ((hi.rolling(c).max() + lo.rolling(c).min()) / 2).shift(b)
        top = np.maximum(spa, spb)
        entry = (df.close > top) & (df.close.shift() <= top.shift()) & (tenkan > kijun)
        return entry, df.close < kijun
    report("Ichimoku cloud breakout, exit below kijun", fr, ichi_break, sl=0.04)
    def bb_trend(df, n=20):
        m, sd = df.close.rolling(n).mean(), df.close.rolling(n).std()
        trend = df.close.ewm(span=4*n).mean().diff(n) > 0
        return (df.close < m - 2 * sd) & trend, df.close >= m
    report("Bollinger 2σ dip in uptrend -> mean", fr, bb_trend, sl=0.03, max_bars=32 if tf=="15min" else 8)
    def supertrend_like(df, n=50):
        e = df.close.ewm(span=n).mean()
        up = (df.close > e) & (df.close.shift() <= e.shift()) & (e.diff(5) > 0)
        return up, df.close < e
    report("EMA50 trend reclaim, exit below EMA", fr, supertrend_like, sl=0.04)
