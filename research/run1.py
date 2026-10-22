from lab import *
import numpy as np
none = lambda df: None
for tf, bars_h in (("5min", 12), ("15min", 4), ("1h", 1)):
    fr = load(tf)
    H = lambda hours: max(1, int(hours * bars_h))
    print(f"===== timeframe {tf}")
    # S1 rebound
    for drop, look in ((-0.04, 4), (-0.03, 2), (-0.05, 4)):
        report(f"S1 rebound drop{drop:.0%}/{look}h tp2 sl4 4h", fr, lambda df, d=drop, lk=look: (np.log(df.close).diff(H(lk)) <= d, None), tp=0.02, sl=0.04, max_bars=H(4))
    # S2 Donchian breakout 24h with trailing stop
    report("S2 breakout 24h-high trail3% sl3%", fr, lambda df: (df.close > df.high.rolling(H(24)).max().shift(1), None), sl=0.03, trail=0.03, max_bars=H(24))
    # S3 EMA trend cross (fast 12h-ish)
    def ema_x(df, f=H(3), s=H(12)):
        ef, es = df.close.ewm(span=f).mean(), df.close.ewm(span=s).mean()
        up = (ef > es) & (ef.shift() <= es.shift()); dn = (ef < es) & (ef.shift() >= es.shift())
        return up, dn
    report("S3 EMA cross 3h/12h, exit on cross down, sl4%", fr, ema_x, sl=0.04)
    # S4 Bollinger extreme mean reversion
    def bb(df, n=H(5)):
        m, sd = df.close.rolling(n).mean(), df.close.rolling(n).std()
        return (df.close < m - 2.5 * sd), (df.close >= m)
    report("S4 BB 2.5σ below -> back to mean, sl3%, 8h", fr, bb, sl=0.03, max_bars=H(8))
    # S5 RSI oversold + uptrend filter (buy the dip in uptrend)
    def dip_up(df):
        d = df.close.diff(); up = d.clip(lower=0).ewm(alpha=1/14).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14).mean()
        rsi = 100 - 100 / (1 + up / dn)
        trend = df.close > df.close.ewm(span=H(24)).mean()
        return (rsi < 25) & trend, rsi > 55
    report("S5 RSI<25 in 24h uptrend, exit RSI>55, sl3%", fr, dip_up, sl=0.03, max_bars=H(12))
