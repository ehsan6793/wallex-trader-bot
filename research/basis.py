"""Strategy 2 candidate: Toman price vs synthetic (COIN/USDT x USDT/TMN). Buy the Toman coin when it trades at a discount."""
from lab import *
import sys
u5 = pd.read_csv(sys.argv[1]); u5.index = pd.to_datetime(u5.t, unit="s", utc=True)
usdt = pd.read_csv(D / "USDTTMN.csv", index_col=0, parse_dates=True).close if (D / "USDTTMN.csv").exists() else pd.read_csv(D / "excluded_alts/USDTTMN.csv", index_col=0, parse_dates=True).close
fr = load("5min")
res = {}
for s, df in fr.items():
    q = s.replace("TMN", "USDT")
    g = u5[u5.symbol == q]
    if g.empty: continue
    syn = (g.close.reindex(df.index).ffill() * usdt.reindex(df.index).ffill())
    basis = np.log(df.close / syn)            # >0: Toman premium, <0: discount
    res[s] = (df, basis)
    print(f"{s}: basis mean {basis.mean()*100:+.2f}% std {basis.std()*100:.2f}% p5 {basis.quantile(.05)*100:+.2f}% p95 {basis.quantile(.95)*100:+.2f}%")
for z in (1.5, 2.0, 2.5):
    for hold in (12, 24, 48):
        allt = []
        for s, (df, b) in res.items():
            dev = b - b.rolling(288, min_periods=100).mean()   # deviation from 1-day mean basis
            sd = b.rolling(288, min_periods=100).std()
            sig = dev < -z * sd
            ex = dev > 0
            allt += [(t, r) for t, r in simulate(df, sig, sl=0.03, max_bars=hold, exit_sig=ex)]
        if not allt: continue
        t = pd.DataFrame(allt, columns=["t", "r"]); end = t.t.max()
        w = [t[(t.t > end - pd.Timedelta(days=a)) & (t.t <= end - pd.Timedelta(days=b))].r.sum() * 100 for a, b in ((21, 14), (14, 7), (7, 0))]
        print(f"basis discount z>{z} hold≤{hold*5}m: n={len(t)} avg={t.r.mean()*100:+.2f}% win={(t.r>0).mean():.0%} weeks={[round(x,2) for x in w]}")
