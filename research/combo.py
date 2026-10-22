from lab import *
import sys
u5 = pd.read_csv(sys.argv[1]); u5.index = pd.to_datetime(u5.t, unit="s", utc=True)
usdt = pd.read_csv(D / "excluded_alts/USDTTMN.csv", index_col=0, parse_dates=True).close
fr5 = load("5min"); fr = load("15min")
rows = []
for s, df in fr.items():
    g = u5[u5.symbol == s.replace("TMN", "USDT")].close
    syn = (g * usdt.reindex(g.index).ffill()).resample("15min").last().reindex(df.index).ffill()
    basis = np.log(df.close / syn)
    gdrop = np.log(syn).diff(8)           # global move (USDT price x dollar) over the same 2h
    sig = np.log(df.close).diff(8) <= -0.03
    for t, r in simulate(df, sig, tp=0.03, sl=0.04, max_bars=32):
        i = df.index.get_loc(t) - 1
        rows.append(dict(t=t, s=s, r=r, basis=basis.iloc[i], gdrop=gdrop.iloc[i]))
d = pd.DataFrame(rows).dropna()
print(d.round(4).to_string())
for name, m in (("TMN at discount (basis<-0.2%)", d.basis < -0.002), ("no discount", d.basis >= -0.002),
                ("global also fell >2%", d.gdrop < -0.02), ("mostly local drop", d.gdrop >= -0.02)):
    x = d[m]; print(f"{name:32s} n={len(x)} avg={x.r.mean()*100:+.2f}% win={(x.r>0).mean():.0%}")
