"""Strategy lab: event-driven simulation per symbol with realistic costs, judged on separate weeks."""
import numpy as np, pandas as pd, pathlib
COST = 2 * (0.0030 + 0.0007)          # real taker fee 0.30% + slippage, both sides
D = pathlib.Path(__file__).resolve().parents[1] / "data_snapshot"
MAJORS = ["BTCTMN", "ETHTMN", "BNBTMN", "SOLTMN", "XRPTMN", "DOGETMN", "TRXTMN", "XAUTTMN"]
ALTS = ["ARBTMN", "ZECTMN", "PEPETMN"]

def load(tf="5min", syms=MAJORS):
    out = {}
    for s in syms:
        p = D / f"{s}.csv"
        if not p.exists(): p = D / "excluded_alts" / f"{s}.csv"
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if tf != "5min":
            df = df.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        out[s] = df
    return out

def simulate(df, entry_sig, tp=None, sl=None, max_bars=None, exit_sig=None, trail=None):
    """entry_sig/exit_sig: boolean Series at bar close. Enter next open. Returns list of (entry_time, net_ret)."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    es = entry_sig.reindex(df.index).fillna(False).to_numpy()
    xs = exit_sig.reindex(df.index).fillna(False).to_numpy() if exit_sig is not None else None
    n, i, trades = len(df), 0, []
    while i < n - 2:
        if not es[i]:
            i += 1; continue
        e = o[i + 1]; peak = e; j = i + 1; ret = None
        while j < n:
            if sl is not None and l[j] <= e * (1 - sl): ret = -sl; break
            if trail is not None and l[j] <= peak * (1 - trail) and peak > e: ret = peak * (1 - trail) / e - 1; break
            if tp is not None and h[j] >= e * (1 + tp): ret = tp; break
            peak = max(peak, h[j])
            if xs is not None and xs[j]: ret = c[j] / e - 1; break
            if max_bars is not None and j - (i + 1) + 1 >= max_bars: ret = c[j] / e - 1; break
            j += 1
        if ret is None: break
        trades.append((df.index[i + 1], ret - COST))
        i = j + 1
    return trades

def report(name, fr, fn, **kw):
    allt = []
    for s, df in fr.items():
        sig, xsig = fn(df)
        allt += [(t, r, s) for t, r in simulate(df, sig, exit_sig=xsig, **kw)]
    if not allt:
        print(f"{name:52s} n=0"); return
    t = pd.DataFrame(allt, columns=["t", "r", "s"])
    end = max(df.index.max() for df in fr.values())
    wk = lambda a, b: t[(t.t > end - pd.Timedelta(days=a)) & (t.t <= end - pd.Timedelta(days=b))].r
    w3, w2, w1 = wk(21, 14), wk(14, 7), wk(7, 0)
    f = lambda x: f"{x.sum()*100:+6.2f}%({len(x):2d})"
    print(f"{name:52s} n={len(t):3d} avg={t.r.mean()*100:+.2f}% win={(t.r>0).mean():.0%} | wk1 {f(w3)} wk2 {f(w2)} wk3 {f(w1)}")
