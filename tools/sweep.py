"""Robustness sweep: every config is judged on TWO separate walk-forward test weeks (no single-period overfit)."""
import itertools, json, sys, tempfile, pathlib
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bot.config import Settings
from bot.backtest import run

import os
SYMS = os.environ.get("SYMS","").split(",") if os.environ.get("SYMS") else [p.stem for p in sorted(pathlib.Path("data_snapshot").glob("*TMN.csv"))]
FR = {s: pd.read_csv(f"data_snapshot/{s}.csv", index_col=0, parse_dates=True) for s in SYMS}
END = max(df.index.max() for df in FR.values())

def one(args):
    cfg, window = args
    s = Settings(); s.log_dir = s.data_dir = pathlib.Path(tempfile.mkdtemp()); s.train_days = 7
    for k, v in cfg.items(): setattr(s, k, v)
    cut = END - pd.Timedelta(days=7) if window == "A" else END
    frames = {k: v[v.index <= cut] for k, v in FR.items()}
    m = run(frames, s, test_days=7, verbose=False)["metrics"]
    return cfg, window, m

grid = []
for d4, d2, tp, sl, sx in itertools.product([-0.04, -0.05], [-0.03], [0.02, 0.03], [0.04], [True, False]):
    grid.append(dict(drop_4h=d4, drop_2h=d2, tp_min=tp, sl_min=sl, signal_exit=sx))
jobs = [(c, w) for c in grid for w in ("A", "B")]
with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
    res = list(ex.map(one, jobs))
rows = {}
for cfg, w, m in res:
    k = json.dumps(cfg)
    rows.setdefault(k, {})[w] = m
out = []
for k, r in rows.items():
    a, b = r["A"], r["B"]
    out.append({**json.loads(k), "retA": a.get("return_pct", 0), "retB": b.get("return_pct", 0), "ddA": a.get("max_drawdown_pct", 0),
                "ddB": b.get("max_drawdown_pct", 0), "nA": a.get("trades", 0), "nB": b.get("trades", 0),
                "winA": a.get("win_rate_pct", 0), "winB": b.get("win_rate_pct", 0)})
df = pd.DataFrame(out)
df["worst"] = df[["retA", "retB"]].min(axis=1)
df["sum"] = df.retA + df.retB
print(df.sort_values(["worst", "sum"], ascending=False).to_string(index=False))
