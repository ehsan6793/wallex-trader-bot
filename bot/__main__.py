"""CLI:  python -m bot <check|backtest|shadow|live|flatten|report>"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .api import WallexAPI, WallexError
from .config import ROOT, Settings


def _api(s):
    return WallexAPI(s.api_key, s.base_url)


def apply_real_fees(api, s, log=print):
    """Use the account's real taker fee for TMN markets if the API exposes it."""
    try:
        fees = api.fees()
        rates = []
        items = fees.items() if isinstance(fees, dict) else []
        for sym, v in items:
            if isinstance(v, dict) and str(sym).endswith("TMN") and v.get("takerFeeRate") is not None:
                rates.append(float(v["takerFeeRate"]))
        if rates:
            r = max(rates)
            r = r / 100 if r > 0.05 else r  # tolerate percent units
            s.taker_fee = r
            log(f"کارمزد واقعی taker بازارهای تومانی: {r:.4%}")
    except Exception as e:
        log(f"کارمزد از API خوانده نشد ({e}); مقدار پیش‌فرض {s.taker_fee:.4%}")


def cmd_check(s, a):
    api = _api(s)
    mk = api.markets()
    tmn = {k: v for k, v in mk.items() if v["quote"] == "TMN"}
    print(f"✓ بازارها: {len(mk)} کل، {len(tmn)} تومانی")
    for sym in s.symbols:
        m = tmn.get(sym)
        print(f"  {sym:10s} {'✓' if m else '✗ ناموجود'}" + (f"  qty_dec={m['qty_dec']} price_dec={m['price_dec']} min_notional={m['min_notional']:,.0f}" if m else ""))
    sym = next((x for x in s.symbols if x in tmn), None)
    if sym:
        d = api.depth(sym)
        print(f"✓ عمق {sym}: bid={d['bids'][0][0]:,.0f} ask={d['asks'][0][0]:,.0f}")
        end = int(time.time())
        c = api.candles(sym, s.resolution, end - 3600 * 6, end)
        print(f"✓ کندل ۵ دقیقه: {len(c)} عدد، آخرین {c.index.max()}")
    if s.api_key:
        b = api.balances()
        nz = {k: v for k, v in b.items() if v["total"] > 0}
        print("✓ موجودی:", json.dumps(nz, ensure_ascii=False))
        apply_real_fees(api, s)
        try:
            print("✓ سفارش‌های باز:", len(api.open_orders()))
        except WallexError as e:
            print("✗ openOrders:", e)
    else:
        print("! WALLEX_API_KEY تنظیم نشده (فقط داده عمومی تست شد)")
    print("پایان چالش:", s.end_time or "تنظیم نشده (CHALLENGE_END)")


def cmd_backtest(s, a):
    from . import backtest
    from .data import fetch
    if a.csv_dir:  # offline, fully reproducible: candles saved as <SYMBOL>.csv (see data_snapshot/)
        frames = {}
        for f in sorted(Path(a.csv_dir).glob("*TMN.csv")):
            if a.symbols and f.stem not in a.symbols.split(","):
                continue
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if s.bar_seconds > 300:  # snapshot is 5m; resample to the strategy timeframe
                df = df.resample(f"{s.bar_seconds // 60}min").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
            frames[f.stem] = df
        print("offline data:", {k: len(v) for k, v in frames.items()})
        res = backtest.run(frames, s, test_days=a.days)
        save_report(res, s, "backtest")
        return
    api = _api(s)
    mk = api.markets()
    syms = [x for x in (a.symbols.split(",") if a.symbols else s.symbols) if x in mk][: s.max_symbols]
    frames = {}
    for sym in syms:
        try:
            frames[sym] = fetch(api, sym, s, a.days + s.train_days)
            print(f"{sym}: {len(frames[sym])} کندل")
        except WallexError as e:
            print(f"{sym}: {e}")
    if s.api_key:
        apply_real_fees(api, s)
    res = backtest.run(frames, s, test_days=a.days)
    save_report(res, s, "backtest")


def save_report(res, s, name):
    out = ROOT / "reports" / f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({**res["metrics"], "settings": s.public()}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    res["trades"].to_csv(out / "trades.csv", index=False)
    res["equity"].to_csv(out / "equity.csv")
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for d in res["decisions"]:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (out / "models.json").write_text(json.dumps(res["models"], ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        res["equity"]["equity"].plot(ax=ax, color="#0f6b66")
        ax.axhline(s.start_capital, color="#999", lw=0.8, ls="--")
        ax.set_title(f"{name}: return {res['metrics'].get('return_pct')}%  maxDD {res['metrics'].get('max_drawdown_pct')}%  trades {res['metrics'].get('trades')}")
        fig.tight_layout()
        fig.savefig(out / "equity.png", dpi=120)
    except Exception as e:
        print("chart skipped:", e)
    print("گزارش ذخیره شد:", out)
    return out


def _engine(s, mode):
    from .broker import LiveBroker, PaperBroker
    from .engine import Engine
    from .journal import Journal
    api = _api(s)
    j = Journal(s.log_dir, mode)
    if s.api_key:
        apply_real_fees(api, s, j.event)
    broker = LiveBroker(api, s, j) if mode == "live" else PaperBroker(api, s, j)
    return Engine(s, api, broker, j, mode)


def cmd_shadow(s, a):
    _engine(s, "shadow").run()


def cmd_live(s, a):
    if not s.api_key:
        sys.exit("WALLEX_API_KEY لازم است")
    if not s.end_time:
        sys.exit("CHALLENGE_END را تنظیم کنید (زمان پایان رسمی چالش، مثلاً 2026-10-23T13:00:00+03:30)")
    if not a.yes:
        print("این حالت سفارش واقعی ثبت می‌کند. برای اجرا: python -m bot live --yes")
        return
    _engine(s, "live").run()


def cmd_flatten(s, a):
    """Sell the bot's markets (SYMBOLS) back to Toman. Other assets in the account are never touched."""
    from .broker import LiveBroker
    from .journal import Journal
    api = _api(s)
    j = Journal(s.log_dir, "live")
    b = LiveBroker(api, s, j)
    bals = api.balances()
    universe = set(s.symbols)
    for sym, m in b.mk.items():
        # only the bot's own markets: never touch other coins the user holds in the same account
        if sym not in universe or m["quote"] != "TMN" or m["base"] not in bals:
            continue
        q = bals[m["base"]]["free"]
        if q <= 0:
            continue
        try:
            bid, ask, _, _ = b.top(sym)
        except WallexError:
            continue
        if q * bid < max(m["min_notional"], 1):
            continue
        if not a.yes:
            print(f"(dry) would sell {q} {m['base']} ≈ {q*bid:,.0f} تومان")
            continue
        f = b.sell(sym, q)
        if f:
            j.trade(symbol=sym, side="SELL", qty=f.qty, price=f.price, toman=round(f.toman), fee_est=round(f.fee), order_id=f.order_id, reason="flatten دستی")
            print(f"sold {f.qty} {sym} @ {f.price:,.0f}")
    if not a.yes:
        print("برای اجرای واقعی: python -m bot flatten --yes")


def cmd_report(s, a):
    """Summary of a live/shadow run from its logs (for the judges)."""
    from .backtest import metrics
    mode = a.mode
    eqf, trf = s.log_dir / f"equity_{mode}.csv", s.log_dir / f"trades_{mode}.csv"
    if not eqf.exists():
        sys.exit(f"لاگی برای {mode} نیست")
    eq = pd.read_csv(eqf, parse_dates=["time"]).set_index("time")
    tr = pd.read_csv(trf) if trf.exists() else pd.DataFrame()
    sells = tr[tr["side"] == "SELL"] if len(tr) else tr
    start = float(eq["equity"].iloc[0])
    final = float(eq["equity"].iloc[-1])
    out = {"mode": mode, "start_equity": round(start), "final_equity": round(final), "return_pct": round((final / start - 1) * 100, 3),
           "max_drawdown_pct": round(float((1 - eq["equity"] / eq["equity"].cummax()).max()) * 100, 3),
           "orders": int(len(tr)), "round_trips": int(len(sells)),
           "win_rate_pct": round(float((sells["pnl"] > 0).mean() * 100), 1) if len(sells) else 0,
           "volume_toman": round(float(tr["toman"].sum())) if len(tr) else 0,
           "avg_order_toman": round(float(tr["toman"].mean())) if len(tr) else 0,
           "period": f"{eq.index[0]} → {eq.index[-1]}"}
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_dashboard(s, a):
    from .dashboard import serve
    serve(s, a.port, not a.no_browser)


def main():
    for stream in (sys.stdout, sys.stderr):  # Persian output on Windows consoles / redirected logs
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="bot", description="Starkoch × Wallex algorithmic trading bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    b = sub.add_parser("backtest")
    b.add_argument("--days", type=float, default=7)
    b.add_argument("--symbols", default="")
    b.add_argument("--csv-dir", default="", help="offline backtest from saved candles (e.g. data_snapshot)")
    sub.add_parser("shadow")
    l = sub.add_parser("live")
    l.add_argument("--yes", action="store_true")
    f = sub.add_parser("flatten")
    f.add_argument("--yes", action="store_true")
    dsh = sub.add_parser("dashboard")
    dsh.add_argument("--port", type=int, default=8765)
    dsh.add_argument("--no-browser", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--mode", default="live")
    a = p.parse_args()
    s = Settings()
    {"check": cmd_check, "backtest": cmd_backtest, "shadow": cmd_shadow, "live": cmd_live, "flatten": cmd_flatten, "report": cmd_report, "dashboard": cmd_dashboard}[a.cmd](s, a)


if __name__ == "__main__":
    main()
