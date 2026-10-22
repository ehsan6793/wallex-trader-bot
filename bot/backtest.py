"""Walk-forward, event-driven backtest using exactly the live decision code (strategy.py + model.py)."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import features as F
from .data import build_dataset
from .model import train
from .strategy import Position, check_exit, evaluate, position_size, trail


def run(frames: dict, s, retrain_every_h: float = 24.0, test_days: float | None = None, verbose=True) -> dict:
    """frames: symbol -> OHLCV (5m). Trains on the first `train_days`, then trades the rest with periodic retraining."""
    feats = {sym: F.compute(df) for sym, df in frames.items()}
    idx = sorted(set().union(*[df.index for df in frames.values()]))
    t_start = idx[0] + pd.Timedelta(days=s.train_days)
    if test_days:
        t_start = max(t_start, idx[-1] - pd.Timedelta(days=test_days))
    if t_start >= idx[-1]:
        raise ValueError("دادهٔ کافی برای بک‌تست نیست (حداقل train_days + چند روز لازم است)")

    cash = s.start_capital
    peak = cash
    positions: dict[str, Position] = {}
    pending: dict[str, tuple] = {}
    cooldown: dict[str, pd.Timestamp] = {}
    trades, equity_curve, decisions = [], [], []
    model, last_train = None, None
    halted = False
    fee, slip = s.taker_fee, s.slippage
    models_meta = []

    pos_in_frame = {sym: {t: i for i, t in enumerate(df.index)} for sym, df in frames.items()}

    for t in idx:
        if t < t_start:
            continue
        # ---------------- retrain (walk-forward, no look-ahead)
        if model is None or (t - last_train) >= pd.Timedelta(hours=retrain_every_h):
            win = {sym: df[(df.index <= t) & (df.index > t - pd.Timedelta(days=s.train_days))] for sym, df in frames.items()}
            ds = build_dataset(win, s, until=t)
            try:
                model = train(ds, s)
                last_train = t
                models_meta.append({"time": str(t), **model.meta, "threshold": model.threshold})
            except ValueError:
                if model is None:
                    continue

        for sym, df in frames.items():
            i = pos_in_frame[sym].get(t)
            if i is None:
                continue
            bar = df.iloc[i]
            # ---- fill pending entry at this bar's open
            if sym in pending and not halted:
                dec, amount = pending.pop(sym)
                price = bar["open"] * (1 + slip)
                spend = min(amount, cash)
                if spend > 1000:
                    qty = spend * (1 - fee) / price
                    cash -= spend
                    positions[sym] = Position(sym, qty, price, str(t), price * (1 + dec.tp), price * (1 - dec.sl), spend,
                                              reason=" | ".join(dec.reasons))
            # ---- manage open position on this bar
            p = positions.get(sym)
            if p:
                p.bars += 1
                x = feats[sym].iloc[i][F.FEATURES].to_numpy(dtype=float)
                prob_now = float(model.proba(x.reshape(1, -1))[0])
                ex = check_exit(p, bar["high"], bar["low"], bar["close"], s, prob_now, model.threshold)
                if ex is None:
                    trail(p, bar["high"], s)
                if ex:
                    px, why = ex
                    if why == "حد سود":
                        px = max(px, bar["open"]) if bar["open"] > px else px
                    elif px == p.sl_price and bar["open"] < px:
                        px = bar["open"]  # gapped through the stop
                    gross = p.qty * px * (1 - slip)
                    net = gross * (1 - fee)
                    cash += net
                    trades.append({"symbol": sym, "entry_time": p.entry_time, "exit_time": str(t), "entry": p.entry,
                                   "exit": px, "qty": p.qty, "cost": p.cost, "proceeds": net, "pnl": net - p.cost,
                                   "ret": net / p.cost - 1, "bars": p.bars, "exit_reason": why, "entry_reason": p.reason})
                    if net < p.cost:
                        cooldown[sym] = t + pd.Timedelta(seconds=s.bar_seconds * s.cooldown_bars)
                    del positions[sym]

        # ---------------- mark to market + risk state
        mtm = 0.0
        for sym, p in positions.items():
            i = pos_in_frame[sym].get(t)
            px = frames[sym].iloc[i]["close"] if i is not None else p.entry
            mtm += p.qty * px
        equity = cash + mtm
        peak = max(peak, equity)
        dd = 1 - equity / peak
        equity_curve.append((t, equity, cash, len(positions)))
        if dd >= s.dd_halt and not halted:
            halted = True  # flatten at next bars via time-stop-like exit
            for sym, p in list(positions.items()):
                i = pos_in_frame[sym].get(t)
                px = frames[sym].iloc[i]["close"] if i is not None else p.entry
                net = p.qty * px * (1 - slip) * (1 - fee)
                cash += net
                trades.append({"symbol": sym, "entry_time": p.entry_time, "exit_time": str(t), "entry": p.entry, "exit": px,
                               "qty": p.qty, "cost": p.cost, "proceeds": net, "pnl": net - p.cost, "ret": net / p.cost - 1,
                               "bars": p.bars, "exit_reason": "توقف کامل: حد افت سرمایه", "entry_reason": p.reason})
                del positions[sym]
        paused = dd >= s.dd_pause

        # ---------------- new entries at bar close -> fill next bar open
        if halted or len(positions) + len(pending) >= s.max_open_positions:
            continue
        cands = []
        for sym, df in frames.items():
            i = pos_in_frame[sym].get(t)
            if i is None or sym in positions or sym in pending:
                continue
            x = feats[sym].iloc[i][F.FEATURES].to_numpy(dtype=float)
            L = s.drop_lookback_bars
            drop = float(df["close"].iloc[i] / df["close"].iloc[i - L] - 1) if i >= L else float("nan")
            ctx = {"paused": paused, "cooldown": cooldown.get(sym, t) > t, "drop": drop}
            dec = evaluate(sym, x, model, s, ctx)
            if dec.action == "BUY":
                cands.append(dec)
        cands.sort(key=lambda d: -d.edge)
        for dec in cands[: s.max_open_positions - len(positions) - len(pending)]:
            amount = position_size(equity, cash - sum(a for _, a in pending.values()), dec.sl, s, dec.size_mult)
            if amount >= 200_000:
                pending[dec.symbol] = (dec, amount)
                decisions.append({"time": str(t), **dec.to_dict(), "amount": round(amount)})

    # close anything left at the last close (challenge: result must be in Toman)
    for sym, p in list(positions.items()):
        px = frames[sym]["close"].iloc[-1]
        net = p.qty * px * (1 - slip) * (1 - fee)
        cash += net
        trades.append({"symbol": sym, "entry_time": p.entry_time, "exit_time": str(frames[sym].index[-1]), "entry": p.entry,
                       "exit": px, "qty": p.qty, "cost": p.cost, "proceeds": net, "pnl": net - p.cost, "ret": net / p.cost - 1,
                       "bars": p.bars, "exit_reason": "تسویه پایان دوره به تومان", "entry_reason": p.reason})
    eq = pd.DataFrame(equity_curve, columns=["time", "equity", "cash", "open_positions"]).set_index("time")
    tr = pd.DataFrame(trades, columns=["symbol", "entry_time", "exit_time", "entry", "exit", "qty", "cost", "proceeds", "pnl", "ret", "bars", "exit_reason", "entry_reason"])
    res = {"metrics": metrics(eq, tr, s, frames, t_start), "trades": tr, "equity": eq, "decisions": decisions, "models": models_meta}
    if verbose:
        print(json.dumps(res["metrics"], ensure_ascii=False, indent=1))
    return res


def metrics(eq: pd.DataFrame, tr: pd.DataFrame, s, frames=None, t_start=None) -> dict:
    if eq.empty:
        return {}
    e = eq["equity"]
    final = float(e.iloc[-1]) if tr.empty else float(eq["cash"].iloc[-1] + 0)  # all positions closed at the end
    if not tr.empty:
        final = s.start_capital + float(tr["pnl"].sum())
    dd = float((1 - e / e.cummax()).max())
    out = {
        "start_capital": s.start_capital, "final_equity": round(final), "return_pct": round((final / s.start_capital - 1) * 100, 3),
        "max_drawdown_pct": round(dd * 100, 3), "trades": int(len(tr)),
        "win_rate_pct": round(float((tr["pnl"] > 0).mean() * 100), 1) if len(tr) else 0,
        "avg_trade_ret_pct": round(float(tr["ret"].mean() * 100), 3) if len(tr) else 0,
        "profit_factor": round(float(tr.loc[tr.pnl > 0, "pnl"].sum() / max(1, -tr.loc[tr.pnl < 0, "pnl"].sum())), 2) if len(tr) else 0,
        "volume_toman": round(float((tr["cost"] + tr["proceeds"]).sum())) if len(tr) else 0,
        "avg_trade_volume_toman": round(float(tr["cost"].mean())) if len(tr) else 0,
        "exposure_pct": round(float((eq["open_positions"] > 0).mean() * 100), 1),
        "period": f"{eq.index[0]} → {eq.index[-1]}",
    }
    r = e.pct_change().dropna()
    if len(r) > 10 and r.std() > 0:
        out["sharpe_annual"] = round(float(r.mean() / r.std() * np.sqrt(365 * 288)), 2)
    if frames and t_start is not None:  # benchmark: equal-weight buy & hold of the same universe
        bh = []
        for df in frames.values():
            d = df[df.index >= t_start]
            if len(d) > 1:
                bh.append(d["close"].iloc[-1] / d["open"].iloc[0] - 1)
        if bh:
            out["buy_hold_benchmark_pct"] = round(float(np.mean(bh)) * 100, 3)
    return out
