"""Decision log (JSONL), trades & equity (CSV), state (JSON). These files are a deliverable for the judges."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    TRADE_COLS = ["time", "mode", "symbol", "side", "qty", "price", "toman", "fee_est", "order_id", "reason", "pnl", "position_ret"]

    def __init__(self, log_dir: Path, mode: str):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.dec = self.dir / f"decisions_{mode}.jsonl"
        self.trades = self.dir / f"trades_{mode}.csv"
        self.equity = self.dir / f"equity_{mode}.csv"
        self.state = self.dir / f"state_{mode}.json"
        self.events = self.dir / f"events_{mode}.log"

    def decision(self, d: dict):
        with self.dec.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": now_iso(), "mode": self.mode, **d}, ensure_ascii=False) + "\n")

    def trade(self, **row):
        new = not self.trades.exists()
        with self.trades.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.TRADE_COLS)
            if new:
                w.writeheader()
            w.writerow({"time": now_iso(), "mode": self.mode, **{k: row.get(k, "") for k in self.TRADE_COLS if k not in ("time", "mode")}})

    def equity_point(self, equity, cash, positions, dd):
        new = not self.equity.exists()
        with self.equity.open("a", encoding="utf-8") as f:
            if new:
                f.write("time,equity,cash,open_positions,drawdown\n")
            f.write(f"{now_iso()},{equity:.0f},{cash:.0f},{positions},{dd:.5f}\n")

    def event(self, msg: str):
        line = f"{now_iso()} {msg}"
        print(line, flush=True)
        with self.events.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_state(self, st: dict):
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.state)

    def load_state(self) -> dict:
        return json.loads(self.state.read_text(encoding="utf-8")) if self.state.exists() else {}
