"""All tunables in one place. Values come from environment / .env so the same code runs for us and for the judges."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides real environment variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


def _f(name, default):
    v = os.environ.get(name)
    return type(default)(v) if v not in (None, "") else default


def _list(name, default):
    v = os.environ.get(name)
    return [s.strip().upper() for s in v.split(",") if s.strip()] if v else list(default)


@dataclass
class Settings:
    # --- connection
    api_key: str = field(default_factory=lambda: os.environ.get("WALLEX_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("WALLEX_BASE_URL", "https://api.wallex.ir"))

    # --- universe: Toman markets only (challenge rule: P&L must be realised in Toman)
    quote: str = "TMN"
    symbols: list = field(default_factory=lambda: _list("SYMBOLS", ["BTCTMN", "ETHTMN", "BNBTMN", "SOLTMN", "XRPTMN", "DOGETMN", "TRXTMN", "XAUTTMN"]))
    max_symbols: int = field(default_factory=lambda: _f("MAX_SYMBOLS", 8))

    # --- data / model
    resolution: str = field(default_factory=lambda: os.environ.get("TIMEFRAME", "15"))  # minutes; 15m chosen by multi-timeframe study
    bar_seconds: int = field(default_factory=lambda: int(os.environ.get("TIMEFRAME", "15")) * 60)
    train_days: int = field(default_factory=lambda: _f("TRAIN_DAYS", 7))
    retrain_hours: float = field(default_factory=lambda: _f("RETRAIN_HOURS", 6.0))
    horizon_bars: int = field(default_factory=lambda: _f("HORIZON_BARS", 32))   # 8 hours (15m bars) max holding for the label

    # --- exits (multiples of ATR%, clamped)
    tp_atr: float = field(default_factory=lambda: _f("TP_ATR", 0.0))
    sl_atr: float = field(default_factory=lambda: _f("SL_ATR", 0.0))
    tp_min: float = field(default_factory=lambda: _f("TP_MIN", 0.030))
    tp_max: float = 0.060
    sl_min: float = field(default_factory=lambda: _f("SL_MIN", 0.040))
    sl_max: float = 0.060
    time_stop_bars: int = field(default_factory=lambda: _f("TIME_STOP_BARS", 32))  # 8h on 15m bars
    breakeven_after: float = 0.020  # move stop to break-even (+fees) after +2%

    # --- costs (overwritten by real values from /v1/account/fee when available)
    taker_fee: float = field(default_factory=lambda: _f("TAKER_FEE", 0.0030))
    slippage: float = field(default_factory=lambda: _f("SLIPPAGE", 0.0007))
    max_spread: float = field(default_factory=lambda: _f("MAX_SPREAD", 0.004))

    # --- signal
    # strict: trade only if out-of-sample validation shows positive return AFTER costs
    # active: also trade the model's top-decile signals when their validated GROSS return is positive (smaller size)
    trade_mode: str = field(default_factory=lambda: os.environ.get("TRADE_MODE", "rebound"))
    # rebound (default): buy capitulation drops in Toman markets (panic-sell overshoot), validated on 21 days of real Wallex data
    drop_trigger: float = field(default_factory=lambda: _f("DROP_TRIGGER", -0.03))      # fall of >=3% ...
    drop_lookback_bars: int = field(default_factory=lambda: _f("DROP_LOOKBACK_BARS", 8))  # ... within 8 bars (2h on 15m)
    approval_mode: bool = field(default_factory=lambda: os.environ.get("APPROVAL_MODE", "0") == "1")  # human approves signals (NOT for the challenge)
    signal_exit: bool = field(default_factory=lambda: os.environ.get("SIGNAL_EXIT", "0") == "1")  # exit early when model prob collapses
    active_size_mult: float = field(default_factory=lambda: _f("ACTIVE_SIZE_MULT", 0.6))
    min_prob: float = field(default_factory=lambda: _f("MIN_PROB", 0.55))
    min_edge: float = field(default_factory=lambda: _f("MIN_EDGE", 0.0020))  # expected value after costs per trade

    # --- risk & money management
    risk_per_trade: float = field(default_factory=lambda: _f("RISK_PER_TRADE", 0.008))  # fraction of equity lost if SL hits
    max_position_frac: float = field(default_factory=lambda: _f("MAX_POSITION_FRAC", 0.30))
    max_open_positions: int = field(default_factory=lambda: _f("MAX_OPEN_POSITIONS", 3))
    min_cash_frac: float = 0.10
    dd_pause: float = field(default_factory=lambda: _f("DD_PAUSE", 0.04))     # stop new entries
    dd_halt: float = field(default_factory=lambda: _f("DD_HALT", 0.06))       # flatten and stop
    cooldown_bars: int = 4                                                     # after a loss on a symbol
    start_capital: float = field(default_factory=lambda: _f("START_CAPITAL", 5_000_000.0))

    # --- challenge clock
    end_time: str = field(default_factory=lambda: os.environ.get("CHALLENGE_END", ""))  # ISO, e.g. 2026-10-23T13:00:00+03:30
    no_entry_before_end_min: int = 90
    flatten_before_end_min: int = 20

    # --- loop
    loop_seconds: int = field(default_factory=lambda: _f("LOOP_SECONDS", 30))

    # --- optional LLM risk supervisor (OpenAI-compatible)
    llm_base: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.metisai.ir/openai/v1"))
    llm_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o"))
    llm_every_min: int = field(default_factory=lambda: _f("LLM_HEARTBEAT_MIN", 180))   # idle heartbeat only when positions are open
    llm_max_calls_day: int = field(default_factory=lambda: _f("LLM_MAX_CALLS_PER_DAY", 30))
    llm_max_tokens: int = 120

    log_dir: Path = field(default_factory=lambda: Path(os.environ.get("LOG_DIR", ROOT / "logs")))
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", ROOT / "data")))

    def end_dt(self):
        if not self.end_time:
            return None
        d = datetime.fromisoformat(self.end_time)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

    def round_trip_cost(self):
        return 2 * (self.taker_fee + self.slippage)

    def public(self) -> dict:
        """Settings without secrets (written into every log/report for reproducibility)."""
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        d["api_key"] = "***" if self.api_key else ""
        d["llm_key"] = "***" if self.llm_key else ""
        return d
