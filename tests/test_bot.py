"""Offline tests: python -m pytest -q  (no network, no API key)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot import features as F
from bot.api import decimals_of, round_down
from bot.backtest import run as backtest
from bot.config import Settings
from bot.data import build_dataset
from bot.model import Model, train
from bot.strategy import Position, check_exit, evaluate, position_size
from tests.fake_wallex import FakeWallex, synth


@pytest.fixture
def s(tmp_path):
    st = Settings()
    st.log_dir = tmp_path / "logs"
    st.data_dir = tmp_path / "data"
    st.api_key = "test"
    st.llm_key = ""
    st.train_days = 7
    st.trade_mode = "strict"   # unit tests exercise the generic path; rebound mode has its own test
    return st


def test_precision_helpers():
    assert decimals_of(8) == 8 and decimals_of("0") == 0 and decimals_of(0.001) == 3
    assert round_down(1.23456789, 4) == 1.2345


def test_features_have_no_lookahead():
    df = synth("X", 5, 1, 1000)
    f_full = F.compute(df)
    f_cut = F.compute(df.iloc[:1000])
    pd.testing.assert_frame_equal(f_full.iloc[:1000], f_cut, check_exact=False, rtol=1e-9)


def test_label_matches_exit_rule(s):
    df = synth("X", 3, 2, 1000)
    f = F.compute(df)
    y = F.label(df, f, s)
    assert set(y.dropna().unique()) <= {0.0, 1.0}
    assert y.iloc[-s.horizon_bars:].isna().all()  # future unknown -> no label


def test_model_is_deterministic(s):
    frames = {k: synth(k, 8, i, 1000) for i, k in enumerate(["A", "B"])}
    ds = build_dataset(frames, s)
    m1, m2 = train(ds, s), train(ds, s)
    assert np.allclose(m1.w, m2.w) and m1.threshold == m2.threshold
    m3 = Model.from_json(m1.to_json())
    x = ds[F.FEATURES].dropna().to_numpy()[:5]
    assert np.allclose(m1.proba(x), m3.proba(x))


def test_sizing_caps(s):
    a = position_size(5_000_000, 5_000_000, 0.01, s)
    assert a <= 5_000_000 * s.max_position_frac + 1
    assert position_size(5_000_000, 400_000, 0.01, s) == 0  # keeps min cash


def test_exit_rules(s):
    p = Position("X", 1, 100, "t", 102, 99, 100)
    assert check_exit(p, 101, 98.5, 100, s)[1] == "حد ضرر"
    assert check_exit(p, 102.5, 99.5, 102, s)[1] == "حد سود"
    p.bars = s.time_stop_bars
    assert "time stop" in check_exit(p, 100.5, 99.5, 100, s)[1]


def test_evaluate_blocks_and_explains(s):
    frames = {k: synth(k, 8, i, 1000) for i, k in enumerate(["A", "B"])}
    m = train(build_dataset(frames, s), s)
    x = F.compute(frames["A"]).iloc[-1][F.FEATURES].to_numpy(dtype=float)
    d = evaluate("A", x, m, s, {"halted": True})
    assert d.action == "HOLD" and any("متوقف" in r for r in d.reasons)
    d2 = evaluate("A", x, m, s, {"llm": {"veto": ["A"], "rationale": "test"}})
    assert d2.action == "HOLD" and any("LLM" in r for r in d2.reasons)
    assert d2.drivers and "feature" in d2.drivers[0]


def test_backtest_runs_and_closes_everything_to_toman(s):
    frames = {k: synth(k, 12, i + 3, 1000 * (i + 1)) for i, k in enumerate(["AAATMN", "BBBTMN", "CCCTMN"])}
    res = backtest(frames, s, test_days=4, verbose=False)
    m = res["metrics"]
    assert "return_pct" in m and res["models"]
    assert m["max_drawdown_pct"] < s.dd_halt * 100 + 3
    assert set(res["trades"]["exit_reason"]).isdisjoint({""})


def test_shadow_and_live_engine_with_fake_exchange(s, monkeypatch):
    from bot.broker import LiveBroker, PaperBroker
    from bot.engine import Engine
    from bot.journal import Journal
    s.symbols = ["BTCTMN", "ETHTMN", "USDTTMN", "SOLTMN"]
    s.min_prob, s.min_edge = 0.0, -1.0       # force entries to exercise the order path
    s.trade_mode = "rebound"; s.drop_trigger = 1.0   # trigger always on
    s.resolution, s.bar_seconds = "15", 900
    for mode, Broker in (("shadow", PaperBroker), ("live", LiveBroker)):
        api = FakeWallex(days=s.train_days + 6)
        j = Journal(s.log_dir, mode)
        LiveBroker.WAIT = 0
        e = Engine(s, api, Broker(api, s, j), j, mode)
        e.select_universe()
        e.restore()
        e.model = None
        assert e.step() in ("ok", "wait")
        assert e.model is not None
        e.model.threshold = 0.0
        e.last_bar = None
        e.step()
        st_sig = [l for l in j.dec.read_text(encoding="utf-8").splitlines() if '"type": "signal"' in l]
        assert st_sig, f"{mode}: no signals evaluated (timeframe/history bug)"
        assert e.positions, f"{mode}: expected at least one position"
        st = json.loads(j.state.read_text(encoding="utf-8"))
        assert st["positions"]
        e.flatten("test")
        assert not e.positions
        if mode == "live":
            b = api.balances()
            # everything back to Toman (only sub-minimum dust from quantity rounding may remain)
            left = sum(v["total"] * api._last(k + "TMN") for k, v in b.items() if k != "TMN")
            assert left < 50_000, left
        assert j.trades.exists() and j.dec.exists()


def test_end_of_challenge_flattens(s):
    from bot.broker import PaperBroker
    from bot.engine import Engine
    from bot.journal import Journal
    api = FakeWallex(days=s.train_days + 3)
    j = Journal(s.log_dir, "shadow")
    b = PaperBroker(api, s, j)
    e = Engine(s, api, b, j, "shadow")
    e.universe = ["BTCTMN"]
    f = b.buy("BTCTMN", 1_000_000)
    e.positions["BTCTMN"] = Position("BTCTMN", f.qty, f.price, "t", f.price * 2, f.price / 2, f.toman)
    s.end_time = pd.Timestamp.now(tz="UTC").isoformat()
    assert e.step() == "ended" and not e.positions


def test_rebound_mode_trigger_and_reasons(s):
    s.trade_mode = "rebound"
    frames = {k: synth(k, 8, i, 1000) for i, k in enumerate(["A", "B"])}
    m = train(build_dataset(frames, s), s)
    x = F.compute(frames["A"]).iloc[-1][F.FEATURES].to_numpy(dtype=float)
    d = evaluate("A", x, m, s, {"drop": -0.01})
    assert d.action == "HOLD" and any("ماشه" in r for r in d.reasons)
    d = evaluate("A", x, m, s, {"drop": -0.05})
    assert d.action == "BUY" and 0.7 <= d.size_mult <= 1.3 and "برگشت" in d.reasons[0] and d.strategy == "rebound"
    d = evaluate("A", x, m, s, {"drop": -0.05, "paused": True})
    assert d.action == "HOLD"


def test_llm_called_only_with_candidates_and_can_veto(s, monkeypatch):
    """Token-efficient supervisor: no call on idle bars; on a BUY candidate it is asked once and its veto is honoured."""
    import bot.llm as L
    from bot.broker import PaperBroker
    from bot.engine import Engine
    from bot.journal import Journal
    calls = []

    class R:
        def __init__(self, veto): self.veto = veto
        def raise_for_status(self): pass
        def json(self):
            return {"usage": {"total_tokens": 210},
                    "choices": [{"message": {"content": json.dumps({"veto": self.veto, "risk": 0.8, "why": "تست"})}}]}

    def fake_post(url, **kw):
        calls.append(kw["json"])
        cand = json.loads(kw["json"]["messages"][1]["content"])["cand"]
        return R(sorted(cand)[:1])           # veto the first candidate
    monkeypatch.setattr(L.httpx, "post", fake_post)
    s.llm_key = "x"; s.symbols = ["BTCTMN", "ETHTMN", "SOLTMN"]
    s.trade_mode = "rebound"; s.resolution, s.bar_seconds = "15", 900
    api = FakeWallex(days=s.train_days + 6)
    j = Journal(s.log_dir, "shadow")
    e = Engine(s, api, PaperBroker(api, s, j), j, "shadow")
    e.select_universe(); e.restore()
    s.drop_trigger = -0.99                   # nothing triggers -> no LLM call
    e.step()
    assert calls == []
    s.drop_trigger = 1.0                     # everything triggers -> exactly one compact call
    e.last_bar = None
    e.step()
    assert len(calls) == 1 and calls[0]["max_tokens"] <= 150
    vetoed = sorted(json.loads(calls[0]["messages"][1]["content"])["cand"])[0]
    assert vetoed not in e.positions and e.positions
