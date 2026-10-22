"""LLM risk supervisor (agentic layer), built to spend as few tokens as possible.

* Event-driven: it is called only when the rules produce a BUY candidate (review that candidate before money moves),
  plus a rare heartbeat while positions are open — never on idle bars.
* Compact prompt: one-line-per-symbol numeric snapshot, short keys, JSON-only answer capped at `llm_max_tokens`.
* Daily call cap (LLM_MAX_CALLS_PER_DAY); token usage is logged and shown on the dashboard.
* It can only VETO a candidate or SCALE DOWN risk; it can never open a trade or raise risk. Any failure = no veto.
"""
from __future__ import annotations

import json
import time

import httpx

PROMPT = ("Risk supervisor for a long-only spot bot on Wallex Toman markets. A rule (15m rebound after a >=3% drop) "
          "proposed buys. You may only reduce risk. Veto a symbol if its data shows a disorderly crash (large 4h/1h drop "
          "still accelerating, all timeframes down with falling RSI, wide spread) or portfolio drawdown is growing. "
          'Reply JSON only: {"veto":[symbols],"risk":0.3-1.0,"why":"<=20 words Persian"}')


class Supervisor:
    def __init__(self, s, journal):
        self.s, self.j = s, journal
        self.enabled = bool(s.llm_key)
        self.state = {"veto": [], "risk_multiplier": 1.0, "rationale": "LLM غیرفعال" if not self.enabled else "هنوز فراخوانی نشده"}
        self.day, self.calls, self.tokens = time.strftime("%Y-%m-%d"), 0, 0

    def _budget_ok(self) -> bool:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.calls = today, 0
        return self.calls < self.s.llm_max_calls_day

    def review(self, snapshot: dict, reason: str) -> dict:
        if not self.enabled:
            return self.state
        if not self._budget_ok():
            self.state = {"veto": [], "risk_multiplier": 1.0, "rationale": "سقف روزانهٔ فراخوانی LLM پر شده؛ قواعد ریسک ثابت فعال است"}
            return self.state
        text = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        try:
            r = httpx.post(f"{self.s.llm_base.rstrip('/')}/chat/completions", timeout=30,
                           headers={"Authorization": f"Bearer {self.s.llm_key}"},
                           json={"model": self.s.llm_model, "temperature": 0, "max_tokens": self.s.llm_max_tokens,
                                 "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]})
            r.raise_for_status()
            data = r.json()
            self.calls += 1
            used = (data.get("usage") or {}).get("total_tokens", 0)
            self.tokens += used
            out = json.loads(data["choices"][0]["message"]["content"])
            syms = set((snapshot.get("cand") or {}).keys())
            veto = [v for v in out.get("veto", []) if v in syms]
            mult = float(min(1.0, max(0.3, float(out.get("risk", 1.0)))))
            self.state = {"veto": veto, "risk_multiplier": mult, "rationale": str(out.get("why", ""))[:200]}
        except Exception as e:  # LLM problems must never block or force trading
            used = 0
            self.state = {"veto": [], "risk_multiplier": 1.0, "rationale": f"خطای LLM ({type(e).__name__})؛ بدون وتو"}
        self.j.decision({"type": "llm_review", "reason": reason, "snapshot": snapshot, "verdict": self.state,
                         "tokens": used, "calls_today": self.calls, "tokens_total": self.tokens})
        self.j.event(f"ناظر LLM ({reason}): ضریب ریسک {self.state['risk_multiplier']}"
                     + (f"، وتو {','.join(self.state['veto'])}" if self.state["veto"] else "") + f" — {self.state['rationale']} [{used} توکن]")
        return self.state
