"""AI core: regularised logistic regression (pure numpy, deterministic) trained walk-forward on triple-barrier labels.

Why this model: it is fast to retrain inside the 3-day window, fully reproducible (no randomness), and every
prediction can be explained by per-feature contributions — which is what the judges asked for.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURES


@dataclass
class Model:
    w: np.ndarray = None
    b: float = 0.0
    mu: np.ndarray = None
    sd: np.ndarray = None
    threshold: float = 0.55
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0) + 1e-9
        Z = self._z(X)
        n, k = Z.shape
        w = np.zeros(k)
        b = np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))
        for _ in range(iters):  # Newton / IRLS
            p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
            g_w = Z.T @ (p - y) / n + l2 * w / n
            g_b = np.mean(p - y)
            r = p * (1 - p)
            Hw = (Z.T * r) @ Z / n + l2 * np.eye(k) / n
            hb = r.mean() + 1e-9
            step = np.linalg.solve(Hw + 1e-6 * np.eye(k), g_w)
            w -= step
            b -= g_b / hb
            if np.abs(step).max() < 1e-6:
                break
        self.w, self.b = w, float(b)
        return self

    def _z(self, X):
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z), -5, 5)

    def proba(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(self._z(X) @ self.w + self.b, -30, 30)))

    def explain(self, x: np.ndarray, top=4) -> list:
        """Top feature contributions (in logit units) for one row — logged with every decision."""
        z = self._z(x.reshape(1, -1))[0]
        contrib = z * self.w
        order = np.argsort(-np.abs(contrib))[:top]
        return [{"feature": FEATURES[i], "value": round(float(x[i]), 5), "logit": round(float(contrib[i]), 3)} for i in order]

    # ------------------------------------------------------------ persistence
    def to_json(self) -> str:
        return json.dumps({"w": self.w.tolist(), "b": self.b, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                           "threshold": self.threshold, "features": FEATURES, "meta": self.meta}, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "Model":
        d = json.loads(s)
        return cls(np.array(d["w"]), d["b"], np.array(d["mu"]), np.array(d["sd"]), d["threshold"], d.get("meta", {}))


def expected_edge(p: float, tp: float, sl: float, cost: float, loss_frac: float = 1.0) -> float:
    """Expected return of one trade after round-trip costs.

    loss_frac: average loss of a non-winning trade as a fraction of the stop (measured on training data;
    trades that hit the time limit usually lose less than the full stop)."""
    return p * tp - (1 - p) * sl * loss_frac - cost


def train(dataset: pd.DataFrame, s, valid_frac: float = 0.25) -> Model:
    """dataset: rows of FEATURES + ['y','tp','sl'] (pooled across symbols, time-ordered).

    The last `valid_frac` of time is held out to choose the probability threshold that maximises
    realised edge (never tuned on the data we later trade on).
    """
    d = dataset.dropna(subset=FEATURES + ["y"]).sort_index()
    if len(d) < 400 or d["y"].nunique() < 2:
        raise ValueError(f"not enough training data ({len(d)} rows)")
    cut = d.index[int(len(d) * (1 - valid_frac))]
    tr, va = d[d.index < cut], d[d.index >= cut]
    m = Model().fit(tr[FEATURES].to_numpy(), tr["y"].to_numpy())
    cost = s.round_trip_cost()
    p = m.proba(va[FEATURES].to_numpy())
    best_t, best_score = s.min_prob, -1e9
    for t in np.arange(0.50, 0.80, 0.01):
        sel = p >= t
        if sel.sum() < max(15, 0.01 * len(va)):
            continue
        ret = va["r"].to_numpy()[sel] - cost  # realised outcome incl. time-limit exits
        score = ret.mean() * np.sqrt(sel.sum())
        if score > best_score:
            best_t, best_score = float(t), score
    # final model on all data, threshold chosen out-of-sample
    m = Model().fit(d[FEATURES].to_numpy(), d["y"].to_numpy())
    m.threshold = max(best_t, s.min_prob)
    # average loss of non-TP outcomes as a fraction of the stop (time-limit exits lose less than a full stop)
    neg = d[d["y"] == 0]
    loss_frac = float(np.clip((-neg["r"] / neg["sl"]).mean(), 0.2, 1.0)) if len(neg) else 1.0
    no_edge = best_score <= 0
    m.meta_mode = "strict"
    if no_edge:
        # top decile of out-of-sample probabilities: is its realised gross return positive?
        q90 = float(np.quantile(p, 0.90))
        top = va["r"].to_numpy()[p >= q90]
        gross = float(top.mean()) if len(top) else -1.0
        if s.trade_mode == "active" and gross > 0:
            m.threshold = q90
            m.meta_mode = "active"
        else:  # no evidence of edge at all -> do not trade with this model
            m.threshold = 0.99
    pv = m.proba(va[FEATURES].to_numpy())
    m.meta = {"rows": int(len(d)), "pos_rate": round(float(d["y"].mean()), 4), "train_end": str(d.index.max()),
              "valid_rows": int(len(va)), "valid_score": round(float(best_score), 5),
              "valid_auc": round(_auc(va["y"].to_numpy(), pv), 4), "no_edge": bool(no_edge), "loss_frac": round(loss_frac, 3), "mode": m.meta_mode}
    return m


def _auc(y, p):
    y = np.asarray(y)
    if y.min() == y.max():
        return 0.5
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos = y.sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))
