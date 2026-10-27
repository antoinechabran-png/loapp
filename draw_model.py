"""Experimental stationary draw-bias model, separate from player popularity.

P(S) = exp(sum(theta[S])) / e_5(exp(theta)), for unordered five-element
subsets S. Chance is an independent, smoothed categorical distribution.
This is a statistical alternative to fairness, not a physical machine model.
"""
from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import loto_engine as E

VERSION = "draw-bias-1"
STRENGTHS = (100.0, 1000.0, 10000.0, float("inf"))
MIN_DRAWS = 400


def partition_and_marginals(theta: np.ndarray, k: int = 5):
    """Exact log normalizer and inclusion probabilities in O(N k^2).

    Prefix/suffix elementary-symmetric-polynomial tables avoid enumerating
    C(49,5) subsets. The shift keeps all polynomial arithmetic bounded.
    """
    theta = np.asarray(theta, float)
    n = len(theta)
    if not 1 <= k <= n or not np.isfinite(theta).all():
        raise ValueError("Finite weights and a valid subset size are required.")
    shift = float(theta.max())
    w = np.exp(theta - shift)
    left, right = np.zeros((n + 1, k + 1)), np.zeros((n + 1, k + 1))
    left[:, 0] = right[:, 0] = 1
    for i in range(n):
        left[i + 1, 1:] = left[i, 1:] + w[i] * left[i, :-1]
    for i in range(n - 1, -1, -1):
        right[i, 1:] = right[i + 1, 1:] + w[i] * right[i + 1, :-1]
    z = left[n, k]
    if not z > 0:
        raise ArithmeticError("Subset normalization underflowed.")
    marginal = np.array([w[i] * np.dot(left[i, :k], right[i + 1, k-1::-1]) / z for i in range(n)])
    return float(np.log(z) + k * shift), marginal


@dataclass
class DrawDistribution:
    theta: np.ndarray
    log_z: float
    chance: np.ndarray
    kind: str
    strength: float

    @classmethod
    def uniform(cls, kind="main"):
        return cls(np.zeros(49), float(np.log(comb(49, 5))), np.full(10, 0.1), kind, float("inf"))

    @classmethod
    def fit(cls, df: pd.DataFrame, strength: float, kind: str):
        if kind not in ("main", "second") or strength <= 0:
            raise ValueError("Choose a valid game and positive shrinkage strength.")
        if not len(df) or np.isinf(strength):
            return cls.uniform(kind)
        counts = np.bincount(E.balls(df).ravel(), minlength=50)[1:].astype(float)
        target = (counts + strength * 5 / 49) / (len(df) + strength)

        def objective(theta):
            log_z, marginal = partition_and_marginals(theta)
            return log_z - np.dot(target, theta), marginal - target

        result = minimize(objective, np.zeros(49), method="L-BFGS-B", jac=True,
                          bounds=[(-8, 8)] * 49,
                          options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9})
        if not result.success:
            raise RuntimeError(f"Draw-bias fit failed: {result.message}")
        theta = result.x - result.x.mean()
        log_z, _ = partition_and_marginals(theta)
        chance = np.full(10, 0.1)
        if kind == "main":
            c = np.bincount(df.chance.to_numpy(int), minlength=11)[1:]
            chance = (c + strength / 10) / (len(df) + strength)
        return cls(theta, log_z, chance, kind, strength)

    def log_prob(self, b: np.ndarray, chance=None) -> np.ndarray:
        b = np.asarray(b)
        if b.ndim != 2 or b.shape[1] != 5 or not np.isfinite(b).all():
            raise ValueError("Expected an array of valid five-number grids.")
        if ((b < 1) | (b > 49) | (b != np.floor(b))).any() or (np.diff(np.sort(b, axis=1), axis=1) == 0).any():
            raise ValueError("Each grid must have five distinct integers from 1 to 49.")
        result = self.theta[b.astype(int) - 1].sum(1) - self.log_z
        if self.kind == "main":
            c = np.asarray(chance)
            if c.shape != (len(b),) or not np.isfinite(c.astype(float)).all() or ((c < 1) | (c > 10) | (c != np.floor(c))).any():
                raise ValueError("Each main-draw grid needs a Chance integer from 1 to 10.")
            result += np.log(self.chance[c.astype(int) - 1])
        return result

    def score_draws(self, df):
        return self.log_prob(E.balls(df), df.chance.to_numpy() if self.kind == "main" else None)

    def top(self, n=10):
        """Exact top n full outcomes by monotone best-first subset search.

        No diversity restriction: changing that would cease to be the top n.
        Main outcomes may have identical five balls and different Chance values.
        """
        if not 1 <= n <= 100:
            raise ValueError("Request between 1 and 100 outcomes.")
        order = np.argsort(-self.theta, kind="stable")
        chance_order = np.argsort(-self.chance, kind="stable") if self.kind == "main" else np.array([0])

        def score(indices, ci):
            value = self.theta[order[list(indices)]].sum() - self.log_z
            return float(value + (np.log(self.chance[chance_order[ci]]) if self.kind == "main" else 0))

        first = (tuple(range(5)), 0)
        heap, seen, results = [(-score(*first), first)], {first}, []
        while len(results) < n:
            negative, (indices, ci) = heapq.heappop(heap)
            grid = np.sort(order[list(indices)] + 1)
            results.append((grid, int(chance_order[ci] + 1) if self.kind == "main" else None, -negative))
            neighbors = []
            for j in range(5):
                upper = indices[j + 1] if j < 4 else 49
                if indices[j] + 1 < upper:
                    nxt = list(indices)
                    nxt[j] += 1
                    neighbors.append((tuple(nxt), ci))
            if ci + 1 < len(chance_order):
                neighbors.append((indices, ci + 1))
            for state in neighbors:
                if state not in seen:
                    seen.add(state)
                    heapq.heappush(heap, (-score(*state), state))
        return results


def prepare_draws(events: pd.DataFrame, kind: str):
    """Uses draw outcomes only: no payouts, winner counts or sales are read."""
    if kind not in ("main", "second"):
        raise ValueError("Choose main or second draw.")
    if events.empty:
        return pd.DataFrame()
    d = events[events.kind == kind].copy()
    if "schema" in d:
        d = d[d.schema != "special_game"]
    d = d.dropna(subset=["date"] + E.BALL_COLS + (["chance"] if kind == "main" else []))
    # Input normally comes from the validated loader. Reject invalid direct API data too.
    valid = ((d[E.BALL_COLS] >= 1) & (d[E.BALL_COLS] <= 49)
             & (d[E.BALL_COLS] == np.floor(d[E.BALL_COLS]))).all(axis=1)
    valid &= (np.diff(np.sort(d[E.BALL_COLS].to_numpy(), axis=1), axis=1) > 0).all(1)
    if kind == "main":
        valid &= d.chance.between(1, 10) & (d.chance == np.floor(d.chance))
    d = d[valid & (d.date >= (pd.Timestamp("2008-10-06") if kind == "main" else E.MODEL_START))]
    # Identical overlapping archives count once; contradictory same-date outcomes are excluded.
    outcome = E.BALL_COLS + (["chance"] if kind == "main" else [])
    d = d.drop_duplicates(["date"] + outcome)
    d = d[~d.date.duplicated(keep=False)]
    return d.sort_values("date").reset_index(drop=True)


class DrawProbabilityModel:
    def __init__(self, kind="main"):
        self.kind, self.fitted, self.eligible = kind, False, False
        self.n_obs, self.strength = 0, float("inf")
        self.distribution = DrawDistribution.uniform(kind)
        self.metrics, self.folds, self.holdout_periods = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.status = "No evidence assessed. Equal probabilities remain the baseline."
        self.data = pd.DataFrame()
        self.snapshot = "no-data"

    def fit(self, events: pd.DataFrame, kind="main"):
        self.__init__(kind)
        d = prepare_draws(events, kind)
        self.data, self.n_obs = d, len(d)
        if len(d):
            # Winner/payout changes cannot alter a draw model or its reproducibility.
            cols = ["date"] + E.BALL_COLS + (["chance"] if kind == "main" else [])
            self.snapshot = hashlib.sha256(d[cols].to_csv(index=False).encode()).hexdigest()
            self.last_date = d.date.max()
        if len(d) < MIN_DRAWS:
            self.status = f"{len(d)} usable draws; at least {MIN_DRAWS} are required. All grids keep equal probabilities."
            return self
        cutoff = int(len(d) * 0.8)
        dev, hold = d.iloc[:cutoff], d.iloc[cutoff:]
        self.holdout_start, self.holdout_end = hold.date.min(), hold.date.max()
        uniform_log = -np.log(comb(49, 5) * (10 if kind == "main" else 1))
        rows, trials = [], []
        splits = list(E.chronological_splits(dev.date))
        for strength in STRENGTHS:
            gains, fold_rows = [], []
            for number, (train, test) in enumerate(splits, 1):
                dist = DrawDistribution.fit(dev.iloc[train], strength, kind)
                gain = dist.score_draws(dev.iloc[test]) - uniform_log
                gains.extend(gain)
                fold_rows.append({"Fold": number, "Train through": dev.iloc[train].date.max().date(),
                                  "Test from": dev.iloc[test].date.min().date(),
                                  "Test through": dev.iloc[test].date.max().date(),
                                  "Test draws": len(test), "Mean log-score gain": float(gain.mean())})
            mean_gain = 0.0 if np.isinf(strength) else float(np.mean(gains))
            trials.append((mean_gain, strength, fold_rows))
            rows.append({"Prior equivalent draws": strength, "Rolling log-score gain": mean_gain})
        # Ties prefer the stronger shrinkage; infinity is exactly the fair model.
        best = max(trials, key=lambda item: (round(item[0], 12), item[1]))
        self.strength, self.folds, self.metrics = best[1], pd.DataFrame(best[2]), pd.DataFrame(rows)
        frozen = DrawDistribution.fit(dev, self.strength, kind)
        gain = frozen.score_draws(hold) - uniform_log
        self.holdout_gain = float(gain.mean())
        self.holdout_interval = E.block_mean_interval(gain, repeats=1000)
        period_rows = []
        for part, indices in enumerate(np.array_split(np.arange(len(hold)), 3), 1):
            period_rows.append({"Holdout part": part, "From": hold.iloc[indices].date.min().date(),
                                "Through": hold.iloc[indices].date.max().date(),
                                "Draws": len(indices), "Mean log-score gain": float(gain[indices].mean())})
        self.holdout_periods = pd.DataFrame(period_rows)
        self.eligible = bool(np.isfinite(self.strength) and best[0] > 0
                             and (self.folds["Mean log-score gain"] > 0).sum() >= 2
                             and self.holdout_interval[0] > 0
                             and (self.holdout_periods["Mean log-score gain"] > 0).sum() >= 2)
        self.fitted = True
        # The displayed estimate may use the newest data only after the independent evaluation.
        self.distribution = DrawDistribution.fit(d, self.strength, kind)
        if self.eligible:
            self.status = ("The experimental model passed this chronological comparison. "
                           "Probabilities remain conditional estimates, not established physical odds or a demonstrated top-ten win advantage.")
        elif np.isinf(self.strength):
            self.status = "The uniform model won the rolling comparison. No supported probability ranking: all combinations are tied."
        else:
            self.status = ("Estimated differences did not pass the held-out validation. "
                           "Equal probabilities are used by default; an explicitly exploratory ranking is available.")
        return self

    def ranking(self, week: str, exploratory=False, n=10, refits=40):
        """Weekly table; seed rotates only statistically tied fallback selections.

        Stability ranges are conditional 10th–90th percentile block-refit ranges,
        not confidence bounds, and do not account for selecting the top outcomes.
        """
        if not 1 <= n <= 100 or not 0 <= refits <= 200:
            raise ValueError("Invalid ranking size or refit count.")
        use_model = self.fitted and np.isfinite(self.strength) and (self.eligible or exploratory)
        baseline = 1 / (comb(49, 5) * (10 if self.kind == "main" else 1))
        # Fairness supplies no reason to replace a week's tied tickets after
        # every upload. Only refit sensitivity uses the outcome fingerprint.
        seed_text = f"{self.snapshot if use_model else 'uniform'}|{week}|{self.kind}"
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        if use_model:
            outcomes = self.distribution.top(n)
        else:
            outcomes, seen = [], set()
            while len(outcomes) < n:
                b = np.sort(rng.choice(np.arange(1, 50), 5, replace=False))
                chance = int(rng.integers(1, 11)) if self.kind == "main" else None
                key = tuple(b) + (chance,)
                if key not in seen:
                    seen.add(key)
                    outcomes.append((b, chance, float(np.log(baseline))))
        b = np.array([x[0] for x in outcomes])
        chances = np.array([x[1] for x in outcomes]) if self.kind == "main" else None
        probabilities = np.exp([x[2] for x in outcomes])
        frame = pd.DataFrame(b, columns=[f"N{i}" for i in range(1, 6)])
        frame.insert(0, "Rank", pd.Series(np.round(np.log(probabilities), 12)).rank(method="min", ascending=False).astype(int))
        if self.kind == "main":
            frame["Chance"] = chances
        frame["Estimated probability"] = probabilities
        frame["Fair-draw probability"] = baseline
        frame["Model / fair"] = probabilities / baseline
        frame["Equivalent 1 in"] = 1 / probabilities
        frame["Evidence"] = ("Experimental: passed comparison" if self.eligible and use_model else
                             "Exploratory: not validated" if use_model else "Uniform baseline: tied")
        if use_model and refits:
            # Resample blocks of whole draws to preserve within-draw dependence.
            sample_p = []
            block = min(10, len(self.data))
            for _ in range(refits):
                starts = rng.integers(0, len(self.data) - block + 1, int(np.ceil(len(self.data) / block)))
                indices = (starts[:, None] + np.arange(block)).ravel()[:len(self.data)]
                dist = DrawDistribution.fit(self.data.iloc[indices], self.strength, self.kind)
                sample_p.append(np.exp(dist.log_prob(b, chances)))
            frame["Refit range 10%"], frame["Refit range 90%"] = np.quantile(sample_p, [0.1, 0.9], axis=0)
        frame["Week"], frame["Draw"] = week, self.kind
        frame["Data through"] = str(self.data.date.max().date()) if len(self.data) else "No data"
        frame["Snapshot"] = self.snapshot[:12]
        return frame


def compare_rankings(current: pd.DataFrame, previous: pd.DataFrame | None):
    """Compare the same outcome, never a row position, across dataset snapshots."""
    out = current.copy()
    if previous is None or previous.empty or current.Draw.iloc[0] != previous.Draw.iloc[0]:
        out["Previous rank"] = pd.Series([pd.NA] * len(out), dtype="Int64")
        out["Change"] = "No previous snapshot"
        return out
    cols = [f"N{i}" for i in range(1, 6)] + (["Chance"] if "Chance" in current else [])
    old = {tuple(row[c] for c in cols): int(row.Rank) for _, row in previous.iterrows()}
    ranks, changes = [], []
    for _, row in out.iterrows():
        prior = old.get(tuple(row[c] for c in cols))
        ranks.append(prior)
        changes.append("New to list" if prior is None else
                       "Unchanged" if prior == row.Rank else f"{'Up' if prior > row.Rank else 'Down'} {abs(prior - row.Rank)}")
    out["Previous rank"], out["Change"] = pd.array(ranks, dtype="Int64"), changes
    return out
