"""
loto_engine.py - data loading, statistics and grid recommendation for the French Loto.

Mathematical stance
-------------------
* Every 5-number combination has exactly the same probability (1 / C(49,5)),
  and draws are independent: past results cannot predict future numbers.
* Loto prizes are pari-mutuel (shared among winners of a rank). Grids that
  look like what most players pick (birthdays, sequences, "lucky" numbers)
  win the same rank as everybody else and must share the pot.
* So the only rational optimisation is to pick grids that are UNPOPULAR,
  i.e. maximise the expected payout *if* you win. We learn popularity from
  the FDJ archives: when a draw contains popular numbers / patterns, many
  players win and the payout per winner ("rapport") drops.
"""
from __future__ import annotations

import io
import re
import zipfile
from math import comb

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

N_MAX = 49
K = 5
CHANCE_MAX = 10

ODDS = {
    "Jackpot (5 numbers + Chance)": comb(N_MAX, K) * CHANCE_MAX,
    "2nd Tirage jackpot (5 numbers)": comb(N_MAX, K),
}
PRICE_GRID = 2.20
PRICE_SECOND = 0.80


# --------------------------------------------------------------------------- #
# 1. Loading FDJ archives
# --------------------------------------------------------------------------- #
def _read_csv_bytes(raw: bytes, name: str) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=";", encoding=enc, dtype=str,
                             index_col=False)
            if df.shape[1] > 3:
                df.columns = [c.strip().lower() for c in df.columns]
                df["__source"] = name
                return df
        except Exception:
            continue
    return None


def read_files(files: list[tuple[str, bytes]]) -> list[pd.DataFrame]:
    """files: list of (filename, raw bytes). Accepts FDJ .zip archives or .csv."""
    frames = []
    for name, raw in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for inner in z.namelist():
                    if inner.lower().endswith(".csv"):
                        df = _read_csv_bytes(z.read(inner), inner)
                        if df is not None:
                            frames.append(df)
        elif name.lower().endswith(".csv"):
            df = _read_csv_bytes(raw, name)
            if df is not None:
                frames.append(df)
    return frames


def _to_num(s: pd.Series) -> pd.Series:
    """FDJ numbers use French formatting: '1 234,50' -> 1234.5"""
    s = s.astype(str).str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    for fmt in ("%Y%m%d", "%d/%m/%y", "%Y-%m-%d"):
        miss = out.isna()
        if not miss.any():
            break
        out[miss] = pd.to_datetime(s[miss], format=fmt, errors="coerce")
    return out


def _rank_cols(cols, prefix: str, second: bool) -> dict[int, str]:
    pat = re.compile(rf"^{prefix}(\d+)(_second_tirage)?$")
    out = {}
    for c in cols:
        m = pat.match(c)
        if m and bool(m.group(2)) == second:
            out[int(m.group(1))] = c
    return out


def normalise(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns a long table with one row per *draw event* (main draw, and 2nd Tirage
    as a separate event) and the list of files skipped (wrong format).

    Columns: date, weekday, kind ('main'|'second'), n1..n5 (sorted), chance,
             payout_r{k} for pari-mutuel ranks, source.
    """
    events, skipped = [], []
    for df in frames:
        cols = list(df.columns)
        main_balls = [f"boule_{i}" for i in range(1, 6)]
        if not all(c in cols for c in main_balls):
            skipped.append(df["__source"].iloc[0])
            continue
        # Current formula: numbers 1..49 + numero_chance 1..10 (since Oct 2008)
        if "numero_chance" not in cols:
            skipped.append(df["__source"].iloc[0] + " (old formula, no N° Chance)")
            continue

        date = _parse_date(df.get("date_de_tirage", pd.Series([None] * len(df))))
        for kind, suffix in (("main", ""), ("second", "_second_tirage")):
            balls = [f"boule_{i}{suffix}" for i in range(1, 6)]
            if not all(c in cols for c in balls):
                continue
            nums = np.column_stack([_to_num(df[c]) for c in balls])
            ok = ~np.isnan(nums).any(axis=1)
            if ok.sum() == 0:
                continue
            nums = np.sort(nums[ok].astype(int), axis=1)
            ev = pd.DataFrame(nums, columns=[f"n{i}" for i in range(1, 6)])
            ev["date"] = date[ok].values
            ev["kind"] = kind
            ev["chance"] = _to_num(df["numero_chance"])[ok].values if kind == "main" else np.nan
            ev["source"] = df["__source"].iloc[0]
            for rank, col in _rank_cols(cols, "rapport_du_rang", kind == "second").items():
                ev[f"payout_r{rank}"] = _to_num(df[col])[ok].values
            events.append(ev)

    if not events:
        return pd.DataFrame(), skipped
    out = pd.concat(events, ignore_index=True)
    nb = out[[f"n{i}" for i in range(1, 6)]]
    out = out[((nb >= 1) & (nb <= N_MAX)).all(axis=1)]
    out["weekday"] = out["date"].dt.day_name()
    out = out.drop_duplicates(subset=["date", "kind", "n1", "n2", "n3", "n4", "n5"])
    return out.sort_values(["date", "kind"]).reset_index(drop=True), skipped


def balls(df: pd.DataFrame) -> np.ndarray:
    return df[[f"n{i}" for i in range(1, 6)]].to_numpy(dtype=int)


# --------------------------------------------------------------------------- #
# 2. Randomness diagnostics
# --------------------------------------------------------------------------- #
def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    b = balls(df)
    counts = np.bincount(b.ravel(), minlength=N_MAX + 1)[1:]
    expected = len(df) * K / N_MAX
    last_seen = {}
    for idx, row in enumerate(b):
        for n in row:
            last_seen[n] = idx
    n_draws = len(df)
    return pd.DataFrame({
        "number": np.arange(1, N_MAX + 1),
        "count": counts,
        "expected": expected,
        "deviation_sd": (counts - expected) / np.sqrt(expected * (1 - K / N_MAX)),
        "draws_since_last": [n_draws - 1 - last_seen.get(n, -1) for n in range(1, N_MAX + 1)],
    })


def chi_square_uniformity(df: pd.DataFrame) -> tuple[float, float]:
    counts = np.bincount(balls(df).ravel(), minlength=N_MAX + 1)[1:]
    chi2, p = stats.chisquare(counts)
    return float(chi2), float(p)


def hot_numbers_backtest(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Does picking the 5 most frequent numbers of the last `window` draws do
    better than chance? Expected matches per draw for any grid = 25/49 = 0.51."""
    b = balls(df)
    rows = []
    for t in range(window, len(b)):
        counts = np.bincount(b[t - window:t].ravel(), minlength=N_MAX + 1)[1:]
        hot = set(np.argsort(-counts)[:K] + 1)
        cold = set(np.argsort(counts)[:K] + 1)
        drawn = set(b[t])
        rows.append((len(hot & drawn), len(cold & drawn)))
    r = np.array(rows) if rows else np.zeros((0, 2))
    return pd.DataFrame({
        "strategy": ["Hot numbers", "Cold numbers", "Theory (any grid)"],
        "avg matches / draw": [r[:, 0].mean() if len(r) else np.nan,
                               r[:, 1].mean() if len(r) else np.nan,
                               K * K / N_MAX],
        "draws tested": [len(r), len(r), np.nan],
    })


# --------------------------------------------------------------------------- #
# 3. Grid features (what makes a grid "popular")
# --------------------------------------------------------------------------- #
PATTERN_FEATURES = [
    "n_le_31", "n_le_12", "sum", "consecutive_pairs", "n_even",
    "n_decades", "max_same_last_digit", "spread", "n_multiple_of_7",
]


def pattern_features(b: np.ndarray) -> np.ndarray:
    b = np.sort(b, axis=1)
    diffs = np.diff(b, axis=1)
    last_digit = b % 10
    same_ld = np.max(np.stack([(last_digit == d).sum(axis=1) for d in range(10)], axis=1), axis=1)
    decades = np.stack([((b // 10) == d).any(axis=1) for d in range(5)], axis=1).sum(axis=1)
    return np.column_stack([
        (b <= 31).sum(axis=1),
        (b <= 12).sum(axis=1),
        b.sum(axis=1),
        (diffs == 1).sum(axis=1),
        (b % 2 == 0).sum(axis=1),
        decades,
        same_ld,
        b[:, -1] - b[:, 0],
        (b % 7 == 0).sum(axis=1),
    ]).astype(float)


def one_hot(b: np.ndarray) -> np.ndarray:
    X = np.zeros((len(b), N_MAX))
    X[np.repeat(np.arange(len(b)), K), b.ravel() - 1] = 1.0
    return X


def design_matrix(b: np.ndarray) -> np.ndarray:
    return np.hstack([one_hot(b), pattern_features(b)])


# --------------------------------------------------------------------------- #
# 4. Popularity model, learned from payouts
# --------------------------------------------------------------------------- #
def payout_target(df: pd.DataFrame, min_cv: float = 0.05) -> tuple[pd.Series, list[str]]:
    """
    Builds a 'payout index' per draw event: for each pari-mutuel rank, log(payout)
    is z-scored within (kind, rank); the index is the mean over available ranks.
    High index = few winners = the drawn grid was UNPOPULAR.
    Ranks with (almost) constant payout are fixed-prize ranks -> excluded.
    Rank 1 is excluded (often no winner).
    """
    pcols = [c for c in df.columns if c.startswith("payout_r") and c != "payout_r1"]
    used, z_parts = [], []
    for kind in df["kind"].unique():
        sub = df["kind"] == kind
        for c in pcols:
            v = df.loc[sub, c].where(df.loc[sub, c] > 0)
            if v.notna().sum() < 50:
                continue
            if v.std() / v.mean() < min_cv:   # fixed prize rank
                continue
            lv = np.log(v)
            z = (lv - lv.mean()) / lv.std()
            z_parts.append(z.rename(f"{kind}_{c}"))
            used.append(f"{kind}: {c.replace('payout_r', 'rank ')}")
    if not z_parts:
        return pd.Series(dtype=float), []
    zdf = pd.concat(z_parts, axis=1)  # main and 2nd-tirage rows don't overlap -> NaN elsewhere
    return zdf.mean(axis=1, skipna=True).reindex(df.index), used


class PopularityModel:
    """Predicts the payout index of any grid. Falls back to a prior built from
    documented player behaviour when the archives have no payout columns."""

    # Prior (sign = effect on payout): birthdays, sequences, 7s are over-played.
    PRIOR = {"n_le_31": -0.35, "n_le_12": -0.20, "consecutive_pairs": -0.25,
             "n_multiple_of_7": -0.10, "max_same_last_digit": -0.10, "n_decades": +0.05}

    def __init__(self):
        self.fitted = False
        self.cv_r2 = np.nan
        self.n_obs = 0
        self.ranks_used: list[str] = []
        self.number_effect = pd.Series(dtype=float)
        self.pattern_effect = pd.Series(dtype=float)

    def fit(self, df: pd.DataFrame) -> "PopularityModel":
        y, self.ranks_used = payout_target(df)
        ok = y.notna()
        if ok.sum() < 200:
            return self
        b = balls(df[ok])
        X = design_matrix(b)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        model = RidgeCV(alphas=np.logspace(-1, 4, 30))
        self.cv_r2 = float(np.mean(cross_val_score(model, Xs, y[ok].values, cv=5, scoring="r2")))
        model.fit(Xs, y[ok].values)
        self.model = model
        self.n_obs = int(ok.sum())
        coef = model.coef_ / self.sd
        self.number_effect = pd.Series(coef[:N_MAX], index=range(1, N_MAX + 1))
        self.pattern_effect = pd.Series(coef[N_MAX:], index=PATTERN_FEATURES)
        self.fitted = True
        return self

    def score(self, b: np.ndarray) -> np.ndarray:
        """Higher = less popular = higher expected payout if the grid wins."""
        if self.fitted:
            return self.model.predict((design_matrix(b) - self.mu) / self.sd)
        F = pattern_features(b)
        w = np.array([self.PRIOR.get(f, 0.0) for f in PATTERN_FEATURES])
        return F @ w


# --------------------------------------------------------------------------- #
# 5. Recommendation
# --------------------------------------------------------------------------- #
def random_grids(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(np.argsort(rng.random((n, N_MAX)), axis=1)[:, :K] + 1, axis=1)


def hard_filters(b: np.ndarray, past: set[tuple] | None) -> np.ndarray:
    """Remove grids that are obviously over-played whatever the model says."""
    F = pd.DataFrame(pattern_features(b), columns=PATTERN_FEATURES)
    keep = (
        (F["n_le_31"] <= 3)                 # at least 2 numbers above 31
        & (F["consecutive_pairs"] <= 1)     # no run of 3
        & (F["max_same_last_digit"] <= 2)
        & (F["n_decades"] >= 3)             # no column / row look on the ticket
        & F["n_even"].between(1, 4)
    ).to_numpy()
    if past:
        keep = keep & np.array([tuple(r) not in past for r in b])
    return keep


def recommend(model: PopularityModel, history: pd.DataFrame | None, n_grids: int = 5,
              n_candidates: int = 200_000, max_overlap: int = 2,
              must_include: list[int] | None = None, exclude: list[int] | None = None,
              seed: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cand = random_grids(n_candidates, rng)
    if exclude:
        cand = cand[~np.isin(cand, exclude).any(axis=1)]
    if must_include:
        cand = cand[np.isin(cand, must_include).sum(axis=1) == len(must_include)]
    if len(cand) == 0:
        raise ValueError("No candidate grid satisfies the include/exclude constraints.")

    # Reference distribution = unconstrained random grids (for percentile display)
    ref_scores = model.score(random_grids(50_000, rng))

    past = {tuple(r) for r in balls(history)} if history is not None and len(history) else None
    cand = cand[hard_filters(cand, past)]
    cand = np.unique(cand, axis=0)
    scores = model.score(cand)
    order = np.argsort(-scores)

    picked, picked_scores = [], []
    for i in order:
        g = set(cand[i])
        if all(len(g & set(p)) <= max_overlap for p in picked):
            picked.append(cand[i])
            picked_scores.append(scores[i])
            if len(picked) == n_grids:
                break

    # Chance number: popularity can't be learnt reliably from 10 values,
    # and it only matters for the jackpot -> uniform random.
    chances = rng.integers(1, CHANCE_MAX + 1, size=len(picked))
    out = pd.DataFrame(np.array(picked), columns=[f"N{i}" for i in range(1, 6)])
    out.insert(0, "Grid", np.arange(1, len(picked) + 1))
    out["Chance"] = chances
    out["Less popular than"] = [f"{stats.percentileofscore(ref_scores, s):.1f}% of random grids"
                                for s in picked_scores]
    F = pattern_features(np.array(picked))
    out["Sum"] = F[:, PATTERN_FEATURES.index("sum")].astype(int)
    out["Nb > 31"] = (K - F[:, PATTERN_FEATURES.index("n_le_31")]).astype(int)
    return out, ref_scores


def expected_value_note(model: PopularityModel) -> str:
    return (f"Price per grid: €{PRICE_GRID:.2f} + €{PRICE_SECOND:.2f} for the 2nd Tirage. "
            f"Odds of the jackpot: 1 in {ODDS['Jackpot (5 numbers + Chance)']:,}; "
            f"2nd Tirage jackpot: 1 in {ODDS['2nd Tirage jackpot (5 numbers)']:,}. "
            "Unpopular grids don't change these odds; they only increase what you'd "
            "receive if you win a shared rank.")
