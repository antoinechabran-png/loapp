"""French Loto diagnostics and experimental winner-count models.

No draw prediction or expected-euro-return claim is made. See README.md.
"""
from __future__ import annotations

import hashlib
import io
import warnings
import zipfile
from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy
from sklearn.linear_model import PoissonRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_MAX, K, CHANCE_MAX = 49, 5, 10
PRICE_GRID, PRICE_SECOND = 2.20, 0.80
RULES_CHECKED = "2026-10-26"
MODEL_START = pd.Timestamp("2019-11-04")
ODDS = {"Jackpot (5 + Chance)": comb(49, 5) * 10, "2nd Tirage: 5 numbers": comb(49, 5)}
BALL_COLS = [f"n{i}" for i in range(1, 6)]
# Combine with/without-Chance ranks. Only the standard November 2019+ schema
# is used for modelling. Older five-ball draws remain available for diagnostics.
RANKS = {"main": {2: (7, 8), 3: (5, 6), 4: (3, 4), 5: (1, 2)},
         "second": {2: (4,), 3: (3,), 4: (2,), 5: (1,)}}
MAX_FILE_BYTES, MAX_TOTAL_BYTES = 30_000_000, 100_000_000


def file_fingerprint(files: list[tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    for name, raw in sorted(files):
        h.update(name.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(raw).digest())
    return h.hexdigest()


def _to_num(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.replace(r"[\s\u00a0\u202f]", "", regex=True)
    return pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")


def _parse_date(s: pd.Series) -> pd.Series:
    text = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%d/%m/%Y", "%Y%m%d", "%d/%m/%y", "%Y-%m-%d"):
        missing = out.isna()
        out.loc[missing] = pd.to_datetime(text[missing], format=fmt, errors="coerce")
    return out


def read_files(files: list[tuple[str, bytes]]) -> tuple[list[pd.DataFrame], list[str]]:
    """Read bounded CSV/ZIP inputs without extracting archives to disk."""
    frames, issues = [], []
    total = 0

    def consume(raw: bytes, name: str):
        nonlocal total
        total += len(raw)
        if len(raw) > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
            issues.append(f"{name}: size limit exceeded; skipped.")
            return
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=";", encoding=encoding, dtype=str)
                if df.empty or len(df.columns) < 5:
                    issues.append(f"{name}: empty or unsupported semicolon CSV.")
                    return
                df.columns = [str(c).strip().lower() for c in df.columns]
                if df.columns.duplicated().any():
                    issues.append(f"{name}: duplicate column names; skipped.")
                    return
                df["__source"] = name
                frames.append(df)
                return
            except UnicodeDecodeError:
                continue
            except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
                issues.append(f"{name}: cannot read CSV ({exc}).")
                return
        issues.append(f"{name}: unsupported encoding.")

    for name, raw in files:
        if len(raw) > MAX_FILE_BYTES:
            issues.append(f"{name}: input exceeds 30 MB; skipped.")
            continue
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
                    if not members:
                        issues.append(f"{name}: no CSV files found.")
                    for member in members:
                        if member.file_size > MAX_FILE_BYTES or total + member.file_size > MAX_TOTAL_BYTES:
                            issues.append(f"{name}/{member.filename}: size limit exceeded; skipped.")
                            continue
                        consume(archive.read(member), f"{name}/{member.filename}")
            except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError) as exc:
                issues.append(f"{name}: unreadable ZIP ({exc}).")
        elif name.lower().endswith(".csv"):
            consume(raw, name)
        else:
            issues.append(f"{name}: only CSV and ZIP inputs are supported.")
    return frames, issues


def normalise(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    events, issues = [], []
    for original in frames:
        if original.empty:
            continue
        df = original.reset_index(drop=True)
        source = str(df["__source"].iloc[0]) if "__source" in df else "input"
        required = [f"boule_{i}" for i in range(1, 6)] + ["date_de_tirage", "numero_chance"]
        if not all(c in df for c in required):
            issues.append(f"{source}: unsupported format (requires five balls, date and Chance).")
            continue
        dates = _parse_date(df["date_de_tirage"])
        chance = _to_num(df["numero_chance"]).to_numpy(float)
        for kind, suffix in (("main", ""), ("second", "_second_tirage")):
            names = [f"boule_{i}{suffix}" for i in range(1, 6)]
            if not all(c in df for c in names):
                continue
            nums = np.column_stack([_to_num(df[c]) for c in names]).astype(float)
            valid = (np.isfinite(nums).all(1) & (nums == np.floor(nums)).all(1)
                     & ((nums >= 1) & (nums <= 49)).all(1)
                     & (np.diff(np.sort(nums, axis=1), axis=1) > 0).all(1)
                     & dates.notna().to_numpy()
                     & (dates >= pd.Timestamp("2008-10-06")).to_numpy())
            if kind == "main":
                valid &= np.isfinite(chance) & (chance == np.floor(chance)) & (chance >= 1) & (chance <= 10)
            else:
                valid &= (dates >= MODEL_START).to_numpy()
            present = np.isfinite(nums).any(1)
            rejected = int((~valid & (present if kind == "second" else True)).sum())
            if rejected:
                issues.append(f"{source}: rejected {rejected} invalid {kind} draw(s).")
            if not valid.any():
                continue
            ev = pd.DataFrame(np.sort(nums[valid].astype(int), axis=1), columns=BALL_COLS)
            ev["date"], ev["kind"] = dates[valid].to_numpy(), kind
            ev["chance"] = chance[valid] if kind == "main" else np.nan
            ev["source"] = source
            special_game = any(word in source.lower() for word in ("super", "grand"))
            ev["schema"] = np.where(ev["date"] >= MODEL_START, "standard_2019", "historical")
            if special_game:
                ev["schema"] = "special_game"
                issues.append(f"{source}: special-game filename; excluded from standard Loto modelling.")
            for rank in range(1, 10):
                for prefix, target, integral in (("nombre_de_gagnant_au_rang", "winners", True),
                                                  ("rapport_du_rang", "payout", False)):
                    col = f"{prefix}{rank}{suffix}"
                    if col not in df:
                        continue
                    values = _to_num(df[col]).to_numpy(float)[valid]
                    bad = ~np.isfinite(values) | (values < 0)
                    if integral:
                        bad |= values != np.floor(values)
                    malformed = bad & df.loc[valid, col].notna().to_numpy()
                    if malformed.any():
                        issues.append(f"{source}: {int(malformed.sum())} invalid {col} value(s) ignored.")
                    values[bad] = np.nan
                    ev[f"{target}_r{rank}"] = values
            # Optional enriched CSV column: simple combinations sold, NOT
            # players, receipts, euros or winning combinations. Never inferred.
            exposure_col = f"total_combinations{suffix}"
            if exposure_col in df:
                values = _to_num(df[exposure_col]).to_numpy(float)[valid]
                bad = ~np.isfinite(values) | (values <= 0) | (values != np.floor(values))
                if bad.any():
                    issues.append(f"{source}: {int(bad.sum())} invalid participation total(s) ignored.")
                values[bad] = np.nan
                ev["entries"] = values
            else:
                ev["entries"] = np.nan
            events.append(ev)
    if not events:
        return pd.DataFrame(), issues
    out = pd.concat(events, ignore_index=True)
    rows = []
    value_cols = [c for c in out if c not in ("date", "kind", "source")]
    for (date, kind), group in out.groupby(["date", "kind"], sort=True):
        conflicts = [c for c in value_cols if group[c].dropna().nunique() > 1]
        if conflicts:
            issues.append(f"{date.date()} {kind}: conflicting duplicate data ({', '.join(conflicts)}); excluded.")
            continue
        row = {"date": date, "kind": kind, "source": " | ".join(sorted(group.source.unique()))}
        for col in value_cols:
            nonnull = group[col].dropna()
            row[col] = nonnull.iloc[0] if len(nonnull) else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(), issues
    out = pd.DataFrame(rows).sort_values(["date", "kind"]).reset_index(drop=True)
    out["weekday"] = out.date.dt.day_name()
    return out, issues


def load_files(files: list[tuple[str, bytes]]) -> tuple[pd.DataFrame, list[str]]:
    frames, issues = read_files(files)
    events, validation = normalise(frames)
    return events, issues + validation


def balls(df: pd.DataFrame) -> np.ndarray:
    return df[BALL_COLS].to_numpy(dtype=int)


def random_grids(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(np.argpartition(rng.random((n, 49)), 4, axis=1)[:, :5] + 1, axis=1)


def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["number", "count", "expected", "deviation_sd"])
    counts = np.bincount(balls(df).ravel(), minlength=50)[1:]
    expected = len(df) * 5 / 49
    return pd.DataFrame({"number": np.arange(1, 50), "count": counts, "expected": expected,
                         "deviation_sd": (counts - expected) / np.sqrt(expected * 44 / 49)})


def chi_square_uniformity(df: pd.DataFrame) -> tuple[float, float]:
    """Without-replacement correction: (48/44) times Pearson -> chi-square(48)."""
    if df.empty:
        return np.nan, np.nan
    ft = frequency_table(df)
    adjusted = float((((ft["count"] - ft.expected) ** 2 / ft.expected).sum()) * 48 / 44)
    return adjusted, float(stats.chi2.sf(adjusted, 48))


def randomness_check(df: pd.DataFrame, simulations: int = 999, seed: int = 2026) -> dict:
    """Monte Carlo p-value simulating entire draws, including non-replacement."""
    if df.empty or simulations < 99:
        raise ValueError("Load draws and use at least 99 simulations.")
    observed, _ = chi_square_uniformity(df)
    rng, exceed = np.random.default_rng(seed), 0
    expected = len(df) * 5 / 49
    for _ in range(simulations):
        counts = np.zeros(49, dtype=int)
        for start in range(0, len(df), 2000):
            sample = random_grids(min(2000, len(df) - start), rng)
            counts += np.bincount(sample.ravel(), minlength=50)[1:]
        statistic = np.sum((counts - expected) ** 2 / expected) * 48 / 44
        exceed += statistic >= observed - 1e-12
    p = (exceed + 1) / (simulations + 1)
    return {"statistic": observed, "p": p,
            "simulation_se": float(np.sqrt(p * (1 - p) / (simulations + 1))), "simulations": simulations}


def block_mean_interval(values: np.ndarray, seed: int = 42, repeats: int = 500) -> tuple[float, float]:
    """Moving-block bootstrap interval for a time-ordered mean."""
    values = np.asarray(values, dtype=float)
    if len(values) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    block = min(10, max(2, len(values) // 5))
    starts = rng.integers(0, len(values) - block + 1, (repeats, int(np.ceil(len(values) / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(repeats, -1)[:, :len(values)]
    return tuple(np.quantile(values[indices].mean(1), [0.025, 0.975]).tolist())


def hot_numbers_backtest(df: pd.DataFrame, window: int = 100, seed: int = 2026) -> pd.DataFrame:
    if window < 1:
        raise ValueError("The look-back window must be positive.")
    b = balls(df.sort_values("date")) if not df.empty else np.empty((0, 5), int)
    rng, rows = np.random.default_rng(seed), []
    for t in range(window, len(b)):
        counts = np.bincount(b[t-window:t].ravel(), minlength=50)[1:]
        tie = rng.random(49)
        hot, cold = np.lexsort((tie, -counts))[:5] + 1, np.lexsort((tie, counts))[:5] + 1
        random = rng.choice(np.arange(1, 50), 5, replace=False)
        rows.append([np.isin(g, b[t]).sum() for g in (hot, cold, random)])
    result = []
    for i, name in enumerate(("Hot numbers", "Cold numbers", "Uniform random")):
        sample = np.asarray(rows)[:, i] if rows else np.array([])
        lo, hi = block_mean_interval(sample)
        result.append({"Strategy": name, "Mean matches": sample.mean() if len(sample) else np.nan,
                       "95% lower": lo, "95% upper": hi, "Draws tested": len(sample)})
    return pd.DataFrame(result)


# Number 49 is the reference category. No duplicate linear aggregates (sum,
# parity, birthday counts, multiples of seven) enter the full model.
PATTERN_FEATURES = ["consecutive_pairs", "max_same_last_digit", "spread", "decades"]


def design_matrix(b: np.ndarray) -> np.ndarray:
    b = np.sort(np.asarray(b, dtype=int).reshape(-1, 5), axis=1)
    onehot = np.zeros((len(b), 49))
    onehot[np.arange(len(b))[:, None], b - 1] = 1
    last = np.stack([(b % 10 == d).sum(1) for d in range(10)], axis=1).max(1)
    decades = np.stack([(b // 10 == d).any(1) for d in range(5)], axis=1).sum(1)
    patterns = np.column_stack([(np.diff(b, axis=1) == 1).sum(1), last, b[:, -1]-b[:, 0], decades])
    return np.column_stack([onehot[:, :48], patterns])


def context_matrix(dates: pd.Series) -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    weekday = np.eye(7)[dates.day_of_week][:, :6]
    time = np.asarray((dates - MODEL_START).days, float) / 365.25
    return np.column_stack([weekday, time])


def model_matrix(df: pd.DataFrame, method: str) -> np.ndarray:
    context = context_matrix(df.date)
    if method == "context":
        return context
    b = balls(df)
    if method == "birthday":
        return np.column_stack([context, (b <= 31).sum(1)])
    return np.column_stack([context, design_matrix(b)])


def prepare_target(df: pd.DataFrame, kind: str, matches: int) -> tuple[pd.DataFrame, str]:
    """Count tickets with exactly m main balls, regardless of Chance."""
    if kind not in RANKS or matches not in RANKS[kind]:
        raise ValueError("Choose main/second and 2, 3, 4 or 5 matching numbers.")
    if df.empty:
        return pd.DataFrame(), "No draw data loaded."
    cols = [f"winners_r{r}" for r in RANKS[kind][matches]]
    if not all(c in df for c in cols):
        return pd.DataFrame(), "Winner-count columns for this match tier are missing."
    d = df[(df.kind == kind) & (df.date >= MODEL_START)].copy()
    if "schema" in d:
        d = d[d.schema == "standard_2019"]
    d["target"] = d[cols].sum(axis=1, min_count=len(cols))
    valid = np.isfinite(d.target) & (d.target >= 0) & (d.target == np.floor(d.target))
    d = d[valid].sort_values("date").reset_index(drop=True)
    if "entries" not in d:
        d["entries"] = np.nan
    d = d[~(d.entries.notna() & (d.target > d.entries))].reset_index(drop=True)
    return d, ""


def chronological_splits(dates: pd.Series, n_splits: int = 3):
    """Split distinct dates so records of the same date stay together."""
    date_array = pd.to_datetime(dates).to_numpy()
    unique = np.sort(np.unique(date_array))
    for train, test in TimeSeriesSplit(n_splits=n_splits).split(unique):
        yield np.flatnonzero(np.isin(date_array, unique[train])), np.flatnonzero(np.isin(date_array, unique[test]))


def poisson_loss(y: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    mu, y = np.maximum(np.asarray(predicted, float), 1e-12), np.asarray(y, float)
    return 2 * (xlogy(y, y / mu) - y + mu)


def _fit_count(df: pd.DataFrame, method: str, alpha: float, exposure: bool):
    scale = df.entries.to_numpy(float) / 1_000_000 if exposure else np.ones(len(df))
    model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-7))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(model_matrix(df, method), df.target.to_numpy() / scale,
                      poissonregressor__sample_weight=scale)
    except ConvergenceWarning as exc:
        raise RuntimeError("The count fit did not converge. Try a different scope; no learned recommendation is available.") from exc
    return model


def _predict_count(model, df: pd.DataFrame, method: str, exposure: bool) -> np.ndarray:
    rates = model.predict(model_matrix(df, method))
    return rates * (df.entries.to_numpy(float) / 1_000_000 if exposure else 1)


class PopularityModel:
    """Per-tier Poisson model with an untouched chronological holdout.

    Predicts matching-ticket rates, NOT exact ticket-choice probabilities,
    prize shares or expected euros. Missing exposure disables learned selection.
    """
    def __init__(self):
        self.fitted = self.eligible = self.exposure_adjusted = False
        self.n_obs = 0
        self.status = "No model fitted. Use random selection or the birthday heuristic."
        self.metrics, self.fold_results = pd.DataFrame(), pd.DataFrame()
        self.ensemble, self.alpha = [], None
        self.kind, self.matches = "main", 3

    def fit(self, df: pd.DataFrame, kind: str = "main", matches: int = 3) -> "PopularityModel":
        self.__init__()
        self.kind, self.matches = kind, matches
        d, problem = prepare_target(df, kind, matches)
        self.n_obs = len(d)
        if len(d) < 320 or d.date.nunique() < 320:
            self.status = problem or f"Only {len(d)} usable draws; at least 320 are required for chronological validation."
            return self
        if d.target.sum() == 0:
            self.status = "No winners observed in this tier; a count model cannot be fitted."
            return self
        self.exposure_adjusted = bool(d.entries.notna().all() and (d.entries > 0).all())
        dates = np.sort(d.date.unique())
        cutoff = dates[int(len(dates) * 0.8)]
        dev, holdout = d[d.date < cutoff].reset_index(drop=True), d[d.date >= cutoff].reset_index(drop=True)
        self.holdout_start, self.holdout_end = pd.Timestamp(cutoff), holdout.date.max()
        splits = list(chronological_splits(dev.date))
        if any(dev.iloc[tr].target.sum() == 0 for tr, _ in splits):
            self.status = "Too few early winners to fit every training fold."
            return self
        # All methods get the same tuning budget. The last 20% of dates never
        # enters alpha selection or fold preprocessing.
        selected, fold_rows = {}, []
        for method in ("context", "birthday", "full"):
            candidates = []
            for alpha in (0.1, 1.0, 10.0):
                losses, rows = [], []
                for fold, (train_idx, test_idx) in enumerate(splits, 1):
                    train, test = dev.iloc[train_idx], dev.iloc[test_idx]
                    model = _fit_count(train, method, alpha, self.exposure_adjusted)
                    loss = poisson_loss(test.target, _predict_count(model, test, method, self.exposure_adjusted))
                    losses.extend(loss)
                    rows.append({"Method": method, "Fold": fold, "Train through": train.date.max().date(),
                                 "Test from": test.date.min().date(), "Test through": test.date.max().date(),
                                 "Draws": len(test), "Deviance": float(loss.mean())})
                candidates.append((float(np.mean(losses)), alpha, rows))
            best = min(candidates, key=lambda c: c[0])
            selected[method] = best
            fold_rows.extend(best[2])
        self.alpha, self.fold_results = selected["full"][1], pd.DataFrame(fold_rows)
        losses, rows = {}, []
        for method in ("context", "birthday", "full"):
            model = _fit_count(dev, method, selected[method][1], self.exposure_adjusted)
            pred = _predict_count(model, holdout, method, self.exposure_adjusted)
            loss = poisson_loss(holdout.target, pred)
            losses[method] = loss
            rows.append({"Method": method, "Chosen alpha": selected[method][1],
                         "Rolling deviance": selected[method][0], "Holdout deviance": float(loss.mean()),
                         "Holdout MAE": float(np.mean(np.abs(holdout.target.to_numpy() - pred)))})
        # Constant count/rate estimated on training data only: no number choices.
        train_scale = dev.entries.to_numpy() / 1e6 if self.exposure_adjusted else np.ones(len(dev))
        test_scale = holdout.entries.to_numpy() / 1e6 if self.exposure_adjusted else np.ones(len(holdout))
        constant = dev.target.sum() / train_scale.sum()
        losses["uniform"] = poisson_loss(holdout.target, constant * test_scale)
        rows.append({"Method": "uniform", "Chosen alpha": np.nan, "Rolling deviance": np.nan,
                     "Holdout deviance": float(losses["uniform"].mean()),
                     "Holdout MAE": float(np.mean(np.abs(holdout.target - constant * test_scale)))})
        self.metrics, self.improvements = pd.DataFrame(rows), {}
        for baseline in ("uniform", "context", "birthday"):
            delta = losses[baseline] - losses["full"]
            lo, hi = block_mean_interval(delta)
            self.improvements[baseline] = {"mean": float(delta.mean()), "lower": lo, "upper": hi}
        folds = self.fold_results.pivot(index="Fold", columns="Method", values="Deviance")
        stable = int(((folds.full < folds.context) & (folds.full < folds.birthday)).sum())
        passes = (all(v["lower"] > 0 for v in self.improvements.values()) and stable >= 2
                  and selected["full"][0] < min(selected["context"][0], selected["birthday"][0]))
        self.eligible, self.fitted, self.anchor_date = bool(passes and self.exposure_adjusted), True, d.date.max()
        # Refit after validation. Recent-window spread measures sensitivity,
        # not a calibrated confidence interval.
        reference = random_grids(4000, np.random.default_rng(162))
        for fraction in (1.0, 0.85, 0.7):
            recent = d.iloc[-int(len(d) * fraction):]
            estimator = _fit_count(recent, "full", self.alpha, self.exposure_adjusted)
            pred = self._log_rate(estimator, reference)
            self.ensemble.append((estimator, float(pred.mean()), max(float(pred.std()), 1e-6)))
        if not self.exposure_adjusted:
            self.status = ("Exploratory raw-count model: participation totals are incomplete. "
                           "Sales volume and player choices cannot be separated reliably; learned selection is disabled.")
        elif not passes:
            self.status = ("Validation did not establish a stable advantage over all baselines. "
                           "Learned selection is disabled; use random selection or the birthday heuristic.")
        else:
            self.status = ("The count model passed the chronological comparison for this scope. "
                           "This is evidence about matching-ticket rates, not proof of higher returns.")
        return self

    def _log_rate(self, estimator, b: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(b, columns=BALL_COLS)
        frame["date"] = self.anchor_date
        return np.log(np.maximum(estimator.predict(model_matrix(frame, "full")), 1e-12))

    def score_details(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.eligible:
            raise ValueError("Learned selection requires complete participation data and a passing validation result.")
        scores = np.vstack([-(self._log_rate(m, b) - center) / scale for m, center, scale in self.ensemble])
        mean, spread = scores.mean(0), scores.std(0)
        return mean, spread, mean - spread


@dataclass
class Recommendation:
    grids: pd.DataFrame
    method: str
    notes: list[str] = field(default_factory=list)


def _validate_numbers(values, label):
    values = list(values) if values is not None else []
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not 1 <= x <= 49 for x in values):
        raise ValueError(f"{label} must contain whole numbers between 1 and 49.")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicate numbers.")
    return sorted(values)


def recommend(model: PopularityModel | None = None, *, strategy: str = "random",
              n_grids: int = 3, n_candidates: int = 20_000, max_overlap: int = 2,
              must_include=None, exclude=None, seed: int | None = None) -> Recommendation:
    if strategy not in ("random", "birthday", "learned"):
        raise ValueError("Unknown selection strategy.")
    if not isinstance(n_grids, int) or not 1 <= n_grids <= 20:
        raise ValueError("Request between 1 and 20 grids.")
    if not isinstance(max_overlap, int) or not 0 <= max_overlap <= 4:
        raise ValueError("Maximum overlap must be between 0 and 4.")
    if not isinstance(n_candidates, int) or not 100 <= n_candidates <= 200_000:
        raise ValueError("Use between 100 and 200,000 candidates.")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("The seed must be a non-negative whole number.")
    must, excluded = _validate_numbers(must_include, "Required numbers"), _validate_numbers(exclude, "Excluded numbers")
    if set(must) & set(excluded):
        raise ValueError("A number cannot be both required and excluded.")
    allowed = sorted(set(range(1, 50)) - set(excluded))
    free = sorted(set(allowed) - set(must))
    r, s = 5 - len(must), len(must)
    if r < 0 or len(allowed) < 5:
        raise ValueError("A grid needs exactly five distinct allowed numbers.")
    possible = comb(len(free), r)
    if possible < n_grids:
        raise ValueError(f"Only {possible} distinct grid(s) satisfy the number constraints.")
    if n_grids > 1:
        minimum_overlap = max(s, 10 - len(allowed))
        if max_overlap < minimum_overlap:
            raise ValueError(f"These constraints force at least {minimum_overlap} shared numbers. Increase maximum overlap.")
        t = max_overlap - s
        bound = comb(len(free), t + 1) // comb(r, t + 1)
        if n_grids > bound:
            raise ValueError(f"At most {bound} grids can satisfy this overlap bound. Reduce the request or increase overlap.")
    rng = np.random.default_rng(seed)
    if possible <= n_candidates:
        cand = np.array([sorted(must + list(c)) for c in combinations(free, r)], dtype=int)
    else:
        indices = np.argpartition(rng.random((n_candidates, len(free))), r - 1, axis=1)[:, :r]
        chosen = np.asarray(free)[indices]
        cand = np.sort(np.column_stack([np.tile(must, (len(chosen), 1)), chosen]), axis=1).astype(int)
        cand = np.unique(cand, axis=0)
    mean, spread = np.full(len(cand), np.nan), np.full(len(cand), np.nan)
    if strategy == "learned":
        if model is None:
            raise ValueError("Load and validate a model first.")
        mean, spread, score = model.score_details(cand)
    elif strategy == "birthday":
        score = (cand > 31).sum(1).astype(float)
    else:
        score = np.zeros(len(cand))
    # Randomize within the best feasible decile, then restart greedy packing
    # when necessary. A failed search is never presented as a complete result.
    best = []
    for _ in range(12):
        remaining, selected = np.arange(len(cand)), []
        while len(remaining) and len(selected) < n_grids:
            pool = remaining if strategy == "random" else remaining[score[remaining] >= np.quantile(score[remaining], 0.9)]
            idx = int(rng.choice(pool))
            selected.append(idx)
            overlaps = np.isin(cand[remaining], cand[idx]).sum(1)
            remaining = remaining[overlaps <= max_overlap]
        if len(selected) > len(best):
            best = selected
        if len(best) == n_grids:
            break
    if len(best) != n_grids:
        raise ValueError(f"The search found only {len(best)} of {n_grids} grids. "
                         "Try another seed, fewer grids or a larger overlap limit; feasibility was not proven.")
    chosen = cand[best]
    out = pd.DataFrame(chosen, columns=[f"N{i}" for i in range(1, 6)])
    out.insert(0, "Grid", np.arange(1, len(out) + 1))
    out["Chance"] = rng.integers(1, 11, len(out))
    out["Numbers above 31"], out["Strategy"] = (chosen > 31).sum(1), strategy
    if strategy == "learned":
        out["Model score"], out["Refit spread"], out["Conservative score"] = mean[best], spread[best], score[best]
    notes = ["Chance is sampled uniformly. It affects the jackpot and several lower ranks.",
             "Overlap constraints change dependence between grids, not any individual grid's draw probability."]
    if strategy == "birthday":
        notes.append("Birthday avoidance is an uncalibrated heuristic, not a measured popularity advantage.")
    if strategy == "learned":
        notes.append("Scores describe a matching-ticket-rate proxy. Refit spread is not a confidence interval or an expected euro gain.")
    return Recommendation(out, strategy, notes)


def expected_value_note() -> str:
    return ("A simple grid costs €2.20. The optional 2nd Tirage adds €0.80. "
            "All valid grids have the same draw probability. This app does not estimate expected euro returns. "
            "Choosing a less popular grid may reduce sharing at shared prize ranks, but that benefit is not established here.")
