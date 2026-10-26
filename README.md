# Loto Grid Explorer

An experimental Streamlit app for standard French Loto. Every valid grid has
the same probability of being drawn. This app does not predict winning numbers,
estimate expected euro returns, or establish that its selections make money.

## Run

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload semicolon CSV/ZIP archives from
[FDJ](https://www.fdj.fr/jeux-de-tirage/loto/historique), or place them in `data/`.
Use ordinary Loto archives, not Super Loto or Grand Loto. Nothing is uploaded
to an external prediction service. The app runs locally.

## Selection methods

- **Uniform random:** default, subject to the user's number and overlap constraints.
- **Birthday avoidance:** favors more numbers above 31. This is a transparent,
  uncalibrated heuristic with no claimed percentage improvement.
- **Experimental model:** available only after complete participation data and
  passing chronological validation. Samples among the highest-scoring feasible
  decile, using mean standardized score minus spread across three refits.
  Scores are not measured popularity percentiles. Refit spread is sensitivity
  to recent training windows, not a confidence interval.

All valid patterns are allowed, including consecutive numbers, one parity,
all numbers above 31, and past winning combinations. Constraints are checked
before sampling. The search returns the full requested set or an explanation;
failure of its bounded greedy search is not a proof of infeasibility.

## Statistical target

Choose one game and one match tier. Standard November 2019+ rank mappings are:

| Matching main balls | Main-draw ranks, combined over Chance | Second-draw rank |
| --- | --- | --- |
| 2 | 7 + 8 | 4 |
| 3 | 5 + 6 | 3 |
| 4 | 3 + 4 | 2 |
| 5 | 1 + 2 | 1 |

Winner counts (`nombre_de_gagnant_au_rang...`) are retained, including zero.
Missing counts are not imputed as zero. Older draws can be inspected in the
frequency diagnostics but are excluded from modelling. Payouts are retained
for inspection; they are not pooled across ranks, fixed/shared prizes or
different payout regimes. Combining Chance ranks models main-ball matching
counts only, not the jackpot's six-number ticket distribution.

For verified participation totals, add these optional columns to the same CSV:

- `total_combinations`: simple main-draw combinations entered that day.
- `total_combinations_second_tirage`: simple combinations entered in the optional draw.

These must cover the same population as the winner counts. They must not be
players, receipts, turnover, or winning tickets. Totals are not inferred from
payouts or substituted from the other game. Missing totals anywhere in the
selected model data mean the entire fit uses raw counts and is exploratory;
learned selection remains disabled. The ordinary archives may not supply these
totals, so this restriction can apply even after uploading valid archives.

With exposure available, the model fits winner rates per million entries using
Poisson regression with exposure-weighted loss. Predicting a count multiplies
that rate by the actual exposure. This is equivalent, up to constants and
regularization scaling, to a Poisson log-exposure offset. Otherwise it predicts
raw counts. Poisson is used as a regularized mean model; independent ticket
choices and exact Poisson variance are not assumed for uncertainty intervals.

The full model uses 48 number indicators (49 is the reference) and four nonlinear
patterns: consecutive pairs, maximum repeated final digits, range and occupied
decades. Linear duplicate aggregates such as sum, parity and birthday count
are omitted. Weekday and a linear time trend are nuisance context features.
No coefficients are presented as causal player preferences.

## Validation

At least 320 distinct usable dates are required. The latest 20% are a final
holdout. Within the earlier 80%, three expanding chronological folds choose
regularization from 0.1, 1 and 10. Every scaler is fitted on its own training
window. Each fit concerns only one draw type; date-based splits keep same-date
records together. Hyperparameters never use the final holdout.

Baselines are a constant count/rate (no grid preferences), weekday/time context,
and context plus birthday count. All fitted methods have the same alpha search.
Metrics are Poisson deviance and count MAE, not realized ticket returns.

Learned selection requires:

1. Complete participation totals.
2. Lower aggregate rolling deviance than both fitted baselines, and an advantage
   in at least two of the three folds.
3. Positive lower 95% block-bootstrap bounds for holdout deviance improvement
   against all three baselines.

The moving-block bootstrap uses blocks of up to ten dates and 500 resamples.
Its intervals are approximate and conditional on the selected scope. Changing
periods, games and tiers repeatedly can overfit this evidence. A pass does not
prove improved ticket returns. The model is refitted after evaluation on all,
the latest 85%, and the latest 70% of the selected data. The holdout evaluation
applies to the fitted full count model, not an independently evaluated portfolio
selection policy or its three-refit stability score.

Matching-subset counts do not uniquely determine exact-grid popularity.
Predicting fewer matching tickets for a hypothetical draw is only a proxy;
lower-tier payouts for a chosen ticket depend on many possible winning draws.
No jackpot-sharing probability or expected-euro uplift is reported. A further
study would require verified sales data, rank-specific prize allocations,
explicit ticket-selection models, and an untouched policy evaluation period.

## Draw diagnostics

The Pearson frequency statistic is multiplied by 48/44 for drawing five of 49
without replacement. Its large-sample reference is chi-square with 48 degrees
of freedom. The simulation option generates 999 complete fair draw histories
and uses the finite-simulation correction `(exceedances + 1)/(simulations + 1)`.
It displays Monte Carlo standard error. This tests aggregate frequencies, not
all forms of dependence or fairness.

Hot/cold tests use only earlier draws, randomize ties reproducibly, include a
random baseline, and show moving-block bootstrap intervals when at least 20
test draws are available. Expected matches under fair independent draws are
25/49 for every past-based choice. Multiple testing still applies.

## Data and UI protections

The loader rejects invalid or missing dates, non-integer/repeated/out-of-range
balls, and invalid main-draw Chance values. Duplicate same-date records are
merged only if their nonmissing data agree; conflicting records are excluded.
ZIP contents are read in memory with size limits. Main and optional-game
winner counts and participation totals are separate.

The cache key hashes every input byte plus the selected scope. Generated grids
are cleared when inputs or options change. Chance is selected uniformly and
affects several main-draw ranks, not just the jackpot. Second draw is optional
and off by default; budgets use the actual option choice.

Prices checked 26 October 2026: EUR 2.20 per simple grid, plus EUR 0.80 for
2nd Tirage. FDJ reports a game-level return of 54.35% from 4 May 2026; this is
not an estimate of any selection strategy's return.

Sources:
- [FDJ prices](https://www.fdj.fr/jeux-de-tirage/loto/comment-jouer)
- [FDJ archives and return-rate notice](https://www.fdj.fr/jeux-de-tirage/loto/historique)
- [FDJ 2025 rank definitions](https://cdn-media.fdj.fr/static/contrib/files/pdf/2025-03-reglement-loto.pdf)

## Tests

```powershell
python -m unittest -v test_loto_engine.py
```

The deterministic synthetic checks cover malformed data, rank mappings, dates,
zero counts, exposure, chronology, holdout isolation, null/signal models,
randomness calibration, constraint feasibility and reproducibility. Synthetic
success is not evidence that the model works on French Loto history.
