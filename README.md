# Loto Grid Explorer

An experimental Streamlit app for standard French Loto. Under a fair draw, every
valid grid has the same probability. The app now includes a separate experiment
testing whether persistent draw biases predict later results better than fairness.
It does not establish that its selections make money or estimate expected returns.

## New: experimental draw probabilities and weekly top ten

Open **Draw probabilities** and enable **Enable draw-probability experiment**.
Select Main draw or 2nd Tirage and the historical scope in the sidebar. At least
400 validated draw dates are needed. Unlike the popularity model, this experiment
needs only drawn numbers and (for the main game) Chance; sales and payouts are
irrelevant. After enabling it, new uploads automatically recompute the cached fit.

Choose a week to display ten outcomes and download the CSV. The table shows:

- Rank, five numbers and Chance (Chance is omitted for the second draw).
- Estimated model probability, fair-draw probability, their ratio and equivalent
  '1 in' odds. These are conditional model estimates, not verified physical odds.
- A sensitivity range across refits, when a nonuniform model is used.
- Changes against the previous dataset/mode in the current browser session,
  the selected week, data cutoff and an outcome-data fingerprint.

If validation fails, the default is a reproducible random list with **all ten
tied at rank 1 and equal probabilities**. When a nonuniform candidate beats the
rolling baseline but fails final validation, you may explicitly show its
unvalidated exploratory ranking. If fairness wins tuning, no forced alternative
ranking is fabricated. Weeks rotate tied fallback lists, not winning odds.
Uploading new data does not replace a week's tied fallback tickets merely for
novelty. Repeated identical uploads reproduce the same result for the same week.

For a nonuniform fit, the list is the exact highest-probability ten complete
outcomes under the fitted model. It has no diversity constraint: outcomes can
share four or five main numbers and differ only in Chance. This is distinct from
the diversified grid selector in the Choose grids tab. A week labels the list;
probabilities apply to one draw, not the chance of winning sometime that week.

### Probability model

For each unordered subset S of five of 49 balls, use

`P(S) = exp(sum(theta[i] for i in S)) / sum(exp(sum(theta[i] for i in T)) for all size-5 T)`.

The denominator is the fifth elementary symmetric polynomial in the positive
weights `exp(theta)`. Dynamic programming computes it exactly, including
sampling without replacement. Equal weights give `1 / C(49,5)`. Chance is an
independent ten-category distribution; full main-draw probabilities multiply
the subset probability by its Chance probability. This conditional-Bernoulli
(also called conditional-Poisson) subset family is a statistical alternative,
not a claim about the mechanical extraction process or every possible bias.
Background: [product-weight subset distribution](https://timvieira.github.io/conditional-poisson-sampling/).

Fit per-ball weights to observed inclusion counts, augmented by 100, 1,000 or
10,000 equivalent fair draws. Chance uses the same equivalent uniform prior
count. An infinite-strength option is exactly uniform. This shrinks small
departures toward fairness and gives every valid outcome nonzero probability.
The fitted process is stationary within the scope: no overdue-number theory,
pair interactions, recency tuning or weekday preference enters this experiment.
It cannot detect every form of bias or account for undocumented machine changes.

### Predictive evaluation

The latest 20% of dates is held out. Three expanding folds within the earlier
80% select prior strength by average log probability assigned to unseen complete
outcomes, with ties preferring stronger shrinkage. Equal probabilities are an
explicit candidate. The chosen fit is trained on the earlier 80%, frozen and
evaluated on the last 20%. Holdout data never choose prior strength.

To pass, the finite-strength model must improve average rolling log score,
improve in at least two of three rolling folds, have a positive lower 95%
moving-block-bootstrap interval for mean holdout log-score gain, and improve
in at least two of three chronological holdout sections. The block interval
uses 1,000 resamples with blocks up to ten whole draws. It is approximate, not
a posterior probability of physical bias. A gain in nats per draw is not a
percentage uplift in a selected ticket's winning chance.

The displayed distribution is refitted on all supplied outcomes after evaluation.
Ranking sensitivity ranges use the 10th/90th percentiles of 40 whole-draw block
refits at fixed chosen strength. They are not calibrated confidence intervals,
do not incorporate hyperparameter uncertainty, and do not adjust for selecting
the highest-scoring outcomes. The normalizer, likelihood gradient and exact
top-ten search are tested against explicit small-case calculations.

Repeated scope searches and reusing a refreshed holdout each week can overfit
the evidence. A pass is conditional on this analysis and must be confirmed on
new, unseen future draws. It evaluates the full probability model, **not** the
realized hit rate of the top ten. No real FDJ predictive advantage has been
demonstrated in the delivered version; the tests use synthetic fair and biased
processes. Historical data were not available locally during development.

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
python -m unittest -v test_loto_engine.py test_draw_model.py
```

The deterministic synthetic checks cover malformed data, rank mappings, dates,
zero counts, exposure, chronology, holdout isolation, null/signal models,
randomness calibration, constraint feasibility and reproducibility. Synthetic
success is not evidence that the model works on French Loto history.
