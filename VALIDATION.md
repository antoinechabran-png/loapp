# Validation report — Loto Grid Explorer v2

Completed on 26 October 2026.

- 26 deterministic engine tests passed.
- Streamlit AppTest flows passed: no-data startup, random generation, changed
  constraint invalidation, impossible constraints, loaded synthetic data,
  model fitting, learned generation, optional second-draw changes, absent
  second-draw data, single-date scope and empty Saturday scope.
- Final Python syntax checks passed.
- Tests covered known-signal and null synthetic count models, missing exposure,
  training-only scaling, holdout isolation, rank mappings, zero/missing winners,
  malformed inputs, special-game exclusion, constrained sampling, reproducibility,
  and corrected frequency-test calibration.

Runtime: Python 3.12; NumPy 2.5.4; pandas 3.1.0; SciPy 1.18.1;
scikit-learn 1.9.1; Streamlit 1.66.1.

No real FDJ historical-performance claim is made. The local archive folder was
empty, and a prior download attempt was blocked by the network. Learned selection
requires verified participation totals and a passing chronological comparison.
The app's validation concerns winner-count prediction, not realized ticket returns.

The archive includes the three original files in original_v1/ for recovery.
The user's existing Downloads/loto_app folder was also updated in place.
