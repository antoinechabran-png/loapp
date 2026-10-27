# Loto Grid Explorer v3 — validation

27 October 2026

44 deterministic engine tests passed (26 existing + 18 draw-experiment checks).
The added tests verify exact subset normalization, its likelihood gradient,
equal-probability odds, shrinking toward fairness, exact top-outcome search,
fair/biased synthetic processes, holdout isolation, incremental updates,
game separation, weekly reproducibility and stable uniform fallback lists.

Streamlit AppTest checks passed for the existing selectors and model flows,
plus the new probability tab, no-data fallback, biased synthetic fit, top ten,
refit ranges, automatic dataset refresh, previous-snapshot comparison and
missing second-draw data. Final source syntax and ZIP integrity were checked.

No real FDJ predictive advantage has been demonstrated. No draw archives were
present in the user's data folder. A prior archive download was blocked by the
network. All probability-model validation during development used synthetic
outcomes; the app evaluates the user's actual data when supplied.

Runtime: Python 3.12; NumPy 2.5.4; pandas 3.1.0; SciPy 1.18.1;
scikit-learn 1.9.1; Streamlit 1.66.1.

The app was updated in the existing Downloads/loto_app folder. The ZIP contains
a backup of v2 under backup_v2/. No scheduler or background weekly task was created.
