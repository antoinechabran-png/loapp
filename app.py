"""Run with: python -m streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

import loto_engine as E

st.set_page_config(page_title="Loto Grid Explorer", page_icon="🎱", layout="wide")
DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(show_spinner="Reading and validating archives…")
def load(files):
    return E.load_files(list(files))


@st.cache_data(show_spinner="Comparing models on later draws…")
def fit_model(content_scope_key: str, _df: pd.DataFrame, kind: str, matches: int):
    # The key includes a SHA-256 hash of ALL input bytes, plus the exact scope.
    # It deliberately does not rely on Streamlit's sampling of large DataFrames.
    return E.PopularityModel().fit(_df, kind=kind, matches=matches)


@st.cache_data(show_spinner="Simulating fair draws…")
def diagnose(content_scope_key: str, _df: pd.DataFrame):
    return E.randomness_check(_df)


st.title("🎱 Loto Grid Explorer")
st.markdown("**Every valid grid has the same chance of being drawn.** "
            "This app explores player-selection patterns and possible prize sharing. "
            "It does not predict winning numbers or promise higher returns.")

st.sidebar.header("Data")
uploads = st.sidebar.file_uploader("Standard FDJ Loto archives (.zip or .csv)",
                                   type=["zip", "csv"], accept_multiple_files=True)
files = [(f.name, f.getvalue()) for f in uploads or []]
if not files and DATA_DIR.exists():
    files = [(p.name, p.read_bytes()) for p in sorted(DATA_DIR.iterdir())
             if p.is_file() and p.suffix.lower() in (".zip", ".csv")]
    if files:
        st.sidebar.caption(f"Loaded {len(files)} file(s) from data/.")
fingerprint = E.file_fingerprint(files)
events, issues = load(tuple(files)) if files else (pd.DataFrame(), [])
if issues:
    with st.sidebar.expander(f"Data checks: {len(issues)} message(s)", expanded=True):
        for message in issues[:30]:
            st.warning(message)
        if len(issues) > 30:
            st.caption("The complete list is available in the Data tab.")

scope = events
period, saturday_only = None, False
if not events.empty:
    st.sidebar.header("Scope")
    dmin, dmax = events.date.min().date(), events.date.max().date()
    if dmin < dmax:
        period = st.sidebar.slider("Period", min_value=dmin, max_value=dmax, value=(dmin, dmax))
    else:
        period = (dmin, dmax)
        st.sidebar.caption(f"One draw date: {dmin}")
    saturday_only = st.sidebar.checkbox("Saturday draws only", value=False)
    scope = events[(events.date.dt.date >= period[0]) & (events.date.dt.date <= period[1])]
    if saturday_only:
        scope = scope[scope.date.dt.day_of_week == 5]
else:
    st.info("Upload the [FDJ Loto archives](https://www.fdj.fr/jeux-de-tirage/loto/historique) "
            "or place them in data/. Random selection and the birthday heuristic work without data.")

kind_label = st.sidebar.selectbox("Model draw", ["Main draw", "2nd Tirage"])
kind = "main" if kind_label == "Main draw" else "second"
matches = st.sidebar.selectbox("Model matching numbers", [3, 2, 4, 5],
    help="Models winner counts for exactly this many matching balls. Main ranks with and without Chance are combined.")
scope_key = f"v2|{fingerprint}|{period}|{saturday_only}|{kind}|{matches}"
if st.session_state.get("model_key") != scope_key:
    st.session_state.pop("model", None)
    st.session_state.pop("recommendation", None)
    st.session_state["model_key"] = scope_key
if st.sidebar.button("Fit and validate model", disabled=scope.empty):
    st.session_state.pop("recommendation", None)
    try:
        st.session_state["model"] = fit_model(scope_key, scope, kind, matches)
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        st.session_state.pop("model", None)
        st.sidebar.error(f"Model fitting failed: {exc}")
model = st.session_state.get("model", E.PopularityModel())

rec_tab, model_tab, random_tab, data_tab, cost_tab = st.tabs(
    ["Choose grids", "Model evidence", "Draw diagnostics", "Data", "Odds & budget"])

with rec_tab:
    if model.eligible:
        st.success(model.status)
    else:
        st.info(model.status)
    strategies = {"Uniform random": "random", "Birthday avoidance (heuristic)": "birthday"}
    if model.eligible:
        strategies["Experimental model (validated count proxy)"] = "learned"
    strategy_label = st.selectbox("Selection method", list(strategies))
    strategy = strategies[strategy_label]
    if strategy == "birthday":
        st.caption("Favors more numbers above 31. No minimum is imposed and no percentage gain is claimed.")
    if strategy == "learned":
        st.caption("Samples among high-scoring feasible grids, penalizing disagreement across three recent-window refits. "
                   "A matching-ticket count is only an indirect guide to ticket popularity and prize sharing.")
    a, b, c = st.columns(3)
    n_grids = a.number_input("Number of grids", min_value=1, max_value=20, value=3, step=1)
    overlap = b.slider("Maximum shared numbers", 0, 4, 2,
                       help="Controls dependence between your grids. It does not improve a grid's individual odds.")
    seed_text = c.text_input("Random seed (optional)", help="A non-negative integer reproduces the selection with the same data and settings.")
    a, b = st.columns(2)
    required = a.multiselect("Required numbers", list(range(1, 50)), max_selections=5)
    excluded = b.multiselect("Excluded numbers", list(range(1, 50)))
    include_second = st.checkbox("Include the optional 2nd Tirage (+€0.80 per grid)", value=False)
    settings = (scope_key, model.fitted, model.eligible, strategy, n_grids, overlap,
                tuple(required), tuple(excluded), seed_text, include_second)
    if st.session_state.get("selection_settings") != settings:
        st.session_state.pop("recommendation", None)
        st.session_state["selection_settings"] = settings
    unit_cost = E.PRICE_GRID + (E.PRICE_SECOND if include_second else 0)
    st.caption(f"Cost for this set, for one draw: €{n_grids * unit_cost:.2f}.")
    if strategy == "learned" and kind == "second" and not include_second:
        st.warning("The selected model concerns 2nd Tirage, but this budget does not include that option.")
    if st.button("Generate grids", type="primary"):
        st.session_state.pop("recommendation", None)
        try:
            seed = int(seed_text.strip()) if seed_text.strip() else None
            result = E.recommend(model, strategy=strategy, n_grids=int(n_grids),
                                 max_overlap=overlap, must_include=required, exclude=excluded, seed=seed)
            st.session_state["recommendation"] = result
        except ValueError as exc:
            st.error(str(exc))
    result = st.session_state.get("recommendation")
    if result is not None:
        for _, row in result.grids.iterrows():
            numbers = " · ".join(f"{int(row[f'N{i}']):02d}" for i in range(1, 6))
            st.markdown(f"**Grid {int(row['Grid'])}: {numbers} — Chance {int(row['Chance'])}**")
            if result.method == "learned":
                st.caption(f"Model score {row['Model score']:.2f}; refit spread {row['Refit spread']:.2f}; "
                           f"conservative score {row['Conservative score']:.2f}. Higher scores favor lower predicted matching-ticket rates.")
        for note in result.notes:
            st.caption(note)
        export = result.grids.copy()
        export["2nd Tirage"] = include_second
        export["Cost per grid EUR"] = unit_cost
        if result.method == "learned":
            export["Model draw"], export["Matching numbers"] = kind, matches
        st.download_button("Download grids (CSV)", export.to_csv(index=False, sep=";").encode("utf-8-sig"),
                           "loto_grids.csv", "text/csv")

with model_tab:
    st.write(model.status)
    st.markdown("The model predicts **winner counts for one match tier**, using a Poisson regression. "
                "Main-draw ranks with and without Chance are combined. Main and second draws are fitted separately.")
    st.caption("Only standard Loto draws from November 2019 onward enter the model. "
               "Payouts are retained for inspection but are not pooled across ranks or rule periods.")
    if model.fitted:
        a, b = st.columns(2)
        a.metric("Usable draws", f"{model.n_obs:,}")
        b.metric("Participation adjusted", "Yes" if model.exposure_adjusted else "No — exploratory only")
        st.write(f"Final holdout: {model.holdout_start.date()} to {model.holdout_end.date()}.")
        st.dataframe(model.metrics.round(4), hide_index=True)
        st.caption("Lower deviance and MAE are better. 'uniform' predicts a constant count/rate without number features; "
                   "'context' adds weekday and time; 'birthday' adds the count of numbers at or below 31; 'full' adds number and pattern features. "
                   "These compare count predictions, not realized ticket returns.")
        intervals = pd.DataFrame(model.improvements).T.rename(
            columns={"mean": "Mean improvement", "lower": "95% lower", "upper": "95% upper"})
        st.write("Holdout deviance improvement over each baseline")
        st.dataframe(intervals.round(4))
        st.caption("Intervals use a moving-block bootstrap over draw dates. Positive values favor the full model. "
                   "They are approximate, depend on the chosen scope, and do not establish a monetary advantage. "
                   "Repeatedly changing periods or tiers to find a passing result weakens the evidence.")
        with st.expander("Chronological folds and feature choices"):
            st.dataframe(model.fold_results.round(4), hide_index=True)
            st.write("Three expanding training windows select regularization before the final 20% holdout. "
                     "Each scaler is fitted on its training window. Number 49 is the reference category; "
                     "the full model uses 48 number indicators plus consecutive pairs, repeated final digits, spread and occupied decades. "
                     "Linear duplicate features such as sum and parity are omitted. Coefficients are not presented as causal effects.")
    with st.expander("Participation totals and limitations", expanded=not model.exposure_adjusted):
        st.markdown("If you have verified totals, enrich the archive CSV with `total_combinations` and, for the optional game, "
                    "`total_combinations_second_tirage`. Each value must be the number of **simple combinations entered** "
                    "for that draw, covering the same population as the winner counts. Do not substitute player counts, receipts or revenue.")
        st.write("Without complete totals, raw-count models remain exploratory. Learned selection is enabled only with complete totals, "
                 "a rolling advantage in at least two folds, and positive lower holdout improvement bounds against every baseline. "
                 "Even then, matching subsets do not reveal exact five-number ticket popularity. "
                 "This app does not estimate jackpot sharing or expected euro returns.")

with random_tab:
    draw_choice = st.radio("Draw to inspect", ["Main draw", "2nd Tirage"], horizontal=True)
    diagnostic_kind = "main" if draw_choice == "Main draw" else "second"
    d = scope[scope.kind == diagnostic_kind] if not scope.empty else pd.DataFrame()
    if d.empty:
        st.info("No draws in this scope.")
    else:
        st.write(f"{len(d):,} draws")
        ft = E.frequency_table(d)
        st.bar_chart(ft.set_index("number")[["count", "expected"]], stack=False)
        statistic, p = E.chi_square_uniformity(d)
        st.write(f"Corrected frequency statistic: {statistic:.2f}; approximate p-value: {p:.3f}.")
        if len(d) < 100:
            st.caption("Small sample: prefer the simulation below to the asymptotic approximation.")
        if st.button("Run fair-draw simulation"):
            result = diagnose(f"{scope_key}|{diagnostic_kind}", d)
            st.write(f"Monte Carlo p-value: {result['p']:.3f} "
                     f"(simulation standard error ≈ {result['simulation_se']:.3f}; {result['simulations']} simulations).")
        st.caption("The test accounts for sampling five balls without replacement. A large p-value is not proof of fairness "
                   "or independence. Repeatedly trying periods and tests can produce low p-values by chance.")
        window = st.slider("Hot/cold look-back window", 20, 500, 100, step=10)
        bt = E.hot_numbers_backtest(d, window)
        st.dataframe(bt.round(4), hide_index=True)
        st.caption("Any fixed or past-based grid has expected matches 25/49 ≈ 0.510 under independent fair draws. "
                   "Intervals are approximate moving-block bootstrap intervals; fewer than 20 test draws gives no interval. "
                   "Frequency ties are randomized reproducibly. These diagnostics do not choose the recommended numbers.")

with data_tab:
    st.write(f"{len(scope):,} valid draw events in scope.")
    if not scope.empty:
        st.dataframe(scope.sort_values("date", ascending=False), hide_index=True)
    if issues:
        st.text("\n".join(issues))
    st.caption("Invalid dates, repeated or fractional balls, invalid Chance values and conflicting duplicate records are rejected. "
               "Zero winner counts are retained; missing winner counts are never replaced with zero. "
               "Use standard Loto archives, not Super Loto or Grand Loto files.")

with cost_tab:
    st.dataframe(pd.DataFrame({"Prize": list(E.ODDS), "Odds: 1 in": list(E.ODDS.values())}), hide_index=True)
    st.write(E.expected_value_note())
    weekly = st.slider("Simple grids per week", 1, 50, 3)
    budget_second = st.checkbox("Add 2nd Tirage to the weekly budget", value=False)
    annual = weekly * 52 * (E.PRICE_GRID + (E.PRICE_SECOND if budget_second else 0))
    st.metric("Annual spend (52 weeks)", f"€{annual:,.2f}")
    st.caption("The Numéro Chance affects the jackpot and several lower main-draw ranks. It is not used for 2nd Tirage. "
               "Not all prize ranks are shared pools.")
    st.markdown("FDJ states a **54.35% game-level return rate from 4 May 2026**. "
                "That is not this app's expected return. "
                "[FDJ notice](https://www.fdj.fr/jeux-de-tirage/loto/historique) · "
                "[Prices and play rules](https://www.fdj.fr/jeux-de-tirage/loto/comment-jouer)")
    st.caption(f"Information checked {E.RULES_CHECKED}. Play within a budget you can afford to lose.")
