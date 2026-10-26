"""
Loto Grid Recommender - Streamlit app
Run:  streamlit run app.py
Data: download the ZIP archives from https://www.fdj.fr/jeux-de-tirage/loto/historique
      and either upload them in the sidebar or drop them in a ./data folder.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import loto_engine as E

st.set_page_config(page_title="Loto Grid Recommender", page_icon="🎱", layout="wide")
DATA_DIR = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading FDJ archives…")
def load(files: tuple[tuple[str, bytes], ...]):
    return E.normalise(E.read_files(list(files)))


@st.cache_resource(show_spinner="Learning which grids players over-play…")
def fit_model(key: str, _df: pd.DataFrame):
    return E.PopularityModel().fit(_df)


st.sidebar.header("1 · Data")
uploads = st.sidebar.file_uploader("FDJ Loto archives (.zip or .csv)",
                                   type=["zip", "csv"], accept_multiple_files=True)
files = [(u.name, u.getvalue()) for u in uploads or []]
if not files and DATA_DIR.exists():
    files = [(p.name, p.read_bytes()) for p in sorted(DATA_DIR.iterdir())
             if p.suffix.lower() in (".zip", ".csv")]
    if files:
        st.sidebar.caption(f"Using {len(files)} file(s) from ./data")

events, skipped = load(tuple(files)) if files else (pd.DataFrame(), [])
if skipped:
    st.sidebar.warning("Skipped (not current 5/49 + N° Chance format):\n\n- " + "\n- ".join(skipped))

if not events.empty:
    st.sidebar.header("2 · Scope")
    saturday_only = st.sidebar.checkbox("Saturday draws only", value=False,
        help="Saturday players may pick slightly differently. Uses fewer draws.")
    dmin, dmax = events["date"].min().date(), events["date"].max().date()
    period = st.sidebar.slider("Period", min_value=dmin, max_value=dmax, value=(dmin, dmax))
    scope = events[(events["date"].dt.date >= period[0]) & (events["date"].dt.date <= period[1])]
    if saturday_only:
        scope = scope[scope["date"].dt.dayofweek == 5]
    main = scope[scope["kind"] == "main"]
    second = scope[scope["kind"] == "second"]
    model = fit_model(f"{len(scope)}-{period}-{saturday_only}", scope)
else:
    scope = main = second = pd.DataFrame()
    model = E.PopularityModel()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🎱 Loto Grid Recommender")
st.markdown(
    "**Principle.** All combinations have the same chance of being drawn, and past draws "
    "don't predict future ones. But prizes are **shared between winners**, so this app "
    "recommends grids that **few other players pick**: same odds, a bigger share if you win."
)
if events.empty:
    st.info("No data loaded yet. Download the Loto archives (ZIP) from "
            "[fdj.fr › Loto › Historique](https://www.fdj.fr/jeux-de-tirage/loto/historique) "
            "and upload them in the sidebar. Without data, recommendations use a prior "
            "based on known player habits (birthdays, sequences, lucky numbers).")

tab_rec, tab_rand, tab_model, tab_data, tab_odds = st.tabs(
    ["🎯 Recommend grids", "🎲 Is it random?", "👥 What players over-play",
     "📄 Data", "💶 Odds & cost"])

# --------------------------------------------------------------------------- #
# Recommend
# --------------------------------------------------------------------------- #
with tab_rec:
    if model.fitted:
        st.success(f"Popularity model learnt from {model.n_obs:,} draws "
                   f"(main + 2nd Tirage) · cross-validated R² = {model.cv_r2:.2f}")
    else:
        st.warning("Using the built-in prior (no payout data available to learn from).")

    c1, c2, c3 = st.columns(3)
    n_grids = c1.number_input("Number of grids", 1, 20, 3)
    max_overlap = c2.slider("Max numbers shared between your grids", 0, 4, 2,
                            help="Lower = more diverse grids, less risk of winning only shared small ranks twice.")
    seed_txt = c3.text_input("Random seed (optional)", "",
                             help="Leave empty for new grids each time; set a value to reproduce a result.")
    c4, c5 = st.columns(2)
    must = c4.multiselect("Numbers you insist on (max 2)", list(range(1, 50)), max_selections=2)
    excl = c5.multiselect("Numbers to exclude", list(range(1, 50)))

    if st.button("Generate my grids", type="primary"):
        try:
            seed = int(seed_txt) if seed_txt.strip() else None
            rec, ref = E.recommend(model, events if not events.empty else None,
                                   n_grids=int(n_grids), max_overlap=max_overlap,
                                   must_include=must, exclude=excl, seed=seed)
            st.session_state["rec"] = rec
        except ValueError as e:
            st.error(str(e))

    if "rec" in st.session_state:
        rec = st.session_state["rec"]
        for _, r in rec.iterrows():
            nums = "  ".join(f"**{int(r[f'N{i}']):02d}**" for i in range(1, 6))
            st.markdown(f"#### Grid {r['Grid']}:  {nums}  ·  Chance **{int(r['Chance'])}**")
            st.caption(f"{r['Less popular than']} · sum {r['Sum']} · {r['Nb > 31']} numbers above 31")
        st.markdown("Tick **2nd Tirage** when validating: the same 5 numbers play a second draw (+€0.80).")
        st.download_button("Download grids (CSV)", rec.to_csv(index=False, sep=";").encode(),
                           "loto_grids.csv", "text/csv")
        st.caption("The N° Chance is drawn uniformly at random: it only matters for the jackpot "
                   "and there is too little data to estimate its popularity.")

# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #
with tab_rand:
    if main.empty:
        st.info("Load data to run the randomness checks.")
    else:
        which = st.radio("Draw", ["Main draw", "2nd Tirage"], horizontal=True)
        d = main if which == "Main draw" else second
        if d.empty:
            st.info("No 2nd Tirage data in the loaded files.")
        else:
            ft = E.frequency_table(d)
            chi2, p = E.chi_square_uniformity(d)
            a, b, c = st.columns(3)
            a.metric("Draws", f"{len(d):,}")
            b.metric("Chi-square (48 df)", f"{chi2:.1f}")
            c.metric("p-value", f"{p:.3f}")
            st.markdown(
                "p-value above 0.05 → frequencies are **consistent with pure chance**. "
                "The \"most drawn numbers\" are just the normal noise of a fair draw."
                if p > 0.05 else
                "p-value below 0.05: worth a look, but with many checks some low p-values "
                "happen by chance alone.")
            st.bar_chart(ft.set_index("number")[["count", "expected"]], stack=False)

            st.subheader("Backtest: do hot or cold numbers work?")
            w = st.slider("Look-back window (draws)", 20, 500, 100, step=10)
            st.dataframe(E.hot_numbers_backtest(d, w).style.format(
                {"avg matches / draw": "{:.3f}", "draws tested": "{:,.0f}"}), hide_index=True)
            st.caption("Any grid matches on average 25/49 ≈ 0.510 numbers per draw. "
                       "If hot or cold numbers had an edge, they would be clearly above that.")
            with st.expander("Full frequency table"):
                st.dataframe(ft.round(2), hide_index=True)

# --------------------------------------------------------------------------- #
# Popularity model
# --------------------------------------------------------------------------- #
with tab_model:
    if not model.fitted:
        st.info("The model needs payout columns (rapport_du_rang…) from the FDJ archives. "
                "Currently using this prior (effect on payout per feature unit):")
        st.dataframe(pd.Series(E.PopularityModel.PRIOR, name="effect").to_frame())
    else:
        st.markdown(
            "Target: average z-score of **log payout** across shared prize ranks. "
            "A low payout means many winners, so the drawn numbers were popular. "
            f"Ranks used: {', '.join(model.ranks_used)}.")
        st.subheader("Effect of each number on payout")
        ne = model.number_effect.rename("effect").to_frame()
        st.bar_chart(ne)
        st.caption("Negative = over-played (you'd share more). Positive = under-played.")
        st.subheader("Effect of grid patterns (per unit)")
        st.dataframe(model.pattern_effect.rename("effect").round(4).to_frame())

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with tab_data:
    if events.empty:
        st.info("No data loaded.")
    else:
        st.write(f"{len(scope):,} draw events in scope "
                 f"({len(main):,} main, {len(second):,} 2nd Tirage).")
        st.dataframe(scope.sort_values("date", ascending=False), hide_index=True)

# --------------------------------------------------------------------------- #
# Odds
# --------------------------------------------------------------------------- #
with tab_odds:
    st.dataframe(pd.DataFrame({"Prize": list(E.ODDS),
                               "Odds (1 in …)": [f"{v:,}" for v in E.ODDS.values()]}),
                 hide_index=True)
    st.markdown(E.expected_value_note(model))
    n_weeks = st.slider("Grids per week", 1, 10, 1)
    yearly = n_weeks * 52 * (E.PRICE_GRID + E.PRICE_SECOND)
    st.metric("Yearly spend (Saturday only, with 2nd Tirage)", f"€{yearly:,.0f}")
    st.caption("Only around 55–60% of stakes are paid back to players, so the expected return "
               "is negative whatever the grid. Play for fun, with money you can afford to lose. "
               "Help: Joueurs Info Service, 09 74 75 13 13.")
