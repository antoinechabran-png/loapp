"""Deterministic checks; synthetic data is not evidence of real-world benefit."""
import io
import unittest
import zipfile

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import loto_engine as E


def synthetic_draws(n=500, signal=False, exposure=True, seed=123):
    rng = np.random.default_rng(seed)
    b = E.random_grids(n, rng)
    d = pd.DataFrame(b, columns=E.BALL_COLS)
    d["date"] = pd.date_range("2020-01-01", periods=n, freq="2D")
    d["kind"], d["schema"] = "main", "standard_2019"
    entries = rng.integers(1_000_000, 4_000_000, n)
    # Known selection effect deliberately independent of birthdays.
    log_rate = 5.5 + (1.0 * (b == 37).any(1) - 0.8 * (b == 42).any(1) if signal else 0)
    winners = rng.poisson(np.exp(log_rate) * entries / 1e6)
    d["winners_r5"], d["winners_r6"] = winners // 10, winners - winners // 10
    d["entries"] = entries if exposure else np.nan
    d["chance"] = rng.integers(1, 11, n)
    return d


def archive_frame():
    return pd.DataFrame({"boule_1": ["1"], "boule_2": ["2"], "boule_3": ["3"],
                         "boule_4": ["4"], "boule_5": ["5"], "numero_chance": ["7"],
                         "date_de_tirage": ["26/10/2026"], "__source": ["loto.csv"],
                         "nombre_de_gagnant_au_rang5": ["0"], "nombre_de_gagnant_au_rang6": ["12"],
                         "total_combinations": ["1 000 000"]})


class LoadingTests(unittest.TestCase):
    def test_bad_draws_rejected_without_truncating_fractional_balls(self):
        variants = []
        for col, value in [("boule_2", "1"), ("boule_1", "1.5"), ("boule_5", "50"),
                           ("numero_chance", "99"), ("numero_chance", "0"), ("date_de_tirage", "nonsense")]:
            row = archive_frame()
            row.loc[0, col] = value
            variants.append(row)
        events, issues = E.normalise([pd.concat(variants, ignore_index=True)])
        self.assertTrue(events.empty)
        self.assertTrue(any("6 invalid main" in x for x in issues))

    def test_zero_counts_and_exposure_preserved(self):
        d, _ = E.normalise([archive_frame()])
        self.assertEqual(d.winners_r5.iloc[0], 0)
        self.assertEqual(d.entries.iloc[0], 1_000_000)
        target, _ = E.prepare_target(d, "main", 3)
        self.assertEqual(target.target.iloc[0], 12)

    def test_missing_count_is_not_zero(self):
        raw = archive_frame()
        raw.loc[0, "nombre_de_gagnant_au_rang5"] = None
        d, _ = E.normalise([raw])
        target, _ = E.prepare_target(d, "main", 3)
        self.assertTrue(target.empty)

    def test_duplicates_merged_but_conflicts_quarantined(self):
        d, _ = E.normalise([archive_frame(), archive_frame()])
        self.assertEqual(len(d), 1)
        other = archive_frame()
        other.loc[0, "numero_chance"] = "8"
        d, issues = E.normalise([archive_frame(), other])
        self.assertTrue(d.empty)
        self.assertTrue(any("conflicting" in x for x in issues))

    def test_second_draw_has_its_own_counts_and_exposure(self):
        raw = archive_frame()
        for i in range(1, 6):
            raw[f"boule_{i}_second_tirage"] = str(i + 20)
        raw["nombre_de_gagnant_au_rang3_second_tirage"] = "41"
        raw["total_combinations_second_tirage"] = "300000"
        d, _ = E.normalise([raw])
        target, _ = E.prepare_target(d, "second", 3)
        self.assertEqual(target.target.iloc[0], 41)
        self.assertEqual(target.entries.iloc[0], 300000)

    def test_csv_zip_and_bad_zip(self):
        raw = archive_frame().drop(columns="__source").to_csv(index=False, sep=";").encode()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("loto.csv", raw)
        d, issues = E.load_files([("loto.zip", buf.getvalue()), ("bad.zip", b"invalid")])
        self.assertEqual(len(d), 1)
        self.assertTrue(any("unreadable ZIP" in x for x in issues))

    def test_content_hash_changes_even_with_equal_length(self):
        self.assertNotEqual(E.file_fingerprint([("a.csv", b"abc")]), E.file_fingerprint([("a.csv", b"abd")]))

    def test_french_number_formats(self):
        out = E._to_num(pd.Series(["1 234,50", "1\u202f234,50", "1\u00a0234,50"]))
        self.assertTrue((out == 1234.5).all())

    def test_special_game_and_old_schema_not_modelled(self):
        special = archive_frame()
        special["__source"] = "superloto.zip/results.csv"
        d, issues = E.normalise([special])
        self.assertEqual(len(d), 1)
        self.assertTrue(E.prepare_target(d, "main", 3)[0].empty)
        self.assertTrue(any("special-game" in x for x in issues))
        old = archive_frame()
        old["date_de_tirage"] = "12/12/2010"
        d, _ = E.normalise([old])
        self.assertEqual(len(d), 1)
        self.assertTrue(E.prepare_target(d, "main", 3)[0].empty)


class StatisticsTests(unittest.TestCase):
    def test_full_feature_matrix_has_no_exact_redundancy(self):
        x = E.design_matrix(E.random_grids(1200, np.random.default_rng(1)))
        self.assertEqual(np.linalg.matrix_rank(x - x.mean(0)), x.shape[1])

    def test_corrected_statistic_has_expected_null_mean(self):
        rng, values = np.random.default_rng(10), []
        for _ in range(300):
            d = pd.DataFrame(E.random_grids(150, rng), columns=E.BALL_COLS)
            q, _ = E.chi_square_uniformity(d)
            values.append(q)
        self.assertAlmostEqual(np.mean(values), 48, delta=2.0)

    def test_simulation_detects_extreme_bias(self):
        d = pd.DataFrame(np.tile([1, 2, 3, 4, 5], (80, 1)), columns=E.BALL_COLS)
        result = E.randomness_check(d, simulations=199)
        self.assertEqual(result["p"], 1 / 200)

    def test_hot_cold_short_and_normal_windows(self):
        d = synthetic_draws(n=160)
        short = E.hot_numbers_backtest(d, 150)
        self.assertTrue(short["95% lower"].isna().all())
        normal = E.hot_numbers_backtest(d, 40)
        self.assertTrue((normal["Draws tested"] == 120).all())
        self.assertTrue((normal["95% lower"] <= normal["95% upper"]).all())

    def test_poisson_loss_handles_zero_winners(self):
        np.testing.assert_allclose(E.poisson_loss([0, 1], [2, 1]), [4, 0])

    def test_chronological_split_groups_dates(self):
        dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=100), 2))
        for train, test in E.chronological_splits(dates):
            self.assertLess(dates.iloc[train].max(), dates.iloc[test].min())
            self.assertEqual(len(train) % 2, 0)
            self.assertEqual(len(test) % 2, 0)


class RecommendationTests(unittest.TestCase):
    def test_constraints_seed_and_complete_result(self):
        kwargs = dict(must_include=[1, 2], exclude=[3, 4], max_overlap=2, n_grids=10, seed=17)
        a, b = E.recommend(**kwargs).grids, E.recommend(**kwargs).grids
        assert_frame_equal(a, b)
        numbers = a[[f"N{i}" for i in range(1, 6)]].to_numpy()
        self.assertEqual(len(a), 10)
        self.assertTrue(np.isin(numbers, [1, 2]).sum(1).min() == 2)
        self.assertFalse(np.isin(numbers, [3, 4]).any())
        for x, y in combinations_local(numbers):
            self.assertLessEqual(len(set(x) & set(y)), 2)

    def test_impossible_requests_raise_actionable_errors(self):
        cases = [({"must_include": [1, 2], "max_overlap": 1}, "force at least 2"),
                 ({"n_grids": 20, "max_overlap": 0}, "At most 9"),
                 ({"must_include": [1], "exclude": [1]}, "both required and excluded"),
                 ({"seed": -1}, "non-negative"),
                 ({"exclude": list(range(1, 47))}, "five distinct")]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                E.recommend(**kwargs)

    def test_no_arbitrary_low_high_parity_or_historical_filters(self):
        for required in ([1, 2, 3, 4, 5], [32, 34, 36, 38, 40]):
            out = E.recommend(n_grids=1, must_include=required, strategy="birthday", seed=1).grids
            self.assertEqual(out[[f"N{i}" for i in range(1, 6)]].iloc[0].tolist(), required)
        out = E.recommend(exclude=list(range(32, 50)), seed=1)
        self.assertEqual(len(out.grids), 3)

    def test_unqualified_model_cannot_recommend(self):
        with self.assertRaisesRegex(ValueError, "complete participation"):
            E.recommend(E.PopularityModel(), strategy="learned")

    def test_single_available_combination(self):
        out = E.recommend(n_grids=1, exclude=list(range(6, 50)), seed=1).grids
        self.assertEqual(out[[f"N{i}" for i in range(1, 6)]].iloc[0].tolist(), [1, 2, 3, 4, 5])


def combinations_local(rows):
    from itertools import combinations
    return combinations(rows, 2)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signal_data = synthetic_draws(signal=True)
        cls.signal_model = E.PopularityModel().fit(cls.signal_data)

    def test_known_signal_survives_validation_and_generates(self):
        m = self.signal_model
        self.assertTrue(m.fitted)
        self.assertTrue(m.eligible, m.status)
        self.assertEqual(len(m.metrics), 4)
        self.assertEqual(len(m.ensemble), 3)
        result = E.recommend(m, strategy="learned", seed=1)
        self.assertEqual(len(result.grids), 3)
        self.assertTrue(np.isfinite(result.grids["Model score"]).all())

    def test_no_signal_does_not_pass_gate(self):
        model = E.PopularityModel().fit(synthetic_draws(signal=False))
        self.assertTrue(model.fitted)
        self.assertFalse(model.eligible)

    def test_missing_participation_disables_learned_selection(self):
        d = self.signal_data.copy()
        d.loc[5, "entries"] = np.nan
        model = E.PopularityModel().fit(d)
        self.assertTrue(model.fitted)
        self.assertFalse(model.exposure_adjusted)
        self.assertFalse(model.eligible)

    def test_holdout_changes_do_not_change_tuning_or_rolling_scores(self):
        d = self.signal_data.copy()
        d.loc[d.date >= self.signal_model.holdout_start, "winners_r6"] *= 3
        changed = E.PopularityModel().fit(d)
        self.assertEqual(changed.alpha, self.signal_model.alpha)
        assert_frame_equal(changed.fold_results, self.signal_model.fold_results)

    def test_scaler_sees_training_only(self):
        d, _ = E.prepare_target(self.signal_data, "main", 3)
        fitted = E._fit_count(d.iloc[:100], "full", 1.0, True)
        np.testing.assert_allclose(fitted.named_steps["standardscaler"].mean_,
                                   E.model_matrix(d.iloc[:100], "full").mean(0))

    def test_zero_and_short_data_fall_back(self):
        self.assertFalse(E.PopularityModel().fit(pd.DataFrame()).fitted)
        self.assertFalse(E.PopularityModel().fit(self.signal_data.iloc[:30]).fitted)
        d = self.signal_data.copy()
        d[["winners_r5", "winners_r6"]] = 0
        self.assertFalse(E.PopularityModel().fit(d).fitted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
