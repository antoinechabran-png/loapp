"""Math, leakage and fair/biased synthetic-data checks for the draw experiment."""
from itertools import combinations
from math import comb
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import loto_engine as E
import draw_model as D


def draw_fixture(n=1000, biased=False, seed=7, kind="main"):
    rng = np.random.default_rng(seed)
    b = E.random_grids(n, rng)
    if biased:
        # Exactly a member of the product-subset family: ball 1 is included
        # with probability .6, and the other selected balls are uniform.
        b = np.array([np.sort(np.r_[1, rng.choice(np.arange(2, 50), 4, replace=False)])
                      if rng.random() < .6 else np.sort(rng.choice(np.arange(2, 50), 5, replace=False))
                      for _ in range(n)])
    d = pd.DataFrame(b, columns=E.BALL_COLS)
    d["date"] = pd.date_range("2020-01-01", periods=n, freq="2D")
    d["kind"], d["schema"] = kind, "standard_2019"
    d["chance"] = rng.integers(1, 11, n)
    if biased and kind == "main":
        p = np.full(10, .5 / 9)
        p[6] = .5
        d["chance"] = rng.choice(np.arange(1, 11), size=n, p=p)
    if kind == "second":
        d["chance"] = np.nan
    return d


class ProbabilityMathTests(unittest.TestCase):
    def test_normalizer_and_marginals_against_enumeration(self):
        theta = np.array([.3, -.8, 1.2, .1, -.2, .6, -.4])
        subsets = np.array(list(combinations(range(7), 3)))
        mass = np.exp(theta[subsets].sum(1))
        z, marginal = D.partition_and_marginals(theta, 3)
        self.assertAlmostEqual(np.exp(z), mass.sum(), places=11)
        exact = np.array([mass[(subsets == i).any(1)].sum() / mass.sum() for i in range(7)])
        np.testing.assert_allclose(marginal, exact, rtol=1e-12)
        self.assertAlmostEqual(marginal.sum(), 3)

    def test_gradient_by_finite_differences(self):
        theta = np.linspace(-.5, .5, 49)
        log_z, marginal = D.partition_and_marginals(theta)
        for i in (0, 20, 48):
            changed = theta.copy()
            changed[i] += 1e-6
            self.assertAlmostEqual((D.partition_and_marginals(changed)[0] - log_z) / 1e-6, marginal[i], places=6)

    def test_uniform_complete_probabilities(self):
        main = D.DrawDistribution.uniform("main")
        second = D.DrawDistribution.uniform("second")
        grid = np.array([[1, 2, 3, 4, 5], [32, 36, 41, 44, 49]])
        np.testing.assert_allclose(np.exp(main.log_prob(grid, np.array([1, 10]))), 1 / (10 * comb(49, 5)))
        np.testing.assert_allclose(np.exp(second.log_prob(grid)), 1 / comb(49, 5))
        _, marginal = D.partition_and_marginals(main.theta)
        np.testing.assert_allclose(marginal, 5 / 49)

    def test_invalid_grids_rejected(self):
        model = D.DrawDistribution.uniform()
        for grid in ([[1, 1, 3, 4, 5]], [[1.1, 2, 3, 4, 5]], [[0, 2, 3, 4, 5]]):
            with self.assertRaises(ValueError):
                model.log_prob(np.array(grid), np.array([2]))
        with self.assertRaises(ValueError):
            model.log_prob(np.array([[1, 2, 3, 4, 5]]), np.array([11]))

    def test_more_shrinkage_moves_toward_uniform(self):
        data = draw_fixture(200, biased=True)
        weak = D.DrawDistribution.fit(data, 100, "main")
        strong = D.DrawDistribution.fit(data, 10000, "main")
        self.assertLess(np.linalg.norm(strong.theta), np.linalg.norm(weak.theta))
        self.assertLess(abs(strong.chance[6] - .1), abs(weak.chance[6] - .1))

    def test_top_search_agrees_with_bruteforce(self):
        theta = np.r_[np.linspace(.8, -.3, 8), np.full(41, -100.)]
        log_z, _ = D.partition_and_marginals(theta)
        chance = np.arange(1, 11, dtype=float) / 55
        model = D.DrawDistribution(theta, log_z, chance, "main", 100)
        candidates = [(tuple(b), c, float(model.log_prob(np.array([b]), np.array([c]))[0]))
                      for b in combinations(range(1, 9), 5) for c in range(1, 11)]
        expected = sorted([x[2] for x in candidates], reverse=True)[:20]
        actual = model.top(20)
        np.testing.assert_allclose([x[2] for x in actual], expected)
        self.assertEqual(len({tuple(x[0]) + (x[1],) for x in actual}), 20)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.biased_data = draw_fixture(biased=True)
        cls.biased = D.DrawProbabilityModel().fit(cls.biased_data)
        cls.fair = D.DrawProbabilityModel().fit(draw_fixture())

    def test_strong_persistent_bias_is_detected(self):
        self.assertTrue(self.biased.eligible, self.biased.status)
        self.assertGreater(self.biased.holdout_interval[0], 0)

    def test_fair_draws_do_not_qualify(self):
        self.assertTrue(self.fair.fitted)
        self.assertFalse(self.fair.eligible, self.fair.status)
        table = self.fair.ranking("2026-W44")
        self.assertEqual(len(table), 10)
        self.assertTrue((table.Rank == 1).all())
        np.testing.assert_allclose(table["Model / fair"], 1)

    def test_top_ten_probabilities_sorted_and_ranges_present(self):
        table = self.biased.ranking("2026-W44", refits=8)
        self.assertEqual(len(table), 10)
        self.assertTrue((np.diff(table["Estimated probability"]) <= 1e-15).all())
        self.assertTrue((table["Refit range 10%"] <= table["Refit range 90%"]).all())
        self.assertTrue((table["Estimated probability"] > 0).all())
        self.assertTrue((table["Estimated probability"] < 1).all())

    def test_holdout_cannot_change_tuning(self):
        changed = self.biased_data.copy()
        mask = changed.date >= self.biased.holdout_start
        changed.loc[mask, E.BALL_COLS] = [45, 46, 47, 48, 49]
        changed.loc[mask, "chance"] = 1
        refit = D.DrawProbabilityModel().fit(changed)
        self.assertEqual(self.biased.strength, refit.strength)
        assert_frame_equal(self.biased.metrics, refit.metrics)
        assert_frame_equal(self.biased.folds, refit.folds)
        self.assertFalse(refit.eligible)

    def test_outcome_snapshot_ignores_payout_data(self):
        a = draw_fixture(20)
        b = a.copy()
        b["payout_r1"], b["winners_r1"], b["entries"] = 1000000, 2, 123456
        first, second = D.DrawProbabilityModel().fit(a), D.DrawProbabilityModel().fit(b)
        self.assertEqual(first.snapshot, second.snapshot)
        assert_frame_equal(first.ranking("2026-W44"), second.ranking("2026-W44"))

    def test_new_draw_updates_snapshot_without_forcing_model_churn(self):
        before = self.biased.ranking("2026-W44", refits=0)
        extra = self.biased_data.iloc[[-1]].copy()
        extra.date += pd.Timedelta(days=2)
        updated = D.DrawProbabilityModel().fit(pd.concat([self.biased_data, extra], ignore_index=True))
        self.assertNotEqual(updated.snapshot, self.biased.snapshot)
        self.assertEqual(updated.n_obs, self.biased.n_obs + 1)
        probe = before[[f"N{i}" for i in range(1, 6)]].to_numpy()
        old = self.biased.distribution.log_prob(probe, before.Chance.to_numpy())
        new = updated.distribution.log_prob(probe, before.Chance.to_numpy())
        self.assertLess(np.abs(new - old).max(), .05)

    def test_short_data_and_empty_input_have_no_rank_advantage(self):
        for data in (pd.DataFrame(), draw_fixture(30)):
            model = D.DrawProbabilityModel().fit(data)
            self.assertFalse(model.fitted)
            table = model.ranking("2026-W44", exploratory=True)
            self.assertTrue((table.Rank == 1).all())

    def test_weekly_fallback_is_reproducible(self):
        a = self.fair.ranking("2026-W44")
        assert_frame_equal(a, self.fair.ranking("2026-W44"))
        b = self.fair.ranking("2026-W45")
        self.assertFalse(a.N1.equals(b.N1))
        np.testing.assert_allclose(a["Estimated probability"], b["Estimated probability"])

    def test_uniform_weekly_list_does_not_churn_on_data_upload(self):
        first = D.DrawProbabilityModel().fit(draw_fixture(30)).ranking("2026-W44")
        second = D.DrawProbabilityModel().fit(draw_fixture(31)).ranking("2026-W44")
        columns = [f"N{i}" for i in range(1, 6)] + ["Chance", "Rank", "Estimated probability"]
        assert_frame_equal(first[columns], second[columns])

    def test_second_draw_needs_no_chance(self):
        model = D.DrawProbabilityModel().fit(draw_fixture(biased=True, kind="second"), "second")
        self.assertTrue(model.eligible)
        table = model.ranking("2026-W44", refits=0)
        self.assertNotIn("Chance", table)
        np.testing.assert_allclose(table["Fair-draw probability"], 1 / comb(49, 5))

    def test_duplicates_and_other_game_excluded(self):
        a = draw_fixture(20)
        b = draw_fixture(20, kind="second")
        data = pd.concat([a, a, b], ignore_index=True)
        model = D.DrawProbabilityModel().fit(data)
        self.assertEqual(model.n_obs, 20)
        altered = a.iloc[[0]].copy()
        altered.chance = altered.chance % 10 + 1
        self.assertEqual(len(D.prepare_draws(pd.concat([a, altered]), "main")), 19)

    def test_comparison_matches_outcomes_not_row_positions(self):
        old = self.biased.ranking("2026-W44", refits=0)
        new = old.iloc[::-1].copy().reset_index(drop=True)
        compared = D.compare_rankings(new, old)
        self.assertTrue((compared.Change == "Unchanged").all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
