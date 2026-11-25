"""Tests for the transcribed reward equation in ``src/reward.py``."""

import unittest

import numpy as np

from src import reward as R


class TestGoalProgress(unittest.TestCase):
    def test_towards_goal_is_positive_one(self):
        # 1 m/s straight at a goal 10 m along +x
        self.assertAlmostEqual(
            R.goal_progress_term([1.0, 0.0, 0.0], [10.0, 0.0, 0.0]), 1.0)

    def test_sideways_is_zero(self):
        self.assertAlmostEqual(
            R.goal_progress_term([0.0, 1.0, 0.0], [10.0, 0.0, 0.0]), 0.0)

    def test_away_is_negative(self):
        self.assertAlmostEqual(
            R.goal_progress_term([-1.0, 0.0, 0.0], [10.0, 0.0, 0.0]), -1.0)

    def test_zero_goal_vector_does_not_divide_by_zero(self):
        self.assertAlmostEqual(
            R.goal_progress_term([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 0.0)


class TestSafetyTerms(unittest.TestCase):
    def test_static_term_is_log_of_gap(self):
        # every beam reads 0 m -> gap is the full 4 m range
        scan = np.zeros((2, 2))
        self.assertAlmostEqual(R.static_safety_term(scan, 4.0), np.log(4.0))

    def test_static_term_is_clamped_at_full_range(self):
        # a beam reading the full range leaves a gap of eps, not log(0)
        scan = np.full((1, 1), 4.0)
        self.assertAlmostEqual(R.static_safety_term(scan, 4.0), np.log(R.EPS))

    def test_static_term_grows_with_clearance(self):
        close = R.static_safety_term(np.full((1, 1), 3.5), 4.0)
        far = R.static_safety_term(np.full((1, 1), 0.5), 4.0)
        self.assertGreater(far, close)

    def test_dynamic_term_is_zero_when_no_obstacles(self):
        self.assertEqual(R.dynamic_safety_term([]), 0.0)

    def test_dynamic_term_is_log_of_surface_distance(self):
        self.assertAlmostEqual(
            R.dynamic_safety_term([2.0], 4.0), np.log(2.0))


class TestPenalties(unittest.TestCase):
    def test_smoothness_is_zero_for_steady_motion(self):
        self.assertAlmostEqual(
            R.smoothness_penalty([1.0, 0.0], [1.0, 0.0]), 0.0)

    def test_smoothness_is_the_velocity_change(self):
        self.assertAlmostEqual(
            R.smoothness_penalty([1.0, 0.0], [0.0, 0.0]), 1.0)

    def test_height_is_zero_inside_the_band(self):
        for z in (0.2, 1.0, 3.9, 4.0):
            self.assertAlmostEqual(R.height_penalty(z, 0.2, 4.0), 0.0)

    def test_height_is_squared_excess_outside_the_band(self):
        self.assertAlmostEqual(R.height_penalty(5.0, 0.2, 4.0), 1.0)
        self.assertAlmostEqual(R.height_penalty(-0.8, 0.2, 4.0), 1.0)


class TestTotalReward(unittest.TestCase):
    def test_reward_is_finite_and_in_range(self):
        r = R.total_reward(
            velocity=[1.0, 0.0, 0.0], prev_velocity=[1.0, 0.0, 0.0],
            goal_rel_pos=[10.0, 0.0, 0.0], z=1.5,
            lidar_scan=np.full((36, 4), 2.0),
            obstacle_distances=[1.5, 2.0, 3.0, 1.0, 2.5])
        self.assertTrue(np.isfinite(r))

    def test_step_bonus_is_present(self):
        # stationary robot, no static obstacle closer than full range, mid altitude:
        # the reward collapses to the constant step bonus plus the log terms
        scan = np.zeros((36, 4))
        r = R.total_reward([0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                           [10.0, 0.0, 0.0], 1.5, scan)
        expected = R.STEP_BONUS + np.log(4.0)
        self.assertAlmostEqual(r, expected)

    def test_moving_away_costs_more_than_moving_towards(self):
        scan = np.zeros((36, 4))
        towards = R.total_reward([1.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], 1.5, scan)
        away = R.total_reward([-1.0, 0, 0], [-1.0, 0, 0], [10.0, 0, 0], 1.5, scan)
        self.assertGreater(towards, away)


if __name__ == "__main__":
    unittest.main()
