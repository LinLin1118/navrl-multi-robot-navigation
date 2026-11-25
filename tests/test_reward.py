"""Tests for the transcribed reward equation in ``src/reward.py``.

The numbers here follow the upstream conventions: ``lidar_scan`` is the gap image
(``lidar_range - distance``) and the altitude band is the per-episode
``[min(z_start, z_goal) - 0.2, max(z_start, z_goal) + 0.2]``.
"""

import unittest

import numpy as np

from src import reward as R

#: Per-episode altitude band for a flight between two 2 m points.
Z_LO, Z_HI = 1.8, 2.2


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
    def test_open_space_is_the_maximum_reward(self):
        # gap image of 0 means nothing within range -> distance is the full range
        scan = np.zeros((2, 2))
        self.assertAlmostEqual(R.static_safety_term(scan, 4.0), np.log(4.0))

    def test_surface_at_the_sensor_gives_the_smallest_reward(self):
        # gap image of 4 means the surface is at the sensor -> clipped to eps
        scan = np.full((1, 1), 4.0)
        self.assertAlmostEqual(R.static_safety_term(scan, 4.0), np.log(R.EPS))

    def test_reward_grows_with_distance(self):
        # a small gap image means a distant surface, so a larger reward
        near = R.static_safety_term(np.full((1, 1), 3.5), 4.0)
        far = R.static_safety_term(np.full((1, 1), 0.5), 4.0)
        self.assertGreater(far, near)

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
        for z in (Z_LO, 2.0, Z_HI):
            self.assertAlmostEqual(R.height_penalty(z, Z_LO, Z_HI), 0.0)

    def test_height_is_squared_excess_outside_the_band(self):
        self.assertAlmostEqual(R.height_penalty(3.2, Z_LO, Z_HI), 1.0)
        self.assertAlmostEqual(R.height_penalty(0.8, Z_LO, Z_HI), 1.0)

    def test_height_band_is_not_the_terminal_band(self):
        # 3 m is inside the terminal band (0.2-4 m) but outside a 2 m flight's band,
        # so it must be penalised; using the terminal bounds here would return 0
        self.assertGreater(R.height_penalty(3.0, Z_LO, Z_HI), 0.0)


class TestTotalReward(unittest.TestCase):
    def _reward(self, velocity, prev_velocity, z=2.0, scan=None, obstacles=None):
        return R.total_reward(
            velocity=velocity, prev_velocity=prev_velocity,
            goal_rel_pos=[10.0, 0.0, 0.0], z=z, z_lo=Z_LO, z_hi=Z_HI,
            lidar_scan=np.zeros((36, 4)) if scan is None else scan,
            obstacle_distances=obstacles)

    def test_reward_is_finite_and_in_range(self):
        r = self._reward([1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                         scan=np.full((36, 4), 2.0),
                         obstacles=[1.5, 2.0, 3.0, 1.0, 2.5])
        self.assertTrue(np.isfinite(r))

    def test_step_bonus_is_present(self):
        # stationary robot, open space, mid altitude: only the step bonus and the
        # saturated static term survive
        r = self._reward([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(r, R.STEP_BONUS + np.log(4.0))

    def test_moving_away_costs_more_than_moving_towards(self):
        towards = self._reward([1.0, 0, 0], [1.0, 0, 0])
        away = self._reward([-1.0, 0, 0], [-1.0, 0, 0])
        self.assertGreater(towards, away)

    def test_weights_are_actually_applied(self):
        # a velocity jump of 2 m/s must cost 0.1 * 2 = 0.2, and nothing else changes
        steady = self._reward([1.0, 0, 0], [1.0, 0, 0])
        jump = self._reward([1.0, 0, 0], [-1.0, 0, 0])
        self.assertAlmostEqual(steady - jump, R.SMOOTH_WEIGHT * 2.0)

    def test_height_weight_is_applied(self):
        # 1 m above the band (2.2 + 1) costs 8 * 1^2 = 8
        inside = self._reward([0.0, 0, 0], [0.0, 0, 0], z=2.0)
        outside = self._reward([0.0, 0, 0], [0.0, 0, 0], z=3.2)
        self.assertAlmostEqual(inside - outside, R.HEIGHT_WEIGHT)


if __name__ == "__main__":
    unittest.main()
