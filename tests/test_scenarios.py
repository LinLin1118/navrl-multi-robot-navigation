"""Tests for the scenario generator. Pure NumPy, no torch needed."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import scenarios as sc  # noqa: E402


class TestRobots(unittest.TestCase):
    def test_same_seed_same_field(self):
        a = sc.make_robots(np.random.default_rng(7))
        b = sc.make_robots(np.random.default_rng(7))
        for p, q in zip(a[0], b[0]):
            self.assertTrue(np.allclose(p, q))

    def test_counts(self):
        starts, goals, vels, dirs = sc.make_robots(np.random.default_rng(1))
        self.assertEqual(len(starts), sc.N_ROBOTS)
        self.assertEqual(len(goals), sc.N_ROBOTS)
        self.assertEqual(len(vels), sc.N_ROBOTS)
        self.assertEqual(len(dirs), sc.N_ROBOTS)

    def test_starts_are_separated(self):
        starts, _, _, _ = sc.make_robots(np.random.default_rng(2))
        for i, p in enumerate(starts):
            for q in starts[i + 1:]:
                self.assertGreaterEqual(np.hypot(*(p - q)), sc.START_MIN_SEP - 1e-9)

    def test_every_goal_is_far_from_its_start(self):
        starts, goals, _, _ = sc.make_robots(np.random.default_rng(3))
        for s, g in zip(starts, goals):
            self.assertGreaterEqual(np.hypot(*(g - s)), sc.GOAL_MIN_DIST - 1e-9)


class TestObstacles(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)
        self.starts, self.goals, _, _ = sc.make_robots(self.rng)

    def test_static_count_and_no_overlap(self):
        obs = sc.make_static(15, self.starts, self.goals, self.rng)
        self.assertEqual(len(obs), 15)
        for i, (x, y, r) in enumerate(obs):
            for (x2, y2, r2) in obs[i + 1:]:
                self.assertGreater(np.hypot(x - x2, y - y2), r + r2)

    def test_static_keeps_away_from_robots(self):
        obs = sc.make_static(15, self.starts, self.goals, self.rng)
        for (x, y, r) in obs:
            for p in self.starts + self.goals:
                self.assertGreater(np.hypot(x - p[0], y - p[1]), r)

    def test_dynamic_count_and_bounds(self):
        pos, vel = sc.make_dynamic(15, self.rng)
        self.assertEqual(len(pos), 15)
        self.assertEqual(len(vel), 15)
        for _ in range(200):
            sc.step_dynamic(pos, vel, self.rng, _)
        for p in pos:
            self.assertLessEqual(abs(p[0]), 15.0 + 1e-9)
            self.assertLessEqual(abs(p[1]), 15.0 + 1e-9)

    def test_radii_range(self):
        r = sc.radii(15, self.rng)
        self.assertEqual(len(r), 15)
        for value in r:
            self.assertGreaterEqual(value, sc.DYN_RADIUS[0])
            self.assertLessEqual(value, sc.DYN_RADIUS[1])


class TestClearance(unittest.TestCase):
    def test_inside_obstacle_is_negative(self):
        d = sc.clearance_to_obstacles(np.array([0.0, 0.0]), [(0.0, 0.0, 1.0)], [], [])
        self.assertAlmostEqual(d, -1.0)

    def test_surface_is_zero(self):
        d = sc.clearance_to_obstacles(np.array([1.0, 0.0]), [(0.0, 0.0, 1.0)], [], [])
        self.assertAlmostEqual(d, 0.0, places=6)

    def test_nearest_obstacle_wins(self):
        static = [(10.0, 0.0, 1.0)]
        dyn_pos = [np.array([2.0, 0.0])]
        d = sc.clearance_to_obstacles(np.array([0.0, 0.0]), static, dyn_pos, [0.5])
        self.assertAlmostEqual(d, 1.5)

    def test_empty_field_is_infinite(self):
        self.assertEqual(sc.clearance_to_obstacles(np.array([0.0, 0.0]), [], [], []), float("inf"))


if __name__ == "__main__":
    unittest.main()
