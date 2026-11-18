"""Tests for the safety filter that vetoes a step into an obstacle."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import scenarios as sc  # noqa: E402


class TestSafetyFilter(unittest.TestCase):
    def test_free_space_keeps_the_command(self):
        vel = np.array([1.0, 0.0])
        out = sc.safety_filter(np.array([0.0, 0.0]), vel, [], [], [])
        self.assertTrue(np.allclose(out, vel))

    def test_blocked_step_is_redirected(self):
        static = [(0.6, 0.0, 0.5)]
        pos = np.array([0.0, 0.0])
        vel = np.array([1.0, 0.0])
        out = sc.safety_filter(pos, vel, static, [], [])
        self.assertFalse(np.allclose(out, vel), "a step into a cylinder must be changed")

        naive = np.hypot(*(pos + vel * sc.DT - np.array([0.6, 0.0])))
        safe = np.hypot(*(pos + out * sc.DT - np.array([0.6, 0.0])))
        self.assertGreater(safe, naive, "the filtered step must end farther from the obstacle")

    def test_leaving_the_arena_is_stopped(self):
        pos = np.array([17.4, 0.0])
        out = sc.safety_filter(pos, np.array([3.0, 0.0]), [], [], [])
        self.assertTrue(np.allclose(out, np.zeros(2)))

    def test_moving_obstacle_is_respected(self):
        pos = np.array([0.0, 0.0])
        vel = np.array([1.0, 0.0])
        out = sc.safety_filter(pos, vel, [], [np.array([0.6, 0.0])], [0.5])
        self.assertFalse(np.allclose(out, vel))


if __name__ == "__main__":
    unittest.main()
