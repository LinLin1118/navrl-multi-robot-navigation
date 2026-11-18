"""Tests for the metric definitions."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import metrics  # noqa: E402


class TestContactClassification(unittest.TestCase):
    def test_clear(self):
        self.assertEqual(metrics.classify_contact(0.4), "clear")

    def test_exactly_at_the_radius_is_still_clear(self):
        # the threshold is exclusive: touching the margin is not a contact
        self.assertEqual(metrics.classify_contact(0.25), "clear")

    def test_inside_the_radius_is_a_near_miss(self):
        self.assertEqual(metrics.classify_contact(0.249), "near_miss")

    def test_negative_is_penetration(self):
        self.assertEqual(metrics.classify_contact(-0.01), "penetration")


class TestSummarise(unittest.TestCase):
    def test_all_arrive(self):
        out = metrics.summarise([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
        self.assertEqual(out["reached"], 3)
        self.assertEqual(out["success_rate"], 100.0)
        self.assertAlmostEqual(out["mean_arrival"], 2.0)

    def test_partial_success_uses_successful_robots_only(self):
        out = metrics.summarise([1.0, None, 3.0], [0.5, 0.5, 0.5])
        self.assertEqual(out["reached"], 2)
        self.assertEqual(out["success_rate"], 66.7)
        self.assertAlmostEqual(out["mean_arrival"], 2.0)

    def test_contacts_are_counted_per_class(self):
        out = metrics.summarise([1, 1, 1, 1], [0.5, 0.1, -0.2, 0.3])
        self.assertEqual(out["near_miss"], 1)
        self.assertEqual(out["penetration"], 1)
        self.assertAlmostEqual(out["worst_min_clearance"], -0.2)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.summarise([1.0], [0.1, 0.2])

    def test_empty_run_is_rejected(self):
        with self.assertRaises(ValueError):
            metrics.summarise([], [])


if __name__ == "__main__":
    unittest.main()
