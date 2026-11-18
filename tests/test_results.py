"""Sanity checks on the shipped results file.

These do not re-run the policy (that needs torch); they verify that the numbers
in results/results.json agree with the metric definitions, so the report and the
code cannot drift apart.
"""

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import metrics  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "results.json")


@unittest.skipUnless(os.path.exists(RESULTS), "results.json not present")
class TestShippedResults(unittest.TestCase):
    def setUp(self):
        with open(RESULTS, encoding="utf-8") as fh:
            self.runs = json.load(fh)

    def test_three_scenes(self):
        self.assertEqual(len(self.runs), 3)
        labels = [r["scenario"] for r in self.runs]
        self.assertIn("Static (15)", labels)
        self.assertIn("Dynamic (15)", labels)
        self.assertIn("Mixed (15)", labels)

    def test_every_scene_has_six_robots_and_fifteen_obstacles(self):
        for run in self.runs:
            self.assertEqual(run["robots"], 6)
            self.assertEqual(run["static"] + run["dynamic"], 15)

    def test_counts_match_the_clearances(self):
        for run in self.runs:
            classes = [metrics.classify_contact(c) for c in run["per_robot_clearance"]]
            self.assertEqual(run["near_miss"], classes.count("near_miss"), run["scenario"])
            self.assertEqual(run["penetration"], classes.count("penetration"), run["scenario"])

    def test_success_rate_matches_arrivals(self):
        for run in self.runs:
            arrived = sum(1 for a in run["arrivals"] if a is not None)
            self.assertEqual(run["reached"], arrived)
            self.assertAlmostEqual(run["success_rate"], 100.0 * arrived / run["robots"], places=1)

    def test_worst_clearance_is_the_minimum(self):
        for run in self.runs:
            self.assertAlmostEqual(run["worst_min_clearance"],
                                   min(run["per_robot_clearance"]), places=3)


if __name__ == "__main__":
    unittest.main()
