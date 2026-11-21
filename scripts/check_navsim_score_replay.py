"""Regression checks for observational official-score replay capture (no SDK)."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ep_driver_navsim.score_replay import wrap_score_function, finite_json


class Occupancy:
    tokens = ["red_light_lane"]

    def __getitem__(self, token):
        return box(0, -1, 3, 1)


class Observation:
    red_light_token = "red_light"

    def __getitem__(self, frame):
        return Occupancy()


class Policy:
    def __init__(self):
        self.calls = 0

    def simulate_environment(self, states, cache):
        self.calls += 1
        obj = SimpleNamespace(track_token="model_actor" if self.calls == 1 else "human_actor",
            tracked_object_type=SimpleNamespace(name="VEHICLE"),
            center=SimpleNamespace(x=float(self.calls), y=0., heading=0.),
            box=SimpleNamespace(length=4., width=2.))
        return [SimpleNamespace(tracked_objects=[obj]) for _ in states]


class Scorer:
    def __init__(self):
        self.calls = 0
        self._config = SimpleNamespace(human_penalty_filter=True)

    def score_proposals(self, states, observation, *unused):
        self.calls += 1
        self._states = states
        self._observation = observation
        self._ego_areas = np.zeros((*states.shape[:2], 3), bool)
        self._ego_areas[:, 1, 1] = True
        self._ego_polygons = np.empty(states.shape[:2], dtype=object)
        for index in np.ndindex(states.shape[:2]):
            self._ego_polygons[index] = box(0, -.5, 1, .5)
        self._collision_time_idcs = np.full(states.shape[0], np.inf)
        self._ttc_time_idcs = np.full(states.shape[0], np.inf)
        if self.calls == 1:
            self._collision_time_idcs[1] = 2
        return [pd.DataFrame([dict(traffic_light_compliance=0., weighted_metrics=np.array([.3, .4]),
                                   pdm_score=float(states[index, -1, 0]))])
                for index in range(len(states))]


def fake_official(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
    """A minimal first model pass followed by the official-style human filter."""
    states = np.stack([model_trajectory + 1., model_trajectory.copy()])
    traffic_agents_policy.simulate_environment(states[1], metric_cache)
    scored = scorer.score_proposals(states, Observation())[1]
    # The official function then overwrites scorer state using a human pass,
    # and edits the SAME model dataframe and its NumPy-valued metrics.
    traffic_agents_policy.simulate_environment(np.zeros_like(model_trajectory), metric_cache)
    scorer.score_proposals(np.zeros_like(states[:1]), Observation())
    scored.at[0, "traffic_light_compliance"] = 1.
    scored.at[0, "weighted_metrics"][0] = 1.
    metric_cache.last_result = scored, states[1]
    return metric_cache.last_result


class ReplayObserverChecks(unittest.TestCase):
    def arguments(self):
        states = np.zeros((4, 11))
        states[:, 0] = np.arange(4)
        return dict(metric_cache=SimpleNamespace(file_path=Path("cache/log/type/token/metric_cache.pkl")),
            model_trajectory=states, future_sampling=None, simulator=None,
            scorer=Scorer(), traffic_agents_policy=Policy())

    def test_result_identity_numerical_parity_and_first_pass_binding(self):
        baseline = fake_official(**self.arguments())
        arguments = self.arguments()
        captured = []
        def retain(args, result, capture, settings):
            captured.append(capture)
        with patch("ep_driver_navsim.score_replay.export_replay", side_effect=retain) as export:
            result = wrap_score_function(fake_official, {})(**arguments)
        self.assertIs(result, arguments["metric_cache"].last_result)
        self.assertEqual(export.call_count, 1)
        np.testing.assert_array_equal(result[1], baseline[1])
        pd.testing.assert_frame_equal(result[0], baseline[0])
        capture = captured[0]
        self.assertEqual(capture["errors"], [])
        self.assertEqual(capture["actors"][0][0]["id"], "model_actor")
        self.assertEqual(capture["raw"]["traffic_light_compliance"], 0.)
        self.assertEqual(capture["raw"]["weighted_metrics"], [.3, .4])
        self.assertEqual(float(result[0].iloc[0]["traffic_light_compliance"]), 1.)
        self.assertEqual(capture["raw_offroad_frames"], [1])
        np.testing.assert_array_equal(capture["ego_states"], result[1])
        self.assertEqual([row["frame_index"] for row in capture["events"]
                          if row["type"] == "raw_first_at_fault_collision"], [2])
        self.assertEqual(arguments["scorer"].calls, 2)
        self.assertEqual(arguments["traffic_agents_policy"].calls, 2)
        self.assertTrue(any(e["type"] == "raw_red_light_polygon_intersection" for e in capture["events"]))

    def test_visual_failure_does_not_change_score_or_validity(self):
        arguments = self.arguments()
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            settings = dict(evaluation_tokens=["token"], output_dir=directory)
            with patch("ep_driver_navsim.score_replay.export_replay", side_effect=RuntimeError("renderer missing")):
                result = wrap_score_function(fake_official, settings)(**arguments)
            self.assertIs(result, arguments["metric_cache"].last_result)
            statuses = list(Path(directory).rglob("capture_status.json"))
            self.assertEqual(len(statuses), 1)
            self.assertIn("visualization_only", statuses[0].read_text(encoding="utf-8"))

    def test_scoring_failure_propagates(self):
        def failing(metric_cache, model_trajectory, future_sampling, simulator, scorer, traffic_agents_policy):
            raise ValueError("real official scoring error")
        with patch("ep_driver_navsim.score_replay.export_replay") as export:
            with self.assertRaisesRegex(ValueError, "real official scoring error"):
                wrap_score_function(failing, {})(**self.arguments())
            export.assert_not_called()

    def test_disabled_and_double_install_do_not_add_calls(self):
        self.assertIs(wrap_score_function(fake_official, dict(bev_video=False)), fake_official)
        enabled = wrap_score_function(fake_official, {})
        self.assertIs(wrap_score_function(enabled, {}), enabled)

    def test_scalar_nonfinite_conversion_and_numpy_detachment(self):
        array = np.array([np.inf, 2.])
        result = finite_json(dict(a=array, b=np.float64(np.nan)))
        array[1] = 3.
        self.assertEqual(result, dict(a=[None, 2.], b=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
