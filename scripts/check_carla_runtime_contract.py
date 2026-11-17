"""Regression checks for source-bound integer control and observation clocks; no CARLA server."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_execution import ControlTimeline
from nine_grid.carla_bridge import CarlaSession


class RuntimeContract(unittest.TestCase):
    def test_float32_world_time_never_changes_segment(self):
        for planner_hz in (1, 2, 10):
            timeline = ControlTimeline(physics_hz=50)
            period = 50//planner_hz
            for tick in range(151):
                if tick % period == 0:
                    timeline.install([[0., .001*i] for i in range(30)],
                        tick*.019999999552965164, .1, 5., source_tick=tick)
                state = timeline.at(tick*.019999999552965164, tick)
                self.assertEqual(state['index'], (tick % period)//5)
                self.assertGreater(state['remaining_segment_s'], 0.)

    def test_exact_expiration_and_async_partial_segment(self):
        timeline = ControlTimeline(physics_hz=50)
        timeline.install([[0., 0.]]*30, 0., .1, 2., source_tick=0)
        self.assertEqual(timeline.at(.499999988824, 25)['index'], 5)
        self.assertEqual(timeline.at(.519999988, 26)['index'], 5)
        self.assertAlmostEqual(timeline.at(.519999988, 26)['remaining_segment_s'], .08)
        self.assertTrue(timeline.at(2.97999993, 149)['available'])
        self.assertEqual(timeline.at(2.99999993, 150)['reason'], 'sequence_exhausted')
        with self.assertRaises(ValueError):
            timeline.at(.5)

    def test_rejected_plan_does_not_replace_running_sequence(self):
        session = CarlaSession()
        session.physics_hz, session.control_step_s = 50, .1
        session.timeline = ControlTimeline(physics_hz=50)
        session._published_sources = {100: (0., 0), 105: (.0999999978, 5)}
        session._install([[0., 0.]]*30, .1, 0., None, 2., 100)
        original = session.timeline
        with self.assertRaises(ValueError):
            session._install([[0., 0.]]*30, .1, .1, None, 2., 100)
        with self.assertRaises(ValueError):
            session._install([[0., 0.]]*30, .1, 0., None, 2., 999)
        self.assertIs(session.timeline, original)
        session._sim_time = .1399999969
        session._runtime_metrics['physical_ticks'] = 7
        session._install([[0., 0.]]*30, .1, .0999999978, None, 2., 105)
        self.assertEqual(session.timeline.source_tick, 5)
        self.assertAlmostEqual(session.timeline.at(session._sim_time, 7)['remaining_segment_s'], .06)

    def test_history_uses_ticks_even_with_large_raw_timestamp(self):
        session = CarlaSession()
        session.physics_hz, session.t0 = 50, 1685.158112
        state = {'id': 'ego', 'kind': 'vehicle', 'position': [0., 0.],
                 'velocity': [1., 0.], 'acceleration': [0., 0.], 'heading': 0.}
        session._entities = lambda: (state, [], 0)
        for tick in range(51):
            session.frame = 400000+tick
            session._runtime_metrics['physical_ticks'] = tick
            session.world = SimpleNamespace(get_snapshot=lambda tick=tick: SimpleNamespace(
                timestamp=SimpleNamespace(elapsed_seconds=session.t0+tick*.019999999552965164)))
            session._sample_history()
        self.assertEqual([row['physical_tick'] for row in session._history['ego']], [0,25,50])


if __name__ == '__main__':
    unittest.main(verbosity=2)
