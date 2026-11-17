"""Bridge/evaluator lifecycle checks with a fake world; no CARLA server.

The real bridge step, tick, publish, pause and close methods are exercised.
Only SDK physics/rendering are replaced; this is not a driving validation.
"""
import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_bridge import CarlaSession
from nine_grid.carla_evaluation import EpisodeEvaluator
from nine_grid.carla_execution import ControlTimeline

SCRATCH = Path(__file__).resolve().parents[1]/'outputs'/'bridge_contract_checks'


class FakeWorld:
    def __init__(self):
        self.frame = 100
        self.tick_calls = 0

    def get_snapshot(self):
        return SimpleNamespace(frame=self.frame, timestamp=SimpleNamespace(
            elapsed_seconds=1000.+(self.frame-100)*.02))

    def tick(self, timeout):
        self.tick_calls += 1
        self.frame += 1
        return self.frame


class FakeDirector:
    def __init__(self):
        self.tick = 0
        self.frame = 100
        self.calls = []
        self.actors, self.events = [], []
        self.closed = False

    def before_tick_tick(self, tick, dt):
        if tick != self.tick or abs(dt-.02) > 1e-12:
            raise AssertionError('director received mismatched source tick')
        self.calls.append(('before', tick))

    def after_tick(self, frame, tick, snapshot):
        if tick != self.tick+1 or frame != self.frame+1 or frame != snapshot.frame:
            raise AssertionError('director missed a physical frame')
        self.tick, self.frame = tick, frame
        self.calls.append(('after', tick))

    def evaluation_state(self):
        return {'scenario_valid': True, 'events': [], 'tick': self.tick, 'frame': self.frame}

    def close(self):
        self.closed = True


class PhysicalProbe:
    def __init__(self, directory):
        self.evaluator = EpisodeEvaluator([[0., 0.], [10., 0.]], directory=directory)

    def observe(self, snapshot, tick, events=(), scenario_events=None):
        ego = {'id': 'ego', 'kind': 'vehicle', 'position': [6.7+.1*tick, 0.],
               'heading': 0., 'length': 4., 'width': 2., 'velocity': [5., 0.],
               'acceleration': [0., 0.]}
        road = {'body_on_road': True, 'wrong_way': False, 'speed_limit_mps': 10.,
                'coverage': {key: True for key in self.evaluator.coverage}}
        return self.evaluator.observe(tick, snapshot.frame, snapshot.timestamp.elapsed_seconds,
            ego, [], road, events, scenario_events)

    def ingest_events(self, events):
        self.evaluator.ingest_events(events)

    def infrastructure_error(self, kind, details):
        self.evaluator.infrastructure_error(kind, details)

    def report(self, reason=None, save=True):
        return self.evaluator.save_report(reason) if save else self.evaluator.report(reason)


def session_fixture(directory):
    session = CarlaSession()
    session.world, session.experiment = FakeWorld(), FakeDirector()
    session.root = Path(directory)
    session.physics_hz, session.planner_hz, session.control_step_s = 50, 10, .1
    session.timeout, session.duration_s, session.runtime_mode = 1., 30., 'sync'
    session.rgb_enabled, session._display_enabled = True, True
    session.timeline = ControlTimeline(physics_hz=50)
    session.frame, session._initial_source_frame, session.t0 = 100, 100, 1000.
    session._sample_history = lambda: None
    session._apply_current = lambda: True
    session._actor_record = lambda actor: {'id': actor.id}
    session.evaluation_probe = PhysicalProbe(directory)
    session._evaluation = session.evaluation_probe.observe(session.world.get_snapshot(), 0,
        scenario_events=session.experiment.evaluation_state())
    session.capture_modes = []
    session.camera_requests = []

    def collect(frame):
        session.camera_requests.append(frame)
        if (frame-100) % 5:
            raise AssertionError('bridge requested nonexistent RGB at a non-camera frame')

    def snapshot(include_cameras=True):
        session.capture_modes.append(include_cameras)
        return {'frame': session.frame, 'time': session._sim_time,
                'runtime': session._runtime_payload(), 'cameras': [],
                'evaluation': copy.deepcopy(session._evaluation)}

    session._collect, session.snapshot = collect, snapshot
    session._publish(snapshot(include_cameras=True))
    return session


class BridgeIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def test_arrival_between_rgb_frames_stops_on_exact_physical_tick(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = session_fixture(directory)
            result = session.step([[0., 0.]]*30, .1, 0., source_speed_mps=5., source_frame=100)
            self.assertEqual(session.world.tick_calls, 3)
            self.assertEqual(result['frame'], 103)
            self.assertEqual(result['runtime']['physical_ticks'], 3)
            self.assertAlmostEqual(result['time'], .06)
            self.assertEqual(result['runtime']['terminal_reason'], 'arrival')
            self.assertTrue(result['evaluation']['arrival'])
            self.assertEqual(result['evaluation']['observations'], 4)
            self.assertEqual(session.experiment.calls,
                [('before', 0), ('after', 1), ('before', 1), ('after', 2), ('before', 2), ('after', 3)])
            self.assertFalse(session.capture_modes[-1])
            self.assertEqual(session.camera_requests, [])
            self.assertEqual(session._published_sources[103][1], 3)

    def test_late_collision_is_merged_after_sensor_stop_without_tick(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = session_fixture(directory)
            session.step([[0., 0.]]*30, .1, 0., source_speed_mps=5., source_frame=100)
            self.assertTrue(session.evaluation_probe.report(save=False)['strict_success'])
            calls = []
            timer = threading.Timer(.04, lambda: session.events.append({
                'kind': 'collision', 'frame': 102, 'phase': 'runtime', 'other_id': 20}))

            def stop():
                calls.append('stop')
                timer.start()

            def destroy():
                calls.append('destroy')
                return True

            session.sensors = [SimpleNamespace(id=9, stop=stop, destroy=destroy)]
            try:
                session.close()
            finally:
                timer.join(1.)
            saved = json.loads((Path(directory)/'evaluation.json').read_text(encoding='utf-8'))
            self.assertEqual(calls, ['stop', 'destroy'])
            self.assertEqual(session.world.tick_calls, 3)
            self.assertTrue(session.experiment.closed)
            self.assertTrue(saved['collision'])
            self.assertFalse(saved['strict_success'])
            self.assertEqual(saved['driving_outcome'], 'collision')
            self.assertEqual(saved['events'][0]['physical_tick'], 2)
            self.assertEqual(saved['events'][0]['late_by_ticks'], 1)
            self.assertEqual(session.cleanup_report['event_drain']['extra_physics_ticks'], 0)

    def test_delayed_initialization_collision_stops_without_becoming_driving_collision(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = session_fixture(directory)
            session.events.append({'kind': 'collision', 'frame': 99,
                                   'phase': 'initialization', 'other_id': 20})
            result = session.step([[0., 0.]]*30, .1, 0., source_speed_mps=5., source_frame=100)
            self.assertEqual(session.world.tick_calls, 1)
            self.assertEqual(result['runtime']['terminal_reason'], 'initialization_collision')
            self.assertFalse(result['evaluation']['collision'])
            self.assertFalse(result['evaluation']['infrastructure_valid'])
            self.assertFalse(result['evaluation']['strict_success'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
