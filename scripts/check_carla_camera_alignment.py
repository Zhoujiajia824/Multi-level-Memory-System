"""Image identity regressions; fake sensor headers, no CARLA server or rendering."""
import json
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_bridge import CarlaSession, CameraMetadataError

SCRATCH = Path(__file__).resolve().parents[1]/'outputs'/'camera_alignment_checks'
VIEWS = ('front_rgb', 'front_left_rgb', 'front_right_rgb',
         'back_rgb', 'back_left_rgb', 'back_right_rgb')
FRAME, STAMP = 30917, 89.97999798879027


def image(frame=FRAME, timestamp=STAMP):
    return SimpleNamespace(frame=frame, timestamp=timestamp)


def fixture(directory):
    session = CarlaSession()
    settings = SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=.02,
        no_rendering_mode=False, substepping=True, max_substep_delta_time=.01, max_substeps=2)
    session.world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(frame=FRAME,
        timestamp=SimpleNamespace(elapsed_seconds=STAMP)), get_settings=lambda: settings)
    session.root, session.frame, session.t0, session.timeout = Path(directory), FRAME, .48, .25
    session.queues = {view: queue.Queue() for view in VIEWS}
    for view, q in session.queues.items():
        session._on_camera_image(view, q, image())
    return session


class CameraAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def check_conflict(self, incoming, reason):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = fixture(directory)
            session.queues['front_rgb'].get_nowait()
            session._on_camera_image('front_rgb', session.queues['front_rgb'], incoming)
            # An empty get after the rejected header would prove an unwanted wait.
            old_get = session.queues['front_rgb'].get
            def get_once(*args, **kwargs):
                if session.queues['front_rgb'].empty():
                    raise AssertionError('metadata conflict fell through to another queue wait')
                return old_get(*args, **kwargs)
            session.queues['front_rgb'].get = get_once
            with self.assertRaises(CameraMetadataError):
                session._collect(FRAME)
            report = json.loads((Path(directory)/'camera_sync_failure.json').read_text(encoding='utf-8'))
            self.assertEqual(report['error_kind'], 'rgb_metadata_conflict')
            self.assertEqual(report['metadata_conflict']['reason'], reason)
            self.assertEqual(report['metadata_conflict']['sensor_frame'], incoming.frame)
            self.assertEqual(report['metadata_conflict']['world_frame'], FRAME)
            self.assertEqual(report['world_timestamp'], STAMP)
            self.assertEqual(report['cameras']['front_rgb']['rejected_metadata_count'], 1)
            self.assertNotIn('discarded_stale_count', report['cameras']['front_rgb'])
            self.assertEqual(session.frame, FRAME)
            self.assertEqual(session._runtime_metrics['physical_ticks'], 0)
            self.assertEqual(session.frames, {})
            # Other images stay queued, rather than being relabelled to fit.
            self.assertEqual(session.queues['back_rgb'].qsize(), 1)

    def test_six_exact_sources_and_repeat_collect(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = fixture(directory)
            session._collect(FRAME)
            original = dict(session.frames)
            session._collect(FRAME)
            self.assertEqual(session.frames, original)
            for key, raw in original.items():
                self.assertEqual((raw.frame, raw.timestamp), (FRAME, STAMP))
                self.assertEqual(session._camera_audit[key]['selected_count'], 1)

    def test_server_observed_older_frame_current_time_is_not_stale(self):
        self.check_conflict(image(FRAME-1), 'old_frame_with_current_or_future_time')

    def test_older_frame_future_time_is_rejected(self):
        self.check_conflict(image(FRAME-1, STAMP+.02), 'old_frame_with_current_or_future_time')

    def test_current_frame_old_time_is_rejected(self):
        self.check_conflict(image(FRAME, STAMP-.02), 'frame_timestamp_disagreement')

    def test_current_frame_future_time_is_rejected(self):
        self.check_conflict(image(FRAME, STAMP+.02), 'frame_timestamp_disagreement')

    def test_future_frame_current_time_is_rejected(self):
        self.check_conflict(image(FRAME+1), 'future_frame')

    def test_nonfinite_header_is_preserved_as_explicit_conflict(self):
        self.check_conflict(image(FRAME, float('nan')), 'nonfinite_timestamp')

    def test_genuinely_stale_image_is_drained_before_matching_source(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = fixture(directory)
            session.queues['front_rgb'] = queue.Queue()
            for raw in (image(FRAME-5, STAMP-.1), image()):
                session._on_camera_image('front_rgb', session.queues['front_rgb'], raw)
            session._collect(FRAME)
            self.assertEqual(session.frames['front_rgb'].frame, FRAME)
            self.assertEqual(session._camera_audit['front_rgb']['discarded_stale_count'], 1)
            self.assertIsNone(session.camera_sync_failure)

    def test_cached_source_cannot_bypass_world_identity_check(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = fixture(directory)
            session._collect(FRAME)
            session.world.get_snapshot = lambda: SimpleNamespace(frame=FRAME+1,
                timestamp=SimpleNamespace(elapsed_seconds=STAMP+.02))
            with self.assertRaisesRegex(RuntimeError, '世界帧与目标帧不一致'):
                session._collect(FRAME)

    def test_selected_peer_and_conflicting_raw_header_survive_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = fixture(directory)
            q = session.queues['back_rgb']
            q.get_nowait()
            session._on_camera_image('back_rgb', q, image(FRAME-1))
            with self.assertRaises(CameraMetadataError):
                session._collect(FRAME)
            report = session.camera_sync_failure
            self.assertEqual(report['accepted_for_request']['front_rgb']['frame'], FRAME)
            self.assertEqual(report['metadata_conflict']['sensor_frame'], FRAME-1)
            self.assertEqual(report['metadata_conflict']['sensor_timestamp'], STAMP)

    def test_startup_probe_rejects_same_timestamp_older_frame_without_more_ticks(self):
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = fixture(directory)
            session.t0, session.physics_hz = None, 50
            stages = []
            def tick(stage):
                stages.append(stage)
                if len(stages) > 2:
                    raise AssertionError('startup continued after metadata conflict')
                if stage == 'camera_phase_probe':
                    session.frame = FRAME+1
                    session.world.get_snapshot = lambda: SimpleNamespace(frame=FRAME+1,
                        timestamp=SimpleNamespace(elapsed_seconds=STAMP+.02))
                    session._on_camera_image('front_rgb', session.queues['front_rgb'],
                        image(FRAME, STAMP+.02))
                return session.world.get_snapshot()
            session._initialization_tick = tick
            with self.assertRaises(CameraMetadataError):
                session._align_initial_camera_phase()
            self.assertEqual(stages, ['camera_bootstrap', 'camera_phase_probe'])
            self.assertEqual(session._rgb_initialization['status'], 'failed')
            self.assertEqual(session.camera_sync_failure['metadata_conflict']['sensor_frame'], FRAME)
            self.assertEqual(session.camera_sync_failure['requested_frame'], FRAME+1)
            self.assertEqual(session._runtime_metrics['physical_ticks'], 0)


if __name__ == '__main__':
    unittest.main()
