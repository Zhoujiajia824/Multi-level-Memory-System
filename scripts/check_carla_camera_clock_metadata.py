"""Metadata-only diagnostic checks; no CARLA connection, ticks, or image pixels."""
import unittest
from types import SimpleNamespace

from check_carla_camera_clock import Recorder


class Image:
    def __init__(self, frame, timestamp):
        self.frame, self.timestamp = frame, timestamp

    @property
    def raw_data(self):
        raise AssertionError('Diagnostic must not read or retain image pixels')


class World:
    def __init__(self):
        self.reads = 0
        self.snapshot = SimpleNamespace(frame=100, timestamp=SimpleNamespace(elapsed_seconds=2.0))

    def get_snapshot(self):
        self.reads += 1
        return self.snapshot


class RecorderMetadata(unittest.TestCase):
    def test_aligned_or_normally_delayed_callbacks_are_not_conflicts(self):
        world = World()
        recorder = Recorder(world)
        callback = recorder.callback('0.1')
        callback(Image(100, 2.0))
        callback(Image(95, 1.9))
        callback(Image(101, 2.0))
        row = recorder.finish()['0.1']
        self.assertEqual(world.reads, 3)
        self.assertEqual(row['count'], 3)
        self.assertEqual(row['callback_errors'], 0)
        self.assertEqual(row['frame_timestamp_conflict_count'], 0)
        self.assertEqual(row['first_12_frame_timestamp_conflicts'], [])
        delayed = row['first_12_samples'][1]
        self.assertEqual(delayed['snapshot_frame_minus_sensor_frame'], 5)
        self.assertAlmostEqual(delayed['snapshot_timestamp_minus_sensor_timestamp_s'], .1)

    def test_old_frame_current_timestamp_uses_fixed_tolerance_and_retains_raw_headers(self):
        world = World()
        recorder = Recorder(world)
        recorder.mark_completed(100)
        callback = recorder.callback('0.1')
        images = [Image(99, 2.), Image(99, 2. - 9e-6), Image(99, 2. + 9e-6),
                  Image(99, 2. - 11e-6), Image(99, 2. + 11e-6)]
        original = [(image.frame, image.timestamp) for image in images]
        for image in images:
            callback(image)
        row = recorder.finish()['0.1']
        self.assertEqual(world.reads, len(images))
        self.assertEqual([(image.frame, image.timestamp) for image in images], original)
        self.assertEqual(row['frame_timestamp_match_tolerance_s'], 1e-5)
        self.assertEqual(row['frame_timestamp_conflict_count'], 3)
        self.assertEqual(row['callback_errors'], 0)
        first = row['first_12_frame_timestamp_conflicts'][0]
        self.assertEqual(first['frame'], 99)
        self.assertEqual(first['timestamp'], 2.)
        self.assertEqual(first['world_snapshot_frame_at_receive'], 100)
        self.assertEqual(first['world_snapshot_timestamp_at_receive'], 2.)
        self.assertTrue(first['frame_timestamp_conflict'])

    def test_conflict_examples_are_bounded_but_every_conflict_is_counted(self):
        recorder = Recorder(World())
        callback = recorder.callback('0.1')
        for frame in range(30, 60):
            callback(Image(frame, 2.))
        row = recorder.finish()['0.1']
        self.assertEqual(row['count'], 30)
        self.assertEqual(row['frame_timestamp_conflict_count'], 30)
        self.assertEqual(len(row['first_12_frame_timestamp_conflicts']), 12)
        self.assertEqual([sample['frame'] for sample in row['first_12_frame_timestamp_conflicts']],
                         list(range(30, 42)))
        self.assertEqual(row['last_sample']['frame'], 59)
        callback(Image(60, 2.))
        self.assertEqual(recorder.finish()['0.1'], row)


if __name__ == '__main__':
    unittest.main(verbosity=2)
