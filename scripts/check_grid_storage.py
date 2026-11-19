"""验证八向无损数组协议；无需CARLA、模型权重，也不修改离线经验库。"""
from pathlib import Path
import copy
import io
import json
import sys
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from check_lane_object_scope import straight, actor
from check_model_object_scope import bind
from nine_grid.grid_storage import (GRID_STORAGE_VERSION, SLOTS, save_grid, load_grid,
                                    grid_to_arrays, grid_from_arrays, grid_tensor16)
from nine_grid.representations import build_grid, build_representations
from nine_grid.retrieval import grid_comparison, validate_grid_schema
from nine_grid.models import scene_text


def fixture(*, crossable=True, observed=True):
    scene = straight(crossable)
    scene.actors = [actor('front_18446744073709551615123', 15.),
                    actor('bike', 12., 3.5, kind='bicycle', lane='l', speed=2.),
                    actor('walker', -10., -3.5, kind='pedestrian', lane='r')]
    if observed:
        bind(scene)
    return build_grid(scene)


def report(grid, other):
    result = grid_comparison(grid, other)
    result.pop('timing_s', None)
    return result


def roundtrip(grid):
    stream = io.BytesIO()
    save_grid(stream, grid)
    stream.seek(0)
    return load_grid(stream)


def change_meta(arrays, **values):
    meta = json.loads(str(arrays['metadata']))
    meta.update(values)
    arrays['metadata'] = np.array(json.dumps(meta, ensure_ascii=False))


class GridStorageTests(unittest.TestCase):
    def test_real_grid_score_and_input_unchanged(self):
        original = fixture()
        before = copy.deepcopy(original)
        restored = roundtrip(original)
        self.assertEqual(original, before)
        self.assertEqual(validate_grid_schema(restored), [])
        self.assertEqual(report(original, original), report(original, restored))
        other = fixture()
        other['ego']['velocity'][0] += 1.125
        self.assertEqual(report(original, other), report(restored, roundtrip(other)))
        self.assertEqual(restored['objects'], [
            {k: value for k, value in obj.items() if k in ('id', 'kind', 'position', 'velocity',
                'acceleration', 'heading', 'slots', 'state_valid')} for obj in original['objects']])

    def test_raw_double_angles_and_large_string_ids(self):
        grid = fixture()
        values = [float(np.nextafter(np.pi, 0)), -np.pi, np.pi]
        for obj, heading in zip(grid['objects'], values):
            obj['heading'] = heading
            for i, row in enumerate(SLOTS):
                for j, name in enumerate(row):
                    if name in obj['slots']:
                        grid['matrices']['heading'][i][j] = heading
        arrays = grid_to_arrays(grid)
        restored = roundtrip(grid)
        self.assertEqual(arrays['values'].dtype, np.dtype('float64'))
        self.assertEqual([o['heading'] for o in grid['objects']], [o['heading'] for o in restored['objects']])
        self.assertIn('front_18446744073709551615123', arrays['object_ids'].tolist())
        self.assertEqual(report(grid, restored)['score'], 1.)
        tensor, mask = grid_tensor16(arrays)
        self.assertEqual(tensor.shape, (3, 3, 16))
        self.assertEqual(mask.shape, (3, 3, 8))
        self.assertTrue(np.all(tensor[..., 12:].sum(axis=2) == 1))
        np.testing.assert_array_equal(tensor[..., 8:12].sum(axis=2), arrays['state'] == 0)

    def test_straddling_object_stored_once_and_duplicate_rows_merge(self):
        scene = straight(); scene.actors = [actor('straddle', 8., 1.7)]
        grid = build_grid(bind(scene))
        self.assertEqual(set(grid['objects'][0]['slots']), {'front', 'left_front'})
        arrays = grid_to_arrays(grid)
        self.assertEqual(len(arrays['object_ids']), 1)
        self.assertEqual(int(np.count_nonzero(arrays['object_index'] == 0)), 2)
        duplicate = copy.deepcopy(grid)
        obj = duplicate['objects'][0]
        duplicate['objects'] = [dict(obj, slots=[name]) for name in obj['slots']]
        self.assertEqual(validate_grid_schema(duplicate), [])
        for key, value in arrays.items():
            np.testing.assert_array_equal(grid_to_arrays(duplicate)[key], value)
        duplicate['objects'][1]['heading'] += .2
        with self.assertRaises(ValueError):
            grid_to_arrays(duplicate)

    def test_unknown_empty_and_nonapplicable_are_not_zero_facts(self):
        for observed, crossable in ((True, True), (False, True), (True, False)):
            grid = fixture(observed=observed, crossable=crossable)
            restored = roundtrip(grid)
            self.assertEqual(restored['slots'], grid['slots'])
            self.assertEqual(restored['matrices'], grid['matrices'])
            self.assertEqual(report(grid, grid), report(restored, restored))
        scene = straight(); scene.diagnostics['lane_map_complete'] = False
        grid = build_grid(scene)
        self.assertEqual({v['state'] for k, v in grid['slots'].items() if k != 'ego'}, {'unknown'})
        self.assertEqual(roundtrip(grid)['matrices'], grid['matrices'])
        self.assertFalse(report(roundtrip(grid), grid)['valid'])

    def test_invalid_minimal_and_invalid_state_source_preserved(self):
        grid = {'version': 'bev-occ-grid-2.1', 'representation_version': 'lane-eight-interaction-2.0',
                'valid': False, 'reason': '自车 Frenet 投影奇异', 'objects': []}
        self.assertEqual(roundtrip(grid), grid)
        self.assertEqual(set(grid_to_arrays(grid)), {'storage_version', 'layout', 'metadata'})
        for field in ('velocity', 'acceleration'):
            scene = straight(); scene.actors = [actor('invalid', 15.)]
            setattr(scene.actors[0], field, np.array([np.nan, 0.]))
            grid = build_grid(bind(scene))
            self.assertFalse(grid['valid'])
            restored = roundtrip(grid)
            self.assertEqual(restored['invalid_state_ids'], ['invalid'])
            self.assertEqual(restored['slots']['front']['state'], 'unknown')
            self.assertIsNone(restored['matrices']['position'][0][1])
            arrays = grid_to_arrays(grid)
            change_meta(arrays, valid=True)
            with self.assertRaises(ValueError):
                grid_from_arrays(arrays)

    def test_zero_rear_range_and_missing_historical_range(self):
        grid = build_grid(bind(straight()), object_range={'front_m': 40., 'rear_m': 0.})
        self.assertEqual(roundtrip(grid)['role_range_m'], {'front_m': 40., 'rear_m': 0.})
        for field in ('role_range_m', 'assignment_protocol', 'assignment_parameters'):
            grid.pop(field)
        restored = roundtrip(grid)
        self.assertNotIn('role_range_m', restored)
        self.assertNotIn('assignment_protocol', restored)
        self.assertEqual(report(restored, grid), report(grid, grid))

    def test_corrupt_arrays_rejected(self):
        original = grid_to_arrays(fixture())
        changes = {
            'float32': lambda a: a.update(values=a['values'].astype(np.float32)),
            'wrong_shape': lambda a: a.update(values=a['values'][0]),
            'pickle': lambda a: a.update(object_ids=a['object_ids'].astype(object)),
            'nonfinite': lambda a: a['values'].__setitem__((0, 1, 0), np.nan),
            'mask': lambda a: a['value_valid'].__setitem__((0, 1, 0), False),
            'bad_reference': lambda a: a['object_index'].__setitem__((0, 1), 7),
            'duplicate_ids': lambda a: a['object_ids'].__setitem__(1, a['object_ids'][0]),
            'occupied_not_clear': lambda a: a['slot_flags'].__setitem__((0, 1, 1), 0),
            'empty_with_state': lambda a: a['values'].__setitem__((2, 1, 0), 1.),
            'missing_coverage': lambda a: a['evidence_present'].__setitem__((2, 1, 0), False),
            'empty_unknown_coverage': lambda a: a['evidence_valid'].__setitem__((2, 1, 0), False),
            'low_coverage': lambda a: a['slot_evidence'].__setitem__((2, 1, 0), .1),
            'signature': lambda a: change_meta(a, metric_signature='wrong'),
            'version': lambda a: a.update(storage_version=np.array('unknown-2.0')),
            'hidden_states': lambda a: change_meta(a, objects=[]),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                arrays = copy.deepcopy(original); change(arrays)
                with self.assertRaises(ValueError):
                    grid_from_arrays(arrays)

    def test_cross_slot_contradiction_rejected(self):
        scene = straight(); scene.actors = [actor('straddle', 8., 1.7)]
        arrays = grid_to_arrays(build_grid(bind(scene)))
        arrays['values'][0, 0, 0] += .1
        with self.assertRaises(ValueError):
            grid_from_arrays(arrays)

    def test_npz_no_pickle_no_duplicate_json_motion_and_broken_archive(self):
        stream = io.BytesIO(); grid = fixture(); save_grid(stream, grid)
        raw = stream.getvalue()
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            self.assertEqual(str(archive['storage_version']), GRID_STORAGE_VERSION)
            self.assertTrue(all(not archive[k].dtype.hasobject for k in archive.files))
            meta = json.loads(str(archive['metadata']))
            self.assertFalse({'objects', 'matrices', 'ego', 'reference', 'lane_roles',
                              'object_lane_assignment'} & set(meta))
        broken = io.BytesIO()
        np.savez_compressed(broken, unsafe=np.array([{'x': 1}], dtype=object)); broken.seek(0)
        with self.assertRaises(ValueError):
            load_grid(broken)
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            load_grid(io.BytesIO(raw[:len(raw)//2]))
        duplicate = io.BytesIO(raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(duplicate, 'a') as archive:
                archive.writestr('values.npy', archive.read('values.npy'))
        duplicate.seek(0)
        with self.assertRaises(ValueError):
            load_grid(duplicate)
        print('fixture storage bytes: JSON=%d NPZ=%d' %
              (len(json.dumps(grid, ensure_ascii=False).encode('utf-8')), len(raw)))

    def test_representation_capture_and_model_text_compatible(self):
        scene = straight(); scene.actors = [actor('a', 12.), actor('b', 20., 3.5, lane='l')]
        bind(scene); scene.rules['map_available'] = {'crosswalk': True}
        output_root = ROOT / 'outputs' / 'validation'
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='grid-storage-check-', dir=output_root) as folder:
            reps = build_representations(scene, Path(folder), extractor=None)
            self.assertTrue(Path(reps.grid_path).is_file())
            self.assertEqual(Path(reps.grid_path).suffix, '.npz')
            self.assertFalse((Path(folder) / 'grid.json').exists())
            before = scene_text(scene, reps)
            reps.grid = load_grid(reps.grid_path)
            self.assertEqual(scene_text(scene, reps), before)
            self.assertEqual(report(reps.grid, reps.grid)['score'], 1.)


if __name__ == '__main__':
    unittest.main()
