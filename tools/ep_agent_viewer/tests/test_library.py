"""Read-only library browsing, integrity, encoding and confinement regressions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from epviz.library import LibraryStore, _grid_arrays


ROLES = (("left_front", "front", "right_front"), ("left_side", "ego", "right_side"),
         ("left_rear", "rear", "right_rear"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def tensor():
    values = np.zeros((3, 3, 7), dtype=np.float64)
    valid = np.zeros((3, 3, 7), dtype=bool)
    state = np.ones((3, 3), dtype=np.int8)
    kind = np.full((3, 3), -1, dtype=np.int8)
    indices = np.full((3, 3), -1, dtype=np.int8)
    flags = np.tile(np.array([1, 1, 0], dtype=np.int8), (3, 3, 1))
    for i, j, index, cls, numbers in ((0, 1, 0, 0, [19., .2, -1., 0., -.1, 0., .05]),
                                      (1, 1, -2, 0, [0., 0., 0., 0., .3, 0., 0.])):
        values[i, j] = numbers
        valid[i, j] = True
        state[i, j], kind[i, j], indices[i, j], flags[i, j] = 0, cls, index, [1, 1, 1]
    return dict(storage_version=np.array("lane-eight-array-1.0"), layout=np.array(1, dtype=np.uint8),
        values=values, value_valid=valid, state=state, kind=kind, object_index=indices,
        slot_flags=flags, slot_evidence=np.zeros((3, 3, 6), dtype=np.float64),
        evidence_present=np.zeros((3, 3, 6), dtype=bool), evidence_valid=np.zeros((3, 3, 6), dtype=bool),
        ego_velocity=np.array([8., 0.]), roads=np.ones((2, 3), dtype=np.int8),
        object_ids=np.array(["lead"]), metadata=np.array(json.dumps(dict(ego_id="ego", valid=True,
            coordinates="common_route_frenet", slot_reasons={"front": {"reason": "observed"}}))))


def legacy_grid():
    slots = {name: {"state": "empty", "id": None, "kind": None, "applicable": True,
                    "clear": True, "state_valid": False} for row in ROLES for name in row}
    slots["ego"] = dict(state="occupied", id="ego", kind="vehicle", applicable=True, clear=True, state_valid=True)
    matrices = {key: [[None] * 3 for _ in range(3)] for key in ("position", "velocity", "acceleration", "heading")}
    for key in ("position", "velocity", "acceleration"):
        matrices[key][1][1] = [0., 0.]
    matrices["heading"][1][1] = 0.
    return {"slots": slots, "matrices": matrices, "coordinates": "common_route_frenet", "valid": True}


class LibraryFixture:
    def __init__(self, root):
        self.root, self.entries = root, []

    def add(self, identity, source="highway", text="沿车道减速等待", declared=True):
        base = self.root / "records" / identity
        folder = base / "evidence" / "0"
        folder.mkdir(parents=True)
        record = {"id": identity, "revision": 3, "status": "active", "source": source,
            "clip_id": "clip-" + identity, "frame_id": identity + ":23", "time": 4.25,
            "text_path": "experience.txt", "feature_dir": "evidence/0", "mpc_tested": False,
            "annotation_is_ground_truth": False, "evidence": [
                {"offset_s": float(i), "time_s": 4.25 + i, "path": "evidence/" + str(i)} for i in range(2)]}
        if declared:
            record["retrieval_eligibility"] = {"eligible": True}
        save_json(base / "record.json", record)
        (base / "experience.txt").write_text(text, encoding="utf-8")
        for name in ("bev.png", "native.png", "grid.png", "feature.png"):
            (folder / name).write_bytes(b"registered-image-" + name.encode())
        (folder / "source").mkdir()
        (folder / "source" / "front.jpg").write_bytes(b"actual-audit-image")
        np.savez_compressed(folder / "grid.npz", **tensor())
        np.save(folder / "bev_feature.npy", np.array([.1, .2, .3], dtype=np.float32))
        occ = np.zeros((6, 2, 2), dtype=np.float32)
        occ[2, 0, 0] = 1.
        known = np.ones_like(occ); known[:, 1, 1] = 0.
        np.savez_compressed(folder / "occupancy.npz", occ=occ, known=known, raw=occ, raw_known=known)
        save_json(folder / "source_assets.json", dict(audit_only=True, used_for_retrieval=False,
            files=[dict(path="source/front.jpg", role="current_source_audit")]))
        save_json(folder / "features.json", dict(status="evidence_insufficient", current_only=True,
            paths=dict(bev="bev.png", feature_image="feature.png", grid="grid.npz", grid_preview="grid.png",
                bev_feature="bev_feature.npy", occupancy="occupancy.npz", source_000="source/front.jpg")))
        next_folder = base / "evidence" / "1"
        save_json(next_folder / "features.json", dict(status="ready", current_only=True, paths=dict(grid="grid.json")))
        save_json(next_folder / "grid.json", legacy_grid())
        self.rehash(identity)
        return base

    def rehash(self, identity):
        base = self.root / "records" / identity
        checksums = {p.relative_to(base).as_posix(): digest(p) for p in base.rglob("*")
                     if p.is_file() and p.name != "checksums.json"}
        save_json(base / "checksums.json", checksums)
        entry = dict(id=identity, path="records/" + identity + "/record.json", checksums_sha256=digest(base / "checksums.json"))
        self.entries = [e for e in self.entries if e["id"] != identity] + [entry]
        self.freeze()

    def freeze(self):
        save_json(self.root / "manifest.json", dict(protocol="retrospective-driving-experience-0.1",
            frozen=True, records=self.entries, compatibility=dict(main_reader_compatible=None,
                current_features_compatible=None, verification_status="not_verified")))
        save_json(self.root / "freeze.json", {"manifest_sha256": digest(self.root / "manifest.json")})


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="library-test-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = LibraryFixture(self.root)
        self.base = self.fixture.add("one", "openscene", "前车停车，沿车道减速等待。唯一全文检索词。").resolve()
        self.fixture.add("two", "highway", "环岛驶出需要合法归道", declared=False)
        self.store = LibraryStore(self.root)

    def add_scene(self, index=0, frame_id=None, timestamp=None):
        frame_id = frame_id or ('one:23' if index == 0 else 'one:next')
        timestamp = 4.25 + index if timestamp is None else timestamp
        folder = self.base / 'evidence' / str(index)
        features = json.loads((folder / 'features.json').read_text(encoding='utf-8'))
        features.update(frame_id=frame_id, time=timestamp)
        save_json(folder / 'features.json', features)
        save_json(folder / 'scene.json', dict(frame_id=frame_id, time=timestamp,
            coordinates='world_xy_m', ego=dict(id='ego', position=[0., 0.], heading=0., length=4., width=2.),
            actors=[], lanes=[], route=[[0., 0.], [30., 0.]]))
        self.fixture.rehash('one')

    def test_scene_snapshot_is_exact_current_or_future_evidence(self):
        self.add_scene(0); self.add_scene(1)
        current = self.store.scene_data('one', 0)
        future = self.store.scene_data('one', 1)
        self.assertTrue(current['available'])
        self.assertTrue(future['available'])
        self.assertEqual(current['frame_id'], 'one:23')
        self.assertEqual(future['frame_id'], 'one:next')
        self.assertEqual(future['scene_source'], 'evidence/1/scene.json')
        self.assertEqual(future['evidence_role'], 'retrospective_review')
        self.assertEqual(future['time'], 5.25)

    def test_scene_missing_tampered_or_wrong_frame_is_not_renderable(self):
        self.assertFalse(self.store.scene_data('one')['available'])
        self.add_scene()
        path = self.base / 'evidence' / '0' / 'scene.json'
        scene = json.loads(path.read_text(encoding='utf-8'))
        scene['frame_id'] = 'another_frame'
        save_json(path, scene)
        tampered = self.store.scene_data('one')
        self.assertFalse(tampered['available'])
        self.assertIn('哈希', tampered['reason'])
        self.fixture.rehash('one')
        misaligned = self.store.scene_data('one')
        self.assertFalse(misaligned['available'])
        self.assertIn('帧号', misaligned['reason'])

    def test_scene_time_and_anchor_record_must_match(self):
        self.add_scene(timestamp=6.25)
        result = self.store.scene_data('one')
        self.assertFalse(result['available'])
        self.assertIn('时间', result['reason'])
        self.add_scene(frame_id='not_the_anchor')
        result = self.store.scene_data('one')
        self.assertFalse(result['available'])
        self.assertIn('锚点', result['reason'])

    def add_future(self):
        self.add_scene(0); self.add_scene(1)
        value = dict(source='recorded_human_driving_not_optimality_proof', ego_trajectory=[
            dict(time_s=4.25, position=[0., 0.], heading=0.),
            dict(time_s=5.25, position=[0., 0.], heading=0.),
            dict(time_s=6.25, position=[2., 0.], heading=0.)])
        save_json(self.base / 'audit/supervision.json', value)
        self.fixture.rehash('one')
        return value

    def add_cameras(self):
        self.add_scene(0)
        folder = self.base / 'evidence/0'
        checksum = digest(folder / 'source/front.jpg')
        paths = dict(source_008='source/008_camera_manifest_original.json',
                     source_009='source/009_cameras_exported.json', source_010='source/010_scene_original.json')
        features = json.loads((folder / 'features.json').read_text(encoding='utf-8'))
        features['paths'].update(paths)
        save_json(folder / 'features.json', features)
        save_json(folder / paths['source_008'], [dict(view='CAM_F0', sample_timestamp_s=4.25)])
        save_json(folder / paths['source_009'], [dict(view='CAM_F0', exported_filename='camera_01.jpg', sha256=checksum)])
        save_json(folder / paths['source_010'], dict(current_context=dict(timestamp_s=4.25)))
        save_json(folder / 'source_assets.json', dict(files=[dict(path='source/front.jpg',
            sha256=checksum, original_relative_path='source/000004/camera_01.jpg')]))
        self.fixture.rehash('one')
        return folder

    def test_future_is_authenticated_and_bound_to_anchor_and_selected_state(self):
        self.add_future()
        anchor = self.store.scene_data('one', 0)['future']
        later = self.store.scene_data('one', 1)['future']
        self.assertTrue(anchor['available'])
        self.assertTrue(later['available'])
        self.assertEqual(later['anchor_frame_id'], 'one:23')
        self.assertEqual(later['evidence_frame_id'], 'one:next')
        self.assertEqual(later['usage'], 'offline_retrospective_display_only')
        self.assertEqual(later['ego_trajectory'][0]['time_s'], 4.25)
        # Existence under another record or on disk does not authorize use.
        self.assertFalse(self.store.scene_data('two', 0)['available'])

    def test_future_tamper_and_wrong_center_are_not_drawn(self):
        value = self.add_future()
        path = self.base / 'audit/supervision.json'
        value['ego_trajectory'][0]['position'] = [1., 0.]
        save_json(path, value)
        scene = self.store.scene_data('one')
        self.assertTrue(scene['available'])
        self.assertFalse(scene['future']['available'])
        self.assertIn('哈希', scene['future']['reason'])
        self.fixture.rehash('one')
        scene = self.store.scene_data('one')
        self.assertFalse(scene['future']['available'])
        self.assertIn('中心不一致', scene['future']['reason'])

    def test_future_wrong_time_order_or_wrong_evidence_state_is_rejected(self):
        value = self.add_future()
        path = self.base / 'audit/supervision.json'
        value['ego_trajectory'][1]['position'] = [9., 0.]
        save_json(path, value); self.fixture.rehash('one')
        self.assertFalse(self.store.scene_data('one', 1)['future']['available'])
        value['ego_trajectory'][1]['time_s'] = 4.25
        save_json(path, value); self.fixture.rehash('one')
        self.assertIn('不严格递增', self.store.scene_data('one')['future']['reason'])

    def test_camera_view_maps_by_export_name_and_hash_not_image_order(self):
        self.add_cameras()
        data = self.store.scene_data('one')['camera_data']
        self.assertTrue(data['available'])
        self.assertEqual(data['frame_id'], 'one:23')
        self.assertEqual(data['time'], 4.25)
        self.assertEqual(data['images'][0]['view'], 'CAM_F0')
        self.assertEqual(data['images'][0]['path'], 'evidence/0/source/front.jpg')
        self.assertTrue(data['images'][0]['sha256_verified'])
        self.assertIn('id=one', data['images'][0]['url'])
        self.add_scene(1)
        self.assertFalse(self.store.scene_data('one', 1)['camera_data']['available'])

    def test_camera_tampered_metadata_or_image_stops_overlay_not_scene(self):
        folder = self.add_cameras()
        (folder / 'source/front.jpg').write_bytes(b'changed')
        scene = self.store.scene_data('one')
        self.assertTrue(scene['available'])
        self.assertFalse(scene['camera_data']['available'])
        self.assertTrue(any('哈希' in reason for reason in scene['camera_data']['warnings']))
        self.fixture.rehash('one')
        save_json(folder / 'source/008_camera_manifest_original.json', [])
        scene = self.store.scene_data('one')
        self.assertFalse(scene['camera_data']['available'])
        self.assertIn('哈希', scene['camera_data']['reason'])

    def test_camera_duplicate_view_or_forged_export_hash_has_no_asset(self):
        folder = self.add_cameras()
        name = folder / 'source/009_cameras_exported.json'
        rows = json.loads(name.read_text(encoding='utf-8'))
        save_json(name, rows + rows); self.fixture.rehash('one')
        self.assertFalse(self.store.scene_data('one')['camera_data']['available'])
        rows[0]['sha256'] = '0' * 64
        save_json(name, rows); self.fixture.rehash('one')
        self.assertFalse(self.store.scene_data('one')['camera_data']['available'])

    def test_public_index_only_and_compatibility_not_inferred(self):
        save_json(self.root / "pending" / "hidden" / "record.json", {"id": "hidden"})
        save_json(self.root / ".history" / "old" / "record.json", {"id": "old"})
        summary = self.store.summary()
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["source_counts"], {"openscene": 1, "highway": 1})
        self.assertEqual(summary["declared_eligibility_counts"], {"true": 1, "missing": 1})
        self.assertIsNone(summary["compatibility"]["main_reader_compatible"])
        self.assertEqual(summary["compatibility"]["verification_status"], "not_verified")
        self.assertTrue(summary["freeze_valid"])
        with self.assertRaises(KeyError):
            self.store.detail("hidden")

    def test_missing_manifest_is_rejected_instead_of_empty_library(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "必须包含可读且合法的 manifest.json"):
            LibraryStore(self.root)

    def test_records_subdirectory_points_to_parent_without_silent_fallback(self):
        with self.assertRaises(ValueError) as raised:
            LibraryStore(self.root / "records")
        self.assertIn("records 子目录", str(raised.exception))
        self.assertIn(str(self.root), str(raised.exception))
        self.assertIn("未自动切换目录", str(raised.exception))

    def test_real_empty_manifest_is_a_valid_empty_library(self):
        save_json(self.root / "manifest.json", {"records": [], "frozen": False})
        empty = LibraryStore(self.root).summary()
        self.assertEqual(empty["count"], 0)
        self.assertTrue(empty["manifest_valid"])
        self.assertEqual(empty["warnings"], [])

    def test_malformed_manifest_and_duplicate_ids_are_rejected(self):
        valid_entry = self.fixture.entries[0]
        cases = [[], {}, {"records": {}}, {"records": [None]},
            {"records": [dict(valid_entry, id="")]}, {"records": [valid_entry, valid_entry]},
            {"records": [dict(valid_entry, path="pending/record.json")]},
            {"records": [dict(valid_entry, checksums_sha256="not-sha256")]},
            {"records": [dict(valid_entry, id="../bad")] }]
        for invalid in cases:
            with self.subTest(manifest=invalid):
                save_json(self.root / "manifest.json", invalid)
                with self.assertRaises(ValueError):
                    LibraryStore(self.root)
        (self.root / "manifest.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            LibraryStore(self.root)

    def test_manifest_damage_after_startup_is_exposed_and_can_recover(self):
        self.assertEqual(self.store.summary()["count"], 2)
        (self.root / "manifest.json").write_text("broken", encoding="utf-8")
        damaged = self.store.summary()
        self.assertFalse(damaged["manifest_valid"])
        self.assertEqual(damaged["count"], 0)
        self.assertTrue(any("manifest" in message for message in damaged["warnings"]))
        self.fixture.freeze()
        recovered = self.store.summary()
        self.assertTrue(recovered["manifest_valid"])
        self.assertEqual(recovered["count"], 2)
        self.assertEqual(recovered["warnings"], [])

    def test_full_text_search_filters_and_bounded_pagination(self):
        result = self.store.records("唯一全文检索词", source="openscene")
        self.assertEqual([r["id"] for r in result["records"]], ["one"])
        self.assertEqual(self.store.records("clip-two TWO:23")["total"], 1)
        self.assertEqual(self.store.records(status="archived")["total"], 0)
        self.assertEqual(len(self.store.records(offset=1, limit=1)["records"]), 1)
        self.assertEqual(self.store.records(limit=10000)["limit"], 100)
        self.assertNotIn("_search", result["records"][0])
        self.assertFalse(result["records"][0]["mpc_tested"])

    def test_npz_slot_decode_uses_validity_and_evidence_identity(self):
        detail = self.store.detail("one")
        self.assertEqual(detail["warnings"], [])
        grid = detail["grid"]
        self.assertEqual(grid["shape"], [3, 3, 7])
        self.assertEqual(grid["slots"][0][1]["id"], "lead")
        self.assertEqual(grid["slots"][1][1]["id"], "ego")
        self.assertEqual(grid["slots"][0][0]["state"], "empty")
        self.assertEqual(grid["values"][0][0], [None] * 7)
        self.assertEqual(grid["values"][1][1][:4], [0., 0., 0., 0.])
        self.assertEqual(grid["channels"][0], "delta_s_m")
        self.assertEqual(detail["features"]["status"], "evidence_insufficient")
        self.assertTrue(detail["record"]["retrieval_eligibility"]["eligible"])
        self.assertEqual(detail["evidence"][0]["role"], "retrieval_anchor")
        self.assertEqual(detail["evidence"][1]["role"], "retrospective_review")

    def test_future_json_is_review_only_even_if_feature_current_only(self):
        detail = self.store.detail("one", 1)
        self.assertEqual(detail["selected_evidence"], 1)
        self.assertTrue(detail["features"]["current_only"])
        self.assertEqual(detail["evidence"][1]["role"], "retrospective_review")
        self.assertEqual(detail["grid"]["storage_version"], "legacy_grid_json_read_only")
        self.assertEqual(detail["grid"]["slots"][1][1]["id"], "ego")
        with self.assertRaises(IndexError):
            self.store.detail("one", 20)

    def test_occ_keeps_unknown_mask_and_raw_array_is_summary_only(self):
        detail = self.store.detail("one")
        self.assertEqual(detail["occupancy"]["shape"], [6, 2, 2])
        self.assertEqual(detail["occupancy"]["known"][2][1][1], 0.)
        self.assertEqual(detail["occupancy"]["values"][2][0][0], 1.)
        self.assertTrue(any(a["name"] == "raw" for a in detail["arrays"]))
        self.assertNotIn("occupancy", detail["assets"])
        self.assertIn("不默认称RGB", detail["assets"]["native"]["label"])
        self.assertEqual(detail["assets"]["source_000"]["role"], "source_audit_image")
        self.assertTrue(detail["source_assets"]["audit_only"])

    def test_registered_images_and_traversal_denial(self):
        path = self.store.resolve_asset("one", "evidence/0/bev.png")
        self.assertEqual(path, self.base / "evidence" / "0" / "bev.png")
        (self.base / "secret.png").write_bytes(b"unregistered")
        for name in ("secret.png", "../two/evidence/0/bev.png", "/absolute.png", "C:/absolute.png",
                     ".history/secret.png", "evidence/0/grid.npz"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.resolve_asset("one", name)

    def test_asset_tamper_invalidates_verification_cache(self):
        path = self.store.resolve_asset("one", "evidence/0/bev.png")
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "哈希不匹配"):
            self.store.resolve_asset("one", "evidence/0/bev.png")
        detail = self.store.detail("one")
        self.assertNotIn("bev", detail["assets"])
        self.assertTrue(any("bev" in w for w in detail["warnings"]))

    def test_record_tamper_invalidates_context_cache(self):
        self.store.detail("one")
        save_json(self.base / "record.json", {"id": "one", "source": "fake"})
        detail = self.store.detail("one")
        self.assertFalse(detail["integrity"]["record_verified"])
        self.assertEqual(detail["assets"], {})
        with self.assertRaises(ValueError):
            self.store.resolve_asset("one", "evidence/0/bev.png")

    def test_freeze_hash_mismatch_is_visible_and_no_asset_url_is_published(self):
        save_json(self.root / "freeze.json", {"manifest_sha256": "0" * 64})
        summary = self.store.summary()
        self.assertFalse(summary["freeze_valid"])
        self.assertGreater(len(summary["warnings"]), 0)
        self.assertEqual(self.store.detail("one")["assets"], {})
        with self.assertRaises(ValueError):
            self.store.resolve_asset("one", "evidence/0/bev.png")

    def test_broken_record_isolated_from_other_records(self):
        (self.base / "checksums.json").write_bytes(b"broken")
        summary = self.store.summary()
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["records_with_warnings"], 1)
        self.assertEqual(self.store.detail("two")["warnings"], [])
        self.assertTrue(self.store.detail("one")["warnings"])

    def test_registered_symbolic_link_escape_denied(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        link = self.base / "escaped.png"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            self.skipTest("OS does not permit symlink creation: " + str(exc))
        self.fixture.rehash("one")
        with self.assertRaisesRegex(ValueError, "边界"):
            self.store.resolve_asset("one", "escaped.png")

    def test_array_enum_codes_are_not_silently_coerced(self):
        arrays = tensor()
        arrays["state"] = arrays["state"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "整数"):
            _grid_arrays(arrays)
        arrays = tensor(); arrays["metadata"] = np.array("[]")
        with self.assertRaisesRegex(ValueError, "必须是对象"):
            _grid_arrays(arrays)

    def test_diagnostic_only_grid_has_no_fabricated_slots(self):
        arrays = {"storage_version": np.array("lane-eight-array-1.0"),
            "layout": np.array(0, dtype=np.uint8),
            "metadata": np.array(json.dumps({"valid": False, "reason": "缺少可用车道参考"}))}
        grid = _grid_arrays(arrays)
        self.assertFalse(grid["available"])
        self.assertEqual(grid["slots"], [])
        self.assertNotIn("values", grid)
        self.assertEqual(grid["reason"], "缺少可用车道参考")

    def test_object_pickle_array_is_never_loaded(self):
        np.savez_compressed(self.base / "evidence" / "0" / "grid.npz", values=np.array([object()], dtype=object))
        self.fixture.rehash("one")
        detail = self.store.detail("one")
        self.assertIsNone(detail["grid"])
        self.assertTrue(any("Object arrays cannot be loaded" in w for w in detail["warnings"]))

    def test_browsing_does_not_write_to_library(self):
        before = {p.relative_to(self.root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        self.store.summary(); self.store.records("停车"); self.store.detail("one"); self.store.detail("one", 1)
        self.store.resolve_asset("one", "evidence/0/bev.png")
        after = {p.relative_to(self.root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
