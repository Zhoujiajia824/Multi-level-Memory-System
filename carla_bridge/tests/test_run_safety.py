"""Regression tests for isolation, readonly memory, frame synchronization and cleanup.

Uses real local FAISS/CARLA bindings, but no server, model download or API request.
"""
import hashlib
import json
import math
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from carla_bridge.control.pure_pursuit import PurePursuit
from carla_bridge.control.trajectory_tracker import TrajectoryTracker
from carla_bridge.memory_adapter.keyframe_builder import KeyframeBuilder
from carla_bridge.run_session import BankLock, RunSession
from carla_bridge.sensors.camera_manager import CameraManager
from src.vla_memory.common.config import Config
from src.vla_memory.memory.faiss_store import FAISSVectorStore
from src.vla_memory.memory.mid_term_memory import MidTermMemory
from src.vla_memory.schemas.memory import MidTermMemoryRecord


def fingerprint(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def bank_at(path, readonly=False):
    return MidTermMemory(FAISSVectorStore(dimension=3), save_dir=str(path),
                         read_only=readonly, persistence_cfg={"enabled": True,
                         "auto_load_on_init": True, "save_on_close": True, "strict_load": True})


def seed_bank(path):
    bank = bank_at(path)
    bank.add_record(MidTermMemoryRecord(record_id="old", scene_text="road"),
                    np.array([1., 0., 0.], dtype=np.float32))
    bank.close()


def session_at(tmp_path, **kwargs):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scenario_name: test\nseed: 42\n", encoding="utf-8")
    cfg = Config(overrides={"output_dir": str(tmp_path / "outputs"), "carla": {}})
    return RunSession(cfg, scenario, kwargs.pop("mode", "memory_on"), **kwargs)


@pytest.mark.parametrize("side", [-1, 1])
def test_circle_curvature_magnitude(side):
    # At 90 degrees on radius 10, chord=(10, +/-10), curvature=+/-0.1.
    angle = PurePursuit().compute_steer(0, 0, 0, 0, [{"x": 10., "y": side * 10.}])
    assert angle == pytest.approx(-side * math.atan(2.8 / 10))


def test_target_progress_and_endpoint_braking():
    tracker = TrajectoryTracker({})
    tracker.set_trajectory([{"x": x, "y": 0, "t": x / 5} for x in range(21)], 0, 0, 0, 5)
    tracker.compute_control(12, 1, 0, 5)
    progress = tracker.pp._progress
    tracker.compute_control(11.5, 1, 0, 5)
    assert tracker.pp._progress >= progress
    control = tracker.compute_control(21, 0, 0, 5)
    assert control.brake == 1 and control.throttle == 0


def test_readonly_search_and_close_do_not_change_memory(tmp_path):
    seed_bank(tmp_path)
    before = fingerprint(tmp_path)
    bank = bank_at(tmp_path, readonly=True)
    state = bank.get_all_records()["old"].model_dump()
    for t in (1, 2, 3):
        assert bank.search(scene_text="road", now_ts=t)["results"]
    assert bank.get_all_records()["old"].model_dump() == state
    for mutation in (lambda: bank.add_record(MidTermMemoryRecord(record_id="new")),
                     lambda: bank.soft_delete("old", "test"), bank.save_full, bank.rebuild_index):
        with pytest.raises(PermissionError):
            mutation()
    bank.close()
    assert fingerprint(tmp_path) == before


def test_accumulate_across_reopens_and_reject_duplicate(tmp_path):
    seed_bank(tmp_path)
    bank = bank_at(tmp_path)
    bank.add_record(MidTermMemoryRecord(record_id="new"), np.array([0., 1., 0.]))
    with pytest.raises(ValueError, match="ID"):
        bank.add_record(MidTermMemoryRecord(record_id="old"), np.array([1., 0., 0.]))
    assert bank.faiss_store.size() == 2
    bank.close()
    loaded = bank_at(tmp_path, readonly=True)
    assert set(loaded.get_all_records()) == {"old", "new"}


def test_runs_and_tokens_are_isolated(tmp_path):
    sessions = [session_at(tmp_path), session_at(tmp_path)]
    try:
        for s in sessions:
            s.prepare()
        assert sessions[0].run_dir != sessions[1].run_dir
        tokens = [KeyframeBuilder("same", run_id=s.run_id).next_token() for s in sessions]
        assert len(set(tokens)) == 2
        for s in sessions:
            assert s.config.get("feature_dir").is_relative_to(s.config.get("memory_db_dir"))
    finally:
        for s in sessions:
            s.close()


def test_readonly_session_does_not_change_bank(tmp_path):
    bank = tmp_path / "bank"
    seed_bank(bank)
    before = fingerprint(bank)
    s = session_at(tmp_path, memory_bank=str(bank), memory_access="read_only")
    try:
        s.prepare()
        assert s.config.get("memory_access") == {"retrieve": True, "write": False, "short_term": True}
        assert not s.config.get("feature_dir").is_relative_to(bank)
    finally:
        s.close()
    assert fingerprint(bank) == before


def test_bank_lock_released_after_close_and_failure(tmp_path):
    bank = tmp_path / "bank"
    first = BankLock(bank).acquire()
    try:
        with pytest.raises(RuntimeError):
            BankLock(bank).acquire()
    finally:
        first.close()
    BankLock(bank).acquire().close()
    session = session_at(tmp_path, memory_bank=str(bank), memory_access="read_only")
    with pytest.raises(ValueError):
        session.prepare()
    BankLock(bank).acquire().close()


def test_snapshot_relocates_bank_assets_and_preserves_source(tmp_path):
    source = tmp_path / "source"
    seed_bank(source)
    asset = source / "assets" / "original.jpg"
    asset.parent.mkdir()
    asset.write_bytes(b"example")
    meta = source / "mid_term_meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["old"]["image_path"] = str(asset)
    meta.write_text(json.dumps(data), encoding="utf-8")
    before = fingerprint(source)
    s = session_at(tmp_path, memory_seed=str(source))
    try:
        s.prepare()
        copied = json.loads((s.config.get("memory_db_dir") / "mid_term_meta.json").read_text(encoding="utf-8"))
        assert Path(copied["old"]["image_path"]).read_bytes() == b"example"
        assert copied["old"]["image_path"] != str(asset)
    finally:
        s.close()
    assert fingerprint(source) == before


def cameras(tmp_path, monkeypatch):
    manager = CameraManager(None, None, [{"name": str(i)} for i in range(6)],
                            str(tmp_path / "images"), str(tmp_path / "mosaic"), capture_timeout_s=0.02)
    manager._installed = True
    manager._queues = {str(i): queue.Queue() for i in range(6)}
    saved = []
    monkeypatch.setattr(manager, "_save_image", lambda image, path: saved.append(image.frame))
    monkeypatch.setattr("carla_bridge.sensors.camera_manager.build_surround_mosaic", Mock())
    return manager, saved


def test_camera_discards_stale_frames_and_matches_all_views(tmp_path, monkeypatch):
    manager, saved = cameras(tmp_path, monkeypatch)
    for q in manager._queues.values():
        q.put(SimpleNamespace(frame=7))
        q.put(SimpleNamespace(frame=8))
    manager.capture("frame", 8)
    assert saved == [8] * 6
    assert manager.last_frame_id == 8


@pytest.mark.parametrize("future", [False, True])
def test_camera_missing_or_future_frame_never_writes_mixed_mosaic(tmp_path, monkeypatch, future):
    manager, saved = cameras(tmp_path, monkeypatch)
    for i, q in enumerate(manager._queues.values()):
        if i != 3:
            q.put(SimpleNamespace(frame=8))
        elif future:
            q.put(SimpleNamespace(frame=9))
    with pytest.raises(TimeoutError):
        manager.capture("frame", 8)
    assert saved == []


def test_close_partial_setup_and_stop_failure_still_destroys(tmp_path):
    from carla_bridge.closed_loop import ClosedLoop
    loop = ClosedLoop(Config(), "unused")
    loop.carla_client = Mock()
    loop.scenario = Mock()
    loop.cameras = Mock()
    loop._collision_sensor = Mock()
    loop._collision_sensor.stop.side_effect = RuntimeError("stop failed")
    loop.close()
    loop.close()
    loop._collision_sensor.destroy.assert_called_once()
    loop.scenario.destroy.assert_called_once()
    loop.carla_client.restore.assert_called_once()


def test_capture_failure_brakes_and_terminates():
    from carla_bridge.closed_loop import ClosedLoop
    loop = ClosedLoop(Config(), "unused")
    loop.tracker = Mock()
    ego = Mock()
    ego.get_transform.return_value = SimpleNamespace(location=SimpleNamespace(x=0, y=0),
                                                      rotation=SimpleNamespace(yaw=0))
    for _ in range(3):
        loop._on_capture_failure(ego, TimeoutError())
    control = ego.apply_control.call_args[0][0]
    assert control.brake == 1 and control.throttle == 0
    assert loop._early_terminated and loop._termination_reason == "sensor_failure"


@pytest.mark.parametrize("access,retrieve,short_term", [
    ("read_only", True, True), ("read_only", True, False),
    ("read_write", True, True), ("read_write", False, False),
])
def test_full_pipeline_memory_policy(tmp_path, monkeypatch, access, retrieve, short_term):
    from src.vla_memory.pipeline import online_loop as module
    # Replace only expensive/external cognition. Exercise the actual setup, retrieval,
    # step admission, short-term updates, event/update policy and persistence.
    monkeypatch.setattr(module, "DINOv2Extractor", Mock())
    monkeypatch.setattr(module, "OpenAICompatibleVLMClient", Mock())
    path = tmp_path / "bank"
    seed_bank(path)
    before = fingerprint(path)
    session = session_at(tmp_path, memory_bank=str(path), memory_access=access,
                         retrieve=retrieve, short_term=short_term)
    loop = None
    try:
        session.prepare()
        cfg = session.config
        cfg.data["feature_extractor"]["feature_dim"] = 3
        cfg.data["mid_term"]["admission"]["enabled"] = False
        cfg.data["mid_term"]["event_memory"]["enabled"] = access == "read_only"
        cfg.data["mid_term"]["capacity"]["enabled"] = False
        loop = module.OnlineDrivingLoop(cfg, "memory_on", resume=False)
        loop.setup()
        if access == "read_only":
            assert not loop._event_memory_enabled and not loop._update_enabled
            assert loop.mid_term._eviction_manager is None
        loop.scene_pipeline.process_frame = Mock(return_value={
            "scene_understanding": {"scene_description": "road", "scene_id": "unknown"}})
        loop.decision_client.decide = Mock(return_value="unused")
        loop._parse_or_fallback = Mock(return_value=({"behavior": "KEEP_LANE", "risk_level": "low",
                                                     "trajectory": [], "target_speed": 5}, [], "success", False))
        record = loop.step({"sample_token": session.run_id + "_1", "timestamp": 100,
                            "scene_token": "scene", "image_path": "", "nav_instruction": "straight"})
        assert bool(record["retrieved_memory_ids"]) == retrieve
        assert record["mid_term_memory_added"] == (access == "read_write")
        assert loop.mid_term.size() == (2 if access == "read_write" else 1)
        assert len(loop.short_term) == (1 if short_term else 0)
        loop.close()
        loop.close()
        if access == "read_only":
            assert fingerprint(path) == before
            assert loop.mid_term.get_all_records()["old"].hit_count == 0
        else:
            assert bank_at(path, readonly=True).size() == 2
    finally:
        if loop is not None:
            loop.close()
        session.close()


def test_setup_failure_preserves_client_for_cleanup(monkeypatch):
    from carla_bridge import closed_loop as module
    client = Mock()
    client.connect.side_effect = RuntimeError("connection setup failed")
    monkeypatch.setattr(module, "CarlaClient", Mock(return_value=client))
    loop = module.ClosedLoop(Config(), "unused")
    try:
        with pytest.raises(RuntimeError):
            loop.setup()
    finally:
        loop.close()
    client.restore.assert_called_once()


def test_scenario_partial_setup_is_cleaned(monkeypatch):
    from carla_bridge import closed_loop as module
    client = Mock()
    scenario = Mock()
    scenario.setup.side_effect = RuntimeError("spawn failed after ego creation")
    monkeypatch.setattr(module, "CarlaClient", Mock(return_value=client))
    monkeypatch.setattr(module.ScenarioManager, "from_yaml", Mock(return_value=scenario))
    loop = module.ClosedLoop(Config(), "unused")
    try:
        with pytest.raises(RuntimeError):
            loop.setup()
    finally:
        loop.close()
    scenario.destroy.assert_called_once()
    client.restore.assert_called_once()


def test_camera_install_failure_keeps_created_sensors_for_cleanup(tmp_path):
    world = Mock()
    first, second = Mock(), Mock()
    first.stop.side_effect = RuntimeError("stop failed")
    world.spawn_actor.side_effect = [first, second, RuntimeError("third camera failed")]
    cfg = [{"name": str(i), "x": 0, "y": 0, "z": 2} for i in range(6)]
    cameras = CameraManager(world, Mock(), cfg, str(tmp_path / "i"), str(tmp_path / "m"))
    try:
        with pytest.raises(RuntimeError):
            cameras.install()
    finally:
        cameras.destroy()
        cameras.destroy()
    first.destroy.assert_called_once()
    second.destroy.assert_called_once()


def test_failed_video_writer_releases_sensor_and_writer(tmp_path, monkeypatch):
    from carla_bridge.video import recorder as module
    world, writer = Mock(), Mock()
    writer.isOpened.return_value = False
    monkeypatch.setattr(module.cv2, "VideoWriter", Mock(return_value=writer))
    recorder = module.VideoRecorder(world, Mock(), str(tmp_path / "test.mp4"))
    try:
        with pytest.raises(RuntimeError):
            recorder.start()
    finally:
        recorder.flush()
        recorder.flush()
    writer.release.assert_called_once()
    world.spawn_actor.return_value.destroy.assert_called_once()


def test_entrypoint_records_setup_failure_and_releases_lock(tmp_path, monkeypatch):
    from carla_bridge import run_carla_demo as module
    helper = session_at(tmp_path)
    cfg = helper.config
    monkeypatch.setattr(module, "load_config", Mock(return_value=cfg))
    monkeypatch.setattr(module, "setup_logger", Mock())
    monkeypatch.setattr(module, "set_log_directory", Mock())
    for block in ("scene_understanding", "decision"):
        monkeypatch.setenv(cfg.get_nested(block, "api_key_env", default="DASHSCOPE_API_KEY"), "test-only")
    loop = Mock()
    loop.setup.side_effect = RuntimeError("model loading failed")
    monkeypatch.setattr(module, "ClosedLoop", Mock(return_value=loop))
    monkeypatch.setattr("sys.argv", ["run_carla_demo", "--scenario", str(helper.scenario_path)])
    with pytest.raises(RuntimeError, match="model loading"):
        module.main()
    loop.close.assert_called_once()
    manifest = json.loads((cfg.get("output_dir") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"
    BankLock(Path(manifest["memory_bank"])).acquire().close()


def test_readonly_results_are_detached_from_library(tmp_path):
    seed_bank(tmp_path)
    bank = bank_at(tmp_path, readonly=True)
    result = bank.search(scene_text="road", now_ts=1)["results"][0]["record"]
    result.hit_count = 100
    bank.get_all_records()["old"].hit_count = 200
    assert bank.get_all_records()["old"].hit_count == 0


def test_snapshot_imports_legacy_external_assets(tmp_path):
    source = tmp_path / "source"
    seed_bank(source)
    external = tmp_path / "legacy.jpg"
    external.write_bytes(b"legacy")
    meta = source / "mid_term_meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["old"]["image_path"] = str(external)
    meta.write_text(json.dumps(data), encoding="utf-8")
    s = session_at(tmp_path, memory_seed=str(source))
    try:
        s.prepare()
        bank = s.config.get("memory_db_dir")
        copied = json.loads((bank / "mid_term_meta.json").read_text(encoding="utf-8"))
        reference = Path(copied["old"]["image_path"])
        assert reference.is_relative_to(bank)
        assert reference.read_bytes() == b"legacy"
    finally:
        s.close()


def test_corrupt_index_mapping_is_rejected_without_writing(tmp_path):
    seed_bank(tmp_path)
    (tmp_path / "mid_term_faiss.ids.json").write_text("[]", encoding="utf-8")
    before = fingerprint(tmp_path)
    with pytest.raises(ValueError, match="ID"):
        bank_at(tmp_path)
    assert fingerprint(tmp_path) == before
