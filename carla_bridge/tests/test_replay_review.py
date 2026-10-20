"""Frame attribution and offline output: no CARLA server or model credentials."""
import json
from pathlib import Path

import pytest

from carla_bridge.replay.review import _asset, build_review


def write_rows(root, name, rows):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return target


def read_data(root):
    return json.loads((root / "replay" / "data.json").read_text(encoding="utf-8"))


def tick(frame, sequence=None, source="tracker"):
    return {"sim_time_s": (frame - 100) * .1, "world_snapshot": {"frame": frame},
            "measured": {"x": frame * .2, "y": 4},
            "command": {"planner_sequence": sequence, "source": source}}


def consultation(cid="c1", frame=101, sample_token="sample1"):
    return [{"event": "consultation_start", "call_id": cid,
             "carla_frame_id": frame, "sim_time_s": (frame - 100) * .1,
             "sample_token": sample_token, "query": {"reason": "blocked_lane"},
             "ego": {"x": 20, "y": 4}},
            {"event": "consultation_end", "call_id": cid, "status": "success",
             "record": {"current_scene": {"scene_description": "车辆遮挡当前车道"},
                        "model_decision_output": {"lateral_decision": "CHANGE_LANE_LEFT"},
                        "raw_response": "完整模型输出"}}]


def request(rid="request1", cid="c1", attempt=1, stage="scene", status="success"):
    common = {"request_id": rid, "call_id": cid, "attempt": attempt, "stage": stage}
    return [{**common, "event": "request_start", "dispatched": True,
             "prompt": "精确用户输入", "system_prompt": "系统提示词", "status": "started"},
            {**common, "event": "request_end", "dispatched": True, "status": status,
             "response": "原始场景响应" if status == "success" else None,
             "error": "超时" if status == "error" else None}]


def plan(sequence, adopted=False, accepted=True, **overrides):
    diagnostic = {"selected_id": 2 if adopted else 1, "baseline_selected_id": 1,
                  "governor": {"adopted": adopted, "authorized": True},
                  "candidates": [{"id": 1, "base_score": .8, "trajectory": ["DO_NOT_EMBED"]},
                                 {"id": 2, "base_score": .75}]}
    diagnostic.update(overrides)
    return {"sim_time_s": sequence * .1, "sequence": sequence,
            "semantic_frame_id": "sample1", "semantic_influence": True,
            "planner_diagnostics": diagnostic,
            "trajectory": ["DO_NOT_EMBED_LARGE_TRAJECTORY"],
            "validation": {"accepted": accepted}}


def test_sdk_stages_retries_and_unique_actual_frames(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101), tick(101), tick(102), tick(103), tick(104)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    rows = request(attempt=1, status="error") + request(attempt=2) + request("request2", stage="decision")
    write_rows(tmp_path, "model_requests.jsonl", rows)
    summary = build_review(tmp_path)
    assert summary["consultation_count"] == 1
    assert summary["logical_model_request_count"] == 2
    assert summary["sdk_attempt_count"] == 3
    assert summary["sdk_failed_attempt_count"] == 1
    assert summary["sdk_incomplete_attempt_count"] == 0
    assert summary["actual_model_call_frame_count"] == 1
    assert summary["total_tick_frame_count"] == 4
    assert summary["model_call_frame_ratio"] == .25
    assert len(read_data(tmp_path)["calls"][0]["requests"]) == 3
    assert (tmp_path / "replay" / "summary.json").is_file()


def test_semantics_execution_uses_preceding_control_sequence(tmp_path):
    # A changed plan at sim=.1 does not affect the measured tick at sim=.1;
    # only ticks explicitly pointing to its applied command can be attributed.
    write_rows(tmp_path, "control.jsonl", [tick(101, 1), tick(102, 2), tick(103, 2),
                                          tick(104, 3), tick(105, 2, source="emergency_brake")])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "model_requests.jsonl", request())
    write_rows(tmp_path, "planner.jsonl", [plan(1), plan(2, True), plan(3, True, accepted=False)])
    summary = build_review(tmp_path)
    assert summary["semantic_effect_plan_ratio"] == pytest.approx(2 / 3)
    assert summary["semantic_effect_accepted_plan_count"] == 1
    assert summary["semantic_effect_tick_frame_count"] == 2
    assert summary["semantic_attributable_tick_frame_count"] == 3
    assert summary["semantic_effect_tick_frame_ratio"] == pytest.approx(2 / 3)
    assert summary["semantic_effect_tick_total_run_ratio"] is None
    data = read_data(tmp_path)
    assert data["calls"][0]["planner"]["related_plan_count"] == 3
    serialized = (tmp_path / "replay" / "data.json").read_text(encoding="utf-8")
    assert "DO_NOT_EMBED" not in serialized


def test_model_recovery_and_shadow_are_accounted_separately(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101, 1), tick(102, 2)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "model_requests.jsonl", [])
    write_rows(tmp_path, "planner.jsonl", [
        plan(1, False, original_baseline_selected_id=9,
             recovery={"authorized": True, "expanded": True, "selected": True}),
        plan(2, True, shadow_only=True),
    ])
    summary = build_review(tmp_path)
    assert summary["semantic_effect_plan_count"] == 1
    assert summary["semantic_effect_tick_frame_count"] == 1


def test_missing_tick_denominator_is_unknown_not_zero(tmp_path):
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "model_requests.jsonl", request())
    summary = build_review(tmp_path)
    assert summary["actual_model_call_frame_count"] == 1
    assert summary["total_tick_frame_count"] is None
    assert summary["model_call_frame_ratio"] is None
    assert summary["warnings"]


def test_instrumented_empty_model_free_run_is_proven_zero(tmp_path):
    (tmp_path / "replay_metadata.json").write_text(json.dumps({
        "audit_enabled": True, "model_inference": False, "audit_errors": []}), encoding="utf-8")
    write_rows(tmp_path, "control.jsonl", [tick(101), tick(102)])
    summary = build_review(tmp_path)
    assert summary["sdk_attempt_count"] == 0
    assert summary["actual_model_call_frame_count"] == 0
    assert summary["model_call_frame_ratio"] == 0
    assert summary["source_mode"] == "instrumented"
    assert summary["audit_complete"] is True


def test_consultation_before_dispatch_is_not_labeled_actual_model_call(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    write_rows(tmp_path, "model_calls.jsonl", consultation()[:1])
    write_rows(tmp_path, "model_requests.jsonl", [])
    summary = build_review(tmp_path)
    assert summary["consultation_count"] == 1
    assert summary["model_call_frame_ratio"] == 0
    assert summary["trigger_counts"] == {}
    assert summary["consultation_trigger_counts"] == {"blocked_lane": 1}
    call = read_data(tmp_path)["calls"][0]
    assert call["actual_sdk_attempt_count"] == 0
    assert call["has_actual_sdk_dispatch"] is False


def test_failed_auditor_does_not_claim_complete_zero_rate(tmp_path):
    (tmp_path / "replay_metadata.json").write_text(json.dumps({
        "audit_enabled": True, "audit_errors": ["disk full"]}), encoding="utf-8")
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    summary = build_review(tmp_path)
    assert summary["model_call_frame_ratio"] is None
    assert summary["audit_complete"] is False
    assert any("disk full" in w for w in summary["warnings"])


def test_legacy_does_not_invent_prompt_requests_or_actual_rate(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    write_rows(tmp_path, "decisions_carla_case_memory_off.jsonl", [{
        "sample_token": "sample1", "carla_frame_id": 101, "sim_time_s": .1,
        "current_scene": {"scene_description": "路口"}, "decision_output": {"behavior": "CRUISE"}}])
    summary = build_review(tmp_path)
    assert summary["consultation_count"] == 1
    assert summary["source_mode"] == "legacy_inferred"
    assert summary["sdk_attempt_count"] is None
    assert summary["model_call_frame_ratio"] is None
    assert read_data(tmp_path)["calls"][0]["requests"] == []


def test_video_alignment_preserves_encoded_mapping_not_sim_time(tmp_path):
    video = tmp_path / "carla_videos" / "drive.mp4"
    video.parent.mkdir()
    video.write_bytes(b"fixture")
    write_rows(tmp_path, "carla_videos/drive.frames.jsonl", [
        {"video_index": 0, "carla_frame_id": 101, "sim_time_s": .1, "video_time_s": 0},
        # One encoded gap and a 20fps video with 10Hz simulation.
        {"video_index": 1, "carla_frame_id": 103, "sim_time_s": .3, "video_time_s": .05},
    ])
    build_review(tmp_path)
    data = read_data(tmp_path)
    assert data["videos"][0]["frames"] == [[0.0, 101, .1], [.05, 103, .3]]
    assert data["videos"][0]["url"] == "../carla_videos/drive.mp4"
    html = (tmp_path / "replay" / "index.html").read_text(encoding="utf-8")
    assert "按逐帧映射精确定位" in html
    assert "目标帧未录入视频" in html


def test_corrupt_or_interrupted_request_audit_remains_reviewable(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    write_rows(tmp_path, "model_calls.jsonl", consultation()[:1])
    p = write_rows(tmp_path, "model_requests.jsonl", request()[:1])
    with p.open("a", encoding="utf-8") as stream:
        stream.write('{"event":')
    summary = build_review(tmp_path)
    assert summary["sdk_attempt_count"] == 1
    assert summary["sdk_incomplete_attempt_count"] == 1
    assert summary["model_call_frame_ratio"] is None
    assert read_data(tmp_path)["calls"][0]["completed"] is False
    assert any("损坏" in w for w in summary["warnings"])


def test_images_survive_linux_to_windows_directory_move(tmp_path):
    image = tmp_path / "model_call_assets" / "test image.jpg"
    image.parent.mkdir()
    image.write_bytes(b"fixture")
    remote = "/data/zjj/outputs/carla_runs/" + tmp_path.name + "/model_call_assets/test image.jpg"
    assert _asset(remote, tmp_path.resolve()) == "../model_call_assets/test%20image.jpg"
    assert _asset("model_call_assets/test image.jpg", tmp_path.resolve()) == "../model_call_assets/test%20image.jpg"
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("private", encoding="utf-8")
    assert _asset(str(outside), tmp_path.resolve()) is None
    assert _asset("../outside_secret.txt", tmp_path.resolve()) is None
    assert _asset("javascript:alert(1)", tmp_path.resolve()) is None
    assert _asset("https://example.com/image.jpg", tmp_path.resolve()) is None


def test_model_output_cannot_end_script_or_execute_html(tmp_path):
    calls = consultation()
    attack = '</script><script>throw new Error("injected")</script>'
    calls[1]["record"]["raw_response"] = attack
    write_rows(tmp_path, "model_calls.jsonl", calls)
    build_review(tmp_path)
    html = (tmp_path / "replay" / "index.html").read_text(encoding="utf-8")
    assert attack not in html
    assert "\\u003c/script\\u003e" in html
    assert read_data(tmp_path)["calls"][0]["record"]["raw_response"] == attack
    assert "fetch(" not in html
    assert "innerHTML" not in html


def test_unmatched_api_frame_does_not_publish_bogus_fraction(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(102)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "model_requests.jsonl", request())
    summary = build_review(tmp_path)
    assert summary["actual_model_call_frame_count"] == 1
    assert summary["actual_model_call_frames_in_tick_log"] == 0
    assert summary["model_call_frame_ratio"] is None


@pytest.mark.parametrize("broken_file", ["control.jsonl", "planner.jsonl"])
def test_corrupt_tick_or_planner_tail_cannot_claim_complete_scene_rate(tmp_path, broken_file):
    write_rows(tmp_path, "control.jsonl", [tick(101, 1), tick(102, 1)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "model_requests.jsonl", request())
    write_rows(tmp_path, "planner.jsonl", [plan(1, True)])
    with (tmp_path / broken_file).open("a", encoding="utf-8") as stream:
        stream.write('{"interrupted":')
    summary = build_review(tmp_path)
    assert summary["total_tick_frame_count"] == 2
    assert summary["semantic_effect_tick_frame_count"] == 2
    assert summary["semantic_effect_tick_frame_ratio"] is None
    if broken_file == "control.jsonl":
        assert summary["model_call_frame_ratio"] is None
        assert summary["control_audit_complete"] is False
    else:
        assert summary["semantic_effect_plan_count"] == 1
        assert summary["semantic_effect_plan_ratio"] is None
        assert summary["planner_audit_complete"] is False


def test_large_planner_is_streamed_and_only_bounded_related_samples_embedded(tmp_path):
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "planner.jsonl", (plan(i, bool(i % 2)) for i in range(100)))
    build_review(tmp_path)
    audit = read_data(tmp_path)["calls"][0]["planner"]
    assert audit["related_plan_count"] == 100
    assert len(audit["samples"]) <= 12
    assert audit["samples"][-1]["sequence"] == 99


def test_build_does_not_require_native_or_model_imports(tmp_path):
    assert build_review(tmp_path)["consultation_count"] == 0
    with pytest.raises(ValueError, match="运行目录不存在"):
        build_review(tmp_path / "does_not_exist")


def test_offline_review_javascript_logic_when_node_available():
    """Optional stdlib-only JS logic checks; this never launches a browser."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is optional; offline review logic can be checked manually where Node is available")
    script = Path(__file__).with_name("replay_logic_test.js")
    result = subprocess.run([node, str(script)], cwd=script.parents[2],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    if result.returncode == 77:
        pytest.skip(result.stdout.strip())
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "replay logic unit tests passed" in result.stdout
