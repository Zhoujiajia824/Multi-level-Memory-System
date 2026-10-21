"""Local generation is visible in replay without being reported as API SDK use."""
import json
from pathlib import Path

from carla_bridge.metrics.run_reporter import write_report
from carla_bridge.model_call_audit import ModelCallAudit
from carla_bridge.replay.review import build_review
from carla_bridge.tests.test_replay_review import (
    consultation, read_data, request, tick, write_rows,
)
from src.vla_memory.common.model_call_audit import describe_model_request, emit_model_request


def backend_request(*, backend, cid, rid, attempt=1, status="success", stage="scene"):
    rows = request(rid, cid, attempt, stage, status)
    for row in rows:
        row.update(backend=backend, provider="local_qwen" if backend == "local" else "qwen")
    return rows


def test_mixed_backends_keep_total_calls_and_api_sdk_counts_separate(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(i) for i in range(101, 105)])
    calls = consultation("local", 101) + consultation("api", 102) + consultation("legacy", 103)
    write_rows(tmp_path, "model_calls.jsonl", calls)
    requests = (
        backend_request(backend="local", cid="local", rid="local-scene", status="failed")
        + backend_request(backend="local", cid="local", rid="local-scene", attempt=2)
        + backend_request(backend="local", cid="local", rid="local-decision", stage="decision")[:1]
        + backend_request(backend="api", cid="api", rid="api-scene", status="error")
        + request("legacy-scene", "legacy")
    )
    write_rows(tmp_path, "model_requests.jsonl", requests)
    summary = build_review(tmp_path)
    assert summary["logical_model_request_count"] == 4
    assert summary["model_attempt_count"] == 5
    assert summary["local_attempt_count"] == 3
    assert summary["sdk_attempt_count"] == 2
    assert summary["model_failed_attempt_count"] == 2
    assert summary["sdk_failed_attempt_count"] == 1
    assert summary["model_incomplete_attempt_count"] == 1
    assert summary["sdk_incomplete_attempt_count"] == 0
    assert summary["actual_model_call_frame_count"] == 3
    assert summary["model_call_frame_ratio"] == .75
    assert summary["trigger_counts"] == {"blocked_lane": 3}
    by_id = {call["call_id"]: call for call in read_data(tmp_path)["calls"]}
    local = by_id["local"]
    assert local["actual_model_attempt_count"] == local["actual_local_attempt_count"] == 3
    assert local["has_actual_model_dispatch"] is True
    assert local["actual_sdk_attempt_count"] == 0
    assert local["has_actual_sdk_dispatch"] is False
    assert by_id["api"]["actual_sdk_attempt_count"] == 1
    assert by_id["legacy"]["requests"][0]["backend"] == "api"


def test_local_preparation_and_unproven_generate_are_not_inference(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    preflight = backend_request(backend="local", cid="c1", rid="preflight", status="failed")
    for row in preflight:
        row["dispatched"] = False
    unproven = backend_request(backend="local", cid="c1", rid="unproven")
    for row in unproven:
        row.pop("dispatched")
    write_rows(tmp_path, "model_requests.jsonl", preflight + unproven)
    summary = build_review(tmp_path)
    assert summary["model_attempt_count"] == summary["sdk_attempt_count"] == 0
    assert summary["local_attempt_count"] == 0
    assert summary["actual_model_call_frame_count"] == 0
    assert summary["model_call_frame_ratio"] == 0
    assert summary["trigger_counts"] == {}
    call = read_data(tmp_path)["calls"][0]
    assert call["has_actual_model_dispatch"] is False
    assert len(call["requests"]) == 2  # Preparation errors remain inspectable.


def test_legacy_response_without_start_is_still_counted_as_api(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    row = request()[1]
    row.pop("dispatched")
    write_rows(tmp_path, "model_requests.jsonl", [row])
    summary = build_review(tmp_path)
    assert summary["model_attempt_count"] == summary["sdk_attempt_count"] == 1
    assert summary["local_attempt_count"] == 0
    assert summary["model_call_frame_ratio"] == 1


def test_local_file_uri_messages_archive_exact_images_for_portable_replay(tmp_path):
    original = tmp_path / "source" / "local-input.jpg"
    original.parent.mkdir()
    original.write_bytes(b"exact-local-input-fixture")
    audit = ModelCallAudit(tmp_path / "run")
    messages = [{"role": "system", "content": "系统提示词"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": original.as_uri()}},
                    {"type": "text", "text": "检查路口"}]}]
    with audit.consultation({"carla_frame_id": 101}, {"reason": "blocked_lane"}, {}):
        with describe_model_request("scene", messages, [str(original)]):
            emit_model_request("request_start", backend="local", provider="local_qwen",
                               model="Qwen2.5-VL-3B-Instruct", attempt=1, dispatched=True)
            emit_model_request("request_end", backend="local", provider="local_qwen",
                               attempt=1, dispatched=True, status="success", response='{"scene":1}')
    rows = [json.loads(line) for line in (audit.run_dir / "model_requests.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert all(row["backend"] == "local" and row["provider"] == "local_qwen" for row in rows)
    archived = rows[0]["messages"][1]["content"][0]["image_url"]["url"]
    assert archived == rows[0]["image_paths"][0]
    assert (audit.run_dir / archived).read_bytes() == original.read_bytes()
    assert rows[0]["prompt"] == "检查路口"
    assert rows[0]["system_prompt"] == "系统提示词"
    assert messages[1]["content"][0]["image_url"]["url"] == original.as_uri()
    original.unlink()
    write_rows(audit.run_dir, "control.jsonl", [tick(101)])
    summary = build_review(audit.run_dir)
    assert summary["model_attempt_count"] == summary["local_attempt_count"] == 1
    assert summary["sdk_attempt_count"] == 0
    assert read_data(audit.run_dir)["calls"][0]["requests"][0]["images"][0]["url"].startswith(
        "../model_call_assets/")


def test_local_report_uses_model_total_and_explicit_backend_breakdown(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    write_rows(tmp_path, "model_calls.jsonl", consultation())
    write_rows(tmp_path, "model_requests.jsonl",
               backend_request(backend="local", cid="c1", rid="local-scene"))
    usage = build_review(tmp_path)
    md, _ = write_report({}, {"scenario": "local", "mode": "memory_off", "model_usage": usage}, str(tmp_path))
    text = Path(md).read_text(encoding="utf-8")
    assert "认知咨询 / 逻辑模型请求 / 实际模型调用尝试: 1 / 1 / 1" in text
    assert "API SDK 0 次；本地推理 1 次" in text


def test_unknown_legacy_evidence_preserves_unknown_model_total(tmp_path):
    write_rows(tmp_path, "control.jsonl", [tick(101)])
    summary = build_review(tmp_path)
    assert summary["model_attempt_count"] is None
    assert summary["local_attempt_count"] is None
    assert summary["sdk_attempt_count"] is None
