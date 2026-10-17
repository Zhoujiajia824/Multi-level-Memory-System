"""Measured stop context and the actual semantic API message contract."""
import pytest

from src.vla_memory.common.config import Config
from src.vla_memory.common.prompt_loader import get_prompt_loader
from src.vla_memory.decision.config_access import get_decision_system_prompt
from src.vla_memory.decision.decision_client import DecisionClient
from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient


def stopped_history():
    return [{"t": round(-4.9 + i * .1, 4), "x": 0, "y": 0} for i in range(50)]


def test_stationary_window_is_measured_even_without_memory():
    prompt = DecisionPromptBuilder().build(
        {}, ego_state={"speed": 0}, history_trajectory=stopped_history(),
        planning_constraints={"decision_mode": "semantic"},
    )
    assert "连续近静止至少 4.90 秒" in prompt
    assert "t=-0.9" in prompt
    assert "## 短期记忆" not in prompt
    assert "以上仅描述实测运动状态，通行权由当前交通情况判断" in prompt


@pytest.mark.parametrize("history", [
    [], [{"x": 0, "y": 0}],
    [{"t": -1, "x": 0, "y": 0}, {"t": -1, "x": 0, "y": 0}],
    [{"t": -1, "x": 0, "y": 0}, {"t": float("nan"), "x": 0, "y": 0}],
    [{"t": -2, "x": 0, "y": 0}, {"t": -1, "x": 0, "y": 0}],
    [{"t": -.1, "x": 0, "y": 0}, {"t": .1, "x": 0, "y": 0}],
    [{"t": -.1, "x": 5, "y": 0}, {"t": 0, "x": 5, "y": 0}],
])
def test_missing_invalid_or_stale_measurements_do_not_prove_stopping(history):
    context = DecisionPromptBuilder._render_motion_context(history, {"speed": 0})
    assert "连续静止时长未知" in context


def test_gap_breaks_stationary_duration_instead_of_counting_unobserved_time():
    history = [{"t": t, "x": 0, "y": 0} for t in (-5, -.2, -.1, 0)]
    assert "连续近静止至少 0.20 秒" in DecisionPromptBuilder._render_motion_context(history, {"speed": 0})


def test_recent_movement_breaks_stationary_duration():
    history = [{"t": -.3, "x": -.5, "y": 0}, {"t": -.2, "x": 0, "y": 0},
               {"t": -.1, "x": 0, "y": 0}, {"t": 0, "x": 0, "y": 0}]
    assert "连续近静止至少 0.20 秒" in DecisionPromptBuilder._render_motion_context(history, {"speed": 0})
    assert "连续近静止至少 0.00 秒" in DecisionPromptBuilder._render_motion_context(history, {"speed": 2})


def test_route_reference_points_do_not_hide_real_actors():
    anchors = [{"semantic_label": "route_center", "position_ego": [i, 0]} for i in range(6)]
    actors = [{"category": "vehicle", "semantic_label": "ACTOR_%02d" % i,
               "position_ego": [20+i, 0]} for i in range(12)]
    prompt = DecisionPromptBuilder().build({}, perception_objects=anchors+actors,
                                          planning_constraints={"decision_mode": "semantic"})
    assert "CARLA GT" in prompt
    assert "nuScenes GT" not in prompt
    assert "导航路线空间参考（仅表示路线位置）" in prompt
    assert "ACTOR_11" in prompt
    assert "共 12 个目标" in prompt


@pytest.mark.parametrize("backend", ["pdm_closed", "vlm_trajectory"])
def test_actual_api_messages_use_the_backend_system_contract(monkeypatch, backend):
    config = Config(overrides={"carla": {"planning": {"backend": backend}},
                               "decision": {"system_prompt": "LEGACY_OUTPUT_WAYPOINTS"}})
    client = OpenAICompatibleVLMClient(system_prompt=get_decision_system_prompt(config))
    sent = []
    monkeypatch.setattr(client, "_call_with_retry", lambda messages: sent.extend(messages) or "{}")
    mode = "semantic" if backend == "pdm_closed" else "trajectory"
    kwargs = {"planning_constraints": {"decision_mode": "semantic"}} if mode == "semantic" else {}
    assert DecisionClient(client).decide(scene_understanding={}, **kwargs) == "{}"
    assert sent[0]["role"] == "system"
    if mode == "semantic":
        assert "LEGACY_OUTPUT_WAYPOINTS" not in str(sent)
        assert "lateral_decision" in sent[0]["content"]
        assert "持续完成导航路线" in sent[0]["content"]
    else:
        assert sent[0]["content"] == "LEGACY_OUTPUT_WAYPOINTS"
        assert "实测运动窗口" not in str(sent)


def test_scene_prompt_renders_without_json_placeholder_errors():
    prompt = get_prompt_loader().render("scene_understanding.user", image_layout="六视角图像")
    assert "单帧图像不能可靠判断自车速度" in prompt
    assert "has_stop_sign" in prompt
    assert "{image_layout}" not in prompt
