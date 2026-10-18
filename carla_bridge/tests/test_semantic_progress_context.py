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
    assert "近静止不是永久保持 STOP 的理由" in prompt


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


def test_recorded_stall_inputs_keep_gt_motion_and_stop_history_above_visual_guess():
    # Reduced from run 78ccccea, t=108.1: the visual description guessed
    # movement, while the nearby Nissan's measured speed was exactly zero.
    prompt = DecisionPromptBuilder().build(
        {"scene_description": "路面有STOP文字，右侧黑色SUV正在行驶。",
         "vehicles": [{"relative_position": "90°", "distance_m": 3.0,
                       "type": "suv", "motion": "unknown"}]},
        ego_state={"speed": 0}, history_trajectory=stopped_history(),
        nav_instruction="right_turn",
        perception_objects=[{
            "category": "vehicle", "semantic_label": "patrol_2021",
            "position_ego": [-.3933, -3.4946], "distance_to_ego": 3.5167,
            "speed": 0.0, "velocity": [0.0, 0.0], "velocity_available": True,
            "kinematics_source": "carla_gt_direct",
        }], planning_constraints={"decision_mode": "semantic"},
    )
    assert "黑色SUV正在行驶" in prompt  # Keep the original evidence auditable.
    assert "patrol_2021] dist=3.5m pos=[-0.4,-3.5] speed=0.00m/s" in prompt
    assert "CARLA GT 已测位置、速度优先于单帧视觉对目标运动的猜测" in prompt
    assert "连续近静止至少 4.90 秒" in prompt
    assert "路面 STOP 文字不等于持续停车指令" in prompt
    assert "有真实冲突则等待，冲突解除则继续路线" in prompt
    assert "未知值不当作静止，也不当作正在接近" in prompt
    assert "导航车道上下文" not in prompt


def test_navigation_context_distinguishes_current_gt_from_prior_candidate_pool():
    prompt = DecisionPromptBuilder().build({}, planning_constraints={
        "decision_mode": "semantic", "navigation_context": {
            "source": "current_carla_gt", "snapshot_time_s": 84.1,
            "ego_lane_key": "24:0:1", "reference_lane_key": "24:0:2",
            "current_side": "left", "planner_source": "previous_pdm_snapshot",
            "planner_snapshot_time_s": 83.9,
            "available_lateral_semantics": ["KEEP_LANE"],
            "unavailable_lateral_request": "FOLLOW_ROUTE",
        },
    })
    assert '"source": "current_carla_gt"' in prompt
    assert '"snapshot_time_s": 84.1' in prompt
    assert '"planner_snapshot_time_s": 83.9' in prompt
    assert '"reference_lane_key": "24:0:2"' in prompt
    assert "不代表当前通行权，缺失不表示道路阻塞" in prompt
    assert "KEEP_LANE 不表示放弃导航目标" in prompt
    assert "FOLLOW_ROUTE 包含进入导航所需车道" in prompt
    assert "{navigation_context_block}" not in prompt


@pytest.mark.parametrize("context", [None, {}, [], {"ignored": "not_a_map_fact"}])
def test_missing_navigation_context_makes_no_blockage_claim(context):
    prompt = DecisionPromptBuilder().build({}, planning_constraints={
        "decision_mode": "semantic", "navigation_context": context,
    })
    assert "导航车道上下文" not in prompt
    assert "not_a_map_fact" not in prompt


def test_navigation_facts_without_provenance_are_explicitly_unknown():
    context = DecisionPromptBuilder._render_navigation_context({
        "ego_lane_key": "24:0:1", "available_lateral_semantics": ["KEEP_LANE"],
    })
    assert '"source": "unspecified"' in context
    assert '"snapshot_time_s": null' in context
    assert '"planner_snapshot_time_s": null' in context
    assert "未标记来源或时间即未知" in context
def test_carla_command_angle_is_not_presented_as_measured_steering_wheel_rotation():
    from src.vla_memory.decision.prompt_builder import DecisionPromptBuilder
    builder = DecisionPromptBuilder.__new__(DecisionPromptBuilder)
    carla = builder._render_ego_block({"source": "carla", "steering_angle": -.2})
    assert "前轮转角近似（控制命令换算，左正）: -0.200 rad" in carla
    assert "方向盘转角" not in carla
    can = builder._render_ego_block({"source": "can_bus", "steering_angle": -.2})
    assert "方向盘转角: -0.200 rad" in can
