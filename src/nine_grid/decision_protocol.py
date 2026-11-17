"""模型四部分回答与旧经验存储之间的显式兼容层；不改变控制或目标语义。"""
from __future__ import annotations

import math

from .contracts import MODEL_OUTPUT_PROTOCOL_VERSION, LEGACY_MODEL_OUTPUT_PROTOCOL_VERSION

LATERAL_ACTIONS = ("保持车道", "左变道", "右变道", "沿导航转向", "车道内避让")
LONGITUDINAL_ACTIONS = ("维持速度", "加速", "减速", "跟车", "停车等待")


def _object(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label}必须且仅包含 " + "、".join(keys))


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}必须为非空字符串")


def normalize_model_output(data):
    """校验完整回答，返回规划字段、旧库说明及实际输出协议版本。

    新回答以数组顺序生成rank，reason不传入纯几何目标校验。
    旧三部分回答仅用于已冻结的旧提示词；两种结构不允许混合。
    身份、原帧坐标和道路合法性仍由models/goals层严格校验。
    """
    if not isinstance(data, dict):
        raise ValueError("回答必须为JSON对象")
    modern = set(data) == {"scene", "decision", "attention", "local_goal"}
    legacy = set(data) == {"attention", "local_goal", "explanation"}
    if not modern and not legacy:
        raise ValueError("回答须包含 scene、decision、attention、local_goal；"
                         "仅兼容完整旧版 attention、local_goal、explanation，不接受混合字段")

    if modern:
        scene = data["scene"]
        _object(scene, ("description", "reason"), "scene")
        for key in ("description", "reason"):
            _text(scene[key], f"scene.{key}")
        decision = data["decision"]
        _object(decision, ("lateral", "longitudinal"), "decision")
        for axis, actions in (("lateral", LATERAL_ACTIONS), ("longitudinal", LONGITUDINAL_ACTIONS)):
            item = decision[axis]
            _object(item, ("action", "reason"), f"decision.{axis}")
            _text(item["action"], f"decision.{axis}.action")
            _text(item["reason"], f"decision.{axis}.reason")
            if item["action"] not in actions:
                raise ValueError(f"decision.{axis}.action不在允许的行动集合中")

    items = data["attention"]
    if not isinstance(items, list) or len(items) > 3:
        raise ValueError("关注列表应包含0至3项")
    attention = []
    for rank, item in enumerate(items, 1):
        keys = ("id", "reason") if modern else ("id", "rank", "reason")
        _object(item, keys, "attention关注项")
        _text(item["id"], "attention.id")
        _text(item["reason"], "attention.reason")
        if legacy and (type(item["rank"]) is not int or item["rank"] != rank):
            raise ValueError("旧版关注排序必须从1连续排列")
        attention.append({"id": item["id"], "rank": rank, "reason": item["reason"]})

    supplied_goal = data["local_goal"]
    keys = ("delta_s_m", "d_m", "reason") if modern else ("delta_s_m", "d_m")
    _object(supplied_goal, keys, "local_goal")
    goal = {key: supplied_goal[key] for key in ("delta_s_m", "d_m")}
    try:
        finite = all(type(value) in (int, float) and math.isfinite(value) for value in goal.values())
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("local_goal坐标必须为有限数值，不接受布尔值")

    if legacy:
        explanation = data["explanation"]
        _object(explanation, ("情境", "依据", "参考答案", "适用条件"), "旧版explanation")
        for value in explanation.values():
            _text(value, "旧版explanation说明")
        return attention, goal, dict(explanation), LEGACY_MODEL_OUTPUT_PROTOCOL_VERSION

    _text(supplied_goal["reason"], "local_goal.reason")
    lateral, longitudinal = decision["lateral"], decision["longitudinal"]
    # 仅供旧经验契约及GoalManager使用的投影。完整四部分另存ModelAnswer.model_output。
    # “适用条件”是程序兼容说明，不冒充模型生成的额外结论。
    explanation = {
        "情境": scene["description"],
        "依据": scene["reason"],
        "参考答案": (
            f"横向：{lateral['action']}；依据：{lateral['reason']}。"
            f"纵向：{longitudinal['action']}；依据：{longitudinal['reason']}。"
            f"局部引导点依据：{supplied_goal['reason']}"
        ),
        "适用条件": "程序兼容说明：仅对应原帧道路权限、对象状态及已知导航；条件变化须重新判断。",
    }
    return attention, goal, explanation, MODEL_OUTPUT_PROTOCOL_VERSION
