"""NAVSIM request guidance and answer consistency, without changing planning.

All geometry is from the same current/history-only scene. Hints describe a
continuous goal domain; they never choose, clamp, snap, or replace a model goal.
"""
import json
import math

import numpy as np

from nine_grid.decision_protocol import LATERAL_ACTIONS
from nine_grid.geometry import (clean_polyline, polyline_lengths, project_polyline,
                                rectangle_corners, sample_polyline)
from nine_grid.goals import _source_lane, goal_reference_context
from nine_grid.quintic_reference import initial_lateral_state


GUIDANCE_PROTOCOL = "navsim-current-frame-guidance-1"


class GuidanceUnavailable(ValueError):
    """A goal-independent current-state precondition prevents this request."""
    def __init__(self, reason_code, detail):
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        super().__init__(self.detail)


class GuidanceMismatch(ValueError):
    """A complete response has a clear identity or decision/goal contradiction."""
    def __init__(self, reason_code, detail):
        self.reason_code = str(reason_code)
        self.detail = str(detail)
        super().__init__(self.detail)


def refine_model_prompt(base_build_prompt):
    """Refine only the current-frame JSON; keep any historical suffix intact.

    Absolute source arc length remains in the internal frozen reference/audit.
    It is not needed to choose a forward *increment* and has been repeatedly
    mistaken for that output by the deployed model. All other current facts,
    including the observed lateral offset, remain in the request.
    """
    prefix = "当前帧\n"
    if not isinstance(base_build_prompt, str) or not base_build_prompt.startswith(prefix):
        raise ValueError("NAVSIM prompt refinement requires the shared current-frame prompt")
    data, end = json.JSONDecoder().raw_decode(base_build_prompt, len(prefix))
    if not isinstance(data, dict) or not isinstance(data.get("规划参考"), dict):
        raise ValueError("NAVSIM current-frame prompt has no planning reference object")
    planning_ref = data["规划参考"]
    source_s = planning_ref.pop("source_s_ego_m", None)
    # Keep original signal/stop_s facts untouched while retaining a readable
    # distance after the nonessential absolute source arc is hidden. Only
    # already associated finite stop lines supply a derived distance.
    distances = []
    rules = planning_ref.get("rules", {})
    lights = rules.get("traffic_lights", []) if isinstance(rules, dict) else []
    if type(source_s) in (int, float) and math.isfinite(source_s) and isinstance(lights, list):
        for light in lights:
            if not isinstance(light, dict):
                continue
            stop_s = light.get("stop_s")
            if type(stop_s) in (int, float) and math.isfinite(stop_s):
                distances.append({**{key: light[key] for key in ("id", "state", "lane_ids") if key in light},
                    "delta_s_m": stop_s-source_s,
                    "scope": "仅适用于原始信号中已关联的导航路线/受控车道，不推广到其他车道"})
    if distances:
        planning_ref["stop_line_distances_from_current"] = distances
    data["目标坐标说明"] = (
        "local_goal.delta_s_m是从当前自车投影向前继续行驶的道路弧长增量，请根据场景独立选择；"
        "不是源参考的绝对s坐标。local_goal.d_m是未来目标车体中心相对规划参考的绝对横向位置，"
        "source_d_ego_m仅表示自车当前横向状态，不是推荐目标；不能仅照抄当前d作为未来目标，"
        "须结合目标处车道中心、整车宽度和合法走廊自行选择；若几何和意图支持，目标d可以与当前d相同。"
        "stop_line_distances_from_current中delta_s_m是从当前投影到已关联停车线的沿参考弧长距离，"
        "正值在前、负值已过线；其原始灯态、受控车道和stop_s仍保留在rules中，未知停车线不推测距离。")
    return prefix + json.dumps(data, ensure_ascii=False, allow_nan=False) + base_build_prompt[end:]


def allowed_lateral_actions(scene, reference):
    """Move an existing, definite absent-neighbour check into the schema.

    Do not infer new restrictions from incomplete neighbour geometry or an
    observed/reference lane identity mismatch, and do not constrain numeric
    continuous goals. Navigation turns remain available without a neighbour.
    """
    actions = list(LATERAL_ACTIONS)
    source, _ = _source_lane(scene, reference)
    if source is not None and scene.ego.lane_id == source.id:
        if source.left is None:
            actions.remove("左变道")
        if source.right is None:
            actions.remove("右变道")
    return actions


def validate_request_state(scene, reference):
    """Use the exact initial-state calculation used by the quintic connector.

    Do not change observed reverse motion, recenter the vehicle, or ask the
    model to solve a defect independent of its chosen target.
    """
    if reference.get("frame_id") != scene.frame_id or reference.get("time") != scene.time:
        raise ValueError("Guidance reference does not belong to the current frozen frame")
    if reference.get("semantic_reference_valid", True) is not True:
        raise GuidanceUnavailable("semantic_reference_invalid",
            reference.get("reference_rejection") or "本帧语义参考不可用")
    line = clean_polyline(reference["points"])
    if len(line) < 2 or not np.isfinite(line).all():
        raise ValueError("Guidance reference must contain finite current-frame geometry")
    s, d, tangent = project_polyline(scene.ego.position, line,
        previous_s=reference["s_ego_m"], heading=scene.ego.heading)
    try:
        state = initial_lateral_state(scene, line, s, d, tangent)
    except ValueError as exc:
        # This is a known target-independent condition in the shared connector,
        # not a blanket catch that hides other malformed geometry or inputs.
        if str(exc) != "当前状态不满足单值前向Frenet连接条件":
            raise
        raise GuidanceUnavailable("current_state_not_forward_frenet",
            "当前实测状态不满足本参考的单值前向Frenet连接条件；"
            "重选目标不能修复该前置条件，保留实测状态并单列参考不可用。") from exc
    if not np.isfinite(state).all():
        raise GuidanceUnavailable("current_state_frenet_boundary_nonfinite",
            "当前实测状态的Frenet连接边界不有限，不能发送可执行目标请求。")
    return dict(protocol=GUIDANCE_PROTOCOL, available=True,
                source_s_m=float(s), source_d_m=float(d),
                initial_boundary=[float(value) for value in state])


def guidance_context(scene, reference):
    """Give body-centre width hints for every legal candidate, including curves.

    Bands are advisory cross-section estimates, not schema constraints or a
    proof that a complete body/path fits the map. In particular, a neighbour's
    future d is not fixed to its offset beside ego at the current instant.
    """
    from shapely.geometry import Polygon
    from shapely.errors import GEOSException
    from nine_grid.local_reference import LocalReferenceManager

    context = goal_reference_context(scene, reference)
    source, lanes = _source_lane(scene, reference)
    manager = LocalReferenceManager({"local_reference": reference["local_reference_parameters"]})
    line = clean_polyline(reference["points"])
    total = float(polyline_lengths(line)[-1])
    lower = float(reference["limits"]["minimum_delta_s_m"])
    upper = min(float(reference["limits"]["maximum_delta_s_m"]),
                total-float(reference["s_ego_m"])-1e-6)
    distances = sorted({float(x) for x in (lower, 10., 20., upper) if lower <= x <= upper})
    rows = []
    for candidate in context["candidate_lanes"]:
        if not candidate["goal_candidate_legal"]:
            continue
        lane = source if source is not None and source.id == candidate["lane_id"] else lanes.get(candidate["lane_id"])
        if lane is None:
            continue
        half_width = (float(lane.width)-float(scene.ego.width))/2
        relation = ("current_reference" if lane.id == reference["source_lane_id"] else
                    "left" if source is not None and lane.id == source.left else
                    "right" if source is not None and lane.id == source.right else "navigation_compatible")
        row = dict(lane_id=lane.id, relation=relation, lane_width_m=float(lane.width),
                   ego_width_m=float(scene.ego.width),
                   lane_local_center_offset_band_m=([-half_width, half_width] if half_width >= 0 else None),
                   fits_width_at_local_section=half_width >= 0, cross_sections=[])
        try:
            path = line if lane.id == reference["source_lane_id"] else manager._continuation(scene, lane, lanes)
            for distance in distances:
                if (candidate.get("forward_geometry_verified") is False or
                        distance >= float(candidate.get("verified_forward_until_m", math.inf))-1e-6):
                    continue
                station = float(reference["s_ego_m"])+distance
                base = sample_polyline(line, station)
                _, _, heading = project_polyline(base, line, previous_s=station, window=5.)
                along, _, tangent = project_polyline(base, path, heading=heading)
                center = sample_polyline(path, along)
                actual_s, center_d, _ = project_polyline(center, line, previous_s=station, window=5.)
                # Do not label a remote/other branch's nearest point as a
                # cross-section at this station. Unknown hints remain absent.
                if abs(actual_s-station) > max(.25, float(lane.width)*.1):
                    continue
                body = Polygon(rectangle_corners(center, tangent, scene.ego.length, scene.ego.width))
                section = dict(delta_s_m=distance, lane_center_d_m=float(center_d),
                    approximate_center_d_band_m=([float(center_d-half_width), float(center_d+half_width)]
                                                 if half_width >= 0 else None),
                    center_body_on_drivable=bool(scene.drivable is not None and scene.drivable.covers(body)))
                row["cross_sections"].append(section)
        except (ValueError, KeyError, GEOSException):
            # A hint must not invent a continuation when source geometry is
            # incomplete. The unchanged core remains the final geometry check.
            row["cross_sections"] = []
        rows.append(row)
    return dict(protocol=GUIDANCE_PROTOCOL, ego_length_m=float(scene.ego.length),
        ego_width_m=float(scene.ego.width), goal_delta_s_range_m=[lower, upper],
        lanes=rows,
        limits="仅为同帧地图的车体中心横截面提示，不保证整个车体、曲线连接、不可跨越边界或动态安全；"
               "弯道/邻道的d随delta_s变化，不得把当前横向位置固定套用到未来。")


def guidance_prompt(scene, reference):
    hints = guidance_context(scene, reference)
    return ("\n本帧目标坐标与决策一致性要求："
        "delta_s_m是你希望从当前投影继续前进的距离，不是source_s_ego_m；不得把源状态数字照抄为目标。"
        "d_m是目标相对冻结参考中心线的绝对横向偏移，不是从当前自车横移的增量。"
        "道路左转/右转已经由导航参考的弯曲表达；沿原车道转弯应使用沿导航转向，不能把它称作左/右变道。"
        "只有实际选择同向合法邻道或朝该邻道的中间点，才声明左/右变道；注意整车体宽度，不只检查中心点。"
        "关注ID不得重复；每项理由应与该ID的相对位置和运动一致。"
        "程序的规划入口仍为连续局部引导点和关注对象，纵向文字本身不构成新的速度目标或停车约束；"
        "引导点不指定速度，也不能用零/负delta_s表示停车。"
        "下面仅提供几何范围信息，不是预选目标；请自行根据同帧图像和真值给出连续坐标。\n"
        + json.dumps(hints, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def validate_answer_consistency(answer, scene, reference):
    """Reject only definite contradictions; keep legitimate intermediate goals."""
    data = getattr(answer, "model_output", {}) or {}
    ids = [row["id"] for row in data.get("attention", [])]
    if len(ids) != len(set(ids)):
        raise GuidanceMismatch("attention_duplicate", "关注对象ID重复，须由模型重新选择不重复的ID。")
    action = data.get("decision", {}).get("lateral", {}).get("action")
    result = dict(protocol=GUIDANCE_PROTOCOL, action=action,
        target_lane_id=answer.target_lane_id, issues=[], longitudinal_policy="goal_and_attention_only")
    if action not in ("左变道", "右变道"):
        return result
    source, _ = _source_lane(scene, reference)
    if source is None or scene.ego.lane_id != source.id:
        result["issues"].append("当前占道与目标参考身份不同，无法仅凭lane标签判定换道文字是否矛盾")
        return result
    intended = source.left if action == "左变道" else source.right
    opposite = source.right if action == "左变道" else source.left
    if intended is None:
        raise GuidanceMismatch("decision_goal_mismatch", action+"没有对应同向邻道；导航弯道应使用沿导航转向。")
    if answer.target_lane_id == opposite:
        raise GuidanceMismatch("decision_goal_mismatch", action+"与目标所在的相反侧车道矛盾。")
    if answer.target_lane_id == source.id:
        direction = 1. if action == "左变道" else -1.
        progress = direction*(float(answer.local_goal["d_m"])-float(reference["d_ego_m"]))
        result["issues"].append("目标仍在当前车道；只有朝声明方向移动的中间点才可能表示本次变道准备")
        result["toward_declared_lane_m"] = progress
        if progress <= 1e-6:
            raise GuidanceMismatch("decision_goal_mismatch",
                action+"的目标仍在当前车道，且没有朝声明侧的有效横向进展；"
                "请更正动作说明或自行选择一致的连续目标，不能只重复相同目标。")
    return result
