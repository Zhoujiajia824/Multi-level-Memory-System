"""决策 Prompt 构建模块
=====================
整合当前感知、记忆、导航、自车状态、历史轨迹，构建决策 VLM 的输入 prompt。
支持 memory_on（包含三层记忆）和 memory_off（仅当前帧感知）两种模式。

通用轨迹模板位于 config/prompts.yaml；CARLA 语义模板与路点边界位于
config/decision.yaml，语义模式复用上下文而不拼接数值轨迹输出要求。
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from src.vla_memory.common.prompt_loader import get_prompt_loader
from src.vla_memory.decision.config_access import (
    get_horizon_and_dt,
    get_semantic_prompt_template,
    get_valid_behaviors,
    get_waypoint_bounds,
)


class DecisionPromptBuilder:
    """决策 Prompt 构建器。

    将所有上下文信息组装为结构化的 VLM 输入 prompt。
    支持 memory_on 模式（包含短期/中期/长期记忆）和
    memory_off 模式（仅当前帧感知、自车状态、历史轨迹、导航语义）。

    模板字面量集中在 `config/prompts.yaml`，其中：
      - 整体框架由 `decision.user` 模板渲染；
      - 中期记忆单条由 `memory_integration.mid_term_item` 渲染；
      - 短期/长期记忆段落由 `memory_integration.short_term_block` / `long_term_block` 渲染。
    """

    def build(
        self,
        scene_understanding: Dict[str, Any],
        ego_state: Optional[Dict] = None,
        history_trajectory: Optional[List[Dict]] = None,
        nav_instruction: str = "",
        short_term_summary: str = "",
        mid_term_memories: Optional[List] = None,
        long_term_rules_text: str = "",
        perception_objects: Optional[List[Dict]] = None,
        image_layout: str = "",
        planning_constraints: Optional[Dict] = None,
    ) -> str:
        """构建完整的决策 prompt。

        当 short_term_summary / mid_term_memories / long_term_rules_text
        均为空时，自动切换为 memory_off 模式（不拼接记忆段落）。

        Args:
            perception_objects: oracle 感知对象列表（nuScenes GT 投影，dict 形式）；
                非空时渲染为独立段落喂给决策 VLM，并显式标注为 oracle 来源。
            image_layout: 输入图像布局描述串（single_front / surround_mosaic）。
            planning_constraints: 可选闭环动力学约束；decision_mode=semantic
                时仅输出横纵向意图，其余轨迹物理约束交由下游规划器处理。
        """
        loader = get_prompt_loader()

        # ------------------------------------------------------------
        # 1. 组装各区块字符串
        # ------------------------------------------------------------
        scene_block = self._render_scene_block(scene_understanding)
        ego_state_block = self._render_ego_block(ego_state)
        nav_block = self._render_nav_block(nav_instruction)
        semantic_mode = (planning_constraints or {}).get("decision_mode") == "semantic"
        history_block = self._render_history_block(history_trajectory, include_time=semantic_mode)
        memory_block = self._render_memory_block(
            loader=loader,
            short_term_summary=short_term_summary,
            mid_term_memories=mid_term_memories,
            long_term_rules_text=long_term_rules_text,
        )
        image_layout_block = self._render_image_layout_block(image_layout)
        perception_objects_block = self._render_perception_objects_block(
            perception_objects, source="CARLA" if semantic_mode else "nuScenes")

        # Keep the complete scene, image, navigation and memory context, but do
        # not render any of the legacy numerical trajectory output contract.
        if (planning_constraints or {}).get("decision_mode") == "semantic":
            from src.vla_memory.decision.semantic_output_parser import (
                LATERAL_DECISIONS, LONGITUDINAL_DECISIONS,
            )
            return get_semantic_prompt_template().format_map({
                "scene_block": scene_block,
                "ego_state_block": ego_state_block,
                "nav_block": nav_block,
                "history_block": history_block,
                "motion_context_block": self._render_motion_context(history_trajectory, ego_state),
                "memory_block": memory_block,
                "image_layout_block": image_layout_block,
                "perception_objects_block": perception_objects_block,
                "lateral_enum": " / ".join(LATERAL_DECISIONS),
                "longitudinal_enum": " / ".join(LONGITUDINAL_DECISIONS),
            })

        # ------------------------------------------------------------
        # 2. 路点边界 / 行为枚举 / horizon （单一来源：decision.yaml）
        # ------------------------------------------------------------
        min_num, max_num = get_waypoint_bounds()
        horizon_seconds, dt = get_horizon_and_dt()
        if planning_constraints is not None:
            # Include both capture-time origin and the full execution horizon
            # without asking for 31 points when the parser accepts at most 30.
            horizon_seconds = max(horizon_seconds,
                                  float(planning_constraints["required_horizon_s"]))
            dt = horizon_seconds / max(1, max_num - 1)
        behavior_enum_str = "/".join(get_valid_behaviors())

        # ------------------------------------------------------------
        # 3. 渲染主模板
        # ------------------------------------------------------------
        prompt = loader.render(
            "decision.user",
            scene_block=scene_block,
            ego_state_block=ego_state_block,
            nav_block=nav_block,
            history_block=history_block,
            memory_block=memory_block,
            waypoint_min_num=min_num,
            waypoint_max_num=max_num,
            horizon_seconds=horizon_seconds,
            dt=dt,
            behavior_enum_str=behavior_enum_str,
            image_layout_block=image_layout_block,
            perception_objects_block=perception_objects_block,
        )
        if planning_constraints is not None:
            prompt += self._render_planning_constraints(
                planning_constraints, ego_state or {}, horizon_seconds, max_num)
        return prompt

    @staticmethod
    def _render_planning_constraints(constraints, ego_state, horizon, point_count):
        return (
            "\n## CARLA 闭环轨迹物理约束（执行前会校验）\n"
            "- 轨迹必须是当前自车坐标系的米制位置：x 前向，y 左向；禁止使用地图全局坐标。\n"
            f"- 输出恰好 {point_count} 个点，首点 t=0,x=0,y=0，末点 t={horizon:.6g} 秒。"
            f"第 i 个点的时间为 i*{horizon:.6g}/({point_count}-1)，i 从 0 开始；t 必须严格递增。\n"
            f"- 当前实测速度是 {float(ego_state.get('speed', 0)):.3f} m/s，"
            "轨迹应从该速度连续演化，不能直接跳到目标速度。\n"
            f"- 目标速度不得超过 {constraints['max_speed_mps']:.3g} m/s，"
            f"纵向加速度不得超过 {constraints['max_acceleration_mps2']:.3g} m/s²，"
            f"制动减速度不得超过 {constraints['max_deceleration_mps2']:.3g} m/s²，"
            f"横向加速度不得超过 {constraints['max_lateral_acceleration_mps2']:.3g} m/s²。\n"
            "- 相邻点距离/时间差必须与速度一致。匀速 6 m/s 行驶 3 秒约走 18 米，"
            "不能输出 80 米轨迹；起步需计入加速，减速需计入当前速度和制动距离。"
            "若提供 optional_v，它必须与位置和时间一致。\n"
            "- route_center 是前方路线的空间参考，不代表必须在本轮时间范围内到达；"
            "只沿其中物理可达的部分规划，不把远处锚点强塞进 3 秒轨迹。\n"
            "- 控制器仅支持前进，禁止倒车、轨迹折返和瞬间横移。"
            "停车目标速度可设 0，但移动车辆仍必须给出连续减速至停止的路径。\n"
        )

    # ================================================================
    # 区块渲染辅助
    # ================================================================

    def _render_scene_block(self, scene_understanding: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("## 当前场景理解")
        lines.append(f"- 场景描述: {scene_understanding.get('scene_description', '未知')}")
        lines.append(f"- 场景类型: {scene_understanding.get('scene_id', '未知')}")
        lines.append(f"- 天气: {scene_understanding.get('weather_id', '未知')}")
        lines.append(f"- 交通密度: {scene_understanding.get('traffic_density', '未知')}")

        # ---- P4 结构化字段：lanes / vehicles / pedestrians / traffic_lights / intersections ----
        lanes = scene_understanding.get("lanes") or []
        if lanes:
            lines.append("- 车道线:")
            for ln in lanes:
                if not isinstance(ln, dict):
                    continue
                parts = [
                    f"side={ln.get('side', 'unknown')}",
                    f"type={ln.get('type', 'unknown')}",
                ]
                if ln.get("color"):
                    parts.append(f"color={ln['color']}")
                if ln.get("direction"):
                    parts.append(f"dir={ln['direction']}")
                lines.append("  - " + ", ".join(parts))

        vehicles = scene_understanding.get("vehicles") or []
        if vehicles:
            lines.append("- 周围车辆:")
            for v in vehicles:
                if not isinstance(v, dict):
                    continue
                bits = [f"pos={v.get('relative_position', 'unknown')}"]
                if v.get("distance_m") is not None:
                    bits.append(f"dist={v['distance_m']:.1f}m")
                if v.get("type"):
                    bits.append(f"type={v['type']}")
                if v.get("motion"):
                    bits.append(f"motion={v['motion']}")
                lines.append("  - " + ", ".join(bits))

        pedestrians = scene_understanding.get("pedestrians") or []
        if pedestrians:
            lines.append("- 周围行人:")
            for p in pedestrians:
                if not isinstance(p, dict):
                    continue
                bits = [f"pos={p.get('relative_position', 'unknown')}"]
                if p.get("distance_m") is not None:
                    bits.append(f"dist={p['distance_m']:.1f}m")
                if p.get("intent"):
                    bits.append(f"intent={p['intent']}")
                lines.append("  - " + ", ".join(bits))

        traffic_lights = scene_understanding.get("traffic_lights") or []
        if traffic_lights:
            lines.append("- 信号灯:")
            for tl in traffic_lights:
                if not isinstance(tl, dict):
                    continue
                bits = [
                    f"state={tl.get('state', 'unknown')}",
                    f"pos={tl.get('relative_position', 'unknown')}",
                ]
                if tl.get("controls_ego_lane") is not None:
                    bits.append(f"controls_ego_lane={tl['controls_ego_lane']}")
                lines.append("  - " + ", ".join(bits))

        intersections = scene_understanding.get("intersections") or {}
        if isinstance(intersections, dict) and intersections.get("present"):
            bits = ["present=true"]
            if intersections.get("type"):
                bits.append(f"type={intersections['type']}")
            if intersections.get("distance_m") is not None:
                bits.append(f"distance={intersections['distance_m']:.1f}m")
            if intersections.get("has_stop_sign") is not None:
                bits.append(f"stop_sign={intersections['has_stop_sign']}")
            lines.append("- 路口: " + ", ".join(bits))

        # ---- 旧字段：仅当未提供新字段时回退展示，避免 prompt 冗余 ----
        if not (lanes or vehicles or pedestrians or traffic_lights):
            lines.append(f"- 车道描述: {scene_understanding.get('lane_description', '未知')}")
            objects = scene_understanding.get("surrounding_objects", []) or []
            if objects:
                lines.append("- 周围物体:")
                for obj in objects:
                    if isinstance(obj, dict):
                        obj_type = obj.get("type", "未知")
                        obj_pos = obj.get("relative_position", "未知")
                        obj_desc = obj.get("description", "")
                    else:
                        obj_type = obj_pos = "未知"
                        obj_desc = ""
                    lines.append(f"  - [{obj_type}] {obj_pos}: {obj_desc}")

        risks = scene_understanding.get("risk_factors", []) or []
        if risks:
            lines.append(f"- 风险因素: {', '.join(str(r) for r in risks)}")
        lines.append("")
        return "\n".join(lines)

    def _render_ego_block(self, ego_state: Optional[Dict]) -> str:
        if not ego_state:
            return ""
        lines = [
            "## 自车状态",
            f"- 位置: x={ego_state.get('x', 0):.2f}, y={ego_state.get('y', 0):.2f}",
            f"- 航向角: {ego_state.get('yaw', 0):.3f} rad",
            f"- 速度: {ego_state.get('speed', 0):.2f} m/s",
            f"- 加速度: {ego_state.get('acceleration', 0):.2f} m/s²",
        ]
        # P3 起，若 CAN bus 提供了额外字段则一并展示
        for key, label, fmt in (
            ("yaw_rate", "偏航角速率", "{:.3f} rad/s"),
            ("steering_angle", "方向盘转角", "{:.3f} rad"),
            ("throttle", "油门", "{:.2f}"),
            ("brake", "刹车", "{:.2f}"),
        ):
            v = ego_state.get(key)
            if v is not None:
                lines.append(f"- {label}: {fmt.format(v)}")
        lines.append("")
        return "\n".join(lines)

    def _render_nav_block(self, nav_instruction: str) -> str:
        if not nav_instruction:
            return ""
        return f"## 导航指令: {nav_instruction}\n"

    def _render_history_block(
        self,
        history_trajectory: Optional[List[Dict]],
        include_time: bool = False,
    ) -> str:
        if not history_trajectory:
            return ""
        lines = [
            "## 最近历史轨迹（ego-centric 坐标，x 前向，y 左向，单位米）"
        ]
        recent = history_trajectory[-10:]
        traj_str = ", ".join(
            (f"(t={p.get('t', '未知')},x={p.get('x', 0):.1f},y={p.get('y', 0):.1f})"
             if include_time else f"({p.get('x', 0):.1f},{p.get('y', 0):.1f})") for p in recent)
        lines.append(f"  {traj_str}")
        lines.append("")
        return "\n".join(lines)

    def _render_memory_block(
        self,
        loader,
        short_term_summary: str,
        mid_term_memories: Optional[List],
        long_term_rules_text: str,
    ) -> str:
        """组装三层记忆段落；任一为空都不出现。memory_off 时整体为空字符串。"""
        sections: List[str] = []

        if short_term_summary:
            sections.append(
                loader.render(
                    "memory_integration.short_term_block",
                    short_term_summary=short_term_summary,
                ).rstrip()
            )

        if mid_term_memories:
            mid_lines: List[str] = ["## 相似历史经验（中期记忆检索）"]
            for i, mem in enumerate(mid_term_memories[:3]):
                record = mem.get("record")
                if record is None:
                    continue
                if isinstance(record, dict):
                    scene_id = record.get("scene_id", "未知")
                    weather_id = record.get("weather_id", "未知")
                    decision_reason = record.get("decision_reason", "未知")
                    behavior = record.get("behavior", "未知")
                else:
                    scene_id = getattr(record, "scene_id", "未知")
                    weather_id = getattr(record, "weather_id", "未知")
                    decision_reason = getattr(record, "decision_reason", "未知")
                    behavior = getattr(record, "behavior", "未知")

                mid_lines.append(
                    loader.render(
                        "memory_integration.mid_term_item",
                        index=i + 1,
                        score=f"{mem.get('final_score', 0):.3f}",
                        scene_id=scene_id,
                        weather_id=weather_id,
                        decision_reason=decision_reason,
                        behavior=behavior,
                    ).rstrip()
                )
                planning_memory = self._render_planning_memory_context(record)
                if planning_memory:
                    mid_lines.append(planning_memory)
            sections.append("\n".join(mid_lines))

        if long_term_rules_text:
            sections.append(
                loader.render(
                    "memory_integration.long_term_block",
                    long_term_rules_text=long_term_rules_text,
                ).rstrip()
            )

        if not sections:
            return ""
        return "\n\n".join(sections) + "\n"

    @staticmethod
    def _render_planning_memory_context(record) -> str:
        """Preserve requested semantics and the selected plan as separate facts.

        A memory is written after plan validation but before the subsequent
        control cycle. Even ``executed_behavior`` names the selected command;
        it does not, by itself, establish successful execution or causality.
        Older records without these optional fields keep their old rendering.
        """
        read = record.get if isinstance(record, dict) else (
            lambda key, default=None: getattr(record, key, default)
        )
        semantic = read("semantic_decision")
        context = read("planning_context")
        if not isinstance(semantic, dict) and not isinstance(context, dict):
            return ""

        lines = []
        if isinstance(semantic, dict):
            requested = {
                key: semantic[key] for key in (
                    "lateral_decision", "longitudinal_decision", "behavior_reason",
                    "risk_level", "safety_notes",
                ) if key in semantic
            }
            lines.append(
                "- 历史模型建议（请求意图，不代表已执行或成功）: "
                + json.dumps(requested, ensure_ascii=False)
            )
        if isinstance(context, dict):
            selected = {
                key: context[key] for key in (
                    "backend", "semantic_influence", "selected_id", "executed_behavior",
                    "selected_lateral", "selected_longitudinal", "record_stage",
                ) if key in context
            }
            lines.append(
                "- 规划器最终选择记录: " + json.dumps(selected, ensure_ascii=False)
            )
            if context.get("semantic_influence") is False:
                lines.append("- 此历史帧关闭语义影响；模型建议未参与该帧候选评分。")
            elif context.get("semantic_influence") is True:
                lines.append("- 此历史帧开启语义影响；模型建议参与该帧候选评分。")
        lines.append("- 上述选择记录不能单独证明动作已完成、驾驶成功或模型建议被采纳。")
        return "\n".join(lines)

    @staticmethod
    def _render_motion_context(history, ego_state) -> str:
        """Summarize measured kinematics, not learned/retrieved memory or right of way."""
        header = "## 实测运动窗口（所有记忆模式共同的传感器状态）\n"
        unknown = header + "连续静止时长未知：有效连续测量不足。\n"
        try:
            rows = [(float(p['t']), float(p['x']), float(p['y'])) for p in history or []]
            speed = float((ego_state or {})['speed'])
            if (len(rows) < 2 or not math.isfinite(speed) or speed < 0
                    or not all(math.isfinite(v) for row in rows for v in row)
                    or any(a[0] >= b[0] for a, b in zip(rows, rows[1:]))
                    or any(t > 0 for t, _, _ in rows) or rows[-1][0] < -0.5
                    or math.hypot(rows[-1][1], rows[-1][2]) > 0.3):
                return unknown
        except (KeyError, TypeError, ValueError, OverflowError):
            return unknown
        duration = 0.0
        distance = sum(math.hypot(b[1]-a[1], b[2]-a[2]) for a, b in zip(rows, rows[1:]))
        if speed <= 0.2:
            for a, b in reversed(list(zip(rows, rows[1:]))):
                dt = b[0] - a[0]
                if dt > 0.5 or math.hypot(b[1]-a[1], b[2]-a[2]) / dt > 0.2:
                    break
                duration += dt
        return (header + f"- 最近测量覆盖 {rows[-1][0]-rows[0][0]:.2f} 秒，累计移动 {distance:.2f} 米。\n"
                f"- 连续近静止至少 {duration:.2f} 秒（速度阈值 0.2m/s，仅统计连续测量；窗口外未知）。\n"
                "- 以上仅描述实测运动状态，通行权由当前交通情况判断。\n")

    def _render_image_layout_block(self, image_layout: str) -> str:
        """渲染输入图像布局说明段（告知决策 VLM 当前图像是单图还是六视角 mosaic）。"""
        if not image_layout:
            return ""
        return f"## 输入图像布局\n{image_layout}\n"

    def _render_perception_objects_block(self, perception_objects, source="nuScenes") -> str:
        """渲染 oracle 感知对象段（nuScenes GT 标注投影，非检测模型预测）。

        显式标注 oracle 来源；速度/加速度标注可用性与来源；无对象时返回空串。
        """
        if not perception_objects:
            return ""
        route_lines = []
        if source == "CARLA":
            anchors = [o for o in perception_objects if isinstance(o, dict)
                       and o.get("semantic_label") == "route_center"]
            perception_objects = [o for o in perception_objects if isinstance(o, dict)
                                  and o.get("semantic_label") != "route_center"]
            if anchors:
                route_lines = ["## 导航路线空间参考（仅表示路线位置）",
                               json.dumps([o.get("position_ego") for o in anchors[:6]], ensure_ascii=False), ""]
        lines = [
            f"## Oracle 感知对象（{source} GT，非检测模型预测）",
            f"共 {len(perception_objects)} 个目标（按到 ego 距离升序），坐标 ego-centric [x 前向, y 左向]：",
        ]
        for i, o in enumerate(perception_objects[:12], 1):  # 限 12 个避免 prompt 过长
            if not isinstance(o, dict):
                continue
            vel = o.get("velocity")
            vstr = (
                f"speed={o.get('speed'):.2f}m/s vel={[round(v, 2) for v in vel]}"
                if (o.get("velocity_available") and vel is not None)
                else "unavailable(无历史)"
            )
            amag = o.get("acceleration_mag")
            astr = (
                f"{amag:.2f}m/s^2"
                if (o.get("acceleration_available") and amag is not None)
                else "unavailable"
            )
            pos = o.get("position_ego") or []
            pos_str = f"[{pos[0]:.1f},{pos[1]:.1f}]" if len(pos) >= 2 else "?"
            lines.append(
                f"  {i}. [{o.get('category', '?')}/{o.get('semantic_label', '?')}] "
                f"dist={o.get('distance_to_ego', 0):.1f}m pos={pos_str} "
                f"{vstr} acc={astr} cams={o.get('visible_cameras', [])} "
                f"kinematics={o.get('kinematics_source', '?')}"
            )
        if len(perception_objects) > 12:
            lines.append(f"  ...（其余 {len(perception_objects) - 12} 个对象略）")
        lines.append("")
        return "\n".join(route_lines + lines)
