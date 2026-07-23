"""决策 Prompt 构建模块
=====================
整合当前感知、记忆、导航、自车状态、历史轨迹，构建决策 VLM 的输入 prompt。
支持 memory_on（包含三层记忆）和 memory_off（仅当前帧感知）两种模式。

P1 重构：所有提示词文本来源迁移到 config/prompts.yaml，路点边界从
config/decision.yaml 读取，做到「一次修改、全链生效」。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.vla_memory.common.prompt_loader import get_prompt_loader
from src.vla_memory.decision.config_access import (
    get_horizon_and_dt,
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
    ) -> str:
        """构建完整的决策 prompt。

        当 short_term_summary / mid_term_memories / long_term_rules_text
        均为空时，自动切换为 memory_off 模式（不拼接记忆段落）。

        Args:
            perception_objects: oracle 感知对象列表（nuScenes GT 投影，dict 形式）；
                非空时渲染为独立段落喂给决策 VLM，并显式标注为 oracle 来源。
            image_layout: 输入图像布局描述串（single_front / surround_mosaic）。
        """
        loader = get_prompt_loader()

        # ------------------------------------------------------------
        # 1. 组装各区块字符串
        # ------------------------------------------------------------
        scene_block = self._render_scene_block(scene_understanding)
        ego_state_block = self._render_ego_block(ego_state)
        nav_block = self._render_nav_block(nav_instruction)
        history_block = self._render_history_block(history_trajectory)
        memory_block = self._render_memory_block(
            loader=loader,
            short_term_summary=short_term_summary,
            mid_term_memories=mid_term_memories,
            long_term_rules_text=long_term_rules_text,
        )
        image_layout_block = self._render_image_layout_block(image_layout)
        perception_objects_block = self._render_perception_objects_block(perception_objects)

        # ------------------------------------------------------------
        # 2. 路点边界 / 行为枚举 / horizon （单一来源：decision.yaml）
        # ------------------------------------------------------------
        min_num, max_num = get_waypoint_bounds()
        horizon_seconds, dt = get_horizon_and_dt()
        behavior_enum_str = "/".join(get_valid_behaviors())

        # ------------------------------------------------------------
        # 3. 渲染主模板
        # ------------------------------------------------------------
        return loader.render(
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
    ) -> str:
        if not history_trajectory:
            return ""
        lines = [
            "## 最近历史轨迹（ego-centric 坐标，x 前向，y 左向，单位米）"
        ]
        recent = history_trajectory[-10:]
        traj_str = ", ".join(
            f"({p.get('x', 0):.1f},{p.get('y', 0):.1f})" for p in recent
        )
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

    def _render_image_layout_block(self, image_layout: str) -> str:
        """渲染输入图像布局说明段（告知决策 VLM 当前图像是单图还是六视角 mosaic）。"""
        if not image_layout:
            return ""
        return f"## 输入图像布局\n{image_layout}\n"

    def _render_perception_objects_block(self, perception_objects) -> str:
        """渲染 oracle 感知对象段（nuScenes GT 标注投影，非检测模型预测）。

        显式标注 oracle 来源；速度/加速度标注可用性与来源；无对象时返回空串。
        """
        if not perception_objects:
            return ""
        lines = [
            "## Oracle 感知对象（nuScenes GT 标注投影，非检测模型预测）",
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
        return "\n".join(lines)
