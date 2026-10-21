"""CARLA 闭环主驱动
==================
同步 tick + 独立数值重规划（默认 MPC，保留 PDM 对照；10Hz tick / 0.2s 重规划）：

选择性模式每 0.5s 检查是否需要认知，只在事件、规划困难或经验提示时咨询。
建议经对象与条件核对后限时生效；模型不可用时沿用当前数值规划。
always 模式保留以下周期认知链路，用于对照：

    每个 replan 周期（默认 3s sim）：
      [1 捕获] tick -> 6 相机 mosaic + ego 状态 + GT 感知 + nav
      [2 认知] loop_runner.step(kf) -> 语义决策 -> 数值轨迹（sim 冻结）
      [3 控制] 10Hz × N tick：MPC优化转向 / PDM LQR + 速度曲线PID回控 CARLA
               其中每周期 5 次 raw 捕获（≈1.67Hz；mosaic+ego+感知，不跑 VLM）
               push 进短期记忆，使队列在控制阶段也保持最新；完整 VLM 仍 3s 一次。
               数值后端每 0.2s 根据最新 world 状态重规划，保持最近合法语义。
              model_inference=false 时跳过认知和六路相机，按同一数值/控制节奏直接运行。
      [4 记录] 逐 tick 指标采集（碰撞/违规/路线/舒适度）+ 卡死检测
      [视频]  每个 CARLA 推进 tick 取一帧写视频（跳过 VLM 冻结期），
              视频时长 = 仿真实际驾驶时间；中断 flush 保存。

复用 ``OnlineDrivingLoop`` 并注入执行前验证，CARLA 数据经 ``KeyframeBuilder`` 组装成
kf dict 喂入。指标由 :class:`MetricsCollector` 采集，run 结束写 Markdown + JSON 报告。
"""
from __future__ import annotations

import queue
import math
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.config import Config
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.schemas.memory import ShortTermMemoryItem
from carla_bridge.env.carla_client import CarlaClient
from carla_bridge.env.scenario_manager import ScenarioManager
from carla_bridge.sensors.camera_manager import CameraManager
from carla_bridge.sensors.perception_provider import PerceptionProvider
from carla_bridge.state.ego_state_provider import EgoStateProvider
from carla_bridge.state.history_buffer import HistoryBuffer
from carla_bridge.state import coords
from carla_bridge.control.trajectory_tracker import TrajectoryTracker
from carla_bridge.control.trajectory_validation import TrajectoryValidator
from carla_bridge.memory_adapter.keyframe_builder import KeyframeBuilder
from carla_bridge.inference_mode import model_inference_enabled
from carla_bridge.metrics.collector import MetricsCollector
from carla_bridge.metrics.run_reporter import write_report

logger = get_logger("carla_closed_loop")

if TYPE_CHECKING:
    from carla_bridge.memory_adapter.loop_runner import LoopRunner


class ClosedLoop:
    """CARLA + 多层次记忆系统 闭环驱动器。"""

    def __init__(self, config: Config, scenario_yaml: str, mode: str = "memory_on"):
        self.config = config
        self.mode = mode
        self.run_id = config.get("run_id", "")
        self.timestamp_origin_us = int(config.get("timestamp_origin_us", 0))
        self.run_status = "setup_failed"
        self._closed = False
        self._capture_failures = 0
        self._capture_failure_total = 0
        self._termination_reason = ""
        self._cleanup_errors = []
        self._final_summary = None
        ccfg = config.get_nested("carla", default={}) or {}
        self.ccfg = ccfg
        self.planning_cfg = dict(ccfg.get("planning") or {})
        self.planner_backend = self.planning_cfg.get("backend", "vlm_trajectory")
        if self.planner_backend not in ("mpc", "pdm_closed", "vlm_trajectory"):
            raise ValueError("Unknown CARLA planning.backend")
        self.semantic_influence = self.planning_cfg.get("semantic_influence", True)
        if type(self.semantic_influence) is not bool:
            raise ValueError("planning.semantic_influence must be a boolean")
        if self.planner_backend == "vlm_trajectory":
            self.semantic_influence = False
        self.model_inference = model_inference_enabled(self.planning_cfg, mode)
        self.semantic_policy = self.planning_cfg.get("semantic_policy", "always")
        if self.semantic_policy not in ("always", "selective", "shadow"):
            raise ValueError("semantic_policy must be always, selective or shadow")
        self.planning_memory_mode = self.planning_cfg.get("planning_memory", "off")
        if self.planning_memory_mode not in ("off", "gate", "full"):
            raise ValueError("planning_memory must be off, gate or full")
        self._selective_execution = (self.planner_backend in ("pdm_closed", "mpc") and self.model_inference
                                    and self.semantic_influence and self.semantic_policy != "always")
        self.experience_memory = None
        self.execution_planner_version = None
        self._experience_signature = None
        self._last_plan_diagnostics = {}
        self._next_query_check_at = 0.0
        self._query_check_interval = .5
        self._wall_started = time.perf_counter()
        self.advisory_stats = {"model_queries": 0, "query_checks_skipped": 0,
            "model_failures": 0, "inference_wall_time_s": 0.0,
            "semantic_changed_plans": 0, "semantic_rejected_plans": 0,
            "shadow_changed_plans": 0, "semantic_stop_proxy_s": 0.0,
            "recovery_expanded_plans": 0, "recovery_selected_plans": 0,
            "recovery_no_solution_plans": 0}
        self.pdm_planner = None
        self._active_semantics = None
        self._semantic_frame_id = None
        self._next_pdm_replan_at = 0.0
        self._pdm_plan_count = 0
        self._control_audit_path = None
        self._last_control_command = None
        self.replan_interval_s = float(ccfg.get("replan_interval_s", 1.0))
        self.fixed_delta = float(ccfg.get("fixed_delta_seconds", 0.05))
        self.max_duration_s = float(ccfg.get("max_duration_s", 300.0))
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.fixed_delta, self.replan_interval_s, self.max_duration_s)):
            raise ValueError("CARLA tick, replan interval and time limit must be finite and positive")
        if self.replan_interval_s < self.fixed_delta:
            raise ValueError("replan_interval_s cannot be shorter than one physics tick")
        self._cycle_ticks = max(1, round(self.replan_interval_s / self.fixed_delta))
        self._execution_horizon = self._cycle_ticks * self.fixed_delta
        self._pdm_interval = float(self.planning_cfg.get("replan_interval_s", 0.2))
        if self.planner_backend in ("pdm_closed", "mpc") and (
                not math.isfinite(self._pdm_interval) or self._pdm_interval < self.fixed_delta
                or self._pdm_interval > self._execution_horizon):
            raise ValueError("Numerical replan interval must be between physics tick and semantic interval")
        if self.planner_backend == "mpc":
            from .planning.mpc_planner import effective_mpc_config
            mpc = effective_mpc_config(self.planning_cfg)
            if self._pdm_interval > .3 + 1e-9:
                raise ValueError("MPC replan_interval_s must be <= 0.3 for fresh optimized controls")
            for name, interval in (("optimization dt", mpc.dt), ("replan interval", self._pdm_interval)):
                ratio = interval / self.fixed_delta
                if ratio < 1.-1e-9 or abs(ratio-round(ratio)) > 1e-8:
                    raise ValueError("MPC " + name + " must be an integer multiple of the physics tick")
            if mpc.dt * mpc.horizon_steps + 1e-8 < self._execution_horizon:
                raise ValueError("MPC optimization horizon must cover the semantic execution interval")
        if (self.planner_backend == "pdm_closed"
                and self.planning_cfg.get("lateral_control", "pure_pursuit") == "native_lqr"):
            from .planning.native_tracking import MAX_PROFILE_LIFETIME_S, SAMPLE_INTERVAL_S
            if abs(self.fixed_delta - SAMPLE_INTERVAL_S) > 1e-8:
                raise ValueError("Native LQR tracking requires fixed_delta_seconds=0.1")
            if self._pdm_interval > MAX_PROFILE_LIFETIME_S + 1e-9:
                raise ValueError("Native LQR tracking requires PDM replan_interval_s <= 0.3")
        self.trajectory_validator = TrajectoryValidator(ccfg.get("trajectory_validation"))
        self._trajectory_rejections = 0
        self._consecutive_trajectory_rejections = 0
        self.route_completion = 0.0
        self.route_completed = False
        self.cameras_cfg = ccfg.get("cameras", []) or []
        self.out_cfg = ccfg.get("output", {}) or {}
        self.controller_cfg = ccfg.get("controller", {}) or {}
        self.scenario_yaml = scenario_yaml

        # 组件
        self.carla_client: Optional[CarlaClient] = None
        self.scenario: Optional[ScenarioManager] = None
        self.cameras: Optional[CameraManager] = None
        self.perception: Optional[PerceptionProvider] = None
        self.ego_provider: Optional[EgoStateProvider] = None
        self.history: Optional[HistoryBuffer] = None
        self.tracker: Optional[TrajectoryTracker] = None
        self.kf_builder: Optional[KeyframeBuilder] = None
        self.loop: Optional[LoopRunner] = None
        self.metrics: Optional[MetricsCollector] = None
        self.recorder = None  # 可选 VideoRecorder
        self.live_view = None  # 与录像共用同帧合成结果；不使用模型相机
        self._model_call_audit = None
        self._replay_summary = None
        self._replay_error = None
        self.trajectory_renderer = None
        self.event_scheduler = None  # 可选 EventScheduler（长尾事件）

        # 碰撞
        self._collision_sensor = None
        self._coll_q: "queue.Queue" = queue.Queue()
        self._collision_count = 0
        self._collision_raw = 0              # 原始事件数（未去重，仅 debug/报告）
        self._collision_actors: set = set()  # 碰撞过的 unique actor id
        self._last_coll_at: dict = {}        # actor_id -> 上次计入的 sim 时刻（2s 冷却去重）

        # 卡死检测
        stuck_cfg = ccfg.get("stuck_detection", {}) or {}
        self._stuck_enabled = bool(stuck_cfg.get("enabled", True))
        self._stuck_speed_thresh = float(stuck_cfg.get("speed_thresh_mps", 0.3))
        self._stuck_warn_s = float(stuck_cfg.get("warn_after_s", 10.0))
        self._stuck_term_s = float(stuck_cfg.get("terminate_after_s", 20.0))
        self._stuck_since: Optional[float] = None
        self._stuck_progress = 0.0
        self._planned_stop = True
        self._early_terminated = False

        self._elapsed = 0.0  # 仿真累计时间（秒）
        # raw 捕获状态（控制阶段短期记忆更新）：场景字段沿用上次 VLM 结果
        self._last_scene_result: dict = {}
        self._last_nav: str = "straight"
        self._raw_counter = 0

        # 路由中心线注入配置（米制回正锚点，见 route_injection）
        ri_cfg = ccfg.get("route_injection", {}) or {}
        self._route_inject_enabled = bool(ri_cfg.get("enabled", True))
        self._route_inject_n = int(ri_cfg.get("num_points", 5))
        self._route_inject_step = float(ri_cfg.get("step_m", 8.0))

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def setup(self) -> None:
        if self.planner_backend == "pdm_closed":
            from carla_bridge.planning.pdm_planner import validate_dependencies
            validate_dependencies()
        elif self.planner_backend == "mpc":
            from carla_bridge.planning.mpc_planner import validate_dependencies
            validate_dependencies()
        # 1. CARLA 连接
        self.carla_client = CarlaClient(
            host=self.ccfg.get("host", "127.0.0.1"),
            port=int(self.ccfg.get("port", 2000)),
            timeout_s=float(self.ccfg.get("timeout_s", 30.0)),
            synchronous=bool(self.ccfg.get("synchronous", True)),
            fixed_delta_seconds=self.fixed_delta,
        )
        self.carla_client.connect()

        # 2. 场景
        self.scenario = ScenarioManager.from_yaml(
            self.carla_client, self.ccfg, self.scenario_yaml
        )
        self.scenario.setup()
        world = self.scenario.world
        ego = self.scenario.ego_vehicle
        if (self.ccfg.get("replay") or {}).get("enabled", True):
            from carla_bridge.model_call_audit import ModelCallAudit
            self._model_call_audit = ModelCallAudit(self.config.get("output_dir") or "outputs")
        if (self.scenario.cfg.get("experiment") or {}).get("suite") == "paper_v1":
            from carla_bridge.env.paper_events import PaperEventScheduler
            # Preserve the partial scheduler for cleanup if actor setup fails.
            self.event_scheduler = PaperEventScheduler(world, ego, self.scenario.route,
                self.scenario.cfg.get("events", []), elapsed=lambda: self._elapsed,
                audit_path=Path(self.config.get("output_dir")) / "events.json")
            self.event_scheduler.setup()
        # 交通流刚 spawn：tick 几帧让 Traffic Manager 与所有 NPC 完成初始化后再挂相机，
        # 否则 6 相机首次 BeginPlay 与大量未初始化 actor 竞态，Town05 上触发引擎崩溃
        for _ in range(5):
            self.carla_client.tick()

        # 3. 相机
        if self.model_inference:
            mosaic_cfg = self.config.get_nested("perception", "mosaic", default={}) or {}
            self.cameras = CameraManager(
                world, ego, self.cameras_cfg,
                image_dir=self.out_cfg.get("image_dir", "outputs/carla_images"),
                mosaic_dir=self.out_cfg.get("mosaic_dir", "outputs/mosaic"),
                mosaic_cell_w=int(mosaic_cfg.get("cell_width", 480)),
                mosaic_cell_h=int(mosaic_cfg.get("cell_height", 270)),
                label_subimages=bool(mosaic_cfg.get("label_subimages", True)),
                capture_timeout_s=float(self.ccfg.get("capture_timeout_s", 2.0)),
            )
            self.cameras.install()
            self.cameras.warmup(n_frames=2)

        # 4. 感知 / 状态 / 控制 / 记忆
        if self.model_inference:
            oracle_cfg = self.config.get_nested("perception", "oracle", default={}) or {}
            self.perception = PerceptionProvider(
                world, max_distance_m=float(oracle_cfg.get("max_distance_m", 50.0))
            )
        self.ego_provider = EgoStateProvider(
            max_steer_rad=float(self.controller_cfg.get("max_steer_rad", 0.6))
        )
        if self.model_inference:
            self.history = HistoryBuffer(
                history_seconds=float(self.config.get("history_seconds", 5.0))
            )
        self.tracker = TrajectoryTracker(self.controller_cfg, control_dt_s=self.fixed_delta)
        if self.planner_backend in ("pdm_closed", "mpc"):
            planner_cfg = dict(self.planning_cfg)
            planner_cfg["horizon_s"] = max(float(planner_cfg.get("horizon_s", 4.0)), self._execution_horizon)
            if self.planner_backend == "mpc":
                from carla_bridge.planning.mpc_planner import CarlaMPCPlanner
                self.pdm_planner = CarlaMPCPlanner(world, ego, self.scenario.route, planner_cfg)
            else:
                from carla_bridge.planning.pdm_planner import CarlaPDMPlanner
                self.pdm_planner = CarlaPDMPlanner(world, ego, self.scenario.route, planner_cfg)
            visual_cfg = ((self.ccfg.get("visualization") or {}).get("trajectories") or {})
            self.pdm_planner.configure_trajectory_visualization(visual_cfg)
            if visual_cfg.get("enabled", False):
                from carla_bridge.visualization.trajectory_renderer import CarlaTrajectoryRenderer
                self.trajectory_renderer = CarlaTrajectoryRenderer(world, visual_cfg)
            if self.planner_backend == "mpc":
                self._bind_mpc_controller(ego)
            else:
                self._bind_pdm_controller(ego)
            from carla_bridge.planning.experience_memory import ExecutionExperienceMemory, execution_planner_version
            if self.planner_backend == "mpc":
                planner = self.pdm_planner
                vehicle_model = planner.execution_vehicle_signature()
                self.execution_planner_version = execution_planner_version(
                    planner_cfg, self.controller_cfg, vehicle_model=vehicle_model)
            else:
                self.execution_planner_version = execution_planner_version(planner_cfg, self.controller_cfg)
            if self.planning_memory_mode != "off" and self.mode != "memory_off":
                access = self.config.get("memory_access", {}) or {}
                self.experience_memory = ExecutionExperienceMemory(self.config.get("memory_db_dir"),
                    memory_mode="read_write" if access.get("write", True) else "read_only",
                    read_enabled=bool(access.get("retrieve", True)), write_enabled=bool(access.get("write", True)),
                    run_id=self.run_id, run_dir=self.config.get("output_dir"),
                    planner_version=self.execution_planner_version)
                self.pdm_planner.experience_memory = self.experience_memory
        if self.model_inference:
            self.kf_builder = KeyframeBuilder(scenario_name=self.scenario.scenario_name(), run_id=self.run_id)

        # 5. 碰撞传感器 + 指标采集
        self._setup_collision_sensor(world, ego)
        self.metrics = MetricsCollector(
            world, self.scenario.route,
            speed_limit_mps=float(self.ccfg.get("speed_limit_mps", 13.9)),
            dt_s=self.fixed_delta,
        )
        self._control_audit_path = Path(self.config.get("output_dir") or "outputs") / "control.jsonl"
        self._control_audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_control_context(world, ego)

        # 6. 记忆 loop
        if self.model_inference:
            from carla_bridge.memory_adapter.loop_runner import LoopRunner
            out_path = self._make_output_path()
            self.loop = LoopRunner(
                self.config, mode=self.mode, output_jsonl_path=out_path, resume=False,
                decision_validator=self._validate_decision,
                decision_processor=self._plan_semantic_decision if self.planner_backend in ("pdm_closed", "mpc") else None,
            )
            self.loop.setup()
        else:
            logger.info("[%s] 模型调用已关闭：跳过场景VLM、决策VLM、DINOv2、文本/视觉记忆和六路认知相机；数值经验启用=%s；保留事件、指标及可选视频",
                        self.planner_backend, self.experience_memory is not None)

        # 7. Traffic Manager 起步：空转几帧
        for _ in range(10):
            self.carla_client.tick()

        # 8. Independent display camera, shared by live view and optional MP4.
        video_cfg = self.ccfg.get("video", {}) or {}
        live_cfg = dict((self.ccfg.get("visualization") or {}).get("live") or {})
        if live_cfg.get("enabled") is None:
            live_cfg["enabled"] = bool(((self.ccfg.get("visualization") or {}).get("trajectories") or {}).get("enabled"))
        if live_cfg.get("enabled"):
            from carla_bridge.visualization.live_view import LiveTrajectoryView
            try:
                self.live_view = LiveTrajectoryView(live_cfg)
                url = self.live_view.start()
                logger.info("实时轨迹观察器: %s （远程服务器可使用SSH端口转发）", url)
            except Exception as exc:
                if self.live_view is not None:
                    self.live_view.close()
                self.live_view = None
                logger.warning("实时观察器启动失败，实验继续: %s", exc)
        if video_cfg.get("enabled", False) or self.live_view is not None:
            from carla_bridge.video.recorder import VideoRecorder
            record_video = bool(video_cfg.get("enabled", False))
            try:
                vout = Path(self.config.get("output_dir") or "outputs") / "carla_videos"
                vout.mkdir(parents=True, exist_ok=True)
                vpath = vout / f"drive_{self.scenario.scenario_name()}_{self.mode}.mp4"
                self.recorder = VideoRecorder(
                    world, ego, str(vpath),
                    fps=float(video_cfg.get("fps", 1.0/self.fixed_delta)),
                    width=int(video_cfg.get("width", 1280)),
                    height=int(video_cfg.get("height", 720)),
                    view=video_cfg.get("view", "chase"),
                    frame_timeout_s=float(video_cfg.get("frame_timeout_s", 0.0)),
                    hud=bool(video_cfg.get("hud", True)),
                    max_pending_frames=int(video_cfg.get("max_pending_frames", 32)),
                    flush_timeout_s=float(video_cfg.get("flush_timeout_s", .25)),
                    record_video=record_video, live_view=self.live_view,
                )
                self.recorder.start()
            except Exception as exc:
                if record_video:
                    raise
                # An optional preview failure must not stop driving. Clean up
                # even when initialization only got as far as opening a file.
                for resource, operation in ((self.recorder, "flush"), (self.live_view, "close")):
                    if resource is not None:
                        try:
                            getattr(resource, operation)()
                        except Exception as cleanup_exc:
                            logger.warning("实时显示清理失败: %s", cleanup_exc)
                self.recorder = self.live_view = None
                logger.warning("实时显示相机启动失败，实验继续: %s", exc)

        # 9. 长尾事件调度器（若场景声明了 events）
        events_cfg = (self.scenario.cfg or {}).get("events") or []
        if events_cfg and self.event_scheduler is None:
            from carla_bridge.env.event_scheduler import EventScheduler
            ctrl_cfg = dict(self.controller_cfg)
            ctrl_cfg["control_dt_s"] = self.fixed_delta  # scripted NPC PID 的真实 dt
            self.event_scheduler = EventScheduler(
                world, ego, events_cfg, ctrl_cfg,
                seed=int(self.scenario.cfg.get("seed", 42)),
            )
            logger.info("长尾事件调度器: %d 个事件", len(events_cfg))

        logger.info(
            "ClosedLoop setup 完成 (scenario=%s, mode=%s, replan=%.1fs, dt=%.3fs, video=%s, live=%s, model_inference=%s)",
            self.scenario.scenario_name(), self.mode, self.replan_interval_s, self.fixed_delta,
            bool(video_cfg.get("enabled", False)), self.live_view is not None, self.model_inference,
        )

    def run(self) -> None:
        assert self.carla_client and self.scenario
        if self.model_inference:
            assert self.loop
        else:
            assert self.pdm_planner
        self.run_status = "running"
        duration = min(self.max_duration_s, self.scenario.duration_s())
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("scenario duration_s must be finite and positive")
        if not self.scenario.route:
            raise ValueError("a non-empty planned route is required for route completion")
        while not self._check_run_end():
            if getattr(self, "_selective_execution", False):
                self._advisory_tick()
            elif self.model_inference:
                self._replan_cycle()
            else:
                self._pdm_only_tick()
            if self._early_terminated:
                break
        self.run_status = self._termination_reason
        self._apply_brake()

    def _check_run_end(self) -> bool:
        """Success is measured route completion; elapsed time is only a guard."""
        if self._early_terminated:
            return True
        from carla_bridge.env.paper_events import PaperEventScheduler
        if isinstance(self.event_scheduler, PaperEventScheduler) and self.event_scheduler.has_failed():
            self._termination_reason = "scenario_invalid"
            self._early_terminated = True
            self._apply_brake()
            return True
        route = self.scenario.route
        if route is not None:
            self.route_completion = route.progress_fraction()
            self.route_completed = route.is_complete()
        if self.route_completed:
            valid_events = (not isinstance(self.event_scheduler, PaperEventScheduler)
                            or self.event_scheduler.summary()["valid"])
            self._termination_reason = "route_completed" if valid_events else "scenario_invalid"
        elif self._elapsed + self.fixed_delta > min(self.max_duration_s, self.scenario.duration_s()) + 1e-8:
            self._termination_reason = "timeout"
        else:
            return False
        self._apply_brake()
        return True

    def _apply_brake(self) -> None:
        if self.scenario is not None and self.scenario.ego_vehicle is not None:
            self._apply_control(self.scenario.ego_vehicle,
                                carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0),
                                source="direct_brake")

    def _apply_control(self, ego, control, *, source="tracker") -> None:
        native_tracking = None
        mpc_tracking = None
        minimum_risk_audit = None
        diagnostics = getattr(self, "_last_plan_diagnostics", {}) or {}
        hold = getattr(self, "_mpc_minimum_risk_hold", None)
        if source == "tracker" and getattr(self, "planner_backend", None) == "mpc" and hold:
            # No trajectory is authorized during an explicitly reported
            # infeasibility. Never execute an older steering/throttle profile,
            # including on the physics tick between two fresh observations.
            control = carla.VehicleControl(throttle=0., steer=0., brake=1.)
            source = "mpc_minimum_risk_hold"
            if self._elapsed >= hold["episode_deadline_s"]-1e-9:
                self._early_terminated = True
                self._termination_reason = "planner_recovery_timeout"
        if source == "mpc_minimum_risk_hold" and hold:
            fresh = hold["issued_at_s"] <= self._elapsed < hold["execution_expires_at_s"]-1e-9
            if not fresh:
                self._next_pdm_replan_at = 0.
            minimum_risk_audit = {"contract_version": hold["contract_version"],
                "observation_lease_fresh": fresh, "issued_at_s": hold["issued_at_s"],
                "execution_expires_at_s": hold["execution_expires_at_s"],
                "episode_deadline_s": hold["episode_deadline_s"],
                "trajectory_validated": False, "optimization_feasible": False}
        if (source == "tracker" and getattr(self, "planner_backend", None) == "pdm_closed"
                and diagnostics.get("native_tracking", {}).get("active") is True):
            from .planning.native_tracking import override_native_tracking
            velocity = ego.get_velocity()
            control, native_tracking = override_native_tracking(control, diagnostics,
                now_s=self._elapsed, speed_mps=math.hypot(velocity.x, velocity.y),
                steering_model=self.pdm_planner.steering_model)
        if source == "tracker" and getattr(self, "planner_backend", None) == "mpc":
            from .planning.mpc_tracking import override_mpc_tracking
            velocity = ego.get_velocity()
            control, mpc_tracking = override_mpc_tracking(control, diagnostics,
                now_s=self._elapsed, speed_mps=math.hypot(velocity.x, velocity.y),
                steering_model=self.pdm_planner.steering_model)
        # Apply before bookkeeping: diagnostics must never delay an emergency.
        ego.apply_control(control)
        if getattr(self, "planner_backend", None) == "mpc":
            # The optimizer's first jerk bound uses the previously dispatched
            # desired input, not noisy CARLA acceleration telemetry or a plan
            # that was selected but never sent to the vehicle.
            feedback = getattr(self.pdm_planner, "record_control_execution", None)
            if callable(feedback):
                feedback(diagnostics, mpc_tracking, now_s=self._elapsed, source=source,
                         tracker_diagnostics=getattr(self.tracker, "last_control_diagnostics", None))
        tracking = getattr(self.tracker, "last_control_diagnostics", None)
        self._last_control_command = {
            "sim_time_s": self._elapsed, "source": source,
            "planner_sequence": self._pdm_plan_count,
            "semantic_frame_id": self._semantic_frame_id,
            **self._control_audit_values(control),
            "tracking": deepcopy(tracking) if source == "tracker" and isinstance(tracking, dict) else None,
            "native_tracking": native_tracking,
            "mpc_tracking": mpc_tracking,
            "minimum_risk_hold": minimum_risk_audit,
        }

    @staticmethod
    def _control_audit_values(control) -> dict:
        return {
            "throttle": float(control.throttle), "brake": float(control.brake),
            "steer": float(control.steer), "gear": int(control.gear),
            "reverse": bool(control.reverse), "hand_brake": bool(control.hand_brake),
        }

    @staticmethod
    def _audit_fields(source, fields, errors, prefix) -> dict:
        """Record unavailable telemetry explicitly; never stringify a client/mock object."""
        values = {}
        for name, kind in fields.items():
            try:
                value = getattr(source, name)
                if kind is float:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("expected a finite number")
                    value = float(value)
                elif type(value) is not kind:
                    raise ValueError("expected " + kind.__name__)
                values[name] = value
            except Exception as exc:
                values[name] = None
                errors[prefix + name] = type(exc).__name__ + ": " + str(exc)
        return values

    def _world_clock_audit(self) -> dict:
        errors = {}
        values = {"frame": None, "elapsed_seconds": None, "delta_seconds": None}
        try:
            snapshot = self.carla_client.get_world().get_snapshot()
            values.update(self._audit_fields(snapshot, {"frame": int}, errors, "snapshot."))
            values.update(self._audit_fields(snapshot.timestamp, {
                "elapsed_seconds": float, "delta_seconds": float,
            }, errors, "snapshot.timestamp."))
        except Exception as exc:
            errors["snapshot"] = type(exc).__name__ + ": " + str(exc)
        if errors:
            values["errors"] = errors
        return values

    def _bind_pdm_controller(self, ego) -> None:
        """Use one vehicle geometry for PDM rollouts and CARLA trajectory tracking."""
        self.tracker.bind_vehicle_model(
            self.pdm_planner.vehicle_parameters.wheel_base,
            steering_model=self.pdm_planner.steering_model,
            rear_axle_actor_xyz_m=self.pdm_planner.rear_axle_actor_xyz_m,
        )
        self.ego_provider.max_steer_rad = self.pdm_planner.max_steer
        self.ego_provider.steering_model = self.tracker.steering_model
        logger.info("PDM控制参数已绑定：轴距 %.4fm，前轮距 %.4fm，等效中心转角上限 %.4frad，后轴参考点跟踪",
                    self.tracker.pp.L, self.tracker.steering_model.front_track_m, self.tracker.max_steer_rad)

    def _bind_mpc_controller(self, ego) -> None:
        """Execute optimized MPC steering with its measured vehicle geometry."""
        planner = self.pdm_planner
        self.tracker.bind_vehicle_model(planner.vehicle_parameters.wheel_base,
            steering_model=planner.steering_model,
            rear_axle_actor_xyz_m=planner.rear_axle_actor_xyz_m)
        self.tracker.vehicle_model_source = "carla_mpc_vehicle"
        self.ego_provider.max_steer_rad = planner.max_steer
        self.ego_provider.steering_model = planner.steering_model
        logger.info("MPC控制参数已绑定：轴距 %.4fm；执行优化转向和速度曲线PID；不使用PDM/LQR",
                    self.tracker.pp.L)

    def _write_control_context(self, world, ego) -> None:
        """Snapshot physics metadata once; these observations never change physics settings."""
        errors = {}
        context = {"schema_version": 1, "world_settings": None, "vehicle_physics": None}
        try:
            context["world_settings"] = self._audit_fields(world.get_settings(), {
                "synchronous_mode": bool, "fixed_delta_seconds": float,
                "substepping": bool, "max_substeps": int, "max_substep_delta_time": float,
            }, errors, "world_settings.")
        except Exception as exc:
            errors["world_settings"] = type(exc).__name__ + ": " + str(exc)
        try:
            physics = ego.get_physics_control()
            values = self._audit_fields(physics, {"mass": float}, errors, "vehicle_physics.")
            values["mass_kg"] = values.pop("mass")
            values["wheels"] = []
            context["vehicle_physics"] = values
            for index, wheel in enumerate(physics.wheels):
                entry = self._audit_fields(wheel, {"max_brake_torque": float, "radius": float},
                                           errors, "vehicle_physics.wheels[%d]." % index)
                values["wheels"].append({"index": index, "max_brake_torque_nm": entry["max_brake_torque"],
                                        "radius_cm": entry["radius"]})
        except Exception as exc:
            errors["vehicle_physics"] = type(exc).__name__ + ": " + str(exc)
        context["errors"] = errors
        tracker = getattr(self, "tracker", None)
        if isinstance(tracker, TrajectoryTracker):
            context["tracking_vehicle_model"] = {
                "source": tracker.vehicle_model_source,
                "wheelbase_m": tracker.pp.L,
                "max_steer_rad": tracker.max_steer_rad,
                "physical_inner_max_steer_rad": tracker.steering_model.physical_max_steer_rad,
                "front_track_m": tracker.steering_model.front_track_m,
                "steering_curve_kmh_scale": tracker.steering_model.curve,
                "mapping": tracker.steering_model.mapping,
                "tracking_reference": ("mpc_rear_axle" if self.planner_backend == "mpc" else "pdm_rear_axle")
                    if tracker.rear_axle_actor_xyz_m is not None else "carla_actor_origin",
                "rear_axle_actor_xyz_m": tracker.rear_axle_actor_xyz_m,
                "input_latency_modeled": False,
            }
        try:
            self._control_audit_path.with_name("control_context.json").write_text(
                json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except Exception:
            self._apply_brake()
            raise

    def _validate_decision(self, decision: dict, kf: dict) -> dict:
        decision = decision or {}
        if decision.get("fallback_used"):
            return {"accepted": False, "reasons": ["rule_fallback_not_executable"], "details": {}}
        diagnostics = decision.get("planner_diagnostics") or {}
        if (diagnostics.get("minimum_risk_hold") or {}).get("active"):
            valid = self._valid_mpc_minimum_risk_hold(decision)
            return {"accepted": False,
                    "reasons": ["mpc_minimum_risk_hold" if valid else "invalid_mpc_minimum_risk_hold_contract"],
                    "details": {"minimum_risk_contract_valid": valid,
                                "execution_mode": "full_brake_reobserve" if valid else None,
                                "nominal_horizon_validated": False,
                                "trajectory_validated": False}}
        if self.planner_backend in ("pdm_closed", "mpc") and diagnostics.get("emergency_brake") is True:
            return {"accepted": False, "reasons": ["mpc_emergency_brake" if self.planner_backend == "mpc" else "pdm_emergency_brake"],
                    "details": {"failure_reason": diagnostics.get("failure_reason")}}
        required_horizon = self._execution_horizon
        degraded = diagnostics.get("degraded_braking") or {}
        if degraded.get("active"):
            # Only this explicit, freshly optimized MPC STOP can use a shorter
            # verified horizon. Ordinary short, stale or unconstrained output
            # remains a rejected trajectory.
            from carla_bridge.planning.mpc_planner import (MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S,
                                                          MPC_BRAKING_RECOVERY_HORIZON_S,
                                                          MPC_BRAKING_RECOVERY_POLICY_VERSION)
            now = getattr(self, "_elapsed", float("nan"))
            tracking = diagnostics.get("mpc_tracking") or {}
            chosen = next((c for c in diagnostics.get("candidates", [])
                           if c.get("id") == diagnostics.get("selected_id")), {})
            numeric_keys = ("verified_horizon_s", "issued_at_s", "episode_started_s", "episode_deadline_s",
                            "execution_expires_at_s", "verified_constraint_tolerance",
                            "verified_constraint_max_violation")
            finite = all(type(degraded.get(k)) in (int, float) and math.isfinite(degraded[k])
                         for k in numeric_keys)
            valid = (finite and self.planner_backend == "mpc" and math.isfinite(now)
                     and degraded.get("optimization_feasible") is True
                     and degraded.get("acceleration_policy_passed") is True
                     and degraded.get("acceleration_policy_version") == MPC_BRAKING_RECOVERY_POLICY_VERSION
                     and degraded.get("constraints_relaxed") is False
                     and degraded.get("fresh_snapshot") is True
                     and decision.get("selected_longitudinal") == "STOP"
                     and chosen.get("optimization_feasible") is True
                     and chosen.get("degraded_braking") is True
                     and chosen.get("safety", {}).get("finite") is True
                     and tracking.get("active") is True)
            if valid:
                valid = (abs(degraded["issued_at_s"]-now) <= 1e-6
                         and degraded["episode_started_s"] <= now < degraded["episode_deadline_s"]
                         and degraded["episode_deadline_s"]-degraded["episode_started_s"]
                             <= MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S+1e-8
                         and now+self.fixed_delta <= degraded["execution_expires_at_s"]+1e-8
                         and tracking.get("issued_at_s") == degraded["issued_at_s"]
                         and tracking.get("expires_at_s") == degraded["execution_expires_at_s"]
                         and self._pdm_interval+self.fixed_delta <= degraded["verified_horizon_s"]
                             <= MPC_BRAKING_RECOVERY_HORIZON_S+1e-8
                         and chosen.get("verified_horizon_s") == degraded["verified_horizon_s"]
                         and diagnostics.get("horizon_s") == degraded["verified_horizon_s"]
                         and abs((chosen.get("solver", {}).get("dt_s", 0)
                                  *chosen.get("solver", {}).get("horizon_steps", 0))
                                 -degraded["verified_horizon_s"]) <= 1e-8
                         and 0 < degraded["verified_constraint_tolerance"] <= .05
                         and 0 <= degraded["verified_constraint_max_violation"]
                             <= degraded["verified_constraint_tolerance"])
            if valid:
                bounds = degraded.get("acceleration_upper_bounds_mps2")
                commands = tracking.get("accelerations_mps2")
                solver_bounds = chosen.get("solver", {}).get("acceleration_upper_bounds_mps2")
                valid = (isinstance(bounds, list) and isinstance(commands, list)
                         and isinstance(solver_bounds, list) and len(bounds) == len(commands) > 0
                         and len(bounds) == len(solver_bounds)
                         and all(type(v) in (int, float) and math.isfinite(v)
                                 for values in (bounds, commands, solver_bounds) for v in values))
                if valid:
                    tolerance = degraded["verified_constraint_tolerance"]
                    valid = all(abs(bound-solver_bound) <= 1e-8 and command <= bound+tolerance
                                for bound, command, solver_bound in zip(bounds, commands, solver_bounds))
            if not valid:
                return {"accepted": False, "reasons": ["invalid_mpc_braking_recovery_contract"], "details": {}}
            required_horizon = degraded["verified_horizon_s"]
        result = self.trajectory_validator.validate(
            decision.get("trajectory"), decision.get("target_speed"),
            (kf.get("ego_state") or {}).get("speed", 0.0), required_horizon,
            use_speed_profile=self.planner_backend in ("pdm_closed", "mpc"),
        ).to_dict()
        if degraded.get("active"):
            result["details"]["execution_mode"] = "verified_short_horizon_braking"
            result["details"]["nominal_horizon_validated"] = False
        return result

    def _valid_mpc_minimum_risk_hold(self, decision):
        """Recognize a bounded control state, never a collision-free trajectory."""
        from .planning.mpc_planner import (MPC_MINIMUM_RISK_HOLD_POLICY_VERSION,
            MPC_MINIMUM_RISK_HOLD_LIFETIME_S, MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S,
            MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S, MPC_BRAKING_RECOVERY_HORIZON_S)
        d = decision.get("planner_diagnostics") or {}
        h, recovery = d.get("minimum_risk_hold") or {}, d.get("degraded_braking") or {}
        now = getattr(self, "_elapsed", float("nan"))
        numeric = ("issued_at_s", "execution_expires_at_s", "episode_started_s",
                   "episode_deadline_s", "max_continuous_s", "nominal_horizon_s", "verified_horizon_s")
        if not (self.planner_backend == "mpc" and math.isfinite(now)
                and h.get("active") is True
                and h.get("contract_version") == MPC_MINIMUM_RISK_HOLD_POLICY_VERSION
                and h.get("reason") == "all_finite_full_and_short_horizon_solutions_infeasible"
                and h.get("actuation") == "full_brake_reobserve"
                and h.get("optimization_feasible") is False and h.get("trajectory_accepted") is False
                and h.get("constraints_relaxed") is False and h.get("fresh_snapshot") is True
                and d.get("emergency_brake") is True and d.get("failure_reason") == "no_feasible_mpc_solution"
                and decision.get("trajectory") == [] and decision.get("target_speed") == 0.
                and recovery.get("active") is False
                and all(type(h.get(k)) in (int, float) and math.isfinite(h[k]) for k in numeric)):
            return False
        if not (abs(h["issued_at_s"]-now) <= 1e-6
                and h["episode_started_s"] <= now < h["execution_expires_at_s"] <= h["episode_deadline_s"]
                and now+self.fixed_delta <= h["execution_expires_at_s"]+1e-8
                and h["execution_expires_at_s"]-now <= MPC_MINIMUM_RISK_HOLD_LIFETIME_S+1e-8
                and h["episode_deadline_s"]-h["episode_started_s"] <= MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S+1e-8
                and h["max_continuous_s"] == MPC_BRAKING_RECOVERY_MAX_CONTINUOUS_S
                and h["verified_horizon_s"] == 0.
                and h["nominal_horizon_s"] == d.get("horizon_s")
                and all(h[k] == recovery.get(k) for k in
                        ("issued_at_s", "episode_started_s", "episode_deadline_s"))):
            return False
        candidates, attempts = d.get("candidates") or [], recovery.get("attempts") or []
        levels = recovery.get("available_horizon_levels_s") or []
        if not (isinstance(candidates, list) and isinstance(attempts, list) and isinstance(levels, list)
                and candidates and attempts and levels
                and all(isinstance(c, dict) for c in candidates+attempts)
                and all(type(n) in (int, float) and math.isfinite(n) for n in levels)
                and all(isinstance(a.get("reference_id"), str)
                        and type(a.get("verified_horizon_s")) in (int, float)
                        and math.isfinite(a["verified_horizon_s"]) for a in attempts)
                and h.get("full_horizon_candidate_count") == len(candidates)
                and h.get("short_horizon_attempt_count") == len(attempts)
                and h.get("attempted_horizon_levels_s") == recovery.get("attempted_horizon_levels_s") == levels
                and set(levels) == {a.get("verified_horizon_s") for a in attempts}):
            return False
        provenance = d.get("provenance") or {}
        config = provenance.get("numerical_config") if isinstance(provenance, dict) else None
        if not isinstance(config, dict):
            return False
        dt, samples = config.get("dt"), config.get("horizon_steps")
        if not (type(dt) in (int, float) and math.isfinite(dt) and 0 < dt <= .5
                and type(samples) is int and samples >= 2):
            return False
        step_levels = []
        for requested in MPC_BRAKING_RECOVERY_HORIZON_LEVELS_S:
            steps = int(math.floor(min(requested, MPC_BRAKING_RECOVERY_HORIZON_S)/dt+1e-9))
            if 2 <= steps < samples and steps not in step_levels and steps*dt < h["nominal_horizon_s"]-1e-9:
                step_levels.append(steps)
        if levels != [steps*dt for steps in step_levels]:
            return False
        def finite_infeasible(item):
            solver = item.get("solver") or {}
            if not isinstance(solver, dict):
                return False
            v, tolerance = solver.get("max_constraint_violation"), solver.get("constraint_tolerance")
            return (item.get("optimization_feasible") is False
                and all(type(n) in (int, float) and math.isfinite(n) for n in (v, tolerance))
                and 0 < tolerance <= .05 and v > tolerance)
        from collections import Counter
        stops = [c.get("id") for c in candidates if c.get("longitudinal") == "STOP"]
        if not stops or any(not isinstance(name, str) for name in stops):
            return False
        expected = Counter((name+"@verified_braking", level) for name in stops for level in levels)
        actual = Counter((a.get("reference_id"), a.get("verified_horizon_s")) for a in attempts)
        return (expected == actual
                and all(isinstance(c.get("safety"), dict)
                        and c["safety"].get("finite") is True and finite_infeasible(c) for c in candidates)
                and all(a.get("finite_result") is True and finite_infeasible(a) for a in attempts))

    def _execute_mpc_minimum_risk_hold(self, decision, validation, tf, yaw):
        if (validation.get("details") or {}).get("minimum_risk_contract_valid") is not True:
            return False
        self._mpc_minimum_risk_hold = deepcopy(decision["planner_diagnostics"]["minimum_risk_hold"])
        self.tracker.set_trajectory([], tf.location.x, tf.location.y, yaw, 0.)
        self._planned_stop = True
        self._apply_control(self.scenario.ego_vehicle,
            carla.VehicleControl(throttle=0., steer=0., brake=1.), source="mpc_minimum_risk_hold")
        memory = getattr(self, "experience_memory", None)
        if memory is not None:
            memory.finish("planner_minimum_risk_hold")
            self._experience_signature = None
        logger.warning("MPC 暂无约束可行轨迹，最小风险制动并重新观测（非可行轨迹，恢复预算剩余 %.1fs）",
                       self._mpc_minimum_risk_hold["episode_deadline_s"]-self._elapsed)
        return True

    def _is_planned_stop(self, decision: dict) -> bool:
        if self.planner_backend not in ("pdm_closed", "mpc"):
            return float(decision.get("target_speed", 0.0)) <= self._stuck_speed_thresh
        # Terminal speed alone may be zero even when this control interval must
        # accelerate towards a distant stop line. Inspect the near-term profile.
        points = decision.get("trajectory") or []
        near = [float(p["optional_v"]) for p in points
                if float(p["t"]) <= self._pdm_interval + self.fixed_delta + 1e-8]
        return not near or max(near) <= self._stuck_speed_thresh

    def _plan_semantic_decision(self, decision: dict, kf: dict) -> dict:
        """Plan and audit before the memory layer can admit this frame."""
        self._active_semantics = None
        if decision.get("parser_status") != "success" or decision.get("decision_mode") != "semantic":
            raise ValueError("A valid semantic decision is required for numerical planning")
        semantic = {key: deepcopy(decision[key]) for key in (
            "lateral_decision", "longitudinal_decision", "behavior_reason", "risk_level", "safety_notes")}
        if "intervention" in decision:
            semantic["intervention"] = deepcopy(decision["intervention"])
        self._semantic_frame_id = kf.get("sample_token")
        try:
            plan = self._pdm_plan(semantic, update=True)
        except Exception as exc:
            # Geometry can fail before a plan dictionary exists. Audit it at
            # both update rates, while preserving the original exception and
            # the memory layer's rejected-decision handling.
            self._apply_brake()
            self._audit_pdm_error(exc, semantic, "semantic_error")
            raise
        result = self._merge_plan(decision, semantic, plan)
        validation = self._validate_decision(result, kf)
        self._audit_pdm_plan(result, validation, "semantic_update")
        if validation["accepted"] or (validation.get("details") or {}).get("minimum_risk_contract_valid"):
            self._active_semantics = semantic
            self._next_pdm_replan_at = self._elapsed + self._pdm_interval
        return result

    def _merge_plan(self, decision: dict, semantic: Optional[dict], plan: dict) -> dict:
        if not isinstance(plan, dict) or not isinstance(plan.get("planner_diagnostics"), dict):
            raise ValueError("Numerical planner returned no auditable plan")
        diagnostics = plan["planner_diagnostics"]
        target = float(plan["target_speed"])
        lateral = plan.get("selected_lateral", diagnostics.get("selected_lateral", "FOLLOW_ROUTE"))
        longitudinal = plan.get("selected_longitudinal", diagnostics.get("selected_longitudinal", "CRUISE"))
        if target <= self._stuck_speed_thresh:
            selected_behavior = "STOP"
        elif lateral in ("CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"):
            selected_behavior = lateral
        else:
            selected_behavior = {"FOLLOW": "FOLLOW", "DECELERATE": "SLOW_DOWN",
                                 "YIELD": "YIELD", "STOP": "STOP"}.get(longitudinal, "KEEP_LANE")
        context = {"backend": self.planner_backend, "semantic_influence": self.semantic_influence,
                   "selected_id": diagnostics.get("selected_id"),
                   "executed_behavior": selected_behavior,
                   "selected_lateral": lateral, "selected_longitudinal": longitudinal,
                   "record_stage": "selected_before_execution"}
        if getattr(self, "semantic_policy", "always") != "always":
            context.update(baseline_selected_id=diagnostics.get("baseline_selected_id"),
                           semantic_policy=self.semantic_policy, governor=deepcopy(diagnostics.get("governor")))
        return {**decision, **plan, "trajectory_source": self.planner_backend,
                "semantic_decision": deepcopy(semantic), "planning_context": context,
                "behavior": selected_behavior,
                "behavior_reason": "%s selected candidate %s; semantic influence=%s. Execution outcome is not yet known."
                                   % (self.planner_backend, context["selected_id"], self.semantic_influence)}

    def _audit_pdm_plan(self, decision: dict, validation: dict, source: str) -> None:
        self._pdm_plan_count += 1
        path = Path(self.config.get("output_dir") or "outputs") / "planner.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        audit = {"sim_time_s": self._elapsed, "sequence": self._pdm_plan_count,
                 "source": source, "semantic_frame_id": self._semantic_frame_id,
                 "semantic_decision": decision.get("semantic_decision"),
                 "semantic_influence": self.semantic_influence,
                 "model_inference": self.model_inference,
                 "planner_diagnostics": decision.get("planner_diagnostics"),
                 "trajectory": decision.get("trajectory"), "validation": validation,
                 "trajectory_visualization": {"enabled": False}}
        diagnostics = decision.get("planner_diagnostics") or {}
        self._last_plan_diagnostics = diagnostics
        renderer = getattr(self, "trajectory_renderer", None)
        if renderer is not None:
            try:
                display_audit = renderer.draw(
                    getattr(self.pdm_planner, "last_visualization_snapshot", None),
                    decision, self.scenario.ego_vehicle.get_transform(),
                    accepted=validation.get("accepted") is True)
                if not isinstance(display_audit, dict):
                    raise TypeError("Trajectory renderer audit must be a mapping")
                # Optional rendering cannot make the planner journal fail because
                # an extension returned a nonfinite/nonserializable display value.
                audit["trajectory_visualization"] = json.loads(
                    json.dumps(display_audit, ensure_ascii=False, allow_nan=False))
            except Exception as exc:
                audit["trajectory_visualization"] = {"enabled": True,
                    "error": {"type": type(exc).__name__, "message": str(exc)}}
                logger.warning("轨迹可视化不可用，继续执行原规划: %s", exc)
        self._last_trajectory_visualization_audit = deepcopy(audit["trajectory_visualization"])
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
        stats = getattr(self, "advisory_stats", None)
        if stats is not None:
            baseline_id = diagnostics.get("original_baseline_selected_id", diagnostics.get("baseline_selected_id"))
            changed = diagnostics.get("selected_id") != baseline_id
            degraded_braking = bool((diagnostics.get("degraded_braking") or {}).get("active"))
            minimum_risk_hold = bool((validation.get("details") or {}).get("minimum_risk_contract_valid"))
            if minimum_risk_hold:
                stats["minimum_risk_hold_plans"] = stats.get("minimum_risk_hold_plans", 0) + 1
            if degraded_braking:
                stats["degraded_braking_plans"] = stats.get("degraded_braking_plans", 0) + 1
            if changed and self.semantic_influence and not degraded_braking and not minimum_risk_hold:
                stats["semantic_changed_plans"] += 1
            if diagnostics.get("shadow_selected_id") != diagnostics.get("baseline_selected_id") and diagnostics.get("shadow_only"):
                stats["shadow_changed_plans"] += 1
            recovery = diagnostics.get("recovery") or {}
            for key, occurred in (
                ("recovery_expanded_plans", recovery.get("expanded")),
                ("recovery_selected_plans", recovery.get("selected")),
                ("recovery_no_solution_plans", recovery.get("expanded") and not recovery.get("positive_recovery_candidates"))):
                if occurred:
                    stats[key] = stats.get(key, 0) + 1
            governor = diagnostics.get("governor", {})
            if governor and not governor.get("authorized") and governor.get("last_submission", {}).get("reason") != "no_submission":
                stats["semantic_rejected_plans"] += 1
        if validation.get("accepted"):
            self._begin_execution_experience(decision)

    def _special_situations(self):
        # Scenario generation parameters/labels are NOT a perception channel.
        # Detect visible obstacles through current GT; never leak future events.
        return ()

    def _pdm_plan(self, semantic, update=False):
        if getattr(self, "semantic_policy", "always") != "always":
            return self.pdm_planner.plan(semantic, semantic_enabled=self.semantic_influence,
                semantic_update=update, sim_time_s=self._elapsed,
                special_situations=self._special_situations())
        return self.pdm_planner.plan(semantic, semantic_enabled=self.semantic_influence,
                                     sim_time_s=self._elapsed)

    def _experience_telemetry(self):
        ego = self.scenario.ego_vehicle
        velocity, acceleration = ego.get_velocity(), ego.get_acceleration()
        return {"sim_time_s": self._elapsed,
                "route_progress_m": float(self.scenario.route.distance_completed_m()),
                "ego_speed_mps": math.hypot(velocity.x, velocity.y),
                "acceleration_mps2": math.hypot(acceleration.x, acceleration.y),
                "jerk_mps3": getattr(self, "_experience_jerk", None),
                "collision_count": self._collision_count,
                "violation_count": (getattr(self.metrics, "red_light_violations", 0)
                                    + getattr(self.metrics, "wrong_way_violations", 0))}

    def _begin_execution_experience(self, decision):
        memory = getattr(self, "experience_memory", None)
        if memory is None:
            return
        d = decision.get("planner_diagnostics") or {}
        if (d.get("degraded_braking") or {}).get("active"):
            memory.finish("planner_degraded_braking")
            self._experience_signature = None
            return
        context = d.get("planner_context")
        candidate = next((c for c in d.get("candidates", []) if c["id"] == d.get("selected_id")), None)
        if not context or not candidate:
            return
        action = {"lateral_decision": candidate["lateral"], "longitudinal_decision": candidate["longitudinal"]}
        parameters = {"target_velocity_mps": candidate["desired_speed_mps"],
                      "headway_time": candidate["idm_headway_s"],
                      "min_gap_to_lead_agent": candidate.get("idm_min_gap_m", 1.5),
                      "accel_max": candidate["idm_accel_max_mps2"], "decel_max": candidate["idm_decel_max_mps2"]}
        signature = (action["lateral_decision"], action["longitudinal_decision"],
                     tuple(sorted((k, round(v, 1)) for k, v in parameters.items())),
                     candidate.get("recovery_recipe"), candidate.get("recovery_transition_m"),
                     candidate.get("lane_offset_m"))
        if self._experience_signature is not None and signature != self._experience_signature:
            memory.finish("policy_changed")
        self._experience_signature = signature
        if memory.pending:
            return
        semantic = decision.get("semantic_decision") or {}
        governor = d.get("governor") or {}
        original_baseline = d.get("original_baseline_selected_id", d.get("baseline_selected_id"))
        memory.begin(context, {"baseline_candidate_id": original_baseline,
            "selected_candidate_id": d.get("selected_id"), "actual_action": action,
            "selected_parameters": parameters, "requested_action": semantic,
            "selected_geometry": {key: candidate.get(key) for key in
                ("recovery", "recovery_recipe", "recovery_transition_m", "lane_offset_m")},
            "adopted": d.get("selected_id") != original_baseline,
            "rejection_reason": governor.get("reason"),
            "semantic_policy": getattr(self, "semantic_policy", "always")}, self._experience_telemetry())

    def _audit_pdm_error(self, exc, semantic, source):
        try:
            self._audit_pdm_plan(
                {"semantic_decision": deepcopy(semantic),
                 "planner_diagnostics": getattr(exc, "diagnostics", None)},
                {"accepted": False, "reasons": ["mpc_planning_error" if self.planner_backend == "mpc" else "pdm_planning_error"],
                 "details": {"error_type": type(exc).__name__, "error": str(exc)}}, source)
        except Exception:
            logger.error("Numerical error audit could not be written; vehicle already braking", exc_info=True)

    def _refresh_pdm_control(self, tf, yaw: float, speed: float) -> None:
        """Replan against current actors between expensive semantic VLM calls."""
        hold = getattr(self, "_mpc_minimum_risk_hold", None)
        if hold and self._elapsed >= hold["execution_expires_at_s"]-1e-9:
            self._next_pdm_replan_at = 0.
        if (self.planner_backend not in ("pdm_closed", "mpc") or (self.model_inference and self._active_semantics is None
                and not getattr(self, "_selective_execution", False)
                and not getattr(self, "_mpc_minimum_risk_hold", None))
                or self._elapsed + 1e-8 < self._next_pdm_replan_at):
            return
        self._next_pdm_replan_at = self._elapsed + self._pdm_interval
        try:
            plan = self._pdm_plan(self._active_semantics)
            result = self._merge_plan({}, self._active_semantics, plan)
            validation = self._validate_decision(result, {"ego_state": {"speed": speed}})
            self._audit_pdm_plan(result, validation, "control_update")
            if self._execute_mpc_minimum_risk_hold(result, validation, tf, yaw):
                return
            if not validation["accepted"]:
                if (result.get("planner_diagnostics") or {}).get("failure_reason") == "planner_recovery_timeout":
                    self._early_terminated = True
                    self._termination_reason = "planner_recovery_timeout"
                raise ValueError("%s trajectory rejected: %s" % (self.planner_backend, validation["reasons"]))
            self._mpc_minimum_risk_hold = None
            self.tracker.set_trajectory(result["trajectory"], tf.location.x, tf.location.y, yaw,
                                        result["target_speed"], follow_speed_profile=True)
            self._planned_stop = self._is_planned_stop(result)
            self._consecutive_trajectory_rejections = 0
        except Exception as exc:
            self._mpc_minimum_risk_hold = None
            # Apply the brake before attempting further logging: a full disk or
            # broken audit path must never leave the previous throttle applied.
            self._apply_brake()
            self.tracker.set_trajectory([], tf.location.x, tf.location.y, yaw, 0.0)
            self._planned_stop = True
            self._trajectory_rejections += 1
            self._consecutive_trajectory_rejections += 1
            self._audit_pdm_error(exc, self._active_semantics, "control_error")
            logger.error("%s control update failed; braking: %s", self.planner_backend, exc)
            if (self._consecutive_trajectory_rejections >= int(self.ccfg.get("max_trajectory_rejections", 3))
                    and not self._early_terminated):
                self._early_terminated = True
                self._termination_reason = "invalid_trajectory"

    def outcome(self) -> dict:
        """Cached result is safe to read after CARLA actors have been destroyed."""
        from carla_bridge.env.paper_events import PaperEventScheduler
        event_result = self.event_scheduler.summary() if isinstance(self.event_scheduler, PaperEventScheduler) else None
        return {"success": self.run_status == "route_completed" and self.route_completed,
                "route_completed": self.route_completed,
                "route_completion": self.route_completion,
                "duration_s": round(self._elapsed, 3),
                "trajectory_rejections": self._trajectory_rejections,
                "planner_backend": self.planner_backend, "semantic_influence": self.semantic_influence,
                "pdm_lateral_control": (getattr(self, "planning_cfg", {}).get("lateral_control", "pure_pursuit")
                                        if self.planner_backend == "pdm_closed" else None),
                "pdm_lateral_preview": (getattr(self, "planning_cfg", {}).get("lqr_lateral_preview", "fixed")
                                        if self.planner_backend == "pdm_closed" else None),
                "numerical_lateral_control": ("optimized_mpc_inputs" if self.planner_backend == "mpc" else
                    getattr(self, "planning_cfg", {}).get("lateral_control", "pure_pursuit")),
                "execution_planner_version": getattr(self, "execution_planner_version", None),
                "lane_quality_enabled": (getattr(self, "planning_cfg", {}).get("lane_quality") or {}).get("enabled", True),
                "lane_maneuver_enabled": (getattr(self, "planning_cfg", {}).get("lane_maneuver") or {}).get("enabled", True),
                "model_inference": self.model_inference,
                "pdm_plan_count": self._pdm_plan_count,
                "numerical_plan_count": self._pdm_plan_count,
                "semantic_policy": getattr(self, "semantic_policy", "always"),
                "planning_memory": getattr(self, "planning_memory_mode", "off"),
                "cognition_scheduling": ("disabled" if not self.model_inference else
                                         "event_triggered" if getattr(self, "_selective_execution", False) else "periodic"),
                "advisory": deepcopy(getattr(self, "advisory_stats", {})),
                "model_usage": deepcopy(getattr(self, "_replay_summary", None)),
                "replay_error": getattr(self, "_replay_error", None),
                "wall_time_s": round(time.perf_counter()-getattr(self, "_wall_started", time.perf_counter()), 3),
                "cleanup_errors": list(self._cleanup_errors), "experiment_events": event_result}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        def cleanup(label, operation):
            try:
                if operation() is not False:
                    return True
            except Exception as exc:
                logger.error("%s: %s", label, exc)
            self._cleanup_errors.append(label)
            return False

        # Snapshot actor-dependent metrics before destroying actors. The final
        # report is written only after cleanup, so it reflects cleanup failures.
        if self.metrics is not None:
            try:
                self._final_summary = self.metrics.summary()
            except Exception as exc:
                logger.error("metrics snapshot: %s", exc)
                self._cleanup_errors.append("metrics_snapshot")
        if getattr(self, "experience_memory", None) is not None:
            cleanup("execution_experience_finalize", lambda: self.experience_memory.finish(self.run_status))
        if self.recorder is not None:
            cleanup("video_flush", self.recorder.flush)
        if getattr(self, "live_view", None) is not None:
            cleanup("live_view_close", self.live_view.close)
        if self.event_scheduler is not None:
            cleanup("event_cleanup", self.event_scheduler.destroy)
        if self.loop:
            try:
                self.loop.close()
            except Exception as e:
                self.run_status = "memory_save_failed"
                logger.error("loop close: %s", e)
        if self._collision_sensor is not None:
            cleanup("collision_stop", self._collision_sensor.stop)
            cleanup("collision_destroy", self._collision_sensor.destroy)
        if self.cameras:
            cleanup("camera_cleanup", self.cameras.destroy)
        scenario_clean = True
        if self.scenario:
            scenario_clean = cleanup("scenario_cleanup", self.scenario.destroy)
        if self.carla_client and scenario_clean:
            cleanup("world_restore", self.carla_client.restore)
        elif self.carla_client:
            self._cleanup_errors.append("world_restore_skipped_for_live_actors")
        if self._cleanup_errors and self.run_status != "memory_save_failed":
            self.run_status = "cleanup_failed"
        self._finalize_replay()
        if self._final_summary is not None:
            try:
                self._write_report()
            except Exception as exc:
                self.run_status = "report_failed"
                self._cleanup_errors.append("report_write")
                logger.error("report write: %s", exc)

    # ------------------------------------------------------------------
    # 单次重规划周期
    # ------------------------------------------------------------------

    def _advisory_tick(self) -> None:
        """Numerical planning runs independently; cognition is an optional consultation.

        CARLA synchronous time remains frozen during a consultation. Model
        failure never replaces the already validated numerical plan with STOP.
        """
        self._pdm_only_tick()
        if self._early_terminated or self._termination_reason:
            self.cameras.drain()
            return
        if self._elapsed+1e-8 < self._next_query_check_at:
            self.cameras.drain()
            return
        self._next_query_check_at = self._elapsed+self._query_check_interval
        query = self.pdm_planner.query_decision(self._elapsed)
        if not query.get("query"):
            self.advisory_stats["query_checks_skipped"] += 1
            self.cameras.drain()
            self._audit_advisory("skip", query)
            return
        ego = self.scenario.ego_vehicle
        started = time.perf_counter()
        try:
            kf = self._advisory_keyframe(ego)
            self.advisory_stats["model_queries"] += 1
            logger.info("[选择性认知] t=%.1fs reason=%s；同步仿真暂停，数值控制方案保留",
                        self._elapsed, query.get("reason"))
            record = self._run_cognition(kf, query) or {}
            self._last_scene_result = record.get("current_scene") or self._last_scene_result
            self._last_nav = kf["nav_instruction"]
            validation = record.get("decision_validation") or {}
            decision = record.get("decision_output") or {}
            tf = ego.get_transform()
            yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
            if self._execute_mpc_minimum_risk_hold(decision, validation, tf, yaw):
                self._audit_advisory("minimum_risk_hold", query)
                return
            if validation.get("accepted") is not True:
                raise ValueError("advisory rejected: %s" % validation.get("reasons", ["missing_validation"]))
            self._mpc_minimum_risk_hold = None
            tf, v = ego.get_transform(), ego.get_velocity()
            yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
            self.tracker.set_trajectory(decision["trajectory"], tf.location.x, tf.location.y, yaw,
                                        decision["target_speed"], follow_speed_profile=True)
            self._planned_stop = self._is_planned_stop(decision)
            self._apply_control(ego, self.tracker.compute_control(tf.location.x, tf.location.y, yaw, math.hypot(v.x, v.y)))
            self._audit_advisory("result", query, decision.get("planner_diagnostics", {}).get("governor"))
        except Exception as exc:
            self.advisory_stats["model_failures"] += 1
            logger.warning("[选择性认知] 建议不可用，继续当前真值下的数值规划: %s", exc, exc_info=True)
            self._active_semantics = None
            self.pdm_planner.governor.submit(None, self.pdm_planner.last_context, self._elapsed)
            self._next_pdm_replan_at = 0.0
            tf, v = ego.get_transform(), ego.get_velocity()
            yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
            self._refresh_pdm_control(tf, yaw, math.hypot(v.x, v.y))
            if not self._early_terminated:
                self._apply_control(ego, self.tracker.compute_control(tf.location.x, tf.location.y, yaw, math.hypot(v.x, v.y)))
            self._audit_advisory("unavailable", query, {"error_type": type(exc).__name__, "error": str(exc)})
        finally:
            self.advisory_stats["inference_wall_time_s"] += time.perf_counter()-started

    def _audit_advisory(self, action, query, result=None):
        path = Path(self.config.get("output_dir") or "outputs") / "advisory.jsonl"
        audit = {"sim_time_s": self._elapsed, "action": action, "query": query,
                 "semantic_policy": self.semantic_policy, "result": result,
                 "context": self.pdm_planner.last_context}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False)+"\n")

    def _run_cognition(self, kf, query):
        audit = getattr(self, "_model_call_audit", None)
        if audit is None:
            return self.loop.step(kf)
        ego = self.scenario.ego_vehicle
        try:
            transform, velocity = ego.get_transform(), ego.get_velocity()
            pose = {"x": float(transform.location.x), "y": float(transform.location.y),
                    "z": float(transform.location.z), "yaw_deg": float(transform.rotation.yaw),
                    "speed_mps": math.hypot(velocity.x, velocity.y)}
        except Exception as exc:
            pose = {"error": type(exc).__name__+": "+str(exc)}
        with audit.consultation(kf, query, pose) as evidence:
            record = self.loop.step(kf)
            evidence["record"] = record
            validation = (record or {}).get("decision_validation") or {}
            evidence["status"] = "accepted" if validation.get("accepted") is True else "unavailable"
            return record

    def _finalize_replay(self):
        if not (getattr(self, "ccfg", {}).get("replay") or {}).get("enabled", True):
            return
        try:
            output = Path(self.config.get("output_dir") or "outputs")
            output.mkdir(parents=True, exist_ok=True)
            metadata = {"schema_version": 1, "model_inference": self.model_inference,
                        "semantic_influence": self.semantic_influence,
                        "semantic_policy": self.semantic_policy,
                        "duration_s": self._elapsed, "run_status": self.run_status,
                        "audit_enabled": self._model_call_audit is not None,
                        "audit_errors": list(self._model_call_audit.errors) if self._model_call_audit else []}
            (output / "replay_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            from carla_bridge.replay.review import build_review
            self._replay_summary = build_review(output)
            logger.info("模型调用回放已生成: %s", output / "replay" / "index.html")
        except Exception as exc:
            self._replay_error = type(exc).__name__+": "+str(exc)
            logger.warning("模型回放报告生成失败，可在运行后重新生成: %s", exc, exc_info=True)

    def _advisory_keyframe(self, ego):
        frame_id = self.carla_client.get_world().get_snapshot().frame
        token = self.kf_builder.next_token()
        mosaic = self.cameras.capture(token, frame_id)
        tf = ego.get_transform()
        yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        self.history.update(ego, self._elapsed)
        state = self.ego_provider.build(ego, self._elapsed, yaw_rate=self.history.yaw_rate(self._elapsed, yaw))
        state.timestamp = self.timestamp_origin_us+int(self._elapsed*1e6)
        objs = self.perception.get_objects(ego, tf.location.x, tf.location.y, yaw)
        objs = self._inject_route_centers(objs, ego, tf, yaw)
        nav = self.scenario.route.nav_instruction()
        kf = self.kf_builder.build(sample_token=token, timestamp_us=state.timestamp, mosaic_path=mosaic,
            ego_state=state.to_ego_centric(),
            history_trajectory=self.history.build_ego_centric(self._elapsed, tf.location.x, tf.location.y, yaw),
            nav_instruction=nav, perception_objects=objs)
        from carla_bridge.planning.navigation_context import build_navigation_context
        kf.update(carla_frame_id=frame_id, sim_time_s=self._elapsed)
        kf["planning_constraints"] = {"decision_mode": "semantic", "semantic_policy": self.semantic_policy,
            "required_horizon_s": self._execution_horizon,
            "planner_context": deepcopy(self.pdm_planner.last_context),
            "navigation_context": build_navigation_context(self.scenario.route, ego, self._elapsed)}
        return kf

    def _pdm_only_tick(self) -> None:
        """Use the normal PDM/control chain every physics tick, without cognition."""
        ego = self.scenario.ego_vehicle
        if self.event_scheduler is not None:
            self.event_scheduler.tick()
        self.carla_client.tick()
        self._elapsed += self.fixed_delta
        self._record_tick(ego)
        if self._check_run_end():
            return
        tf = ego.get_transform()
        yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        velocity = ego.get_velocity()
        speed = math.hypot(velocity.x, velocity.y)
        self._refresh_pdm_control(tf, yaw, speed)
        if self._early_terminated:
            self._apply_brake()
            return
        self._apply_control(ego, self.tracker.compute_control(tf.location.x, tf.location.y, yaw, speed))
        if self.history is not None:
            self.history.update(ego, self._elapsed)
        if self._elapsed + 1e-8 >= getattr(self, "_next_pdm_progress_log_at", 0.0):
            self._check_stuck(speed)
            logger.info("[%s] t=%.1fs 路线完成度=%.1f%% speed=%.2fm/s 碰撞=%d",
                        ("选择性" if getattr(self, "_selective_execution", False) else "纯") + self.planner_backend.upper(),
                        self._elapsed, self.route_completion * 100.0, speed, self._collision_count)
            self._next_pdm_progress_log_at = self._elapsed + 3.0

    def _replan_cycle(self) -> None:
        ego = self.scenario.ego_vehicle

        # ---- [1 捕获] ----
        if self.event_scheduler is not None:
            self.event_scheduler.tick()
        frame_id = self.carla_client.tick()
        self._elapsed += self.fixed_delta
        self._record_tick(ego)
        if self._check_run_end():
            self.cameras.drain()
            return
        sample_token = self.kf_builder.next_token()
        try:
            mosaic_path = self.cameras.capture(sample_token, frame_id)
        except (TimeoutError, OSError) as exc:
            self._on_capture_failure(ego, exc)
            return
        self._capture_failures = 0

        tf = ego.get_transform()
        ego_yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        self.history.update(ego, self._elapsed)
        yr = self.history.yaw_rate(self._elapsed, ego_yaw)
        ego_state = self.ego_provider.build(ego, self._elapsed, yaw_rate=yr)
        ego_state.timestamp = self.timestamp_origin_us + int(self._elapsed * 1e6)
        hist = self.history.build_ego_centric(
            self._elapsed, tf.location.x, tf.location.y, ego_yaw
        )
        objs = self.perception.get_objects(ego, tf.location.x, tf.location.y, ego_yaw)
        nav = self.scenario.route.nav_instruction() if self.scenario.route else "straight"
        objs = self._inject_route_centers(objs, ego, tf, ego_yaw)

        kf = self.kf_builder.build(
            sample_token=sample_token,
            timestamp_us=self.timestamp_origin_us + int(self._elapsed * 1e6),
            mosaic_path=mosaic_path,
            ego_state=ego_state.to_ego_centric(),
            history_trajectory=hist,
            nav_instruction=nav,
            perception_objects=objs,
        )
        kf["carla_frame_id"] = frame_id
        kf["sim_time_s"] = self._elapsed
        kf["planning_constraints"] = {
            "decision_mode": "semantic" if self.planner_backend in ("pdm_closed", "mpc") else "trajectory",
            "required_horizon_s": self._execution_horizon,
            **{key: getattr(self.trajectory_validator, key) for key in (
                "max_speed_mps", "max_acceleration_mps2", "max_deceleration_mps2",
                "max_lateral_acceleration_mps2")},
        }
        if self.planner_backend in ("pdm_closed", "mpc"):
            from carla_bridge.planning.navigation_context import build_navigation_context
            kf["planning_constraints"]["navigation_context"] = build_navigation_context(
                self.scenario.route, ego, self._elapsed)
            kf["planning_constraints"]["semantic_policy"] = getattr(self, "semantic_policy", "always")
            kf["planning_constraints"]["planner_context"] = deepcopy(getattr(self.pdm_planner, "last_context", None))
        logger.info(
            "[replan] t=%.1fs token=%s nav=%s objs=%d speed=%.1f",
            self._elapsed, sample_token, nav, len(objs), ego_state.speed,
        )

        # ---- [2 认知]（sim 冻结，约 5-30s，期间无日志属正常，请勿 Ctrl+C）----
        logger.info("[认知] 开始：场景VLM + 三层记忆检索 + 决策VLM（耗时 5-30s，请耐心等待）")
        inference_started = time.perf_counter()
        self.advisory_stats["model_queries"] += 1
        try:
            record = self._run_cognition(kf, {"query": True, "reason": "periodic_cognition"})
            # 记录上次 VLM 场景结果 + nav，供控制阶段 raw 捕获沿用
            self._last_scene_result = (record or {}).get("current_scene") or self._last_scene_result
            self._last_nav = nav
            decision = (record or {}).get("decision_output") or {}
            validation = (record or {}).get("decision_validation") or {}
            accepted = validation.get("accepted") is True
            minimum_risk = self._execute_mpc_minimum_risk_hold(decision, validation, tf, ego_yaw)
            if minimum_risk:
                traj, target_speed = [], 0.
            elif accepted:
                self._mpc_minimum_risk_hold = None
                self._consecutive_trajectory_rejections = 0
                traj = decision.get("trajectory") or []
                target_speed = float(decision.get("target_speed", 0.0) or 0.0)
                self._planned_stop = self._is_planned_stop(decision)
            else:
                self._mpc_minimum_risk_hold = None
                recovery_timeout = (decision.get("planner_diagnostics") or {}).get("failure_reason") == "planner_recovery_timeout"
                if recovery_timeout:
                    self._early_terminated = True
                    self._termination_reason = "planner_recovery_timeout"
                self.advisory_stats["model_failures"] += 1
                traj, target_speed = [], 0.0
                self._planned_stop = True
                self._trajectory_rejections += 1
                self._consecutive_trajectory_rejections += 1
                logger.error("轨迹被拒绝，本周期全刹: %s", validation.get("reasons", ["missing_validation"]))
                if (self._consecutive_trajectory_rejections >= int(self.ccfg.get("max_trajectory_rejections", 3))
                        and not self._early_terminated):
                    self._early_terminated = True
                    self._termination_reason = "invalid_trajectory"
            self.tracker.set_trajectory(
                traj, tf.location.x, tf.location.y, ego_yaw, target_speed,
                **({"follow_speed_profile": True} if self.planner_backend in ("pdm_closed", "mpc") and accepted else {})
            )
        except Exception as e:
            self.advisory_stats["model_failures"] += 1
            # 认知步骤异常（如 VLM 调用失败）：打印完整 traceback 便于定位，本周期全刹，继续下一周期
            logger.error("认知步骤异常（本周期全刹，继续下一周期）: %s", e, exc_info=True)
            self.tracker.set_trajectory([], tf.location.x, tf.location.y, ego_yaw, 0.0)
            self._planned_stop = True
            self._active_semantics = None
        finally:
            self.advisory_stats["inference_wall_time_s"] += time.perf_counter() - inference_started

        if self._early_terminated:
            self._apply_brake()
            return

        # ---- [3 控制] 10Hz × N tick，其中每周期 5 次 raw 捕获 push 进短期记忆 ----
        # One capture tick belongs to the next cycle. Together these ticks match
        # the validated trajectory horizon exactly (no extra stale-control tick).
        n_ticks = self._cycle_ticks - 1
        raw_every = max(1, int(round(n_ticks / 5.0)))  # 每周期 5 次 raw 捕获（≈1.67Hz @3s/10Hz）
        # Apply the new decision before advancing physics (including emergency braking).
        v = ego.get_velocity()
        speed = (v.x * v.x + v.y * v.y) ** 0.5
        self._apply_control(ego, self.tracker.compute_control(tf.location.x, tf.location.y, ego_yaw, speed))
        for i in range(n_ticks):
            if self._check_run_end():
                return
            if self.event_scheduler is not None:
                self.event_scheduler.tick()
            frame_id = self.carla_client.tick()
            self._elapsed += self.fixed_delta
            self._record_tick(ego)
            if self._check_run_end():
                self.cameras.drain()
                return
            self.history.update(ego, self._elapsed)
            if i % raw_every == 0:
                if not self._raw_capture(ego, frame_id):
                    return  # raw 捕获：mosaic+ego+感知 push 进短期记忆
            else:
                self.cameras.drain()  # 其余 tick 丢弃相机帧，防堆积
            tf2 = ego.get_transform()
            yaw2 = coords.carla_yaw_deg_to_rad(tf2.rotation.yaw)
            v = ego.get_velocity()
            speed = (v.x * v.x + v.y * v.y) ** 0.5
            self._refresh_pdm_control(tf2, yaw2, speed)
            if self._early_terminated:
                self._apply_brake()
                return
            ctrl = self.tracker.compute_control(tf2.location.x, tf2.location.y, yaw2, speed)
            self._apply_control(ego, ctrl)

        self._check_stuck(speed)
        if self._early_terminated:
            return

        # ---- [4 记录] ----
        prog = (self.scenario.route.progress_fraction() * 100.0) if self.scenario.route else 0.0
        logger.info(
            "[cycle end] t=%.1fs 路线完成度=%.1f%% 碰撞=%d",
            self._elapsed, prog, self._collision_count,
        )

    def _check_stuck(self, speed: float) -> None:
        """Preserve the same no-progress guard in both cognition modes."""
        diagnostics = getattr(self, "_last_plan_diagnostics", {}) or {}
        degradation = diagnostics.get("planner_degradation") or {}
        health = degradation.get("native_health") or {}
        planner_deadlock = (degradation.get("sustained") is True
                            and health.get("positive_progressing_count") == 0
                            and health.get("drivable_area_failure_count", 0) > 0)
        # A numerical-planner deadlock is not an explained traffic wait merely
        # because its winning trajectory is stationary. Context-grounded
        # degradation has already excluded current red lights/queues/conflicts.
        if self._stuck_enabled and (not self._planned_stop or planner_deadlock):
            prog = (self.scenario.route.progress_fraction()
                    if self.scenario.route else 0.0)
            if speed < self._stuck_speed_thresh and (prog - self._stuck_progress) < 0.005:
                if self._stuck_since is None:
                    self._stuck_since = self._elapsed
                stuck_for = self._elapsed - self._stuck_since
                if stuck_for >= self._stuck_term_s:
                    logger.error(
                        "[卡死] 低速无进展 %.0fs（speed=%.2f, progress=%.1f%%），提前终止本场景",
                        stuck_for, speed, prog * 100.0,
                    )
                    self._early_terminated = True
                    self._termination_reason = "stuck"
                    return
                if stuck_for >= self._stuck_warn_s:
                    logger.warning(
                        "[卡死预警] 已低速无进展 %.0fs（speed=%.2f, progress=%.1f%%），"
                        "%.0fs 后将提前终止", stuck_for, speed, prog * 100.0,
                        self._stuck_term_s - stuck_for,
                    )
            else:
                self._stuck_since = None
            self._stuck_progress = prog
        else:
            # A deliberate stop/yield can legitimately outlast the stuck timer.
            # The overall route timeout still bounds repeated stop decisions.
            self._stuck_since = None

    def _record_tick(self, ego) -> None:
        # Read the server clock before observers/metrics, in the same callback as
        # the measured response. Keep the local scheduler's sim_time_s unchanged.
        world_clock = self._world_clock_audit() if self._control_audit_path is not None or self.recorder is not None else None
        self._drain_collisions()
        self.carla_client.set_spectator_follow(ego)
        if self.recorder is not None:
            audit = getattr(self, "_model_call_audit", None)
            latest = audit.latest if audit is not None else None
            overlay = {"model_call_count": audit.consultation_count if audit else 0,
                       "run_id": self.config.get("run_id"),
                       "planner_sequence": (getattr(self, "_last_control_command", {}) or {}).get("planner_sequence"),
                       "latest_call_id": latest.get("call_id") if latest else None,
                       "latest_trigger": (latest.get("query") or {}).get("reason") if latest else None,
                       "selected_id": (getattr(self, "_last_plan_diagnostics", {}) or {}).get("selected_id"),
                       "trajectory_visualization": deepcopy(
                           getattr(self, "_last_trajectory_visualization_audit", {"enabled": False}))}
            try:
                display = getattr(getattr(self, "trajectory_renderer", None), "last_display_snapshot", None)
                visual = overlay["trajectory_visualization"]
                if (isinstance(display, dict) and visual.get("enabled") is True
                        and not visual.get("error")):
                    # Display-only copies never enter decisions, scoring or
                    # controller inputs. Delayed images retain their own plan.
                    overlay["trajectory_display"] = deepcopy(display)
                    try:
                        vertices = ego.bounding_box.get_world_vertices(ego.get_transform())
                        overlay["trajectory_display"]["ego_box_world"] = [
                            [float(v.x), float(v.y), float(v.z)] for v in vertices]
                    except Exception as exc:
                        # Self-vehicle masking is optional annotation metadata;
                        # unsupported telemetry cannot remove the retained plan.
                        logger.debug("视频自车遮挡轮廓不可用: %s", exc)
            except Exception as exc:
                logger.warning("视频轨迹快照已跳过: %s", exc)
            self.recorder.on_tick(
                frame_id=(world_clock or {}).get("frame"), sim_time_s=self._elapsed,
                overlay=overlay)
        if self.metrics is not None or self._control_audit_path is not None:
            v, a, tf = ego.get_velocity(), ego.get_acceleration(), ego.get_transform()
            yaw = math.radians(tf.rotation.yaw)  # Native CARLA axes for acceleration projections.
            state = {
                "speed": math.hypot(v.x, v.y),
                "velocity_x": v.x, "velocity_y": v.y,
                "speed_longitudinal_mps": v.x * math.cos(yaw) + v.y * math.sin(yaw),
                "acceleration": math.hypot(a.x, a.y),
                "acceleration_x": a.x, "acceleration_y": a.y, "yaw_rad": yaw,
            }
            previous = getattr(self, "_experience_kinematics", None)
            self._experience_jerk = (math.hypot(a.x-previous[1], a.y-previous[2])/(self._elapsed-previous[0])
                                    if previous and self._elapsed > previous[0] else None)
            self._experience_kinematics = (self._elapsed, a.x, a.y)
            previous_command = getattr(self, "_last_control_command", {}) or {}
            if (previous_command.get("source") == "mpc_minimum_risk_hold"
                    and getattr(self, "_minimum_risk_measured_at_s", None) != self._elapsed):
                self._minimum_risk_measured_at_s = self._elapsed
                stats = self.advisory_stats
                stats["minimum_risk_hold_ticks"] = stats.get("minimum_risk_hold_ticks", 0) + 1
                stats["minimum_risk_hold_s"] = stats.get("minimum_risk_hold_s", 0.) + self.fixed_delta
            if self.metrics is not None:
                self.metrics.on_tick(ego, self._elapsed, state)
            memory = getattr(self, "experience_memory", None)
            if memory is not None:
                memory.observe(self._experience_telemetry())
            diagnostics = getattr(self, "_last_plan_diagnostics", {})
            if diagnostics and state["speed"] <= .3 and getattr(self, "advisory_stats", None) is not None:
                original_id = diagnostics.get("original_baseline_selected_id", diagnostics.get("baseline_selected_id"))
                baseline = next((c for c in (diagnostics.get("original_baseline_candidates") or diagnostics.get("candidates", []))
                                 if c["id"] == original_id), {})
                if (diagnostics.get("selected_id") != original_id
                        and baseline.get("speed_at_0_2s_mps", 0) > .5):
                    # Attribution proxy, not a claim that the stop was unnecessary.
                    self.advisory_stats["semantic_stop_proxy_s"] += self.fixed_delta
            if self._control_audit_path is not None:
                actual = ego.get_control()
                audit = {
                    "sim_time_s": self._elapsed,
                    "world_snapshot": world_clock,
                    "measured": {**state, "x": tf.location.x, "y": tf.location.y,
                                 "acceleration_longitudinal_mps2": a.x * math.cos(yaw) + a.y * math.sin(yaw),
                                 "acceleration_lateral_mps2": -a.x * math.sin(yaw) + a.y * math.cos(yaw)},
                    "applied_control": self._control_audit_values(actual),
                    # This is the command preceding this measured physics tick,
                    # not the newly replanned command for the next tick.
                    "command": self._last_control_command,
                }
                try:
                    with self._control_audit_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
                except Exception:
                    self._apply_brake()
                    raise

    # ------------------------------------------------------------------
    # 路由中心线注入（米制回正锚点）
    # ------------------------------------------------------------------

    def _inject_route_centers(self, objs: list, ego, ego_tf, ego_yaw: float) -> list:
        """把 A* 路由前方中心点转 ego-centric 后作为 route_center 伪对象注入。

        决策 prompt 的感知对象段逐条渲染 ``position_ego=[前方,左侧]``（米），
        注入后 VLM 拿到车道中心的米制坐标，KEEP_LANE 轨迹可对齐回正，不再沿
        偏移位置平行直行。保持整体按距离升序（prompt 只渲染前 12 条且声明排序）。
        """
        if not (self._route_inject_enabled and self.scenario.route):
            return objs
        try:
            pts = self.scenario.route.center_points_ahead(
                ego_loc=ego.get_location(),
                n=self._route_inject_n,
                step_m=self._route_inject_step,
            )
        except Exception as e:
            logger.debug("路由中心点获取失败（本帧不注入）: %s", e)
            return objs
        merged = list(objs)
        for k, p in enumerate(pts):
            fwd, left = coords.global_to_ego(
                p.x, p.y, ego_tf.location.x, ego_tf.location.y, ego_yaw
            )
            dist = (fwd * fwd + left * left) ** 0.5
            merged.append({
                "annotation_token": f"route_{k}",
                "instance_token": f"route_{k}",
                "category": "route",
                "category_name_raw": "route_center",
                "semantic_label": "route_center",
                "size": [],
                "position_global": [round(p.x, 4), round(p.y, 4), round(p.z, 4)],
                "position_ego": [round(fwd, 4), round(left, 4)],
                "distance_to_ego": round(dist, 4),
                "heading_global": ego_yaw,
                "heading_ego": 0.0,
                "velocity": None,
                "speed": None,
                "acceleration": None,
                "acceleration_mag": None,
                "velocity_available": False,
                "acceleration_available": False,
                "kinematics_source": "route_center",
            })
        if pts:
            merged.sort(key=lambda o: o.get("distance_to_ego", 0.0))
            logger.info("[route] 注入 %d 个车道中心锚点: %s",
                        len(pts),
                        ", ".join(f"({o['position_ego'][0]:.0f},{o['position_ego'][1]:.1f})"
                                  for o in merged if o.get("semantic_label") == "route_center"))
        return merged

    # ------------------------------------------------------------------
    # raw 捕获（控制阶段，不跑 VLM）
    # ------------------------------------------------------------------

    def _raw_capture(self, ego, frame_id: int) -> bool:
        """控制阶段 raw 捕获：存 mosaic + ego + GT 感知，push 进短期记忆。

        不跑 DINOv2/VLM；场景字段（scene_id/weather/description）沿用上次 VLM 结果，
        使短期记忆队列在控制阶段也时刻保持最新（图像+ego+感知），完整 VLM 按 replan_interval_s 运行。
        """
        self._raw_counter += 1
        token = f"carla_raw_{self.run_id or self.scenario.scenario_name()}_{self._raw_counter:06d}"
        try:
            mosaic_path = self.cameras.capture(token, frame_id)
        except (TimeoutError, OSError) as exc:
            self._on_capture_failure(ego, exc)
            return False
        self._capture_failures = 0
        tf = ego.get_transform()
        ego_yaw = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        yr = self.history.yaw_rate()
        ego_state = self.ego_provider.build(ego, self._elapsed, yaw_rate=yr)
        ego_state.timestamp = self.timestamp_origin_us + int(self._elapsed * 1e6)
        try:
            objs = self.perception.get_objects(ego, tf.location.x, tf.location.y, ego_yaw)
        except Exception:
            objs = []
        sc = self._last_scene_result or {}
        item = ShortTermMemoryItem(
            frame_id=token,
            timestamp=self.timestamp_origin_us + int(self._elapsed * 1e6),
            image_path=mosaic_path,
            image_feature_path="",
            scene_description=sc.get("scene_description", ""),
            scene_id=sc.get("scene_id", "unknown"),
            weather_id=sc.get("weather_id", "unknown"),
            nav_instruction=self._last_nav,
            ego_state=ego_state.to_ego_centric(),
            history_trajectory=[],
            scene_understanding_result=sc,
        )
        try:
            self.loop.add_short_term_item(item)
        except Exception as e:
            logger.debug("raw short_term push 失败: %s", e)
        return True

    def _on_capture_failure(self, ego, exc):
        self._capture_failures += 1
        self._capture_failure_total += 1
        tf = ego.get_transform()
        self.tracker.set_trajectory([], tf.location.x, tf.location.y,
                                    coords.carla_yaw_deg_to_rad(tf.rotation.yaw), 0.0)
        self._apply_control(ego, carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0),
                            source="capture_failure")
        logger.error("同步采集失败，已制动 (%d): %s", self._capture_failures, exc)
        if self._capture_failures >= int(self.ccfg.get("max_capture_failures", 3)):
            self._early_terminated = True
            self._termination_reason = "sensor_failure"

    # ------------------------------------------------------------------
    # 碰撞
    # ------------------------------------------------------------------

    def _setup_collision_sensor(self, world, ego) -> None:
        bp = world.get_blueprint_library().find("sensor.other.collision")
        # try_spawn_actor：失败返回 None 不抛异常，降级为「不计碰撞」而非崩掉整个 setup
        sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=ego)
        if sensor is None:
            logger.warning("碰撞传感器 spawn 失败，本次运行不计碰撞指标")
            return
        self._collision_sensor = sensor
        sensor.listen(self._coll_q.put)

    def _drain_collisions(self) -> None:
        """碰撞事件去重：同 actor 2s 冷却内只计 1 次。

        collision sensor 每个接触 tick 都发事件，一次碰撞被反复计成几十次，
        会污染 memory_on/off 对比。原始事件数保留在 ``_collision_raw`` 供报告。
        """
        try:
            while True:
                ev = self._coll_q.get_nowait()
                self._collision_raw += 1
                try:
                    aid = ev.other_actor.id
                except Exception:
                    aid = None
                self._collision_actors.add(aid)
                if aid is not None:
                    last = self._last_coll_at.get(aid, -1e9)
                    if self._elapsed - last < 2.0:
                        continue
                    self._last_coll_at[aid] = self._elapsed
                self._collision_count += 1
                if self.metrics is not None:
                    self.metrics.on_collision()
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    def _make_output_path(self) -> str:
        out_dir = self.config.get("output_dir") or Path("outputs")
        return str(
            Path(out_dir)
            / f"decisions_carla_{self.scenario.scenario_name()}_{self.mode}.jsonl"
        )

    def _write_report(self) -> None:
        if self.metrics is None:
            return
        summary = self._final_summary if self._final_summary is not None else self.metrics.summary()
        run_info = {
            "run_id": self.run_id,
            "status": self.run_status,
            **self.outcome(),
            "success_criterion": "ordered_route_completion_100_percent",
            "capture_failures": self._capture_failure_total,
            "scenario": self.scenario.scenario_name(),
            "mode": self.mode,
            "duration_s": round(self._elapsed, 2),
            "replan_interval_s": (self.replan_interval_s if self.model_inference
                                  and not getattr(self, "_selective_execution", False) else None),
            "query_check_interval_s": (self._query_check_interval
                                       if getattr(self, "_selective_execution", False) else None),
            "pdm_replan_interval_s": self._pdm_interval if self.planner_backend == "pdm_closed" else None,
            "numerical_replan_interval_s": self._pdm_interval if self.planner_backend in ("pdm_closed", "mpc") else None,
            "collision_raw_events": self._collision_raw,
            "collision_unique_actors": len(
                {a for a in self._collision_actors if a is not None}
            ),
            "early_terminated": self._early_terminated,
            "early_terminated_reason": self._termination_reason,
        }
        out_dir = self.config.get("output_dir") or Path("outputs")
        report_dir = Path(out_dir)
        md_path, _json_path = write_report(summary, run_info, str(report_dir))
        logger.info("闭环报告已写: %s", md_path)

    @property
    def collision_count(self) -> int:
        return self._collision_count
