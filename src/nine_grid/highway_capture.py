"""实验场景的原始素材采集：复用基础驾驶，只导出离线工作台已有记录格式。"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import time

import numpy as np
from PIL import Image

from .contracts import plain
from .geometry import actor_shape
from .highway import HighwayAdapter
from .paths import filesystem_path
from .planner_factory import create_planner
from .planning_context import prepare_planning_context
from .prediction import Predictor
from .quality import execution_audit

PROTOCOL = "ep-driver-highway-capture-1.0"
logger = logging.getLogger(__name__)


class _CaptureAdapter(HighwayAdapter):
    """沿用原执行前缀，在每个真实控制步后记录；不额外推进世界。"""
    on_frame = None

    def step(self, control):
        result = super().step(control)
        if self.on_frame is not None:
            self.on_frame(self.snapshot(), result)
        return result


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _line(stream, value):
    stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def record_frame(adapter, scene, folder, index, frames, future):
    """同一冻结时刻的真实原生图与完整可观测快照；完整真值只进独立审计流。"""
    now = float(scene.time)
    snapshot = plain(scene)
    # 保留适配器的公开道路WKT（含多面与孔洞），不重建成旧单一可见多边形。
    snapshot["diagnostics"]["driver_source"] = "ep_driver_base_planner_actual_execution_not_expert"
    # 不向教师显示主规划的预测线；render不推进物理世界。
    adapter.env._planned_trajectory = None
    image = np.asarray(adapter.render())
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or not image.size or not image.any() or image.min() == image.max()):
        raise RuntimeError("原生图像缺失、全黑或单色，不保存伪图像")
    if not math.isclose(float(adapter.snapshot().time), now, abs_tol=1e-8):
        raise RuntimeError("原生图像与状态时间不同步")
    native = f"native/{index:06d}.png"
    Image.fromarray(image).save(folder / native)

    observed = {"ego"} | {obj.id for obj in scene.actors}
    actors, collisions = [], []
    for vehicle in [*adapter.env.road.vehicles, *adapter.env.road.objects]:
        item = adapter._actor(vehicle)
        item.visible = item.id in observed
        actors.append(plain(item))
        if getattr(vehicle, "crashed", False):
            collisions.append(item.id)
    ego_collision = bool(adapter.env.vehicle.crashed)
    offroad = not adapter._audit_drivable.covers(actor_shape(scene.ego))
    audit = {"collision": ego_collision, "collision_actor_ids": collisions, "offroad": offroad}
    _line(frames, {"index": index, "time": now, "snapshot": snapshot,
                   "native_image": native, "audit": audit})
    _line(future, {"index": index, "time": now, "actors": actors,
                   "audit_only": True, "used_for_current_representation": False})
    return audit


def capture_episode(config, seed, output_dir, *, provenance=None, live=False):
    """保存每个真实控制步及终帧。中断标记complete=false；拒绝覆盖已有目录。"""
    config = copy.deepcopy(config)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed必须为非负整数")
    if config.get("platform", "highway") != "highway":
        raise ValueError("本入口只采集highway-env")
    duration = float(config.get("duration", config["clock"]["episode_seconds"]))
    step_s = float(config["mpc"]["step_s"])
    if (not math.isfinite(duration) or duration <= 0
            or abs(duration / step_s - round(duration / step_s)) > 1e-7):
        raise ValueError("时长必须为正且为控制步长的整数倍")
    folder = filesystem_path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "native").mkdir()
    identity = folder.name
    run = {
        "protocol": PROTOCOL, "normalization_protocol": PROTOCOL,
        "id": identity, "scenario": config["scenario"], "source": "highway-env",
        "source_version": importlib.metadata.version("highway-env"),
        "record_origin": "closed_loop_simulation", "runtime_mode": "sync",
        "reference_driver": "ep_driver_base_planner_actual_execution_not_expert",
        "seed": seed, "duration_s": duration, "record_hz": 1. / step_s,
        "planner_hz": config["clock"]["planner_hz"], "physics_hz": config["clock"]["physics_hz"],
        "parameters": config, "provenance": provenance or {},
        "parameters_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
        "frames_path": "frames.jsonl", "future_path": "future_states.jsonl",
        "camera_kind": "native_simulator_render_not_camera_sensor", "plan_overlay": False,
        "read_future_during_current_representation": False,
        "model_calls": 0, "retrieval_enabled": False, "write_memory_enabled": False,
        "feature_extraction_enabled": False, "reference_mpc_enabled": False,
        "quality_scoring_enabled": False, "complete": False,
        "frame_count": 0, "planning_steps": 0, "simulated_seconds": 0.,
        "collision": False, "offroad": False, "emergency_steps": 0,
        "note": "与正式场景共用初始化、规划、执行审计和控制；动作是实际结果，不是最优示范。未来仅供离线回顾。",
    }
    _json(folder / "run.json", run)
    adapter = None
    started = time.monotonic()
    result = None
    try:
        adapter = _CaptureAdapter(config, seed, identity, live=live)
        planner = create_planner(config)
        predictor = Predictor(config)
        with (folder / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as frames, (
                folder / "future_states.jsonl").open("w", encoding="utf-8", newline="\n") as future:
            def save(scene, execution=None):
                run["simulated_seconds"] = float(scene.time)
                audit = record_frame(adapter, scene, folder, run["frame_count"], frames, future)
                run["frame_count"] += 1
                run["collision"] |= audit["collision"]
                run["offroad"] |= audit["offroad"]
                if (math.isclose(scene.time, round(scene.time), abs_tol=1e-7)
                        or (execution and execution["terminated"])):
                    logger.info("%s：已保存 %.1fs，原帧%d", identity, scene.time, run["frame_count"])

            adapter.on_frame = save
            save(adapter.snapshot())
            while True:
                scene = adapter.snapshot()
                terminal = bool(result and result["terminated"])
                if terminal:
                    run.update(complete=True, termination_reason=result["reason"],
                               arrival=result["arrival"], progress_m=result["progress"])
                    break
                scoped, predictions = prepare_planning_context(scene, config, predictor=predictor)
                plan = planner.solve(scoped, predictions, None)
                audit = execution_audit(scoped, plan, predictions, config)
                if audit.get("alarm") and not plan.diagnostics.get("emergency", False):
                    plan = planner.emergency_brake(scoped, predictions,
                                                   reason="正常候选执行前几何审计报警")
                    execution_audit(scoped, plan, predictions, config)
                if not plan.valid:
                    raise RuntimeError("基础规划未产生有效控制；保留已采集记录，不伪造后续")
                run["emergency_steps"] += int(plan.diagnostics.get("emergency", False))
                result = adapter.step_plan(plan)
                run["planning_steps"] += 1
                run["collision"] |= bool(result["collision"])
                run["offroad"] |= bool(result["offroad"])
    except BaseException as error:
        run.update(complete=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if adapter is not None:
                adapter.close()
        except BaseException as error:
            run["complete"] = False
            run["cleanup_error"] = type(error).__name__ + ": " + str(error)
            # 若已有运行错误，保留原始异常；正常运行的清理失败则明确报错。
            if "error_type" not in run:
                raise
        finally:
            run["wall_seconds"] = time.monotonic() - started
            _json(folder / "run.json", run)
    return {"run_path": str(folder / "run.json"), **run}
