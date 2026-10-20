"""轨迹跟踪器：决策轨迹 -> carla.VehicleControl
================================================
把决策的 ego-centric 轨迹在**捕获时刻**一次性转成全局 waypoint（``coords``），
之后每个控制 tick 用 Pure Pursuit(横向) + PID(纵向) 算 ``carla.VehicleControl``。

默认目标速度取决策的 ``target_speed``；规划器可通过 ``follow_speed_profile=True``
启用 waypoint 的 ``t`` / ``optional_v`` 速度曲线。曲线按控制 tick 的仿真时间推进，
重规划时重新计时，超出曲线有效期立即刹车。
方向盘左正(项目约定)经 ``steer_sign`` 转成 CARLA 归一化 steer ∈ [-1,1]：
CARLA ``VehicleControl.steer`` 正值=右转，与项目左正相反，故 ``steer_sign=-1``。
"""
from __future__ import annotations

from bisect import bisect_right
import math
from typing import List, Optional, Tuple

import carla  # 须在 mulmem_carla(3.9) 运行

from carla_bridge.state import coords
from carla_bridge.control.pure_pursuit import PurePursuit
from carla_bridge.control.pid import PID
from carla_bridge.control.steering_model import SteeringModel


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class TrajectoryTracker:
    """决策轨迹 -> 每_tick 的 carla.VehicleControl。"""

    def __init__(self, controller_cfg: dict, control_dt_s: float = 0.05):
        self.pp = PurePursuit(
            wheelbase_m=controller_cfg.get("wheelbase_m", 2.8),
            lookahead_min=controller_cfg.get("lookahead_min_m", 3.0),
            lookahead_max=controller_cfg.get("lookahead_max_m", 12.0),
            lookahead_k=controller_cfg.get("lookahead_k", 0.5),
        )
        pid_cfg = controller_cfg.get("pid", {}) or {}
        self.pid = PID(
            kp=pid_cfg.get("kp", 1.0),
            ki=pid_cfg.get("ki", 0.05),
            kd=pid_cfg.get("kd", 0.05),
            max_throttle=pid_cfg.get("max_throttle", 0.75),
            max_brake=pid_cfg.get("max_brake", 0.8),
            deadband_mps=pid_cfg.get("deadband_mps", 0.15),
        )
        self.max_steer_rad = controller_cfg.get("max_steer_rad", 0.6)
        self.steering_model = SteeringModel(self.max_steer_rad)
        self.rear_axle_actor_xyz_m = None
        self.vehicle_model_source = "controller_config"
        # 项目 steer_rad 左正；CARLA VehicleControl.steer 右正 → 必须取反（+yaw=右转，实测）
        self.steer_sign = controller_cfg.get("steer_sign", -1)
        self.control_dt = control_dt_s
        self._global_wps: Optional[List[dict]] = None
        self._actor_global_wps: Optional[List[dict]] = None
        self._target_speed: float = 0.0
        self._speed_profile: Optional[List[Tuple[float, float]]] = None
        self._profile_elapsed_s: float = 0.0
        self.last_control_diagnostics: dict = {}

    def bind_vehicle_model(self, wheelbase_m, max_steer_rad=None, steering_curve=(), *,
                           steering_model=None, rear_axle_actor_xyz_m=None):
        """Bind PDM and execution to the same vehicle, without changing physics."""
        wheelbase = float(wheelbase_m)
        if not math.isfinite(wheelbase) or not 1 < wheelbase < 6:
            raise ValueError("tracker wheelbase must be between 1 and 6 metres")
        model = steering_model if steering_model is not None else SteeringModel(max_steer_rad, steering_curve)
        if not isinstance(model, SteeringModel):
            raise ValueError("tracker requires a SteeringModel")
        if model.wheelbase_m is not None and abs(model.wheelbase_m - wheelbase) > 1e-6:
            raise ValueError("tracker and steering model wheelbases must agree")
        rear = None if rear_axle_actor_xyz_m is None else tuple(float(x) for x in rear_axle_actor_xyz_m)
        if rear is not None and (len(rear) != 3 or not all(math.isfinite(x) for x in rear)
                                 or abs(rear[0]) > 6 or abs(rear[1]) > 3):
            raise ValueError("tracker requires a finite actor-local rear axle point")
        self.pp.L = wheelbase
        self.max_steer_rad = model.max_steer_rad
        self.steering_model = model
        self.rear_axle_actor_xyz_m = rear
        self.vehicle_model_source = "carla_pdm_vehicle"

    def set_trajectory(
        self,
        ego_centric_trajectory: List[dict],
        capture_ego_x: float,
        capture_ego_y: float,
        capture_ego_yaw_rad: float,
        target_speed: float,
        *,
        follow_speed_profile: bool = False,
    ) -> None:
        """设定本周期轨迹；启用速度曲线时每个点必须有有效的时间和非负速度。

        无效曲线抛出 ValueError，并清除上一周期轨迹，避免继续执行过期计划。
        空轨迹始终表示全刹车，不要求速度曲线字段。
        """
        previous_profile = self._speed_profile is not None and bool(self._global_wps)
        self._global_wps = None
        self._actor_global_wps = None
        self._speed_profile = None
        self._profile_elapsed_s = 0.0
        self.pp.reset()
        if not ego_centric_trajectory:
            self.pid.reset()
        elif not (follow_speed_profile and previous_profile):
            self.pid.soft_reset()
        if follow_speed_profile and ego_centric_trajectory:
            if not math.isfinite(self.control_dt) or self.control_dt <= 0:
                raise ValueError("Speed-profile tracking requires a positive finite control dt")
            profile = []
            for index, wp in enumerate(ego_centric_trajectory):
                try:
                    t, speed = float(wp["t"]), float(wp["optional_v"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid speed-profile waypoint {index}: t and optional_v are required") from exc
                if not math.isfinite(t) or not math.isfinite(speed) or t < 0 or speed < 0:
                    raise ValueError(f"Invalid speed-profile waypoint {index}: time and speed must be finite and nonnegative")
                if profile and t <= profile[-1][0]:
                    raise ValueError("Speed-profile waypoint times must be strictly increasing")
                profile.append((t, speed))
            self._speed_profile = profile
        actor_points = coords.trajectory_ego_to_global(
            ego_centric_trajectory, capture_ego_x, capture_ego_y, capture_ego_yaw_rad
        )
        tracking_points = actor_points
        if self.rear_axle_actor_xyz_m is not None:
            tracking_points = []
            rear_x, rear_y, _ = self.rear_axle_actor_xyz_m
            for index, (wp, actor_point) in enumerate(zip(ego_centric_trajectory, actor_points)):
                try:
                    relative_heading = float(wp["optional_heading_rad"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"PDM rear-axle waypoint {index} requires optional_heading_rad") from exc
                if not math.isfinite(relative_heading):
                    raise ValueError(f"PDM rear-axle waypoint {index} heading must be finite")
                # PDM supplies body orientation in the capture frame, left-positive.
                # A trajectory tangent is not an interchangeable body orientation.
                heading = capture_ego_yaw_rad - relative_heading
                c, s = math.cos(heading), math.sin(heading)
                tracking_points.append({**actor_point,
                    "x": actor_point["x"] + c * rear_x - s * rear_y,
                    "y": actor_point["y"] + s * rear_x + c * rear_y})
        self._actor_global_wps = actor_points
        self._global_wps = tracking_points
        self._target_speed = float(target_speed or 0.0)

    def has_trajectory(self) -> bool:
        return bool(self._global_wps)

    def compute_control(
        self, ego_x: float, ego_y: float, ego_yaw_rad: float, current_speed: float
    ) -> "carla.VehicleControl":
        """根据当前自车位姿/速度算本 tick 控制量。无轨迹则全刹。"""
        self.last_control_diagnostics = {
            "profile_acceleration_mps2": None,
            "endpoint_distance_m": None,
            "endpoint_forward_m": None,
        }
        if not self._global_wps:
            return self._full_brake("no_trajectory", current_speed)
        endpoint = self._actor_global_wps[-1]
        self.last_control_diagnostics.update(
            endpoint_distance_m=math.hypot(endpoint["x"] - ego_x, endpoint["y"] - ego_y),
            endpoint_forward_m=coords.global_to_ego(
                endpoint["x"], endpoint["y"], ego_x, ego_y, ego_yaw_rad)[0],
        )
        target_speed = self._target_speed
        changing_profile_speed = False
        if self._speed_profile is not None:
            # This control will be applied during the next simulation tick. Looking
            # one tick ahead lets a trajectory starting at v(0)=0 accelerate.
            previous_target = self._profile_speed_at(self._profile_elapsed_s)
            self._profile_elapsed_s += self.control_dt
            if self._profile_elapsed_s > self._speed_profile[-1][0] + 1e-9:
                return self._full_brake("profile_expired", current_speed)
            target_speed = self._profile_speed_at(self._profile_elapsed_s)
            profile_acceleration = (target_speed - previous_target) / self.control_dt
            changing_profile_speed = abs(profile_acceleration) > 1e-6
            self.last_control_diagnostics["profile_acceleration_mps2"] = profile_acceleration
        planned_stop = target_speed <= 1e-6
        if planned_stop and current_speed <= 0.05:
            # Braking a stationary car holds it. A moving car follows its valid
            # deceleration profile below; zero speed is not itself an emergency.
            return self._full_brake("stationary_hold", current_speed, target_speed)
        tracking_x, tracking_y = ego_x, ego_y
        if self.rear_axle_actor_xyz_m is not None:
            rear_x, rear_y, _ = self.rear_axle_actor_xyz_m
            c, s = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
            tracking_x += c * rear_x - s * rear_y
            tracking_y += s * rear_x + c * rear_y
        steer_rad = self.pp.compute_steer(
            tracking_x, tracking_y, ego_yaw_rad, current_speed, self._global_wps
        )
        if not self.pp.target_valid:
            return self._full_brake("no_forward_target", current_speed, target_speed)
        throttle, brake = self.pid.compute(
            target_speed - current_speed, dt=self.control_dt,
            current_speed=current_speed if self._speed_profile is not None else None,
            # A fixed cruise target can coast inside the noise deadband. A
            # planned gentle acceleration/deceleration must not disappear just
            # because the next-tick speed change is smaller than that deadband.
            allow_coast=not (planned_stop or changing_profile_speed),
        )
        steering_limit = self.steering_model.limit_at(current_speed)
        steer_norm = self.steering_model.bicycle_to_normalized(steer_rad, current_speed) * self.steer_sign
        self.last_control_diagnostics.update(
            requested_wheel_angle_rad=steer_rad,
            steering_limit_at_speed_rad=steering_limit,
            physical_inner_limit_at_speed_rad=self.steering_model.physical_limit_at(current_speed),
            requested_inner_wheel_angle_rad=self.steering_model.bicycle_to_inner(
                max(-steering_limit, min(steering_limit, steer_rad))),
            steering_mapping=self.steering_model.mapping,
            tracking_reference=("mpc_rear_axle" if self.vehicle_model_source == "carla_mpc_vehicle" else "pdm_rear_axle")
                if self.rear_axle_actor_xyz_m is not None else "carla_actor_origin",
            tracking_position_xy_m=[tracking_x, tracking_y],
            wheelbase_m=self.pp.L,
            vehicle_model_source=self.vehicle_model_source,
        )
        return self._control(throttle, steer_norm, brake, current_speed, target_speed,
                             "planned_stop" if planned_stop else "speed_tracking")

    def _full_brake(self, reason: str, current_speed: float, target_speed=None):
        # A hold/failsafe bypasses PID. Discard its state so a later release does
        # not inherit accumulated throttle or a stale acceleration measurement.
        self.pid.reset()
        return self._control(0.0, 0.0, 1.0, current_speed, target_speed, reason)

    def _control(self, throttle, steer, brake, current_speed, target_speed, reason):
        self.last_control_diagnostics = {
            **self.last_control_diagnostics,
            "reason": reason,
            "target_speed_mps": target_speed,
            "current_speed_mps": current_speed,
            "speed_error_mps": None if target_speed is None else target_speed - current_speed,
            "profile_elapsed_s": self._profile_elapsed_s if self._speed_profile is not None else None,
            "pid_integral": self.pid._i,
            "throttle": throttle, "brake": brake, "steer": steer,
        }
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def _profile_speed_at(self, elapsed_s: float) -> float:
        """Linearly interpolate the validated profile at simulation time."""
        profile = self._speed_profile
        index = bisect_right([sample[0] for sample in profile], elapsed_s)
        if index == 0:
            return profile[0][1]
        if index >= len(profile):
            return profile[-1][1]
        t0, v0 = profile[index - 1]
        t1, v1 = profile[index]
        fraction = (elapsed_s - t0) / (t1 - t0)
        return v0 + fraction * (v1 - v0)
