"""场景管理器
============
加载场景 YAML：设置地图 -> 天气 -> 生成 ego -> 全局路由规划(A*) -> 生成交通流；提供
统一清理。对应"YAML + Python 场景类"方案：标准场景一个 YAML 搞定，复杂自定义事件
可继承 :class:`ScenarioManager` 覆写钩子。

路由：``route.start`` + ``route.end`` 用 A* 全局规划（GlobalRoutePlanner），模拟地图
导航；``end`` 为空时回退沿路前进。起点/终点支持 ``{x,y}`` 或 ``{spawn_point: N}``（地图
预定义生成点，保证在路上）。
"""
from __future__ import annotations

from typing import Optional
import hashlib
import json
from pathlib import Path

import carla
import yaml

from src.vla_memory.common.logging_utils import get_logger
from carla_bridge.env.weather_controller import WeatherController
from carla_bridge.env.traffic_spawner import TrafficSpawner
from carla_bridge.env.route_planner import RoutePlanner

logger = get_logger("carla_scenario")


class ScenarioManager:
    """场景生命周期管理。

    Args:
        carla_client: :class:`CarlaClient`（用于 ``load_map`` / 取 client）。
        carla_cfg: ``carla.yaml -> carla`` 配置块（含 ``python_api_dir`` 等）。
        scenario_cfg: 场景 YAML 解析后的 dict。
    """

    def __init__(self, carla_client, carla_cfg: dict, scenario_cfg: dict):
        self.carla_client = carla_client
        self.client = carla_client.client
        self.carla_cfg = carla_cfg
        self.cfg = scenario_cfg
        self.world = None
        self.ego_vehicle = None
        self.traffic: Optional[TrafficSpawner] = None
        self.route: Optional[RoutePlanner] = None
        self.paper = (scenario_cfg.get("experiment") or {}).get("suite") == "paper_v1"
        self.experiment_snapshot = None

    @classmethod
    def from_yaml(cls, carla_client, carla_cfg: dict, yaml_path: str) -> "ScenarioManager":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(carla_client, carla_cfg, cfg)

    def setup(self) -> "ScenarioManager":
        cfg = self.cfg
        if self.paper:
            from carla_bridge.paper_suite import validate_prepared
            validate_prepared(cfg)
        # 地图（load_map 会重应用同步设置）
        map_name = cfg.get("map")
        if map_name:
            self.world = self.carla_client.load_map(map_name)
        else:
            self.world = self.carla_client.get_world()
        if self.paper:
            expected = cfg["experiment"]
            actual_hash = hashlib.sha256(self.world.get_map().to_opendrive().encode()).hexdigest()
            if actual_hash != expected["map_sha256"]:
                raise ValueError("map OpenDRIVE differs from prepared experiment")
            if self.client.get_server_version() != expected["server_version"]:
                raise ValueError("server version differs from prepared experiment")
        # 【关键】同步模式下 load_map 后必须 tick：让地图与 Weather actor 完成初始化。
        # 否则后续 spawn 相机时，相机 BeginPlay 访问 Weather 的 post-process 会触发引擎崩溃
        # （Pure virtual function called / access violation，CARLA 0.9.15 在 Town05 等大图高发，
        #  Town04 负载低侥幸不触发）。见 carla issue #8554。
        for _ in range(2):
            self.carla_client.tick()
        # 天气
        WeatherController(self.world).apply(cfg.get("weather", {}))
        # apply 天气后 tick 一帧，让 post-process effects 生效，防相机初始化时 CheckWeatherPostProcessEffects 崩
        self.carla_client.tick()
        # 路由配置（先解析，ego 可能要 spawn 在路由起点）
        route_cfg = cfg.get("route", {}) or {}
        start_loc = self._loc_from_cfg(route_cfg.get("start"))
        end_loc = self._loc_from_cfg(route_cfg.get("end"))
        # ego（ego.spawn 优先；未指定但 route.start 给了 -> spawn 在路由起点）
        self.ego_vehicle, spawn_tf = self._spawn_ego(start_loc)
        # ego spawn 后 tick 让其完成 BeginPlay（同步模式 spawn 后未 tick 读 location 是 (0,0)）
        self.carla_client.tick()
        # ego 若因碰撞回退到其他 spawn 点，路由起点改用 spawn transform 位置
        # （不读 ego.get_location()：同步模式 spawn 后未 tick 会返回 (0,0) 原点）
        if start_loc is not None:
            if spawn_tf.location.distance(start_loc) > 5.0:
                logger.warning(
                    "ego 碰撞回退，路由起点改用 spawn 位置: (%.1f,%.1f)",
                    spawn_tf.location.x, spawn_tf.location.y,
                )
                start_loc = spawn_tf.location
        # 全局路由规划：A* start->end（end 为空回退沿路前进）
        self.route = RoutePlanner(
            self.world, self.ego_vehicle,
            python_api_dir=self.carla_cfg.get("python_api_dir"),
            completion_tolerance_m=float(self.carla_cfg.get("route_completion_tolerance_m", 2.0)),
            tracking_tolerance_m=float(self.carla_cfg.get("route_tracking_tolerance_m", 5.0)),
        )
        if self.paper:
            self.route.load_fixed_points(route_cfg["points"])
            if not self.route._progress.started:
                raise ValueError("ego did not spawn near the prepared route start")
            if self.world.get_weather().sun_altitude_angle < 0:
                self.ego_vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        else:
            self.route.plan(
                start_loc=start_loc,
                end_loc=end_loc,
                sampling_resolution=float(route_cfg.get("sampling_resolution", 2.0)),
                length_m=float(route_cfg.get("length_m", 200.0)),
                turns=route_cfg.get("turns", []),
            )
        # 交通流
        self.traffic = TrafficSpawner(
            self.world, self.client, self.ego_vehicle, seed=int(cfg.get("seed", 42))
        )
        tcfg = cfg.get("traffic", {}) or {}
        if self.paper:
            self.traffic.configure_reproducibility(int(cfg["seed"]), [
                carla.Location(x=p["x"], y=p["y"], z=p["z"]) for p in cfg.get("reserved_event_sites", [])])
        spawned_vehicles = spawned_walkers = 0
        if int(tcfg.get("num_vehicles", 0)) > 0:
            spawned_vehicles = self.traffic.spawn_vehicles(int(tcfg["num_vehicles"]), radius=float(tcfg.get("vehicle_radius_m", 80)))
        if int(tcfg.get("num_walkers", 0)) > 0:
            spawned_walkers = self.traffic.spawn_walkers(int(tcfg["num_walkers"]), radius=float(tcfg.get("walker_radius_m", 60)))
        if self.paper:
            from carla_bridge.paper_suite import digest
            self.experiment_snapshot = {
                "experiment": cfg["experiment"], "case_config_sha256": digest(cfg), "seed": cfg["seed"],
                "map": self.world.get_map().name,
                "route_length_m": self.route._progress.lengths[-1],
                "spawned_vehicles": spawned_vehicles, "spawned_walkers": spawned_walkers,
                "requested_vehicles": tcfg["num_vehicles"], "requested_walkers": tcfg["num_walkers"],
                "initial_actors": self.traffic.spawn_manifest,
                "weather": {key: getattr(self.world.get_weather(), key) for key in (
                    "cloudiness", "precipitation", "precipitation_deposits", "sun_altitude_angle",
                    "sun_azimuth_angle", "fog_density", "fog_distance", "wetness")},
            }
            output = self.carla_cfg.get("experiment_output_dir")
            if output:
                Path(output).mkdir(parents=True, exist_ok=True)
                (Path(output) / "experiment_setup.json").write_text(
                    json.dumps(self.experiment_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            if spawned_vehicles != tcfg["num_vehicles"] or spawned_walkers != tcfg["num_walkers"]:
                raise RuntimeError("paper case invalid: actual traffic counts differ from requested counts; see experiment_setup.json")
        logger.info(
            "场景就绪: %s (地图=%s, ego=%s)",
            cfg.get("scenario_name"), map_name, self.ego_vehicle.id,
        )
        return self

    # ------------------------------------------------------------------

    def _loc_from_cfg(self, loc_cfg) -> Optional[carla.Location]:
        """从 YAML 解析一个位置：``{x,y,z}`` 或 ``{spawn_point: N}``（地图生成点）。"""
        if not loc_cfg:
            return None
        if loc_cfg.get("spawn_point") is not None:
            sp = self.world.get_map().get_spawn_points()
            idx = int(loc_cfg["spawn_point"])
            if 0 <= idx < len(sp):
                return sp[idx].location
            logger.warning("spawn_point 索引 %d 越界（共 %d 个）", idx, len(sp))
            return None
        if loc_cfg.get("x") is not None:
            return carla.Location(
                x=float(loc_cfg["x"]),
                y=float(loc_cfg["y"]),
                z=float(loc_cfg.get("z", 0.0)),
            )
        return None

    def _spawn_ego(self, start_loc=None):
        bp_lib = self.world.get_blueprint_library()
        bp_name = self.carla_cfg.get("ego", {}).get(
            "blueprint", (self.cfg.get("ego") or {}).get("blueprint", "vehicle.tesla.model3")
        )
        if self.paper:
            bp_name = self.cfg["ego"]["blueprint"]
        bp = bp_lib.find(bp_name)
        spawn_cfg = (self.cfg.get("ego", {}) or {}).get("spawn", {}) or {}
        # 候选 spawn transform：优先目标点，再备选所有 spawn_points（碰撞时回退）
        candidates = []
        if spawn_cfg.get("x") is not None:
            loc = carla.Location(
                x=float(spawn_cfg["x"]),
                y=float(spawn_cfg["y"]),
                z=float(spawn_cfg.get("z", 1.0)),
            )
            candidates.append(carla.Transform(loc, carla.Rotation(yaw=float(spawn_cfg.get("yaw", 0.0)))))
        elif start_loc is not None:
            wp = self.world.get_map().get_waypoint(start_loc, project_to_road=True)
            candidates.append(wp.transform if wp is not None else carla.Transform(start_loc))
        # 追加所有地图 spawn_points 作备选（目标点碰撞时自动找空位）
        if not self.paper:
            candidates.extend(self.world.get_map().get_spawn_points())
        # try_spawn_actor：碰撞返回 None 而非抛异常；逐个候选直到成功
        for tf in candidates:
            ego = self.world.try_spawn_actor(bp, tf)
            if ego is not None:
                return ego, tf
        raise RuntimeError(
            "ego 生成失败：所有候选 spawn 点都碰撞。请重启 CARLA 服务器清场后重试，"
            "或在场景 YAML 改 route.start 的 spawn_point 索引 / ego.spawn 坐标。"
        )

    # ------------------------------------------------------------------

    def duration_s(self) -> float:
        return float(self.cfg.get("duration_s", 300.0))

    def scenario_name(self) -> str:
        return self.cfg.get("scenario_name", "carla_scenario")

    def destroy(self) -> bool:
        """报告清理是否全部完成；单项失败不阻断其他资源，失败引用保留。"""
        traffic_ok = True
        if self.traffic is not None:
            try:
                traffic_ok = self.traffic.destroy() is not False
            except Exception as exc:
                traffic_ok = False
                logger.warning("traffic cleanup: %s", exc)
        if self.ego_vehicle is not None:
            try:
                if not self.ego_vehicle.is_alive or self.ego_vehicle.destroy() is not False:
                    self.ego_vehicle = None
                else:
                    logger.warning("ego cleanup: destroy 返回 False，将保留引用供重试")
            except Exception as exc:
                logger.warning("ego cleanup: %s", exc)
        return traffic_ok and self.ego_vehicle is None
