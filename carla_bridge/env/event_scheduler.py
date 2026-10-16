"""长尾事件调度器
================
按 ego 行驶距离触发长尾事件，支持：

* ``pedestrian_crossing`` - 行人过马路（路边 spawn -> 对侧）
* ``pedestrian_roadside`` - 路边行人沿路走
* ``scripted_vehicle`` - NOA+ACC 交通车（路由跟随 + 跟车减速）
* ``obstacle`` - 静态障碍物（barrier/cone）
* ``traffic_light_force`` - 强制前方信号灯状态（red/green/yellow）

事件在场景 YAML 的 ``events`` 列表声明，每个事件 ``at_distance`` = ego 离起点的距离
（米）达到时触发。触发后生成的 scripted 车/行人由本调度器每 tick 驱动。
"""
from __future__ import annotations

from typing import List, Optional

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger
from carla_bridge.control.scripted_vehicle import ScriptedVehicleController
from carla_bridge.env.walker_controller import WalkerController, lateral_offset

logger = get_logger("carla_events")

_TL_STATE = {
    "red": carla.TrafficLightState.Red,
    "green": carla.TrafficLightState.Green,
    "yellow": carla.TrafficLightState.Yellow,
}


class EventScheduler:
    """长尾事件调度器：按 ego 距离触发 + 每 tick 驱动 scripted 车/行人。"""

    def __init__(
        self,
        world,
        ego_vehicle,
        events_cfg: List[dict],
        controller_cfg: dict,
        seed: int = 42,
    ):
        self.world = world
        self.ego = ego_vehicle
        self.controller_cfg = controller_cfg
        self.events: List[dict] = []
        for e in (events_cfg or []):
            ev = dict(e)
            ev["_triggered"] = False
            self.events.append(ev)
        self.scripted_vehicles: List[ScriptedVehicleController] = []
        self.walker_ctrl = WalkerController(world, seed=seed)
        self._static_props: List = []
        self._scripted_actors: List = []
        self._start_loc = ego_vehicle.get_location()

    # ------------------------------------------------------------------

    def tick(self) -> None:
        """每 CARLA 推进 tick 调用：检查触发 + 驱动 scripted 车。"""
        try:
            ego_loc = self.ego.get_location()
        except Exception:
            return
        dist = ego_loc.distance(self._start_loc)
        for ev in self.events:
            if ev["_triggered"]:
                continue
            if dist >= float(ev.get("at_distance", 0)):
                ev["_triggered"] = True
                self._execute(ev)
        for svc in self.scripted_vehicles:
            try:
                svc.tick(self.world)
            except Exception as e:
                logger.debug("scripted vehicle tick: %s", e)

    # ------------------------------------------------------------------

    def _execute(self, ev: dict) -> None:
        etype = ev.get("type")
        try:
            ego_tf = self.ego.get_transform()
        except Exception:
            return
        if etype == "pedestrian_crossing":
            self.walker_ctrl.spawn_crossing(
                ego_tf, ahead_m=float(ev.get("ahead_m", 20)),
                side=ev.get("side", "right"), cross_dist=float(ev.get("cross_dist", 8)),
            )
        elif etype == "pedestrian_roadside":
            self.walker_ctrl.spawn_roadside(
                ego_tf, ahead_m=float(ev.get("ahead_m", 15)),
                side=ev.get("side", "right"), walk_ahead_m=float(ev.get("walk_ahead_m", 30)),
            )
        elif etype == "scripted_vehicle":
            svc = self._spawn_scripted_vehicle(ev, ego_tf)
            if svc is not None:
                self.scripted_vehicles.append(svc)
        elif etype == "obstacle":
            self._spawn_obstacle(ev, ego_tf)
        elif etype == "traffic_light_force":
            self._force_traffic_light(ev, ego_tf)
        else:
            logger.warning("未知事件类型: %s", etype)
            return
        logger.info("[event] %s 触发 @ ego_dist=%.1fm", etype, float(ev.get("at_distance", 0)))

    def _spawn_scripted_vehicle(self, ev: dict, ego_tf) -> Optional[ScriptedVehicleController]:
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find(ev.get("blueprint", "vehicle.audi.etron"))
        ahead_m = float(ev.get("ahead_m", 25))
        side_sign = -1.0 if ev.get("lane", "left") == "left" else 1.0  # CARLA +y=右，left=-y
        fwd = ego_tf.get_forward_vector()
        ldx, ldy = lateral_offset(ego_tf, side_sign, 3.5)
        spawn_loc = carla.Location(
            x=ego_tf.location.x + fwd.x * ahead_m + ldx,
            y=ego_tf.location.y + fwd.y * ahead_m + ldy,
            z=ego_tf.location.z + 1.0,
        )
        spawn_tf = carla.Transform(spawn_loc, ego_tf.rotation)
        veh = self.world.try_spawn_actor(bp, spawn_tf)
        if veh is None:
            logger.warning("scripted 车 spawn 碰撞，跳过")
            return None
        self._scripted_actors.append(veh)
        # NOA 路由：沿 spawn 点所在车道向前走（在路上）
        wp = self.world.get_map().get_waypoint(spawn_loc, project_to_road=True)
        wps: List[carla.Location] = []
        step = 3.0
        length = float(ev.get("route_len_m", 150))
        traveled = 0.0
        cur = wp
        while cur is not None and traveled < length:
            wps.append(cur.transform.location)
            nxt = cur.next(step)
            if not nxt:
                break
            cur = nxt[0]
            traveled += step
        if not wps:
            wps = [spawn_loc]
        return ScriptedVehicleController(
            veh, wps, float(ev.get("speed", 8.0)),
            self.controller_cfg, acc_enabled=bool(ev.get("acc", True)),
            dt=float(self.controller_cfg.get("control_dt_s", 0.1)),
        )

    def _spawn_obstacle(self, ev: dict, ego_tf) -> None:
        prop = ev.get("prop", "static.prop.barrier01")
        try:
            bp = self.world.get_blueprint_library().find(prop)
        except Exception:
            logger.warning("未知障碍物蓝图: %s", prop)
            return
        ahead_m = float(ev.get("ahead_m", 20))
        fwd = ego_tf.get_forward_vector()
        loc = carla.Location(
            x=ego_tf.location.x + fwd.x * ahead_m,
            y=ego_tf.location.y + fwd.y * ahead_m,
            z=ego_tf.location.z,
        )
        try:
            self._static_props.append(self.world.spawn_actor(bp, carla.Transform(loc)))
        except Exception as e:
            logger.warning("障碍物 spawn 失败: %s", e)

    def _force_traffic_light(self, ev: dict, ego_tf) -> None:
        state_str = ev.get("state", "red")
        state = _TL_STATE.get(state_str, carla.TrafficLightState.Red)
        ahead_m = float(ev.get("ahead_m", 30))
        fwd = ego_tf.get_forward_vector()
        target = carla.Location(
            x=ego_tf.location.x + fwd.x * ahead_m,
            y=ego_tf.location.y + fwd.y * ahead_m,
            z=ego_tf.location.z,
        )
        best, best_d = None, float("inf")
        try:
            lights = self.world.get_actors().filter("traffic.traffic_light*")
        except Exception:
            return
        for tl in lights:
            try:
                d = tl.get_location().distance(target)
            except Exception:
                continue
            if d < best_d and d < 30.0:
                best, best_d = tl, d
        if best is not None:
            best.set_state(state)
            dur = float(ev.get("duration", 20))
            if state_str == "red":
                best.set_red_time(dur)
            elif state_str == "green":
                best.set_green_time(dur)
            elif state_str == "yellow":
                best.set_yellow_time(dur)

    # ------------------------------------------------------------------

    def destroy(self) -> None:
        try:
            self.walker_ctrl.destroy()
        except Exception as exc:
            logger.warning("walker cleanup: %s", exc)
        for actor in self._scripted_actors + self._static_props:
            try:
                actor.destroy()
            except Exception as exc:
                logger.warning("event actor cleanup: %s", exc)
        self.scripted_vehicles.clear()
        self._scripted_actors.clear()
        self._static_props.clear()
