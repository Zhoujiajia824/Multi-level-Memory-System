"""自车历史位姿缓冲
==================
滚动保存过去 N 秒的自车全局位姿，按捕获时刻用 ``coords`` 转成 ego-centric 历史
轨迹（``{t, x, y}``，t 为负，y=左）。

⚠️ **不复用** ``src.vla_memory.data.trajectory_builder.TrajectoryBuilder``：它的
ego_y 实为"右"，与项目决策轨迹约定（y=左）手性不一致。此处直接用 ``coords``
生成，保证历史轨迹 / 感知对象 / 决策轨迹三者都在 y=左 系下。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

from carla_bridge.state import coords


class HistoryBuffer:
    """滚动过去 N 秒自车全局位姿 ``(elapsed_s, x, y, yaw_rad)``。"""

    def __init__(self, history_seconds: float = 5.0, max_len: int = 400):
        self.history_seconds = history_seconds
        self._buf: Deque[Tuple[float, float, float, float]] = deque(maxlen=max_len)

    def update(self, ego_vehicle, elapsed_sim_s: float) -> None:
        """记录当前帧自车全局位姿。每个控制 tick 调一次。"""
        tf = ego_vehicle.get_transform()
        yaw_rad = coords.carla_yaw_deg_to_rad(tf.rotation.yaw)
        self._buf.append((elapsed_sim_s, tf.location.x, tf.location.y, yaw_rad))

    def build_ego_centric(
        self,
        current_elapsed_s: float,
        current_x: float,
        current_y: float,
        current_yaw_rad: float,
    ) -> List[dict]:
        """构造 ego-centric 历史轨迹 ``[{t, x, y}, ...]``（t 为负，y=左）。"""
        pts: List[dict] = []
        for (el, x, y, yaw) in self._buf:
            dt = el - current_elapsed_s  # 负值（过去）
            if dt > 0:
                continue
            if -dt > self.history_seconds:
                continue
            fwd, left = coords.global_to_ego(x, y, current_x, current_y, current_yaw_rad)
            pts.append({"t": round(dt, 4), "x": round(fwd, 4), "y": round(left, 4)})
        pts.sort(key=lambda p: p["t"])  # t 升序（最负在前）
        return pts

    def yaw_rate(self, current_elapsed_s: float = None, current_yaw_rad: float = None) -> float:
        """用缓冲区最近两个位姿差分估偏航角速率（rad/s）。不足两帧返回 0.0。

        注意：调用方通常先 ``update()`` 压入当前位姿再调用，故用 ``_buf[-1]`` 与
        ``_buf[-2]`` 差分；传入的 current_yaw_rad 恒等于 ``_buf[-1]`` 的 yaw，忽略。
        CARLA 侧 yaw/yaw_rate 为右正约定（+yaw=右转），与 nuScenes 左正相反；
        因 CARLA 用独立记忆库（memory_db_carla）且检索只在同源状态间比较，无实际影响。
        """
        if len(self._buf) < 2:
            return 0.0
        el, _, _, yaw = self._buf[-1]
        pel, _, _, pyaw = self._buf[-2]
        dt = el - pel
        if dt <= 0:
            return 0.0
        diff = yaw - pyaw
        # 归一化到 [-pi, pi]
        while diff > 3.141592653589793:
            diff -= 2 * 3.141592653589793
        while diff < -3.141592653589793:
            diff += 2 * 3.141592653589793
        return diff / dt
