"""CARLA控制序列的源时标消费；Python 3.7标准库实现。"""
import math


class ControlTimeline:
    """只执行已经规划的控制；耗尽后显式报告，不无限保持末项。"""
    def __init__(self, acceleration_min=-6., acceleration_max=3., steering_max=.6):
        values = (acceleration_min, acceleration_max, steering_max)
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in values)
                or acceleration_min >= acceleration_max or steering_max <= 0):
            raise ValueError('控制物理边界必须为有限数字，最小加速度小于最大值且转角界为正')
        self.minimum = float(acceleration_min)
        self.maximum = float(acceleration_max)
        self.steering_max = float(steering_max)
        self.controls = []
        self.source_time = None
        self.step_s = None
        self.revision = 0
        self.endpoint_speeds = []

    def install(self, controls, source_time, step_s, source_speed_mps=None):
        if (isinstance(source_time, bool) or not isinstance(source_time, (int, float))
                or not math.isfinite(source_time) or source_time < -1e-5):
            raise ValueError('控制序列需要有限非负源仿真时间')
        if (isinstance(step_s, bool) or not isinstance(step_s, (int, float))
                or not math.isfinite(step_s) or step_s <= 0):
            raise ValueError('控制序列采样间隔必须为有限正数')
        if not isinstance(controls, (list, tuple)) or not controls:
            raise ValueError('控制序列不能为空')
        checked = []
        for control in controls:
            if (not isinstance(control, (list, tuple)) or len(control) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in control)):
                raise ValueError('控制序列必须由有限的加速度和转角组成')
            acceleration, steering = map(float, control)
            if not self.minimum <= acceleration <= self.maximum or abs(steering) > self.steering_max + 1e-8:
                raise ValueError('控制序列超出规划物理边界')
            checked.append([acceleration, steering])
        if self.source_time is not None and source_time < self.source_time - 1e-8:
            raise ValueError('旧源帧控制不能覆盖新控制序列')
        endpoints = []
        if source_speed_mps is not None:
            if (type(source_speed_mps) not in (int, float) or
                    not math.isfinite(source_speed_mps) or source_speed_mps < 0):
                raise ValueError('控制序列的源帧速度必须为有限非负值')
            speed = float(source_speed_mps)
            for acceleration, _ in checked:
                # 单向车辆在零速处停止；与规划动力学的非负速度语义一致。
                speed = max(0., speed + acceleration * float(step_s))
                endpoints.append(speed)
        self.controls = checked
        self.endpoint_speeds = endpoints
        self.source_time = float(source_time)
        self.step_s = float(step_s)
        self.revision += 1

    def at(self, sim_time):
        if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time):
            raise ValueError('控制消费需要真实有限仿真时间')
        if self.source_time is None:
            return {'available': False, 'reason': 'no_plan', 'index': None, 'control': None}
        age = float(sim_time) - self.source_time
        if age < -1e-6:
            return {'available': False, 'reason': 'future_source', 'index': None, 'control': None, 'age_s': age}
        index = int(math.floor((max(0., age) + 1e-8) / self.step_s))
        if index >= len(self.controls):
            return {'available': False, 'reason': 'sequence_exhausted', 'index': index,
                    'control': None, 'age_s': age, 'revision': self.revision}
        return {'available': True, 'reason': None, 'index': index,
                'control': list(self.controls[index]), 'age_s': age,
                'target_speed_mps': self.endpoint_speeds[index] if self.endpoint_speeds else None,
                'revision': self.revision, 'remaining_s': len(self.controls)*self.step_s-max(0., age)}
