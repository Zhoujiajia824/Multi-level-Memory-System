"""Source-frame control timeline and signed actuation, Python 3.10 standard library.
Extracted from committed CARLA execution; official evaluator owns world ticks.
"""
import math


def planning_previous_control(previous_execution, initial_steering_rad=0.):
    """MPC jerk applies to planned acceleration, never the actuator's feedback effort.

    Steering uses the previously returned, rate-limited wheel angle. Both values
    describe the last applied control segment rather than a new measured state.
    """
    if previous_execution is None:
        acceleration, steering = 0., initial_steering_rad
    else:
        acceleration = previous_execution['planned_acceleration_mps2']
        steering = previous_execution['steering_rad_left']
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (acceleration, steering)):
        raise ValueError('规划连续性锚点必须是有限的前次规划加速度与已返回转角')
    return [float(acceleration), float(steering)]


class ControlTimeline:
    """只执行已经规划的控制；耗尽后显式报告，不无限保持末项。"""
    def __init__(self, acceleration_min=-6., acceleration_max=3., steering_max=.6, physics_hz=None):
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
        self.source_speed_mps = None
        if physics_hz is not None and (type(physics_hz) is not int or physics_hz <= 0):
            raise ValueError('控制时钟物理频率必须为正整数')
        self.physics_hz = physics_hz
        self.source_tick = None
        self.step_ticks = None
        self.emergency_braking = False

    def install(self, controls, source_time, step_s, source_speed_mps=None, source_tick=None,
                emergency_braking=False):
        if type(emergency_braking) is not bool:
            raise ValueError('紧急制动序列标志必须为布尔值')
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
            if emergency_braking and acceleration > 1e-8:
                raise ValueError('紧急制动序列不能包含正加速度')
            if not self.minimum <= acceleration <= self.maximum or abs(steering) > self.steering_max + 1e-8:
                raise ValueError('控制序列超出规划物理边界')
            checked.append([acceleration, steering])
        if self.source_time is not None and source_time < self.source_time - 1e-8:
            raise ValueError('旧源帧控制不能覆盖新控制序列')
        step_ticks = None
        if self.physics_hz is not None:
            if type(source_tick) is not int or source_tick < 0:
                raise ValueError('在线控制必须绑定非负整数源物理帧')
            ticks = step_s * self.physics_hz
            if abs(ticks-round(ticks)) > 1e-8 or ticks < 1:
                raise ValueError('控制段必须覆盖完整物理帧')
            step_ticks = int(round(ticks))
            if self.source_tick is not None and source_tick < self.source_tick:
                raise ValueError('旧源物理帧不能覆盖新控制序列')
        elif source_tick is not None:
            raise ValueError('源物理帧需要同时配置物理频率')
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
        self.source_speed_mps = None if source_speed_mps is None else float(source_speed_mps)
        self.endpoint_speeds = endpoints
        self.source_time = float(source_time)
        self.step_s = float(step_s)
        self.source_tick = source_tick
        self.step_ticks = step_ticks
        self.emergency_braking = emergency_braking
        self.revision += 1

    def at(self, sim_time, physical_tick=None):
        if isinstance(sim_time, bool) or not isinstance(sim_time, (int, float)) or not math.isfinite(sim_time):
            raise ValueError('控制消费需要真实有限仿真时间')
        if self.source_time is None:
            return {'available': False, 'reason': 'no_plan', 'index': None, 'control': None}
        raw_age = float(sim_time) - self.source_time
        age = raw_age
        age_ticks = None
        if self.physics_hz is not None:
            if type(physical_tick) is not int or physical_tick < 0:
                raise ValueError('在线控制消费必须提供整数物理帧')
            age_ticks = physical_tick-self.source_tick
            age = age_ticks / float(self.physics_hz)
        if age < -1e-6:
            return {'available': False, 'reason': 'future_source', 'index': None, 'control': None, 'age_s': age}
        # 秒数接口仅用于旧离线记录；在线执行完全由整数帧决定控制段。
        index = (age_ticks // self.step_ticks if age_ticks is not None else
                 int(math.floor((max(0., age) + 1e-8) / self.step_s)))
        if index >= len(self.controls):
            return {'available': False, 'reason': 'sequence_exhausted', 'index': index,
                    'control': None, 'age_s': age, 'revision': self.revision}
        return {'available': True, 'reason': None, 'index': index,
                'control': list(self.controls[index]), 'age_s': age,
                'raw_age_s': raw_age, 'source_tick': self.source_tick,
                'physical_tick': physical_tick,
                'remaining_segment_s': (((index+1)*self.step_ticks-age_ticks)/float(self.physics_hz)
                    if age_ticks is not None else (index+1)*self.step_s-age),
                'target_speed_mps': self.endpoint_speeds[index] if self.endpoint_speeds else None,
                'segment_start_speed_mps': (self.source_speed_mps if index == 0 else
                    self.endpoint_speeds[index-1]) if self.endpoint_speeds else None,
                'revision': self.revision, 'remaining_s': len(self.controls)*self.step_s-max(0., age)}


class SignedAccelerationController:
    """20Hz有符号加速度前馈和有界反馈；状态跨计划保留，不积累虚构速度目标。

    throttle/brake是执行器归一化输入，前馈增益仍须真实CARLA闭环验收。
    不依赖Ackermann内部将速度误差两次积分后才改变踏板的响应。
    """
    VERSION = 'signed-acceleration-feedforward-1.1'

    def __init__(self, maximum_steering_rad, dt=.02, acceleration_min=-6.,
                 acceleration_max=3., steering_rate=.8, settings=None):
        settings = dict(settings or {})
        defaults = {'throttle_full_acceleration_mps2': 6., 'brake_full_deceleration_mps2': 8.,
                    'speed_kp': 1.5, 'acceleration_kp': .2, 'acceleration_ki': .4,
                    'integral_limit_mps2': 1., 'stop_hold_brake': .5,
                    'stop_speed_mps': .1, 'intent_deadband_mps2': .05}
        if set(settings)-set(defaults):
            raise ValueError('未知纵向执行参数：'+','.join(sorted(set(settings)-set(defaults))))
        defaults.update(settings)
        for name, value in defaults.items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('纵向执行参数必须为有限非负数字：'+name)
        if (defaults['throttle_full_acceleration_mps2'] <= 0 or
                defaults['brake_full_deceleration_mps2'] <= 0 or
                not 0 < defaults['stop_hold_brake'] <= 1):
            raise ValueError('执行器前馈增益及停车保持参数非法')
        values = (maximum_steering_rad, dt, acceleration_min, acceleration_max, steering_rate)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values) or
                maximum_steering_rad <= 0 or dt <= 0 or acceleration_min >= 0 or
                acceleration_max <= 0 or steering_rate <= 0):
            raise ValueError('执行控制器物理参数非法')
        self.settings = defaults
        self.maximum_steering = float(maximum_steering_rad)
        self.dt, self.minimum, self.maximum = float(dt), float(acceleration_min), float(acceleration_max)
        self.steering_rate = float(steering_rate)
        self.integral = 0.
        self.previous_intent = 0
        self.last_tick = None
        self.last_result = None
        self._before_step = None
        self._last_inputs = None
        self.steering = None

    def step(self, tick, acceleration, steering, target_speed, measured_speed,
             measured_acceleration, actual_steering_rad=0., emergency_braking=False):
        if type(tick) is not int or tick < 0:
            raise ValueError('执行反馈需要非负整数物理帧')
        numeric = (acceleration, steering, target_speed, measured_speed,
                   measured_acceleration, actual_steering_rad)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in numeric):
            raise ValueError('执行反馈必须为有限物理量')
        if type(emergency_braking) is not bool:
            raise ValueError('紧急制动执行标志必须为布尔值')
        if emergency_braking and acceleration > 1e-8:
            raise ValueError('紧急制动计划包含正加速度，禁止将其作为制动执行')
        values = numeric + (emergency_braking,)
        if self.last_tick is not None:
            if tick < self.last_tick:
                raise ValueError('执行反馈物理帧倒退')
            if tick == self.last_tick:
                if values == self._last_inputs:
                    return dict(self.last_result)
                # 异步等待期间同帧接纳新计划：可替换本帧命令，但反馈积分只算一次。
                self.integral, self.previous_intent, self.steering = self._before_step
            elif tick != self.last_tick+1:
                raise ValueError('执行反馈不能漏过物理帧')
        if tick != self.last_tick:
            self._before_step = (self.integral, self.previous_intent, self.steering)
        cfg = self.settings
        acceleration = max(self.minimum, min(self.maximum, acceleration))
        speed_error = max(0., target_speed)-max(0., measured_speed)
        acceleration_error = acceleration-measured_acceleration
        deadband = cfg['intent_deadband_mps2']
        intent = 1 if acceleration > deadband else -1 if acceleration < -deadband else 0
        # 制动/起步切换不继承相反方向的积分，不限制真实物理减速响应。
        if intent and intent != self.previous_intent:
            self.integral = 0.
        proposed_integral = max(-cfg['integral_limit_mps2'], min(cfg['integral_limit_mps2'],
            self.integral+cfg['acceleration_ki']*acceleration_error*self.dt))
        without_integral = acceleration+cfg['speed_kp']*speed_error+cfg['acceleration_kp']*acceleration_error
        unsaturated = without_integral+proposed_integral
        # 到达加速度限幅时，只允许能解除饱和的积分变化。
        if not ((unsaturated > self.maximum and acceleration_error > 0) or
                (unsaturated < self.minimum and acceleration_error < 0)):
            self.integral = proposed_integral
        requested_acceleration = max(self.minimum, min(self.maximum, without_integral+self.integral))
        if emergency_braking:
            # A zero/release segment of an emergency stop must not acquire
            # throttle from velocity feedback or an earlier positive integral.
            requested_acceleration = min(0., requested_acceleration)
            self.integral = min(0., self.integral)
        if intent < 0:
            # 有明确制动意图时不能由滞后的正速度误差或积分生成油门。
            # 仍允许反馈减弱过强制动；下界只保留明确的制动方向，不能反向给油。
            requested_acceleration = min(requested_acceleration, .25*acceleration)
        hold = (intent <= 0 and target_speed <= cfg['stop_speed_mps'] and
                abs(measured_speed) <= cfg['stop_speed_mps'])
        throttle = max(0., min(1., requested_acceleration/cfg['throttle_full_acceleration_mps2']))
        brake = max(0., min(1., -requested_acceleration/cfg['brake_full_deceleration_mps2']))
        if hold:
            throttle, brake, self.integral = 0., max(brake, cfg['stop_hold_brake']), 0.
        elif intent < 0 or emergency_braking:
            throttle = 0.
        # 与CARLA Ackermann相同的最大轮转角归一化；保留原左正坐标约定。
        previous_steering = actual_steering_rad if self.steering is None else self.steering
        change = self.steering_rate*self.dt
        self.steering = max(-self.maximum_steering, min(self.maximum_steering,
            max(previous_steering-change, min(previous_steering+change, steering))))
        result = {'controller': self.VERSION, 'control_period_s': self.dt,
            'planned_acceleration_mps2': acceleration, 'planned_steering_rad_left': steering,
            'target_speed_mps': max(0., target_speed), 'measured_speed_mps': measured_speed,
            'measured_longitudinal_acceleration_mps2': measured_acceleration,
            'speed_error_mps': speed_error, 'acceleration_error_mps2': acceleration_error,
            'integral_mps2': self.integral, 'requested_acceleration_mps2': requested_acceleration,
            'emergency_braking': emergency_braking,
            'continuity_acceleration_source': 'planned_acceleration_mps2',
            'throttle': throttle, 'brake': brake, 'stop_hold': hold,
            'steer_normalized_right': -self.steering/self.maximum_steering,
            'steering_rad_left': self.steering, 'maximum_steering_rad': self.maximum_steering}
        self.previous_intent, self.last_tick, self.last_result = intent, tick, result
        self._last_inputs = values
        return dict(result)
