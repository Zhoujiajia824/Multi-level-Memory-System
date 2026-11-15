"""CARLA 图像消费者策略；兼容 SDK 的 Python 3.7，仅依赖标准库。

本模块只决定应创建哪些相机，不导入 CARLA，也不读取环境、图像或密钥。
RGB采样/观看10Hz，原始图片归档1Hz；两者均独立于规划频率。
"""


SENSOR_POLICY_VERSION = 'carla-rgb-consumers-2.0'
RGB_HZ = 10.0
RGB_VIEW_NAMES = ('front', 'front_left', 'front_right',
                  'back', 'back_left', 'back_right')


def sensor_policy(fast_enabled=False, reflection_enabled=False, video=False, live=False):
    """按明确图像消费者生成相机创建策略；所有开关必须是真正的布尔值。

    ``rgb_hz`` 和 ``sensor_tick_s`` 是启用相机后的配置值；关闭时由
    ``rgb_enabled=False``、空视角集合及零相机数量明确表示不创建相机。
    录像或显示是独立消费者，不会因为两个模型关闭而被静默禁用。
    """
    switches = (('fast_enabled', fast_enabled),
                ('reflection_enabled', reflection_enabled),
                ('video', video), ('live', live))
    for name, value in switches:
        if type(value) is not bool:
            raise ValueError(name + ' 必须为布尔值，不能使用字符串或数字代替')

    consumers = [name for name, enabled in switches if enabled]
    enabled = bool(consumers)
    return {
        'version': SENSOR_POLICY_VERSION,
        'rgb_enabled': enabled,
        'depth_enabled': False,
        'rgb_hz': RGB_HZ,
        'sensor_tick_s': 1.0 / RGB_HZ,
        'rgb_view_names': list(RGB_VIEW_NAMES) if enabled else [],
        'rgb_sensor_count': len(RGB_VIEW_NAMES) if enabled else 0,
        'depth_sensor_count': 0,
        'consumers': consumers,
        'reason': ('存在明确图像消费者，创建六路10Hz RGB相机、原帧1Hz归档；不创建深度相机'
                   if enabled else '没有模型、录像或显示图像消费者；不创建RGB或深度相机'),
    }
