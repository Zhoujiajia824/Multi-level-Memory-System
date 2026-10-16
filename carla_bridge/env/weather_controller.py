"""CARLA 天气控制
================
按场景 YAML 设置天气。**动态**枚举 ``carla.WeatherParameters`` 预设，兼容不同
CARLA 版本（0.9.15 的预设名如 ``MidRainyNoon`` / ``MidRainSunset`` 不统一，硬编码
易踩 ``AttributeError``），另加几个友好别名。也支持自定义参数。
"""
from __future__ import annotations

import carla  # 须在 mulmem_carla(3.9) 运行

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("carla_weather")


def _build_presets() -> dict:
    """动态收集当前 CARLA 版本可用的天气预设 + 友好别名。"""
    names = {
        x for x in dir(carla.WeatherParameters)
        if not x.startswith("_") and x[:1].isupper()
    }
    presets = {n: getattr(carla.WeatherParameters, n) for n in names}
    # 友好别名 -> 真实预设（仅当真实预设存在时才加）
    aliases = {
        "Night": "ClearNight",
        "RainNoon": "MidRainyNoon",
        "RainSunset": "MidRainSunset",
        "RainNight": "MidRainyNight",
        "Fog": "CloudyNoon",
    }
    for alias, real in aliases.items():
        if real in presets:
            presets[alias] = presets[real]
    return presets


_PRESETS = _build_presets()


class WeatherController:
    """设置 CARLA 天气。"""

    def __init__(self, world):
        self.world = world

    def apply(self, weather_cfg: dict) -> None:
        if not weather_cfg:
            return
        preset = weather_cfg.get("preset")
        if preset and preset in _PRESETS:
            self.world.set_weather(_PRESETS[preset])
            return
        if preset and preset not in _PRESETS:
            logger.warning(
                "未知天气预设 '%s'，可选: %s；改用自定义参数或 Default",
                preset, sorted(k for k in _PRESETS.keys()),
            )
        # 自定义参数
        self.world.set_weather(carla.WeatherParameters(
            cloudiness=float(weather_cfg.get("cloudiness", 0.0)),
            precipitation=float(weather_cfg.get("precipitation", 0.0)),
            precipitation_deposits=float(weather_cfg.get("precipitation_deposits", 0.0)),
            wind_intensity=float(weather_cfg.get("wind_intensity", 0.0)),
            sun_azimuth_angle=float(weather_cfg.get("sun_azimuth", 0.0)),
            sun_altitude_angle=float(weather_cfg.get("sun_altitude", 70.0)),
            fog_density=float(weather_cfg.get("fog_density", 0.0)),
            fog_distance=float(weather_cfg.get("fog_distance", 0.0)),
            wetness=float(weather_cfg.get("wetness", 0.0)),
        ))
