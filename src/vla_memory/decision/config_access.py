"""决策配置访问辅助
==================
提供从 config/decision.yaml 读取关键决策约束（如路点边界、行为枚举）的
单一入口，避免硬编码常量在多个模块中漂移。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import yaml

# 项目根目录：src/vla_memory/decision -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DECISION_YAML_PATH = PROJECT_ROOT / "config" / "decision.yaml"

# 兜底常量：仅在 config/decision.yaml 缺失或字段缺失时使用
_FALLBACK_WAYPOINT_MIN = 20
_FALLBACK_WAYPOINT_MAX = 30
_FALLBACK_HORIZON_SECONDS = 3.0
_FALLBACK_DT = 0.1


@lru_cache(maxsize=1)
def _load_decision_yaml() -> dict:
    """加载 config/decision.yaml，结果缓存。"""
    if not DECISION_YAML_PATH.exists():
        return {}
    with open(DECISION_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def reset_cache() -> None:
    """清除缓存，主要供测试在改写 decision.yaml 后调用。"""
    _load_decision_yaml.cache_clear()


def get_waypoint_bounds() -> Tuple[int, int]:
    """返回 (waypoint_min_num, waypoint_max_num)。

    单一权威来源：config/decision.yaml -> trajectory.waypoint_min_num / waypoint_max_num
    """
    data = _load_decision_yaml()
    traj = data.get("trajectory") or {}
    min_num = int(traj.get("waypoint_min_num", _FALLBACK_WAYPOINT_MIN))
    max_num = int(traj.get("waypoint_max_num", _FALLBACK_WAYPOINT_MAX))
    if min_num > max_num:
        # 配置异常时回退到默认值并保持顺序
        return _FALLBACK_WAYPOINT_MIN, _FALLBACK_WAYPOINT_MAX
    return min_num, max_num


def get_horizon_and_dt() -> Tuple[float, float]:
    """返回 (horizon_seconds, dt)，供模板渲染使用。"""
    data = _load_decision_yaml()
    traj = data.get("trajectory") or {}
    horizon = float(traj.get("horizon_seconds", _FALLBACK_HORIZON_SECONDS))
    dt = float(traj.get("dt", _FALLBACK_DT))
    return horizon, dt


def get_valid_behaviors() -> List[str]:
    """返回行为枚举列表（用于校验与提示词渲染）。

    优先 config/decision.yaml -> behaviors.valid_behaviors，
    缺失时回退到 schemas/decision.py 的 VALID_BEHAVIORS 常量。
    """
    data = _load_decision_yaml()
    behaviors = (data.get("behaviors") or {}).get("valid_behaviors")
    if behaviors:
        return [str(b) for b in behaviors]
    # 延迟导入避免循环依赖
    from src.vla_memory.schemas.decision import VALID_BEHAVIORS
    return list(VALID_BEHAVIORS)
