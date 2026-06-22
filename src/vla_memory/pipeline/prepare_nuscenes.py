"""nuScenes 数据准备流水线
=========================
加载 nuScenes 数据集，构建索引，提取关键帧列表。
如果数据集不存在，必须明确报错并停止。
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.vla_memory.common.config import Config, load_config
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.data.nuscenes_adapter import NuScenesAdapter
from src.vla_memory.keyframes.nuscenes_keyframe_sampler import NuScenesKeyframeSampler
from src.vla_memory.common.path_utils import ensure_dir

logger = get_logger("prepare_nuscenes")


def run_prepare_nuscenes(config: Optional[Config] = None) -> Dict[str, Any]:
    """执行 nuScenes 数据准备流水线。

    步骤:
    1. 检查数据集目录是否存在。
    2. 加载 nuScenes 数据集。
    3. 按 keyframe_step 采样关键帧。
    4. 输出关键帧索引。

    Args:
        config: 配置实例（None 则自动加载）。

    Returns:
        包含关键帧索引等信息的字典。

    Raises:
        FileNotFoundError: 数据集目录不存在。
        RuntimeError: 数据集加载失败。
    """
    if config is None:
        config = load_config()

    dataroot = config.get("dataroot")
    version = config.get("version", "v1.0-mini")
    keyframe_step = config.get_nested("keyframe", "step", default=2)

    logger.info("=" * 60)
    logger.info("nuScenes 数据准备流水线")
    logger.info(f"数据目录: {dataroot}")
    logger.info(f"版本: {version}")
    logger.info(f"关键帧步长: {keyframe_step}")
    logger.info("=" * 60)

    # 检查数据目录
    if not Path(dataroot).exists():
        raise FileNotFoundError(
            f"nuScenes 数据集目录不存在: {dataroot}\n"
            f"请将 nuScenes {version} 数据集放置到该目录。\n"
            f"目录结构应为:\n"
            f"  {dataroot}/\n"
            f"    {version}/\n"
            f"    samples/\n"
            f"    maps/\n"
            f"    ..."
        )

    # 加载数据集（P3 起：按 yaml 注入 CanBusLoader）
    can_bus_loader = _maybe_build_can_bus_loader(config)
    adapter = NuScenesAdapter(
        dataroot=dataroot,
        version=version,
        can_bus_loader=can_bus_loader,
        fallback_to_pose_diff=config.get_nested("can_bus", "fallback_to_pose_diff", default=True),
    )
    adapter.load()
    logger.info(f"nuScenes 数据集加载成功: {adapter.get_sample_count()} 个样本")
    if can_bus_loader is not None:
        logger.info("CAN bus 真值已启用 (data_nuscenes.yaml -> can_bus.enabled=true)")

    # 关键帧采样
    sampler = NuScenesKeyframeSampler(step=keyframe_step)

    # 子集配置
    subset_cfg = config.get("subset", {})
    max_scenes = subset_cfg.get("max_scenes") if subset_cfg.get("enabled") else None
    max_samples = subset_cfg.get("max_samples_per_scene") if subset_cfg.get("enabled") else None

    # 采样所有场景
    keyframe_index = sampler.sample_all_scenes(
        adapter=adapter,
        max_scenes=max_scenes,
        max_samples_per_scene=max_samples,
    )

    total_keyframes = sum(len(kfs) for kfs in keyframe_index.values())
    logger.info(f"关键帧索引构建完成: {len(keyframe_index)} 个场景, {total_keyframes} 个关键帧")

    return {
        "adapter": adapter,
        "keyframe_index": keyframe_index,
        "total_keyframes": total_keyframes,
        "sample_rate": sampler.get_sample_rate(),
    }


def _maybe_build_can_bus_loader(config: Config):
    """根据 config/data_nuscenes.yaml 构造 CanBusLoader；关闭或缺失时返回 None。

    两个开关任一为 False 就关闭：
      - ego_state.use_can_bus
      - can_bus.enabled
    """
    use_can_bus = config.get_nested("ego_state", "use_can_bus", default=False)
    can_bus_enabled = config.get_nested("can_bus", "enabled", default=False)
    if not (use_can_bus and can_bus_enabled):
        return None
    root = config.get_nested("can_bus", "root", default="data/nuscenes/raw/can_bus")
    tolerance_us = int(config.get_nested("can_bus", "tolerance_us", default=60_000))
    root_path = Path(root)
    if not root_path.is_absolute():
        # 相对路径以项目根（dataroot 的祖父 .. 视为不可靠）解析；用 config.root 更稳
        root_path = config.root / root_path
    try:
        from src.vla_memory.data.can_bus_loader import CanBusLoader
        return CanBusLoader(can_bus_root=root_path, tolerance_us=tolerance_us)
    except FileNotFoundError as e:
        logger.warning("CAN bus loader 初始化失败，关闭真值通路: %s", e)
        return None
