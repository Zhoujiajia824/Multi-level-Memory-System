"""
图像 IO 工具模块
================
提供图像读取、预处理等工具函数。
使用 PIL 和 OpenCV 实现图像操作。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image


def load_image(
    path: Union[str, Path],
    mode: str = "RGB",
    size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """加载图像文件。

    Args:
        path: 图像文件路径。
        mode: 颜色模式（RGB / BGR / L）。
        size: 可选的目标尺寸 (width, height)。

    Returns:
        PIL Image 对象。

    Raises:
        FileNotFoundError: 图像文件不存在。
        ValueError: 不支持的颜色模式。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    img = Image.open(str(path))

    if mode == "RGB":
        img = img.convert("RGB")
    elif mode == "L":
        img = img.convert("L")
    elif mode == "BGR":
        img = img.convert("RGB")
        # BGR 模式下交换通道（如果需要 numpy 数组）
        img_array = np.array(img)
        img_array = img_array[:, :, ::-1]
        img = Image.fromarray(img_array)
    else:
        raise ValueError(f"不支持的颜色模式: {mode}，请使用 RGB / BGR / L")

    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)

    return img


def image_to_tensor(
    image: Image.Image,
) -> np.ndarray:
    """将 PIL Image 转换为 numpy 数组。

    Args:
        image: PIL Image 对象。

    Returns:
        形状为 (H, W, C) 的 numpy 数组，float32 类型，值域 [0, 1]。
    """
    arr = np.array(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def validate_image_file(path: Union[str, Path]) -> bool:
    """验证文件是否为有效的图像文件。

    Args:
        path: 文件路径。

    Returns:
        是否为有效图像文件。
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with Image.open(str(path)) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(path: Union[str, Path]) -> dict:
    """获取图像文件基本信息。

    Args:
        path: 图像文件路径。

    Returns:
        包含宽度、高度、模式、格式等信息的字典。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    with Image.open(str(path)) as img:
        return {
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
            "file_size": path.stat().st_size,
            "file_name": path.name,
        }
