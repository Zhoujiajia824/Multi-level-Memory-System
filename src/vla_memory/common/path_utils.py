"""
路径工具模块
============
提供跨平台路径操作工具。使用 pathlib 保证 Windows 路径兼容。
所有路径均使用 pathlib.Path 对象，避免字符串拼接导致路径问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/vla_memory/common -> 项目根


def get_project_root() -> Path:
    """获取项目根目录。

    Returns:
        项目根目录的 Path 对象。
    """
    return PROJECT_ROOT


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，不存在则递归创建。

    Args:
        path: 目录路径。

    Returns:
        创建后的 Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relative_to_project(path: Union[str, Path]) -> Path:
    """将绝对路径转换为相对于项目根目录的路径。

    Args:
        path: 绝对路径或相对路径。

    Returns:
        相对于项目根目录的 Path 对象。
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT)
    except ValueError:
        # 不在项目目录下，返回原路径
        return p


def safe_resolve(path: Union[str, Path], base: Union[str, Path, None] = None) -> Path:
    """安全解析路径。

    如果路径是相对路径且提供了 base，则相对于 base 解析。
    否则相对于当前工作目录解析。

    Args:
        path: 待解析的路径。
        base: 基准目录。

    Returns:
        解析后的绝对路径。
    """
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    if base is not None:
        return (Path(base) / p).resolve()
    return p.resolve()


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """查找目录中匹配模式的文件。

    Args:
        directory: 搜索目录。
        pattern: 文件名匹配模式（glob 格式）。
        recursive: 是否递归搜索子目录。

    Returns:
        匹配的文件路径列表，按文件名排序。
    """
    d = Path(directory)
    if not d.exists():
        return []
    if recursive:
        return sorted(d.rglob(pattern))
    return sorted(d.glob(pattern))


def validate_path_exists(
    path: Union[str, Path],
    description: str = "路径",
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """验证路径是否存在。

    Args:
        path: 待验证的路径。
        description: 路径描述（用于错误信息）。
        must_be_file: 是否必须是文件。
        must_be_dir: 是否必须是目录。

    Returns:
        解析后的 Path 对象。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 路径类型不匹配。
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{description}不存在: {p}")
    if must_be_file and not p.is_file():
        raise ValueError(f"{description}不是文件: {p}")
    if must_be_dir and not p.is_dir():
        raise ValueError(f"{description}不是目录: {p}")
    return p
