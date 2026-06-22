"""
日志工具模块
============
提供统一的日志配置和管理。支持控制台输出和文件输出。
所有日志信息使用中文，便于调试和排查问题。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "vla_memory",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    console_enabled: bool = True,
    file_enabled: bool = True,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """配置并返回日志记录器。

    Args:
        name: 日志记录器名称。
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）。
        log_dir: 日志文件目录。
        log_file: 日志文件名。默认自动生成。
        console_enabled: 是否启用控制台输出。
        file_enabled: 是否启用文件输出。
        fmt: 日志格式字符串。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 日志格式
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台 handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件 handler
    if file_enabled and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        file_path = log_dir / log_file
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "vla_memory") -> logging.Logger:
    """获取已配置的日志记录器。

    如果记录器尚未配置，则使用默认配置进行初始化。

    Args:
        name: 日志记录器名称。

    Returns:
        Logger 实例。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # 使用默认配置
        return setup_logger(
            name=name,
            level="INFO",
            log_dir=Path("outputs/logs"),
            console_enabled=True,
            file_enabled=True,
        )
    return logger
