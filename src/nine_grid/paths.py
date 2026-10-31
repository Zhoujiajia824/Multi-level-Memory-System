"""本地持久化路径：Windows 长中文目录也使用同一文件系统语义。"""
from __future__ import annotations

import os
from pathlib import Path


def filesystem_path(value) -> Path:
    """返回已解析的绝对路径；Windows 显式启用扩展长度盘符或 UNC 路径。

    前缀作用于目录入口，因此追加文件名、临时目录和原子重命名也保留支持。
    不通过截短用户目录或文件名规避 MAX_PATH，也不修改系统长路径设置。
    """
    resolved = Path(value).resolve()
    if os.name != "nt":
        return resolved
    text = str(resolved)
    if text.startswith("\\\\?\\"):
        return resolved
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)
