"""决策结果 JSONL append / resume 辅助
======================================
为在线循环（OnlineDrivingLoop）提供：

* ``append_decision_record(path, record)``  ——
  单条 append + ``flush`` + ``os.fsync``，保证进程中断后 jsonl 不丢已写帧。
* ``load_processed_sample_tokens(path)``  ——
  扫描已存在的 jsonl，收集已处理的 ``sample_token``，
  用于 resume 时跳过重复 VLM 调用。

约定：每条记录是一行 JSON；``ensure_ascii=False`` 让中文 reason / scene_description
直接可读；``default=str`` 兜底 numpy / pathlib 等非原生可序列化对象。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("decision_record_io")


def append_decision_record(path: Path | str, record: Dict[str, Any]) -> None:
    """以 utf-8 + ensure_ascii=False append 一条 JSON 行；fsync 防中断丢数据。

    Args:
        path: 目标 jsonl 路径。父目录若不存在会自动创建。
        record: 要写入的字典。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    # 用 "a" 模式 + 文件级 fsync，保证 OS 缓冲也落盘
    with open(str(path), "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # 个别 Windows / 虚拟文件系统不支持 fsync；记 warning 不中断
            logger.debug("fsync 不支持 (path=%s)，跳过", path)


def load_processed_sample_tokens(path: Path | str) -> Set[str]:
    """扫描已存在的 jsonl，收集所有已处理的 ``sample_token``。

    用法（OnlineDrivingLoop.setup 中）：

        seen = load_processed_sample_tokens(jsonl_path)
        if sample_token in seen:
            return None   # resume 跳过

    Args:
        path: jsonl 路径。

    Returns:
        ``Set[str]`` —— 已处理 sample_token 集合。文件不存在或全部解析失败时返回空集。
    """
    path = Path(path)
    if not path.exists():
        return set()

    out: Set[str] = set()
    with open(str(path), "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("resume 扫描跳过第 %d 行 (JSON 解析失败): %s", line_num, e)
                continue
            # 支持新格式 sample_token / 旧格式 frame_id
            tok = rec.get("sample_token") or rec.get("frame_id")
            if tok:
                out.add(str(tok))

    if out:
        logger.info("resume 扫描完成: %s 中已存在 %d 个 sample_token", path, len(out))
    return out


def count_decision_records(path: Path | str) -> int:
    """统计 jsonl 中已写入的有效记录条数（用于日志 / 测试）。"""
    return len(load_processed_sample_tokens(path))
