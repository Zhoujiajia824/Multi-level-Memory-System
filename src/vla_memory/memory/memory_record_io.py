"""记忆记录 IO 模块
==================
提供记忆记录的序列化、反序列化、保存、加载功能。
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
from src.vla_memory.schemas.memory import ShortTermMemoryItem, MidTermMemoryRecord
from src.vla_memory.common.logging_utils import get_logger

logger = get_logger("memory_record_io")


def save_short_term_memory(items: List[ShortTermMemoryItem], path: str) -> None:
    """保存短期记忆到文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "wb") as f:
        # pydantic v2: 用 model_dump 代替 v1 的 dict
        pickle.dump([item.model_dump() for item in items], f)
    logger.info(f"短期记忆已保存: {path} ({len(items)} 条)")


def load_short_term_memory(path: str) -> List[ShortTermMemoryItem]:
    """从文件加载短期记忆。"""
    path = Path(path)
    if not path.exists():
        return []
    with open(str(path), "rb") as f:
        data_list = pickle.load(f)
    return [ShortTermMemoryItem(**d) for d in data_list]


def save_mid_term_meta(records: Dict[str, Any], path: str) -> None:
    """保存中期记忆元数据到 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 转换记录为可序列化格式（pydantic v2 用 model_dump）
    serializable = {}
    for rid, record in records.items():
        if hasattr(record, "model_dump"):
            serializable[rid] = record.model_dump()
        elif hasattr(record, "dict"):
            serializable[rid] = record.dict()
        else:
            serializable[rid] = record
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"中期记忆元数据已保存: {path} ({len(records)} 条)")


def load_mid_term_meta(path: str) -> Dict[str, Any]:
    """从 JSON 文件加载中期记忆元数据。

    反序列化为 MidTermMemoryRecord 对象，保持与 MidTermMemory._records 一致的类型。

    Args:
        path: JSON 文件路径。

    Returns:
        Dict[str, MidTermMemoryRecord]: record_id 到 MemoryRecord 对象的映射。
            文件不存在时返回空字典。
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(str(path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 将原始字典反序列化为 MidTermMemoryRecord 对象
    records = {}
    for rid, data in raw.items():
        if isinstance(data, dict):
            records[rid] = MidTermMemoryRecord(**data)
        else:
            records[rid] = data
    logger.info(f"中期记忆元数据已加载: {path} ({len(records)} 条)")
    return records
