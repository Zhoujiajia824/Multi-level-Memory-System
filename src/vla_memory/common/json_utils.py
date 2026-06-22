"""
JSON 工具模块
=============
提供 JSON 解析、校验、清洗工具函数。
所有 VLM 输出必须经过本模块校验，确保格式正确。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json_from_text(text: str) -> Optional[dict]:
    """从文本中提取 JSON 对象。

    VLM 输出可能包含 Markdown 代码块标记或其他多余文本，
    本函数尝试从文本中提取第一个有效的 JSON 对象。

    Args:
        text: 可能包含 JSON 的文本。

    Returns:
        解析后的字典，如果解析失败则返回 None。
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # 尝试 1：直接解析整个文本
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 尝试 2：移除 Markdown 代码块标记
    # 匹配 ```json ... ``` 或 ``` ... ```
    markdown_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(markdown_pattern, text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 尝试 3：查找第一个 { 和最后一个 } 之间的内容
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace : last_brace + 1]
        try:
            result = json.loads(json_str)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # 尝试 4：修复常见的 JSON 格式问题
    # 移除控制字符
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    cleaned = cleaned.strip()
    if cleaned != text:
        return extract_json_from_text(cleaned)

    return None


def validate_json_schema(
    data: dict,
    required_fields: list[str],
    optional_fields: Optional[list[str]] = None,
) -> tuple[bool, list[str]]:
    """验证 JSON 数据是否包含所有必需字段。

    Args:
        data: 待验证的字典。
        required_fields: 必需字段列表。
        optional_fields: 可选字段列表（用于日志提示，不影响验证结果）。

    Returns:
        (是否通过验证, 缺失字段列表)
    """
    if not isinstance(data, dict):
        return False, ["输入数据不是字典类型"]

    missing = []
    for field in required_fields:
        if field not in data:
            missing.append(field)

    if missing:
        return False, missing
    return True, []


def safe_get(
    data: dict,
    *keys: str,
    default: Any = None,
) -> Any:
    """安全获取嵌套字典值。

    Args:
        data: 字典数据。
        *keys: 嵌套键名序列。
        default: 默认值。

    Returns:
        获取到的值或默认值。
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def validate_enum_field(
    data: dict,
    field: str,
    valid_values: list[str],
    default: str = "unknown",
) -> tuple[str, bool]:
    """验证枚举类型字段。

    Args:
        data: 字典数据。
        field: 字段名。
        valid_values: 有效值列表。
        default: 无效时的默认值。

    Returns:
        (实际值或默认值, 是否为有效值)
    """
    value = data.get(field, default)
    if value not in valid_values:
        return default, False
    return value, True


def to_json_str(data: Any, ensure_ascii: bool = False, indent: int = 2) -> str:
    """将数据序列化为 JSON 字符串。

    Args:
        data: 待序列化的数据。
        ensure_ascii: 是否转义非 ASCII 字符。
        indent: 缩进空格数。

    Returns:
        JSON 字符串。
    """
    return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent, default=str)
