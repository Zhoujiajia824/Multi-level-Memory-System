#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长期记忆候选沉淀脚本（Phase 7）
================================
从中期记忆库（``outputs/memory_db/mid_term_meta.json``）挖掘高价值、稳定、可泛化的经验，
总结为长期记忆**候选**规则，写入 ``outputs/long_term_candidates/candidate_rules.yaml``
（``status=pending_review``，**不覆盖**正式长期记忆库 ``data/knowledge/long_term_rules.yaml``）。

离线批处理：只读中期记忆元数据，不依赖 faiss / VLM API。

用法::
    python scripts/08_consolidate_long_term_candidates.py
    python scripts/08_consolidate_long_term_candidates.py --output outputs/my_candidates.yaml
    python scripts/08_consolidate_long_term_candidates.py --min-evidence-count 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vla_memory.common.config import Config, load_config
from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.memory.consolidation import MemoryConsolidationManager
from src.vla_memory.memory.memory_record_io import load_mid_term_meta

logger = get_logger("consolidate_candidates")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="中期记忆 → 长期记忆候选沉淀（pending_review，不覆盖正式长期库）",
    )
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选，默认加载 config/ 下所有 YAML）")
    parser.add_argument("--mid-term-db", type=str, default=None,
                        help="中期记忆 meta.json 路径（默认 config persistence.save_dir/mid_term_meta_file）")
    parser.add_argument("--output", type=str, default=None,
                        help="候选规则输出路径（默认 config consolidation.output_path）")
    parser.add_argument("--min-evidence-count", type=int, default=None,
                        help="覆盖 min_evidence_count（最小证据数）")
    parser.add_argument("--source-memory-type", type=str, default=None,
                        help="覆盖 source_memory_type（event_memory / frame_memory）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()

    # 合并 CLI 覆盖到 consolidation 配置
    cons_cfg = dict(config.get_nested("mid_term", "consolidation", default={}) or {})
    if args.min_evidence_count is not None:
        cons_cfg["min_evidence_count"] = args.min_evidence_count
    if args.source_memory_type is not None:
        cons_cfg["source_memory_type"] = args.source_memory_type

    # 定位中期记忆 meta.json
    persist_cfg = config.data.get("persistence", {}) or {}
    save_dir = args.mid_term_db
    if save_dir is None:
        save_dir = Path(persist_cfg.get("save_dir", "outputs/memory_db"))
        meta_file = persist_cfg.get("mid_term_meta_file", "mid_term_meta.json")
        save_dir = Path(save_dir) / meta_file
    meta_path = Path(save_dir)
    if not meta_path.exists():
        logger.error("中期记忆元数据不存在: %s（先跑 memory_on demo 生成 memory_db）", meta_path)
        return 1

    # 加载中期记忆
    records = load_mid_term_meta(str(meta_path))
    logger.info("加载中期记忆: %d 条 (%s)", len(records), meta_path)

    # 沉淀
    manager = MemoryConsolidationManager(cons_cfg)
    candidates = manager.consolidate(records)
    if not candidates:
        logger.warning("未生成任何候选规则（可能高价值 %s 不足 %d 条/价值分不达标）",
                       manager.source_memory_type, manager.min_evidence_count)
        return 0

    # 保存（不覆盖正式长期记忆库）
    output_path = manager.save(candidates, output_path=args.output)

    # 摘要
    print("\n========== 长期记忆候选沉淀完成 ==========")
    print(f"来源中期记忆: {len(records)} 条 (memory_type={manager.source_memory_type})")
    print(f"生成候选规则: {len(candidates)} 条 (status=pending_review)")
    print(f"输出文件: {output_path}")
    print("（不覆盖正式长期库 data/knowledge/long_term_rules.yaml；需人工审核后晋升）\n")
    by_type = {}
    for c in candidates:
        by_type[c.get("candidate_type", "?")] = by_type.get(c.get("candidate_type", "?"), 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n} 条")
    for c in candidates[:5]:
        print(f"  - {c['rule_id']} | {c['candidate_type']} | event={c['condition']['event_types']} "
              f"| conf={c['confidence']} | evidence={c['evidence']['evidence_count']}")
    if len(candidates) > 5:
        print(f"  ... 共 {len(candidates)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
