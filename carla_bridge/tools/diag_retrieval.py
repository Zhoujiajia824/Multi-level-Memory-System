"""C2 诊断：CARLA 中期记忆检索 score=0.000 问题定位（只读，不改 src）。

用法（项目根目录，mulmem_carla 环境）：
    python -m carla_bridge.tools.diag_retrieval [--db outputs/memory_db_carla] [--features-dir outputs/features]

已知疑点（诊断输入，不预设结论）：
  1. ids.json 存在重复 id（event_carla_NOA2_00000_3200000 出现 2 次，FAISS 13 vs meta 12）
  2. 存量记录 scene_description=None → text 相似度恒 0
  3. 存量 scene_id='straight_road' vs VLM 输出 'car_following' 词表不一致
  4. weather/nav 均匹配时 final_score 应≥0.2，实测 0.0 —— 疑运行时权重全 0 或查询侧字段空
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# 项目根入 sys.path（与 run_carla_demo 同款注入方式）
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="outputs/memory_db_carla")
    ap.add_argument("--features-dir", default="outputs/features")
    args = ap.parse_args()

    db_dir = Path(args.db)
    print(f"=== CARLA 记忆检索诊断 (db={db_dir}) ===\n")

    # ---- 1. 静态检查：meta.json / ids.json 对齐 ----
    meta_path = db_dir / "mid_term_meta.json"
    ids_path = db_dir / "mid_term_faiss.ids.json"
    idx_path = db_dir / "mid_term_faiss.index"
    if not meta_path.exists():
        print(f"[SKIP] {meta_path} 不存在（记忆库为空？）—— 重建后重跑本诊断")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ids = []
    if ids_path.exists():
        with open(ids_path, "r", encoding="utf-8") as f:
            ids = json.load(f)

    print(f"[1] meta.json 记录数 = {len(meta)}, ids.json 长度 = {len(ids)}")
    dup = [i for i in set(ids) if ids.count(i) > 1]
    if dup:
        print(f"    ⚠️ ids.json 重复 id: {dup}")
    if len(set(ids)) != len(meta):
        print(f"    ⚠️ ids({len(set(ids))}) 与 meta({len(meta)}) 不对齐")
    else:
        print("    ✓ ids/meta 对齐")

    # ---- 2. 存量记录字段完整性 ----
    print("\n[2] 存量记录字段完整性：")
    field_stats = {}
    for rid, rec in meta.items():
        if not isinstance(rec, dict):
            continue
        for k in ("scene_id", "weather_id", "nav_instruction", "scene_text",
                  "source_dataset"):
            v = rec.get(k)
            field_stats.setdefault(k, {"non_null": 0, "values": {}})
            if v not in (None, ""):
                field_stats[k]["non_null"] += 1
                vs = field_stats[k]["values"]
                vs[str(v)[:30]] = vs.get(str(v)[:30], 0) + 1
    for k, st in field_stats.items():
        print(f"    {k}: non_null={st['non_null']}/{len(meta)}  值分布={st['values']}")

    # ---- 3. 实例化 MidTermMemory 并打真实查询 ----
    print("\n[3] 用真实特征向量查询（若 features 缺失则跳过）：")
    npy_files = sorted(glob.glob(str(Path(args.features_dir) / "carla*.npy")))
    if not npy_files:
        print(f"    [SKIP] {args.features_dir}/carla*.npy 不存在")
        return
    print(f"    找到 {len(npy_files)} 个 carla 特征文件")

    from src.vla_memory.common.config import load_config
    from src.vla_memory.memory.faiss_store import FAISSVectorStore
    from src.vla_memory.memory.mid_term_memory import MidTermMemory

    carla_yaml = _ROOT / "carla_bridge" / "config" / "carla.yaml"
    overrides = {}
    if carla_yaml.exists():
        from src.vla_memory.common.config import load_yaml
        overrides = load_yaml(carla_yaml) or {}
    config = load_config(overrides=overrides)
    weights = (config.get_nested("mid_term", "weights", default={}) or {})
    persistence = {
        "enabled": True, "save_on_close": False, "auto_load_on_init": True,
        "strict_load": False,
    }
    retrieval_cfg = (config.get_nested("mid_term", "retrieval", default={})
                     or {})
    mem = MidTermMemory(
        faiss_store=FAISSVectorStore(dimension=768),
        weights=weights or None,
        persistence_cfg=persistence,
        save_dir=str(db_dir),
        retrieval_cfg=retrieval_cfg,
    )
    print(f"    实例化完成: size={mem.size()}, faiss={mem.faiss_store.size()}")
    print(f"    运行时权重 = {mem.weights}")

    # 用第一份特征做查询（模拟典型查询字段）
    q = np.load(npy_files[0])
    print(f"\n    query 特征: shape={q.shape}, norm={np.linalg.norm(q):.4f}")
    res = mem.search(
        query_feature=q,
        scene_text="直路上有前方车辆跟随行驶",
        scene_id="car_following",
        weather_id="sunny",
        nav_instruction="lane_follow",
        ego_state={"speed": 8.0},
    )
    print(f"    stats = {res['stats']}")
    for i, r in enumerate(res.get("results", [])[:3], 1):
        print(f"\n    #{i} {r.get('record').record_id}")
        print(f"       final_score={r.get('final_score'):.4f} sub_scores={r.get('sub_scores')}")

    # ---- 4. 结论提示 ----
    print("\n[4] 判读要点：")
    print("    - 若 visual_score 全 0 → 特征没进 FAISS / 索引损坏（重建库）")
    print("    - 若 scene/weather/nav_score=0 → 记录字段为空或词表不匹配（见[2]值分布）")
    print("    - 若 weights 全 0 → 配置加载路径问题（config memory.mid_term.retrieval_weights）")
    print("    - 重复 id → add_record 二次写入路径，重建库可暂时消除")


if __name__ == "__main__":
    main()
