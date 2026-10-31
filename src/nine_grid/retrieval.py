"""三通道精确检索：各自召回，再给候选补齐所有证据，最后等权融合。"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contracts import Representations, RetrievalResult
from .representations import FEATURE_VERSION,valid_dino_signature
from .configuration import grid_distance_config,grid_metric_signature


def bev_similarity(a, b):
    if a is None or b is None:
        return None
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != (768,) or b.shape != (768,) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/denominator,0,1)) if denominator > 1e-12 else None


def occupancy_similarity(a, b, known_a, known_b):
    """共同已知部分内计算软 IoU；双空跳过、单空计零、未知不当空。"""
    a,b,ka,kb = (np.asarray(x,dtype=float) for x in (a,b,known_a,known_b))
    if a.shape != (6,50,25) or a.shape != b.shape or ka.shape != a.shape or kb.shape != a.shape:
        return None
    if not all(np.all(np.isfinite(x)) for x in (a,b,ka,kb)):
        return None
    if not all(np.all((x>=0)&(x<=1)) for x in (a,b,ka,kb)):
        return None
    values = []
    for channel in (0,2,3,4,5):
        known = np.minimum(ka[channel],kb[channel])
        if np.mean(known) < .1:
            continue
        union = np.sum(known*np.maximum(a[channel],b[channel]))
        if union*4 < .01:
            continue
        intersection = np.sum(known*np.minimum(a[channel],b[channel]))
        values.append(intersection/union)
    return float(np.mean(values)) if values else None


def _heading_difference(a,b):
    if not np.isfinite(float(a)) or not np.isfinite(float(b)):
        raise ValueError('道路相对朝向必须为有限角度')
    return abs((float(a)-float(b)+math.pi) % (2*math.pi)-math.pi)


def _object_cost(a,b,ego=False,config=None):
    config=grid_distance_config(config)
    scales,weights=config['scales'],config['weights']
    terms = []
    for key,normalizer in (("position",[scales['s_m'],scales['d_m']]),("velocity",[scales['vs_mps'],scales['vd_mps']]),("acceleration",[scales['as_mps2'],scales['ad_mps2']])):
        if ego and key == "position":
            continue
        av,bv = np.asarray(a[key],dtype=float),np.asarray(b[key],dtype=float)
        if av.shape != (2,) or bv.shape != (2,) or not np.all(np.isfinite(av)) or not np.all(np.isfinite(bv)):
            raise ValueError("九宫格状态包含非法数值")
        value = min(1.,float(np.linalg.norm((av-bv)/normalizer)/math.sqrt(2)))
        terms.append((weights[key],value))
    terms.append((weights['heading'],min(1.,_heading_difference(a["heading"],b["heading"])/scales['heading_rad'])))
    if sum(w for w,_ in terms)<=0:
        return 0.
    return sum(w*v for w,v in terms)/sum(w for w,_ in terms)


def _unique_objects(grid):
    objects = {}
    for item in grid.get("objects",[]):
        key = str(item["id"])
        if key not in objects:
            objects[key] = dict(item,slots=set(item.get("slots",[])))
        else:
            objects[key]["slots"].update(item.get("slots",[]))
    return list(objects.values())


def grid_similarity(a, b, return_matches=False):
    """公共格子约束的一对一匹配；一个跨格实体仅贡献一次。"""
    missing = (None,[]) if return_matches else None
    if not isinstance(a,dict) or not isinstance(b,dict):
        return missing
    if not a.get("valid",False) or not b.get("valid",False) or a.get("version") != FEATURE_VERSION or b.get("version") != FEATURE_VERSION:
        return missing
    try:
        config=grid_distance_config(a.get('distance_config'))
        signature=grid_metric_signature(config)
        if signature!=grid_metric_signature(b.get('distance_config')):
            return missing
        if a.get('metric_signature',signature)!=signature or b.get('metric_signature',signature)!=signature:
            return missing
        left,right = _unique_objects(a),_unique_objects(b)
    except (ValueError,TypeError,KeyError,AttributeError):
        return missing
    n,m = len(left),len(right)
    if not n and not m:
        return missing
    # 增广指派允许左右两侧各自不匹配，避免强制把不同车道/类型配成一辆车。
    costs = np.full((n+m,n+m),1e6,dtype=float)
    try:
        for i,obj_a in enumerate(left):
            for j,obj_b in enumerate(right):
                if obj_a["kind"] == obj_b["kind"] and obj_a["slots"] & obj_b["slots"]:
                    costs[i,j] = _object_cost(obj_a,obj_b,config=config)
        for i in range(n):
            costs[i,m+i] = config['unmatched_cost']
        for j in range(m):
            costs[n+j,j] = config['unmatched_cost']
        costs[n:,m:] = 0.
        rows,cols = linear_sum_assignment(costs)
        cost = min(1.,float(costs[rows,cols].sum()/max(n,m)))
        ego_cost = _object_cost(a["ego"],b["ego"],ego=True,config=config)
    except (KeyError,TypeError,ValueError):
        return missing
    ego_weight=config['ego_weight']
    score = (1-ego_weight)*(1-cost)+ego_weight*(1-ego_cost)
    matches = [{"query_id":left[i]["id"],"memory_id":right[j]["id"],"cost":float(costs[i,j]),
                "shared_slots":sorted(left[i]["slots"] & right[j]["slots"])}
               for i,j in zip(rows,cols) if i<n and j<m and costs[i,j]<1e5]
    return (float(score),matches) if return_matches else float(score)


def _record_reps(record):
    reps = record.get("_representations")
    if isinstance(reps,Representations):
        return reps
    return None


def retrieve(query: Representations, records, top_k=3, min_score=.5, snapshot_id="") -> RetrievalResult:
    """候选未进入另一通道 Top3 不等于缺证据，所有候选均重新完整评分。"""
    if type(top_k) is not int or top_k < 0 or not 0 <= min_score <= 1:
        raise ValueError("Top-K 必须非负且最低分位于 [0,1]")
    if not valid_dino_signature(query.dino_signature):
        return RetrievalResult(query.frame_id,snapshot_id=snapshot_id,time=query.time)
    scored = []
    identities = set()
    for record in records:
        identity = (record.get("id"),record.get("revision",1))
        if identity in identities or record.get("status","active") != "active":
            continue
        identities.add(identity)
        reps = _record_reps(record)
        if reps is None or reps.version != query.version or not valid_dino_signature(reps.dino_signature) or reps.dino_signature!=query.dino_signature:
            continue
        bev = bev_similarity(query.bev_feature,reps.bev_feature)
        occ = occupancy_similarity(query.occ,reps.occ,query.occ_known,reps.occ_known)
        grid,matches = grid_similarity(query.grid,reps.grid,return_matches=True)
        scored.append({"record":record,"bev":bev,"occupancy":occ,"grid":grid,"matches":matches})
    candidate_indices = set()
    for channel in ("bev","occupancy","grid"):
        ordered = sorted((i for i,item in enumerate(scored) if item[channel] is not None),
                         key=lambda i:(-scored[i][channel],str(scored[i]["record"].get("id")),scored[i]["record"].get("revision",1)))
        candidate_indices.update(ordered[:3])
    eligible = []
    for i in candidate_indices:
        item = scored[i]
        scores = [item[channel] for channel in ("bev","occupancy","grid")]
        if any(value is None for value in scores):
            continue
        item["fused"] = float(np.mean(scores))
        if item["fused"] >= min_score:
            eligible.append(item)
    eligible.sort(key=lambda item:(-item["fused"],str(item["record"].get("id")),item["record"].get("revision",1)))
    chosen = eligible[:top_k]
    return RetrievalResult(query.frame_id,[item["record"] for item in chosen],
                           [{"id":item["record"].get("id"),"revision":item["record"].get("revision",1),
                             "bev":item["bev"],"occupancy":item["occupancy"],"grid":item["grid"],
                             "fused":item["fused"],"object_matches":item["matches"]} for item in chosen],snapshot_id,time=query.time)
