"""自包含、无种子的经验存储；只读快照与后台写库明确分离。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import numpy as np

from .contracts import ANSWER_PROTOCOL_VERSION, ExperienceRecord, Representations, VERSION, plain
from .representations import FEATURE_VERSION,valid_dino_signature
from .configuration import grid_metric_signature
from .paths import filesystem_path


def _sha(path):
    digest = hashlib.sha256()
    with filesystem_path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path,payload):
    with filesystem_path(path).open("w",encoding="utf-8",newline="\n") as stream:
        json.dump(plain(payload),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_metadata(record):
    """只验收已保存契约，不调用模型或用重算结果替换原始验收证据。"""
    def number(value):
        try:
            return type(value) in (int,float) and math.isfinite(value)
        except OverflowError:
            return False
    def vector(value,size):
        return isinstance(value,list) and len(value)==size and all(number(x) for x in value)
    def same_frame(value):
        return (value.get('frame_id')==record['frame_id'] and number(value.get('time'))
                and value['time']==record['time'])
    if (not isinstance(record,dict) or any(not isinstance(record.get(k),str) or not record[k].strip() for k in ('id','frame_id'))
        or not number(record.get('time')) or type(record.get('revision')) is not int or record['revision']<1):
        raise ValueError('经验 ID、原帧时间戳或正整数 revision 缺失/非法')
    if record.get('answer_protocol_version')!=ANSWER_PROTOCOL_VERSION:
        raise ValueError('经验答案协议不匹配；旧多点经验保留但不参与检索')
    reference=record.get('goal_reference')
    if (not isinstance(reference,dict) or not same_frame(reference)
        or not isinstance(reference.get('reference_id'),str) or not reference['reference_id'].strip()
        or type(reference.get('reference_version')) is not int or reference['reference_version']<0
        or type(reference.get('request_version')) is not int or reference['request_version']<0
        or not number(reference.get('s_ego_m')) or not number(reference.get('d_ego_m'))
        or not isinstance(reference.get('points'),list) or len(reference['points'])<2
        or any(not vector(point,2) for point in reference['points'])
        or not vector(reference.get('source_pose'),3)
        or not isinstance(reference.get('source_lane_id'),str) or not reference['source_lane_id'].strip()
        or not isinstance(reference.get('route_lane_ids'),list)
        or any(not isinstance(identity,str) or not identity.strip() for identity in reference['route_lane_ids'])
        or not isinstance(reference.get('definition'),str) or not reference['definition'].strip()
        or not isinstance(reference.get('limits'),dict)):
        raise ValueError('经验目标参考缺失、非法或与原帧不一致')
    situation,answer,quality=(record.get(k) for k in ('situation','answer','quality'))
    if (not isinstance(situation,dict) or not same_frame(situation)
        or any(not isinstance(situation.get(k),str) or not situation[k].strip() for k in ('scenario','坐标','情境'))
        or not vector(situation.get('原点世界姿态'),3) or not number(situation.get('自车速度')) or situation['自车速度']<0
        or not isinstance(situation.get('可见对象'),list)):
        raise ValueError('经验情境缺失、数值非法或与原帧不一致')
    if not np.allclose(reference['source_pose'],situation['原点世界姿态'],rtol=0,atol=1e-8):
        raise ValueError('经验目标参考的原帧位姿与情境不一致')
    identities=set()
    for actor in situation['可见对象']:
        if (not isinstance(actor,dict) or not isinstance(actor.get('id'),str) or not actor['id'].strip()
            or actor['id'] in identities or actor.get('类型') not in ('vehicle','bicycle','pedestrian','static_obstacle')
            or not vector(actor.get('世界位置'),2) or not vector(actor.get('速度'),2)
            or not vector(actor.get('尺寸'),2) or any(x<=0 for x in actor['尺寸'])):
            raise ValueError('经验情境中的对象信息不完整或数值非法')
        identities.add(actor['id'])
    if not isinstance(answer,dict) or set(answer)!={'attention','local_goal','explanation'}:
        raise ValueError('经验参考答案结构不完整')
    explanation=answer['explanation']
    if (not isinstance(explanation,dict) or set(explanation)!={'情境','依据','参考答案','适用条件'}
        or any(not isinstance(x,str) or not x.strip() for x in explanation.values())
        or explanation['情境']!=situation['情境']):
        raise ValueError('经验参考答案四段不完整或与情境不一致')
    attention,goal=answer['attention'],answer['local_goal']
    if not isinstance(attention,list) or len(attention)>3:
        raise ValueError('经验关注列表非法')
    identities=set()
    for rank,item in enumerate(attention,1):
        if (not isinstance(item,dict) or set(item)!={'id','rank','reason'}
            or not isinstance(item['id'],str) or not item['id'].strip() or item['id'] in identities
            or type(item['rank']) is not int or item['rank']!=rank
            or not isinstance(item['reason'],str) or not item['reason'].strip()):
            raise ValueError('经验关注项 ID、排序或依据非法')
        identities.add(item['id'])
    if (not isinstance(goal,dict) or set(goal)!={'delta_s_m','d_m'}
        or not number(goal['delta_s_m']) or goal['delta_s_m']<=0 or not number(goal['d_m'])):
        raise ValueError('经验单Frenet目标缺失、非有限或字段非法')
    limits=reference['limits']
    if (any(not number(limits.get(key)) or limits[key]<=0 for key in
            ('minimum_delta_s_m','maximum_delta_s_m','maximum_lateral_slope'))
        or limits['maximum_delta_s_m']<limits['minimum_delta_s_m']
        or not limits['minimum_delta_s_m']<=goal['delta_s_m']<=limits['maximum_delta_s_m']
        or abs(goal['d_m']-reference['d_ego_m'])>goal['delta_s_m']*limits['maximum_lateral_slope']):
        raise ValueError('经验单目标违反原帧固定范围或横向几何界')
    reference_length=float(np.linalg.norm(np.diff(np.asarray(reference['points']),axis=0),axis=1).sum())
    if reference['s_ego_m']<0 or reference['s_ego_m']+goal['delta_s_m']>reference_length-1e-6:
        raise ValueError('经验单目标超出原帧参考覆盖')
    if (not isinstance(quality,dict) or not same_frame(quality) or quality.get('version')!=record.get('version')
        or quality.get('valid') is not True or not number(quality.get('score')) or not 0<=quality['score']<=1
        or quality.get('collision') is not False or quality.get('precollision') is not False
        or quality.get('reasons')!=[] or not isinstance(quality.get('metrics'),dict)):
        raise ValueError('经验质量验收证据缺失、非法或与原帧不一致')
    metrics=quality['metrics']
    if (not isinstance(metrics.get('gates'),dict) or any(metrics['gates'].get(k) is not True for k in ('collision_free','on_road','correct_direction','rules'))
        or any(not number(metrics.get(k)) or not 0<=metrics[k]<=1 for k in ('progress','ttc','comfort'))
        or metrics['ttc']!=1 or any(not number(metrics.get(k)) or metrics[k]<0 for k in ('progress_m','reference_progress_m'))):
        raise ValueError('经验质量指标缺失、非有限或与验收结论矛盾')
    if len(json.dumps({'situation':situation,'answer':answer},ensure_ascii=False))>10000:
        raise ValueError('经验文本超出 10000 字符，不截断经验')


class MemoryStore:
    """snapshot 返回已载入的独立副本；之后 write 不会改变本轮可检索集合。"""
    def __init__(self, read_dir, write_dir=None):
        self.read_dir = filesystem_path(read_dir) if read_dir is not None else None
        self.write_dir = filesystem_path(write_dir) if write_dir is not None else None
        self.snapshot_id = ""
        self.audit = []
        self._lock = threading.RLock()

    def _invalidated(self, root):
        invalid = set()
        path = root/"invalidations.jsonl"
        if not path.exists():
            return invalid
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                    invalid.add((str(item["id"]),item.get("revision")))
                except (ValueError,KeyError,TypeError):
                    # 审计无法完整恢复时不能悄悄将可能失效的经验重新放行。
                    raise ValueError(f"失效日志损坏，停止读取记忆库：{path}")
        return invalid

    def snapshot(self):
        with self._lock:
            self.snapshot_id = hashlib.sha256(b"[]").hexdigest()
            self.audit = []
            if self.read_dir is None or not self.read_dir.exists():
                return []
            invalid = self._invalidated(self.read_dir)
            result = []
            for record_path in sorted((self.read_dir/"records").glob("*/record.json")):
                if record_path.parent.name.startswith("."):
                    continue
                try:
                    record = self._read_record(record_path)
                    identity = (str(record["id"]),record.get("revision",1))
                    if identity in invalid or (identity[0],None) in invalid:
                        self.audit.append({"id":record["id"],"reason":"经验已停用"})
                        continue
                    result.append(record)
                except (ValueError,KeyError,TypeError,OSError) as exc:
                    self.audit.append({"path":str(record_path),"reason":str(exc)})
            evidence = [{"id":r["id"],"revision":r["revision"],"checksums":r["checksums"],
                         "metadata_sha256":r['_metadata_sha256']} for r in result]
            self.snapshot_id = hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()
            return result

    def _read_record(self, path):
        metadata=path.read_bytes()
        record = json.loads(metadata.decode('utf-8'))
        if not isinstance(record,dict) or not isinstance(record.get('checksums'),dict):
            raise ValueError('经验元数据必须包含有效的校验映射')
        if record.get("version") != VERSION or record.get("feature_version") != FEATURE_VERSION:
            raise ValueError("经验或特征版本不匹配")
        if record.get('answer_protocol_version')!=ANSWER_PROTOCOL_VERSION:
            raise ValueError('经验答案协议不匹配；旧多点经验保留但不参与检索')
        if record.get("status") != "active":
            raise ValueError("经验未处于有效状态")
        _validate_metadata(record)
        if not valid_dino_signature(record.get('dino_signature')):
            raise ValueError('经验缺少有效的 DINO 权重身份')
        base = filesystem_path(path.parent)
        for name,expected in record.get("checksums",{}).items():
            asset = filesystem_path(base/name)
            if not asset.is_relative_to(base) or not asset.is_file() or _sha(asset) != expected:
                raise ValueError(f"经验资产缺失或校验不符：{name}")
        required = {"features.npz","grid.json","assets/bev.png","assets/feature.png"}
        if not required.issubset(record.get("checksums",{})):
            raise ValueError("经验缺少必要资产校验项")
        with np.load(base/"features.npz",allow_pickle=False) as features:
            bev = features["bev"].copy()
            occ,known = features["occ"].copy(),features["known"].copy()
        if bev.shape != (768,) or not np.all(np.isfinite(bev)) or np.linalg.norm(bev)<1e-8:
            raise ValueError("BEV 特征无效")
        if occ.shape != (6,50,25) or known.shape != occ.shape or not np.all(np.isfinite(occ)) or not np.all(np.isfinite(known)):
            raise ValueError("占据特征无效")
        if not np.all((occ>=0)&(occ<=1)) or not np.all((known>=0)&(known<=1)):
            raise ValueError('占据及已知掩码必须位于 [0,1]')
        grid = json.loads((base/"grid.json").read_text(encoding="utf-8"))
        if not isinstance(grid,dict) or grid.get("version") != FEATURE_VERSION or not grid.get("valid",False):
            raise ValueError("九宫格版本或有效性不符")
        if record.get('metric_signature')!=grid_metric_signature(grid.get('distance_config')):
            raise ValueError('九宫格距离参数签名不符')
        record["_representations"] = Representations(record["frame_id"],record["time"],str(base/"assets/bev.png"),
                                                     str(base/"assets/feature.png"),occ,known,grid,bev,
                                                     dino_signature=record['dino_signature'])
        record["_asset_root"] = str(base)
        record['_metadata_sha256'] = hashlib.sha256(metadata).hexdigest()
        return record

    def write(self, record: ExperienceRecord | dict, reps: Representations):
        if self.write_dir is None:
            raise RuntimeError("记忆写入已关闭：未配置写入目录")
        payload = plain(record)
        payload = {k:v for k,v in payload.items() if not k.startswith("_")}
        if payload.get("version") != VERSION or reps.version != VERSION or payload.get("frame_id") != reps.frame_id:
            raise ValueError("经验与表示的原帧或版本不一致")
        if (type(payload.get('time')) not in (int,float) or
            not isinstance(reps.time,(int,float,np.integer,np.floating)) or isinstance(reps.time,(bool,np.bool_)) or
            not np.isfinite(payload['time']) or not np.isfinite(reps.time) or payload['time']!=reps.time):
            raise ValueError('经验与表示必须具有相同的有限原帧时间戳')
        if not payload.get("id") or payload.get("status","active") != "active":
            raise ValueError("入库经验必须具有 ID 且处于有效状态")
        if reps.bev_feature is None or np.asarray(reps.bev_feature).shape != (768,) or not np.all(np.isfinite(reps.bev_feature)):
            raise ValueError("无有效 DINOv2 特征，拒绝生成可检索经验")
        if np.linalg.norm(reps.bev_feature) < 1e-8:
            raise ValueError("零向量不是有效 DINOv2 特征")
        if not valid_dino_signature(reps.dino_signature):
            raise ValueError('没有有效 DINO 权重身份，拒绝生成可检索经验')
        if reps.occ.shape != (6,50,25) or reps.occ_known.shape != reps.occ.shape:
            raise ValueError("占据及已知掩码尺寸不匹配")
        if not all(np.all(np.isfinite(x)) and np.all((x>=0)&(x<=1)) for x in (reps.occ,reps.occ_known)):
            raise ValueError("占据及已知掩码必须在 [0,1] 内")
        if reps.grid.get("version") != FEATURE_VERSION or not reps.grid.get("valid",False):
            raise ValueError("九宫格特征无效")
        metric_signature=grid_metric_signature(reps.grid.get('distance_config'))
        if reps.grid.get('metric_signature',metric_signature)!=metric_signature:
            raise ValueError('九宫格距离参数签名不符')
        length = len(json.dumps({"situation":payload.get("situation"),"answer":payload.get("answer")},ensure_ascii=False))
        if length > 10000:
            raise ValueError("经验文本超出 10000 字符，不截断经验")
        payload.setdefault('revision',1)
        _validate_metadata(payload)
        revision = payload['revision']
        token = hashlib.sha256(f"{payload['id']}\0{revision}".encode("utf-8")).hexdigest()
        with self._lock:
            records_dir = self.write_dir/"records"
            records_dir.mkdir(parents=True,exist_ok=True)
            target = records_dir/token
            if target.exists():
                raise FileExistsError("同一经验 ID 与修订号已存在，禁止覆盖")
            temporary = filesystem_path(tempfile.mkdtemp(prefix=".pending-",dir=records_dir))
            try:
                assets = temporary/"assets"
                assets.mkdir()
                for source,name in ((reps.bev_path,"bev.png"),(reps.feature_image_path,"feature.png")):
                    source = filesystem_path(source)
                    if not source.is_file():
                        raise ValueError(f"必要原始资产缺失：{source}")
                    shutil.copyfile(source,assets/name)
                copied = {"bev":"assets/bev.png","feature_image":"assets/feature.png"}
                original_bev = filesystem_path(reps.bev_path)
                raw_occ = original_bev.with_name(original_bev.name.replace("_bev.png","_occupancy.npz"))
                if raw_occ != original_bev and raw_occ.is_file():
                    shutil.copyfile(raw_occ,assets/"occupancy_raw.npz")
                    copied["raw_occupancy"] = "assets/occupancy_raw.npz"
                def copy_original(value,key):
                    if isinstance(value,dict):
                        return {str(k):copy_original(v,key+"_"+str(k)) for k,v in value.items()}
                    if isinstance(value,list):
                        return [copy_original(v,key+"_"+str(i)) for i,v in enumerate(value)]
                    if isinstance(value,str) and Path(value).suffix.lower() in (".png",".jpg",".jpeg",".webp",".npy",".npz",".json"):
                        source = filesystem_path(value)
                        if not source.is_file():
                            raise ValueError(f"原始资产缺失：{value}")
                        name = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]+source.suffix.lower()
                        shutil.copyfile(source,assets/name)
                        return "assets/"+name
                    return value
                copied["original"] = copy_original(payload.get("assets",{}),"original")
                payload["assets"] = copied
                payload["status"] = "active"
                payload["revision"] = revision
                payload["feature_version"] = FEATURE_VERSION
                payload['dino_signature'] = reps.dino_signature
                payload["metric_signature"] = grid_metric_signature(reps.grid.get('distance_config'))
                payload["features"] = {"path":"features.npz","bev_dimension":768,"occ_shape":[6,50,25],"grid":"grid.json"}
                np.savez_compressed(temporary/"features.npz",bev=np.asarray(reps.bev_feature,dtype=np.float32),
                                    occ=np.asarray(reps.occ,dtype=np.float32),known=np.asarray(reps.occ_known,dtype=np.float32))
                _json_write(temporary/"grid.json",reps.grid)
                payload["checksums"] = {str(p.relative_to(temporary)).replace("\\","/"):_sha(p) for p in temporary.rglob("*") if p.is_file()}
                _json_write(temporary/"record.json",payload)
                # 同一文件系统目录原子替换；只在完整资产和元数据均写完后对读取者可见。
                os.rename(temporary,target)
                return copy.deepcopy(payload)
            except Exception:
                if temporary.exists():
                    if not filesystem_path(temporary).is_relative_to(filesystem_path(records_dir)):
                        raise RuntimeError("临时经验目录超出允许清理的目录")
                    shutil.rmtree(temporary)
                raise

    def invalidate(self, record_id, reason, revision=None):
        """仅供已确认错误的经验停用；事故关联本身不调用此方法。"""
        if self.write_dir is None:
            raise RuntimeError("未配置可写记忆库")
        if not reason.strip():
            raise ValueError("停用经验必须填写原因")
        self.write_dir.mkdir(parents=True,exist_ok=True)
        with self._lock, (self.write_dir/"invalidations.jsonl").open("a",encoding="utf-8",newline="\n") as stream:
            stream.write(json.dumps({"id":str(record_id),"revision":revision,"reason":reason},ensure_ascii=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
