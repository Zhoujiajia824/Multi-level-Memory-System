"""Independent, read-only browser for published driving-experience libraries.

Only manifest-listed records are indexed. Integrity checks follow freeze ->
manifest -> checksums -> record/asset, and do not assert retrieval compatibility.
This module never imports a driving program, migrates a grid, or writes a file.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import threading
from urllib.parse import urlencode
import zipfile

import numpy as np

_HASH = re.compile(r"^[0-9a-f]{64}$")
_IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_SLOTS = (("left_front", "front", "right_front"), ("left_side", "ego", "right_side"),
          ("left_rear", "rear", "right_rear"))
_KINDS = ("vehicle", "bicycle", "pedestrian", "static_obstacle")
_STATES = ("occupied", "empty", "unknown", "not_applicable")
_CHANNELS = ("delta_s_m", "delta_d_m", "delta_vs_mps", "delta_vd_mps", "as_mps2", "ad_mps2", "heading_rad")
_EVIDENCE = ("coverage", "raw_coverage", "unobserved_area_m2", "region_area_m2", "known_area_m2", "clearance")


def _disk(path):
    value = str(Path(path).absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        return "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return value


def _inside(path, root):
    try:
        return os.path.normcase(os.path.commonpath([str(path), str(root)])) == os.path.normcase(str(root))
    except ValueError:
        return False


def _relative(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("文件路径必须是非空库内相对路径")
    name = value.replace("\\", "/")
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ":" in name or not parts or any(p in ("..", ".history", "pending") for p in parts):
        raise ValueError("拒绝绝对路径、父级路径或未发布目录")
    return PurePosixPath(*parts).as_posix()


def _stat(path):
    try:
        st = os.stat(_disk(path))
        return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
    except OSError:
        return None


def _bytes(path, maximum=16 * 1024 * 1024):
    stat = _stat(path)
    if stat is None or stat[0] > maximum:
        raise ValueError("文件缺失或超过只读展示的大小限制")
    with open(_disk(path), "rb") as stream:
        return stream.read(maximum + 1)


def _json(path):
    return json.loads(_bytes(path).decode("utf-8-sig"))


def _sha(path):
    digest = hashlib.sha256()
    with open(_disk(path), "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _array_summary(array, path, name):
    result = {"path": path, "name": name, "shape": list(array.shape), "dtype": str(array.dtype),
              "elements": int(array.size), "bytes": int(array.nbytes)}
    if array.dtype.hasobject:
        raise ValueError("禁止pickle/object数组")
    if array.dtype.kind in "biuf" and array.size:
        finite = np.isfinite(array)
        result["finite_count"] = int(finite.sum())
        if finite.any():
            result["minimum"], result["maximum"] = float(array[finite].min()), float(array[finite].max())
    return result


def _load_arrays(path):
    # Bound decompressed size before opening an NPZ; never enable pickle.
    if str(path).lower().endswith(".npz"):
        with zipfile.ZipFile(_disk(path)) as package:
            if len(package.infolist()) > 64 or sum(p.file_size for p in package.infolist()) > 64 * 1024 * 1024:
                raise ValueError("NPZ解压后过大，拒绝在浏览器中加载")
        with np.load(_disk(path), allow_pickle=False) as package:
            return {name: package[name] for name in package.files}
    array = np.load(_disk(path), allow_pickle=False)
    if array.nbytes > 64 * 1024 * 1024 or array.dtype.hasobject:
        raise ValueError("数组过大或包含pickle对象")
    return {"array": array}


def _grid_arrays(arrays):
    """Decode the published seven-channel storage without importing its writer."""
    version = arrays.get("storage_version")
    if version is None or version.shape != () or version.dtype.kind != "U" or version.item() != "lane-eight-array-1.0":
        raise ValueError("未知八向数组存储版本，保留原始数组摘要而不猜测编码")
    meta = arrays.get("metadata")
    if meta is None or meta.shape != () or meta.dtype.kind != "U":
        raise ValueError("八向metadata必须是无pickle的Unicode JSON标量")
    metadata = json.loads(meta.item())
    if not isinstance(metadata, dict):
        raise ValueError("八向metadata JSON必须是对象")
    if not isinstance(metadata.get("slot_reasons", {}), dict):
        raise ValueError("八向槽位原因必须是映射")
    layout = arrays.get("layout")
    if layout is None or layout.shape != () or layout.dtype.kind not in "iu" or int(layout) not in (0, 1):
        raise ValueError("八向数组布局无效")
    if int(layout) == 0:
        return {"available": False, "storage_version": version.item(), "metadata": metadata,
                "reason": metadata.get("reason", "归档仅含无效状态诊断，没有完整张量"), "slots": []}
    required = {"values": (3, 3, 7), "value_valid": (3, 3, 7), "kind": (3, 3), "state": (3, 3),
                "object_index": (3, 3), "slot_flags": (3, 3, 3), "slot_evidence": (3, 3, 6),
                "evidence_present": (3, 3, 6), "evidence_valid": (3, 3, 6)}
    for key, shape in required.items():
        if key not in arrays or arrays[key].shape != shape or arrays[key].dtype.hasobject:
            raise ValueError("八向数组缺失或形状错误：" + key)
    ids = arrays.get("object_ids")
    if ids is None or ids.ndim != 1 or ids.dtype.kind != "U" or len(ids) > 8:
        raise ValueError("八向对象ID索引无效")
    values, valid = arrays["values"], arrays["value_valid"]
    if values.dtype.kind != "f" or valid.dtype.kind != "b" or not np.isfinite(values).all():
        raise ValueError("八向连续量或有效性掩码无效")
    if any(arrays[key].dtype.kind not in "iu" for key in ("kind", "state", "object_index", "slot_flags")):
        raise ValueError("八向枚举和标志必须是整数编码")
    if any(arrays[key].dtype.kind != "b" for key in ("evidence_present", "evidence_valid")):
        raise ValueError("八向区域证据掩码必须是布尔值")
    if arrays["slot_evidence"].dtype.kind != "f" or not np.isfinite(arrays["slot_evidence"]).all():
        raise ValueError("八向区域证据连续量无效")
    slots = []
    for i, names in enumerate(_SLOTS):
        row = []
        for j, name in enumerate(names):
            state, kind, index = (int(arrays[key][i, j]) for key in ("state", "kind", "object_index"))
            if not 0 <= state < len(_STATES) or not -1 <= kind < len(_KINDS):
                raise ValueError("八向状态/类型码越界")
            occupied = state == 0
            identity = None
            if occupied:
                if kind < 0:
                    raise ValueError("占用区域缺少对象类别")
                if name == "ego" and index == -2:
                    identity = metadata.get("ego_id")
                elif 0 <= index < len(ids):
                    identity = str(ids[index])
                else:
                    raise ValueError("八向对象索引越界")
            elif np.any(valid[i, j]):
                raise ValueError("空/未知/不适用格不能包含有效运动数值")
            if not occupied and (kind != -1 or index != -1):
                raise ValueError("非占用区域不能含隐藏对象类别或索引")
            flags = arrays["slot_flags"][i, j]
            if not -1 <= flags[0] <= 1 or np.any(flags[1:] < 0) or np.any(flags[1:] > 1):
                raise ValueError("八向区域标志码无效")
            continuous = [float(v) if bool(ok) else None for v, ok in zip(values[i, j], valid[i, j])]
            item = {"role": name, "state": _STATES[state], "id": identity,
                "kind": _KINDS[kind] if kind >= 0 else None, "values": continuous,
                "position": continuous[:2], "velocity": continuous[2:4],
                "acceleration": continuous[4:6], "heading": continuous[6],
                "applicable": None if flags[0] < 0 else bool(flags[0]),
                "clear": bool(flags[1]), "state_valid": bool(flags[2]),
                }
            reason = metadata.get("slot_reasons", {}).get(name, {})
            if isinstance(reason, dict):
                item.update(reason)
            for k, field in enumerate(_EVIDENCE):
                if arrays["evidence_present"][i, j, k]:
                    item[field] = float(arrays["slot_evidence"][i, j, k]) if arrays["evidence_valid"][i, j, k] else None
            row.append(item)
        slots.append(row)
    return {"available": True, "storage_version": version.item(), "shape": [3, 3, 7],
        "channels": list(_CHANNELS), "values": [[slot["values"] for slot in row] for row in slots],
        "value_valid": valid.tolist(), "slots": slots, "metadata": metadata,
        "coordinates": metadata.get("coordinates"), "valid": metadata.get("valid"),
        "note": "仅解码归档数组供查看；不重新计算对象匹配或检索资格。掩码无效的0保留为null"}


def _grid_json(value):
    if not isinstance(value, dict):
        raise ValueError("旧八向JSON不是对象")
    matrices = value.get("matrices") or {}
    rows, tensor, masks = [], [], []
    for i, names in enumerate(_SLOTS):
        row, vals, valid_row = [], [], []
        for j, name in enumerate(names):
            original = value.get("slots", {}).get(name)
            if not isinstance(original, dict):
                raise ValueError("旧八向JSON缺少区域：" + name)
            values = []
            for key, n in (("position", 2), ("velocity", 2), ("acceleration", 2), ("heading", 1)):
                matrix = matrices.get(key)
                cell = matrix[i][j] if isinstance(matrix, list) and len(matrix) == 3 else None
                if n == 1:
                    values.append(cell)
                elif cell is None:
                    values.extend([None] * n)
                elif isinstance(cell, list) and len(cell) == n:
                    values.extend(cell)
                else:
                    raise ValueError("旧八向运动矩阵形状错误")
            if original.get("state") != "occupied" and any(v is not None for v in values):
                raise ValueError("旧八向非占用区域包含运动值")
            values = _finite(values)
            row.append({**original, "role": name, "values": values, "position": values[:2],
                        "velocity": values[2:4], "acceleration": values[4:6], "heading": values[6]})
            vals.append(values)
            valid_row.append([v is not None for v in values])
        rows.append(row); tensor.append(vals); masks.append(valid_row)
    return {"available": True, "storage_version": "legacy_grid_json_read_only", "shape": [3, 3, 7],
        "channels": list(_CHANNELS), "values": tensor, "value_valid": masks, "slots": rows,
        "metadata": {k: v for k, v in value.items() if k not in ("slots", "matrices", "objects")},
        "coordinates": value.get("coordinates"), "valid": value.get("valid"),
        "note": "原始旧JSON只读展开供查看，未迁移、重匹配或保存张量"}


class LibraryStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError("离线记忆库目录不存在")
        self._lock = threading.RLock()
        self._stamp = None
        self._entries, self._contexts, self._verified = {}, OrderedDict(), {}
        self._index = None
        self._refresh()
        if self._manifest_error is not None:
            hint = ""
            if self.root.name.casefold() == "records" and (self.root.parent / "manifest.json").is_file():
                hint = ("；当前选择的是 records 子目录，请将 --memory-root 设为父目录 "
                        + str(self.root.parent) + "（未自动切换目录）")
            raise ValueError("离线记忆库根目录必须包含可读且合法的 manifest.json："
                             + str(self.root) + hint + "；原因：" + self._manifest_error)

    def _path(self, base, relative):
        name = _relative(relative)
        path = (base / Path(*PurePosixPath(name).parts)).resolve()
        if not _inside(path, base) or not _inside(path, self.root):
            raise ValueError("符号链接或文件路径越过库/经验目录边界")
        if any(part in (".history", "pending") for part in path.relative_to(self.root).parts):
            raise ValueError("符号链接指向未发布目录")
        return path

    def _refresh(self):
        stamp = (_stat(self.root / "manifest.json"), _stat(self.root / "freeze.json"))
        if stamp == self._stamp:
            return
        self._stamp = stamp
        self._contexts.clear(); self._verified.clear(); self._index = None
        self._warnings, self._entries = [], {}
        self._manifest, self._manifest_sha = {}, None
        self._manifest_error = None
        self._freeze_valid = False
        try:
            manifest_path = self._path(self.root, "manifest.json")
            content = _bytes(manifest_path)
            self._manifest_sha = hashlib.sha256(content).hexdigest()
            manifest = json.loads(content.decode("utf-8-sig"))
            if not isinstance(manifest, dict) or not isinstance(manifest.get("records"), list):
                raise ValueError("manifest缺少公开records列表")
            entries = {}
            for row in manifest["records"]:
                if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
                    raise ValueError("manifest记录必须包含非空字符串ID")
                identity = row["id"]
                if _relative(identity) != identity or "/" in identity:
                    raise ValueError("manifest记录ID不是合法单层目录名：" + identity)
                if identity in entries:
                    raise ValueError("manifest出现重复ID：" + identity)
                if row.get("path") != "records/" + identity + "/record.json":
                    raise ValueError("manifest记录路径必须是records/ID/record.json：" + identity)
                checksum = row.get("checksums_sha256")
                if not isinstance(checksum, str) or not _HASH.fullmatch(checksum):
                    raise ValueError("manifest记录缺少合法checksums_sha256：" + identity)
                entries[identity] = dict(row)
            self._manifest, self._entries = manifest, entries
            try:
                freeze = _json(self._path(self.root, "freeze.json"))
                self._freeze_valid = freeze.get("manifest_sha256") == self._manifest_sha
            except (OSError, ValueError, TypeError, AttributeError):
                self._freeze_valid = False
            if manifest.get("frozen") and not self._freeze_valid:
                self._warnings.append("冻结manifest哈希校验失败；可审阅元数据，禁止发布图片资产")
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            self._manifest_error = str(exc)
            self._manifest, self._entries = {}, {}
            self._warnings.append("无法读取库manifest：" + self._manifest_error)

    def _context(self, identity):
        if identity not in self._entries:
            raise KeyError("经验ID未登记在公开manifest中")
        if identity in self._contexts:
            cached = self._contexts[identity]
            if cached.get("_tracked_stats") == [(str(p), _stat(p)) for p in cached.get("_tracked_paths", [])]:
                self._contexts.move_to_end(identity)
                return cached
            del self._contexts[identity]
        entry = self._entries[identity]
        warnings, record, checksums, base = [], {}, {}, None
        integrity = {"freeze_valid": self._freeze_valid, "checksums_verified": False,
                     "record_verified": False, "text_verified": False, "assets": "on_demand"}
        text, tracked = "", []
        try:
            relative = _relative(entry.get("path"))
            if _relative(identity) != identity or "/" in identity or relative != "records/" + identity + "/record.json":
                raise ValueError("公开记录路径必须是records/ID/record.json")
            record_path = self._path(self.root, relative)
            base = record_path.parent
            expected_base = self.root / "records" / identity
            if os.path.normcase(str(base)) != os.path.normcase(str(expected_base)):
                raise ValueError("公开记录目录不能通过链接指向其他经验")
            checksum_path = self._path(base, "checksums.json")
            tracked.extend((record_path, checksum_path))
            declared = entry.get("checksums_sha256")
            if not isinstance(declared, str) or not _HASH.fullmatch(declared) or _sha(checksum_path) != declared:
                raise ValueError("manifest登记的checksums.json哈希不匹配")
            value = _json(checksum_path)
            if not isinstance(value, dict):
                raise ValueError("checksums.json不是文件摘要映射")
            for name, checksum in value.items():
                name = _relative(name)
                if not isinstance(checksum, str) or not _HASH.fullmatch(checksum):
                    raise ValueError("登记的资产SHA256格式无效")
                checksums[name] = checksum
            integrity["checksums_verified"] = True
            if checksums.get("record.json") != _sha(record_path):
                raise ValueError("record.json未登记或内容哈希不匹配")
            record = _json(record_path)
            if not isinstance(record, dict) or record.get("id") != identity:
                raise ValueError("记录ID与manifest不一致")
            integrity["record_verified"] = True
            text_name = _relative(record.get("text_path", "experience.txt"))
            text_path = self._verified_path(base, checksums, text_name)
            tracked.append(text_path)
            text = _bytes(text_path, 2 * 1024 * 1024).decode("utf-8-sig")
            integrity["text_verified"] = True
        except (OSError, ValueError, TypeError, UnicodeError) as exc:
            warnings.append(str(exc))
        context = dict(entry=entry, base=base, record=record, checksums=checksums,
                       integrity=integrity, warnings=warnings, experience=text,
                       _tracked_paths=tracked, _tracked_stats=[(str(p), _stat(p)) for p in tracked])
        self._contexts[identity] = context
        if len(self._contexts) > 64:
            self._contexts.popitem(last=False)
        return context

    def _verified_path(self, base, checksums, relative):
        name = _relative(relative)
        if name not in checksums:
            raise ValueError("文件未登记在该经验checksums中")
        path = self._path(base, name)
        stat = _stat(path)
        if stat is None or not path.is_file():
            raise ValueError("登记文件缺失")
        key = (str(path), checksums[name], stat)
        if key not in self._verified:
            if _sha(path) != checksums[name]:
                raise ValueError("登记文件内容哈希不匹配：" + name)
            self._verified[key] = True
            if len(self._verified) > 8192:
                self._verified.clear()
                self._verified[key] = True
        return path

    def _make_index(self):
        if self._index is not None:
            return self._index
        rows = []
        for identity in self._entries:
            context = self._context(identity)
            record, text = context["record"], context["experience"]
            declaration = record.get("retrieval_eligibility")
            row = {"id": identity, "source": record.get("source", "unknown"),
                "clip_id": record.get("clip_id"), "frame_id": record.get("frame_id"),
                "time": record.get("time"), "status": record.get("status", "unreadable"),
                "preview": text[:230], "text_preview": text[:230],
                "mpc_tested": record.get("mpc_tested"), "pdm_tested": record.get("pdm_tested"),
                "declared_retrieval_eligible": declaration.get("eligible") if isinstance(declaration, dict) else None,
                "integrity": context["integrity"], "warnings": context["warnings"],
                "_search": " ".join(str(v) for v in (identity, record.get("source", ""), record.get("clip_id", ""),
                    record.get("frame_id", ""), text)).casefold()}
            rows.append(row)
        self._index = rows
        return rows

    def summary(self):
        with self._lock:
            self._refresh()
            rows = self._make_index()
            sources = dict(Counter(str(row["source"]) for row in rows))
            statuses = dict(Counter(str(row["status"]) for row in rows))
            eligibility = Counter("missing" if row["declared_retrieval_eligible"] is None else
                                  "true" if row["declared_retrieval_eligible"] is True else "false" for row in rows)
            return {"configured": True, "manifest_valid": self._manifest_error is None,
                "root": str(self.root), "protocol": self._manifest.get("protocol"),
                "frozen": self._manifest.get("frozen", False), "freeze_valid": self._freeze_valid,
                "manifest_sha256": self._manifest_sha, "count": len(rows), "source_counts": sources,
                "status_counts": statuses, "sources": sorted(sources), "statuses": sorted(statuses),
                "declared_eligibility_counts": dict(eligibility), "compatibility": self._manifest.get("compatibility"),
                "verification_note": "可浏览与冻结完整性不等同主程序检索兼容；eligible仅为记录声明，未重新执行检索验证",
                "read_only": True, "warnings": list(self._warnings),
                "records_with_warnings": sum(bool(row["warnings"]) for row in rows)}

    def records(self, query="", source="", status="", offset=0, limit=40):
        with self._lock:
            self._refresh()
            if isinstance(offset, bool) or isinstance(limit, bool):
                raise ValueError("分页参数必须是整数")
            offset, limit = max(0, int(offset)), max(1, min(100, int(limit)))
            terms = str(query).casefold().split()
            selected = [row for row in self._make_index() if (not source or row["source"] == source)
                and (not status or row["status"] == status) and all(term in row["_search"] for term in terms)]
            rows = [{key: deepcopy(value) for key, value in row.items() if key != "_search"}
                    for row in selected[offset:offset + limit]]
            return _finite({"records": rows, "total": len(selected), "offset": offset, "limit": limit, "read_only": True})

    def _asset(self, identity, context, name, label, role=None):
        if self._manifest.get("frozen") and not self._freeze_valid:
            raise ValueError("冻结manifest校验失败，拒绝发布图片")
        if not context["integrity"]["record_verified"]:
            raise ValueError("经验元数据未通过完整性校验，拒绝发布图片")
        self._verified_path(context["base"], context["checksums"], name)
        return {"path": name, "url": "/api/library/asset?" + urlencode({"id": identity, "path": name}),
                "label": label, "role": role, "registered": True, "sha256_verified": True}

    def detail(self, identity, evidence_index=0):
        with self._lock:
            self._refresh()
            context = self._context(str(identity))
            record = context["record"]
            warnings = list(context["warnings"])
            evidence = []
            anchor = record.get("feature_dir")
            for i, row in enumerate(record.get("evidence") or []):
                if not isinstance(row, dict):
                    continue
                evidence.append({**row, "index": i,
                    "role": "retrieval_anchor" if row.get("path") == anchor else "retrospective_review",
                    "note": "当前检索锚点" if row.get("path") == anchor else "归档后续观测，仅供离线经验审阅，不是在线未来预测或当前模型输入"})
            if not evidence and anchor:
                evidence = [{"index": 0, "path": anchor, "time_s": record.get("time"), "offset_s": 0., "role": "retrieval_anchor"}]
            if isinstance(evidence_index, bool) or int(evidence_index) < 0:
                raise ValueError("证据序号必须是非负整数")
            index = int(evidence_index)
            selected = next((row for row in evidence if row["index"] == index), None)
            result = {"id": str(identity), "record": deepcopy(record), "experience": context["experience"],
                "evidence": evidence, "selected_evidence": index, "assets": {}, "features": {},
                "arrays": [], "grid": None, "occupancy": None, "integrity": dict(context["integrity"]),
                "warnings": warnings, "read_only": True}
            if not record or not context["integrity"]["record_verified"]:
                return result
            if selected is None:
                raise IndexError("该经验不存在所选证据序号")
            try:
                folder = _relative(selected["path"])
                feature_name = folder + "/features.json"
                features = _json(self._verified_path(context["base"], context["checksums"], feature_name))
                if not isinstance(features, dict):
                    raise ValueError("features.json不是元数据对象")
                result["features"] = features
                source_meta = None
                source_name = folder + "/source_assets.json"
                if source_name in context["checksums"]:
                    source_meta = _json(self._verified_path(context["base"], context["checksums"], source_name))
                result["source_assets"] = source_meta
                paths = dict(features.get("paths") or {})
                for semantic, leaf in (("native", "native.png"), ("grid", "grid.png")):
                    if folder + "/" + leaf in context["checksums"]:
                        paths.setdefault("native_preview" if semantic == "native" else "grid_preview", leaf)
                for key, relative in paths.items():
                    try:
                        name = _relative(folder + "/" + _relative(relative))
                        suffix = PurePosixPath(name).suffix.lower()
                        if suffix in _IMAGES:
                            normalized = {"feature_image": "feature", "grid_preview": "grid", "native_preview": "native"}.get(key, key)
                            label = {"native": "原始native预览（类型按归档，不默认称RGB相机）", "bev": "BEV", "grid": "八向图预览", "feature": "BEV特征输入图"}.get(normalized, key)
                            role = "source_audit_image" if key.startswith("source_") else "recorded_preview"
                            result["assets"][normalized] = self._asset(str(identity), context, name, label, role)
                        elif suffix in (".npz", ".npy") and key in ("occupancy", "bev_feature", "grid"):
                            file_path = self._verified_path(context["base"], context["checksums"], name)
                            arrays = _load_arrays(file_path)
                            result["arrays"].extend(_array_summary(array, name, member) for member, array in arrays.items())
                            if key == "grid":
                                result["grid"] = _grid_arrays(arrays)
                            elif key == "occupancy":
                                occ, known = arrays.get("occ"), arrays.get("known")
                                if (occ is not None and known is not None and occ.shape == known.shape and
                                        occ.ndim == 3 and occ.size <= 30000 and occ.dtype.kind == "f" and known.dtype.kind == "f"):
                                    result["occupancy"] = {"available": True, "shape": list(occ.shape),
                                        "values": _finite(occ.tolist()), "known": _finite(known.tolist()),
                                        "channels": ["crosswalk", "ego", "vehicle", "bicycle", "pedestrian", "static_obstacle"] if occ.shape[0] == 6 else None,
                                        "source": "stored_occ_and_known", "note": "原始归档占据量及已知掩码；未重建或补全未知区域，没有生成新图片"}
                        elif suffix == ".json" and key == "grid":
                            result["grid"] = _grid_json(_json(self._verified_path(context["base"], context["checksums"], name)))
                    except (OSError, ValueError, TypeError, KeyError, IndexError, zipfile.BadZipFile) as exc:
                        warnings.append(str(key) + "：" + str(exc))
                result["camera_note"] = "source_*只显示实际归档的源审计图；不推断同步曝光，不套用CARLA标定或新增3D相机投影"
            except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
                warnings.append("证据读取失败：" + str(exc))
            # Return convenient aliases for a UI without duplicating tensor storage.
            result["feature_summary"] = result["arrays"]
            result["grid_tensor"] = result["grid"]
            return _finite(result)

    def _future_data(self, context, selected, scene):
        """Authenticated retrospective execution, separate from every online input."""
        name = 'audit/supervision.json'
        missing = dict(available=False, source_path=name,
                       reason='该经验未登记未来实驶监督；不以导航参考代替轨迹')
        if name not in context['checksums']:
            return missing
        try:
            value = _json(self._verified_path(context['base'], context['checksums'], name))
            if not isinstance(value, dict) or not isinstance(value.get('source'), str) or not value['source']:
                raise ValueError('未来实驶监督缺少明确来源')
            rows = value.get('ego_trajectory')
            if not isinstance(rows, list) or not 1 <= len(rows) <= 512:
                raise ValueError('未来实驶监督轨迹点数量无效')
            times = []
            for row in rows:
                if not isinstance(row, dict) or isinstance(row.get('time_s'), bool):
                    raise ValueError('未来实驶监督轨迹点无效')
                t = float(row['time_s'])
                xy = row.get('position')
                if not isinstance(xy, list) or len(xy) < 2 or any(isinstance(v, bool) for v in xy[:2]):
                    raise ValueError('未来实驶监督缺少世界坐标')
                if not math.isfinite(t) or not all(math.isfinite(float(v)) for v in xy[:2]):
                    raise ValueError('未来实驶监督含非有限坐标/时间')
                times.append(t)
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError('未来实驶监督时间不严格递增')
            record = context['record']
            anchor_name = _relative(record['feature_dir']) + '/scene.json'
            anchor = scene if selected.get('role') == 'retrieval_anchor' else _json(
                self._verified_path(context['base'], context['checksums'], anchor_name))
            if (anchor.get('frame_id') != record.get('frame_id') or
                    abs(float(anchor['time']) - float(record['time'])) > 1e-5 or
                    abs(times[0] - float(record['time'])) > 1e-5):
                raise ValueError('未来实驶监督不属于该经验锚点时间')
            anchor_xy = anchor['ego']['position']
            if math.dist([float(v) for v in rows[0]['position'][:2]],
                         [float(v) for v in anchor_xy[:2]]) > .05:
                raise ValueError('未来实驶监督锚点与scene自车中心不一致')
            current = next((row for t, row in zip(times, rows)
                            if abs(t - float(scene['time'])) <= 1e-4), None)
            if current is None or math.dist([float(v) for v in current['position'][:2]],
                    [float(v) for v in scene['ego']['position'][:2]]) > .05:
                raise ValueError('所选证据与未来实驶记录没有同一自车状态，停止裁剪显示')
            return {**value, 'available': True, 'source_path': name,
                    'anchor_frame_id': record['frame_id'], 'anchor_time': float(record['time']),
                    'evidence_frame_id': scene['frame_id'], 'evidence_time': float(scene['time']),
                    'usage': 'offline_retrospective_display_only', 'sha256_verified': True}
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
            return {**missing, 'reason': '未来实驶监督不可显示：' + str(exc)}

    def _camera_data(self, identity, context, selected, scene, features, source_assets):
        """Bind calibration and exported images within this evidence's hash chain.

        Original absolute dataset paths are never opened. View associations must
        agree in the export manifest, registered source mapping and image hash.
        """
        result = dict(available=False, source=context['record'].get('source'),
                      frame_id=scene['frame_id'], time=scene['time'],
                      evidence_index=selected['index'], images=[], warnings=[])
        try:
            paths = features.get('paths') or {}
            folder = _relative(selected['path'])
            def source_json(leaf):
                names = [_relative(folder + '/' + _relative(relative))
                         for relative in paths.values() if isinstance(relative, str)
                         and PurePosixPath(relative.replace('\\', '/')).name.endswith(leaf)]
                if len(names) != 1 or not names[0].startswith(folder + '/source/'):
                    raise ValueError('当前证据缺少唯一已登记标定源：' + leaf)
                return _json(self._verified_path(context['base'], context['checksums'], names[0]))
            manifest = source_json('camera_manifest_original.json')
            exports = source_json('cameras_exported.json')
            original = source_json('scene_original.json')
            if not isinstance(manifest, list) or not isinstance(exports, list) or not isinstance(original, dict):
                raise ValueError('相机标定或原始场景格式无效')
            files = source_assets.get('files') if isinstance(source_assets, dict) else None
            if not isinstance(files, list):
                raise ValueError('缺少本证据的源图像登记映射')
            view_counts = Counter(row.get('view') for row in exports if isinstance(row, dict)
                                  and isinstance(row.get('view'), str))
            for export in exports:
                try:
                    if not isinstance(export, dict) or not isinstance(export.get('view'), str):
                        raise ValueError('相机导出记录缺少view')
                    view, filename, checksum = export['view'], export.get('exported_filename'), export.get('sha256')
                    if view_counts[view] != 1:
                        raise ValueError('相机view重复，拒绝不明确的映射')
                    if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
                        raise ValueError('相机导出文件名无效')
                    if not isinstance(checksum, str) or not _HASH.fullmatch(checksum):
                        raise ValueError('相机导出记录未保存有效图像哈希')
                    candidates = [row for row in files if isinstance(row, dict)
                        and isinstance(row.get('original_relative_path'), str)
                        and PurePosixPath(row['original_relative_path'].replace('\\', '/')).name == filename
                        and row.get('sha256') == checksum]
                    if len(candidates) != 1:
                        raise ValueError('导出图像与本证据源映射不唯一或哈希不同')
                    name = _relative(folder + '/' + _relative(candidates[0]['path']))
                    if not name.startswith(folder + '/source/') or PurePosixPath(name).suffix.lower() not in _IMAGES:
                        raise ValueError('相机图片不在本证据source内')
                    if context['checksums'].get(name) != checksum:
                        raise ValueError('导出图像哈希与checksums不一致')
                    image = self._asset(identity, context, name, view, 'calibrated_camera_source')
                    result['images'].append({**image, 'view': view, 'sha256': checksum})
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    result['warnings'].append('相机资产：' + str(exc))
            if not result['images']:
                raise ValueError('没有通过归档映射和哈希校验的相机图片')
            result.update(available=True, manifest=manifest, exports=exports, original_scene=original,
                          integrity='verified_current_evidence_assets')
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
            result['reason'] = '归档相机暂不可投影：' + str(exc)
        return result

    def scene_data(self, identity, evidence_index=0):
        """Read one verified evidence snapshot; never treat later evidence as a plan."""
        with self._lock:
            self._refresh()
            context = self._context(str(identity))
            detail = self.detail(identity, evidence_index)
            selected = next((row for row in detail['evidence']
                             if row['index'] == int(evidence_index)), None)
            result = dict(available=False, record_id=str(identity), evidence_index=int(evidence_index),
                          source=context['record'].get('source'), read_only=True,
                          evidence_role=selected.get('role') if selected else None)
            try:
                if not context['integrity']['record_verified']:
                    raise ValueError('经验记录未通过完整性校验')
                if self._manifest.get('frozen') and not self._freeze_valid:
                    raise ValueError('冻结清单校验未通过，停止场景渲染')
                if selected is None:
                    raise ValueError('未登记所选证据帧')
                name = _relative(selected['path']) + '/scene.json'
                result['scene_source'] = name
                scene = _json(self._verified_path(context['base'], context['checksums'], name))
                if not isinstance(scene, dict):
                    raise ValueError('归档scene.json不是场景对象')
                features = detail.get('features') or {}
                frame_id, timestamp = scene.get('frame_id'), scene.get('time')
                if not isinstance(frame_id, str) or not frame_id or frame_id != features.get('frame_id'):
                    raise ValueError('场景帧号与该证据的特征帧号不一致或缺失')
                def same_time(value):
                    try:
                        return not isinstance(value, bool) and not isinstance(timestamp, bool) and \
                            math.isfinite(float(value)) and math.isfinite(float(timestamp)) and \
                            abs(float(value) - float(timestamp)) <= 1e-5
                    except (ValueError, TypeError):
                        return False
                if not same_time(features.get('time')) or not same_time(selected.get('time_s')):
                    raise ValueError('场景时间与该证据的特征或登记时间不一致或缺失')
                record = context['record']
                if selected['role'] == 'retrieval_anchor' and (
                        frame_id != record.get('frame_id') or not same_time(record.get('time'))):
                    raise ValueError('检索锚点场景与经验登记的原帧不一致')
                result.update(available=True, scene=scene, grid=detail.get('grid'),
                              frame_id=frame_id, time=timestamp,
                              future=self._future_data(context, selected, scene),
                              camera_data=self._camera_data(str(identity), context, selected, scene,
                                  features, detail.get('source_assets')))
            except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
                result['reason'] = str(exc)
            return _finite(result)

    def resolve_asset(self, identity, relative):
        with self._lock:
            self._refresh()
            if self._manifest.get("frozen") and not self._freeze_valid:
                raise ValueError("冻结manifest校验失败，拒绝发布图片")
            context = self._context(str(identity))
            if not context["integrity"]["record_verified"]:
                raise ValueError("经验元数据未通过完整性校验，拒绝发布图片")
            name = _relative(relative)
            if PurePosixPath(name).suffix.lower() not in _IMAGES:
                raise ValueError("该接口只允许已登记的图片资产")
            path = self._verified_path(context["base"], context["checksums"], name)
            return Path(_disk(path)) if os.name == "nt" and len(str(path)) >= 260 else path
