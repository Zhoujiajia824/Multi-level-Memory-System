"""八向状态的无损数组存储；不改变对象选择或检索指标。

磁盘连续通道为 P2/V2/A2/H1，保留原始 float64 角度，避免 sin/cos
往返或 float32 降精度影响匹配阈值。逻辑 3×3×16 张量按需派生，不重复落盘。
metadata 仅保存协议、参数及原因；对象运动、自车状态、槽位掩码只存数组。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
from .configuration import (grid_distance_config, grid_metric_signature,
                            grid_observation_config, grid_observation_signature)

GRID_STORAGE_VERSION = 'lane-eight-array-1.0'
SLOTS = (('left_front', 'front', 'right_front'), ('left_side', 'ego', 'right_side'),
         ('left_rear', 'rear', 'right_rear'))
KINDS = ('vehicle', 'bicycle', 'pedestrian', 'static_obstacle')
STATES = ('occupied', 'empty', 'unknown', 'not_applicable')
CONTINUOUS_CHANNELS = ('delta_s_m', 'delta_d_m', 'delta_vs_mps', 'delta_vd_mps',
                       'as_mps2', 'ad_mps2', 'heading_rad')
EVIDENCE_FIELDS = ('coverage', 'raw_coverage', 'unobserved_area_m2',
                   'region_area_m2', 'known_area_m2', 'clearance')
_META_FIELDS = ('version', 'representation_version', 'valid', 'reason', 'invalid_state_ids',
                'coordinates', 'distance_config', 'metric_signature', 'observation_config',
                'observation_signature', 'role_range_m', 'assignment_protocol', 'assignment_parameters')
_SLOT_META = ('reason', 'observation_reason', 'selected_candidate_id')
_MOTION = (('position', slice(0, 2)), ('velocity', slice(2, 4)),
           ('acceleration', slice(4, 6)), ('heading', slice(6, 7)))
_BASE_KEYS = {'storage_version', 'layout', 'metadata'}
_ARRAY_SHAPES = {'values': ((3, 3, 7), 'f', 8), 'value_valid': ((3, 3, 7), 'b', 1),
    'kind': ((3, 3), 'i', 1), 'state': ((3, 3), 'i', 1), 'object_index': ((3, 3), 'i', 1),
    'slot_flags': ((3, 3, 3), 'i', 1), 'slot_evidence': ((3, 3, 6), 'f', 8),
    'evidence_present': ((3, 3, 6), 'b', 1), 'evidence_valid': ((3, 3, 6), 'b', 1),
    'ego_velocity': ((2,), 'f', 8), 'roads': ((2, 3), 'i', 1)}


def _error(message):
    raise ValueError('八向数组格式错误：' + message)


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('八向元数据必须为有限JSON基本类型') from exc


def _scalar_text(value, name):
    a = np.asarray(value)
    if a.shape != () or a.dtype.kind != 'U':
        _error(name + '必须是Unicode标量，禁止pickle对象数组')
    return str(a.item())


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _error('元数据包含重复键：' + key)
        result[key] = value
    return result


def _metadata(grid):
    meta = {key: copy.deepcopy(grid[key]) for key in _META_FIELDS if key in grid}
    return _json_copy(meta)


def _validate_metadata(meta, complete):
    """元数据只允许声明过的配置和文本，不成为第二份运动状态容器。"""
    for key in ('version', 'representation_version', 'reason', 'coordinates',
                'metric_signature', 'observation_signature', 'assignment_protocol'):
        if key in meta and not isinstance(meta[key], str):
            _error(key + '必须为字符串')
    if 'invalid_state_ids' in meta and (not isinstance(meta['invalid_state_ids'], list) or
        any(not isinstance(value, str) or not value for value in meta['invalid_state_ids'])):
        _error('无效对象来源必须为字符串ID列表')
    if meta.get('valid') and meta.get('invalid_state_ids'):
        _error('存在无效对象时不能声明八向有效')
    for key, validator, signature, signed in (
        ('distance_config', grid_distance_config, grid_metric_signature, 'metric_signature'),
        ('observation_config', grid_observation_config, grid_observation_signature, 'observation_signature')):
        if complete or key in meta or signed in meta:
            value = meta.get(key)
            if not isinstance(value, dict) or value != validator(value) or meta.get(signed) != signature(value):
                _error(key + '不完整或参数签名不一致')
    for key, fields in (
        ('role_range_m', {'front_m', 'rear_m'}),
        ('assignment_parameters', {'connection_gap_m', 'connection_angle_rad', 'geometry_tolerance_m',
            'owner_heading_weight_m_per_rad', 'owner_motion_min_speed_mps', 'branch_sample_step_m', 'max_chain_segments'})):
        if key in meta:
            value = meta[key]
            if not isinstance(value, dict) or set(value) != fields:
                _error(key + '参数字段不符')
            for name, number in value.items():
                number = _finite_number(number, key + '.' + name)
                if number < 0 or (number == 0 and not (key == 'role_range_m' and name == 'rear_m')):
                    _error(key + '参数必须为正，只有后向范围允许为零')


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        _error(name + '不是数值')
    if not np.isfinite(value):
        _error(name + '不是有限数值')
    return float(value)


def _vector(value, count, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.asarray(value).shape != (count,):
        _error(name + '维数不符')
    return np.array([_finite_number(v, name) for v in value], dtype=np.float64)


def _flag(value, nullable, name):
    if value is None and nullable:
        return -1
    if type(value) is not bool:
        _error(name + '必须是布尔值' + ('或null' if nullable else ''))
    return int(value)


def _motion(row):
    return np.concatenate((_vector(row.get('position'), 2, 'position'),
                           _vector(row.get('velocity'), 2, 'velocity'),
                           _vector(row.get('acceleration'), 2, 'acceleration'),
                           [_finite_number(row.get('heading'), 'heading')]))


def grid_to_arrays(grid):
    """把检索所需信息编码为无对象dtype数组；不保存重复JSON物理状态。

非检索诊断（完整参考线、全部候选归属、世界坐标、对象尺寸）不进入此资产。
缺失的范围或分配协议保持缺失，绝不根据当前默认配置补写历史语义。
"""
    if not isinstance(grid, dict):
        _error('输入不是字典')
    if type(grid.get('valid')) is not bool:
        _error('缺少valid布尔值')
    meta = _metadata(grid)
    base = {'storage_version': np.array(GRID_STORAGE_VERSION), 'layout': np.array(1, dtype=np.uint8)}
    required = ('matrices', 'slots', 'objects', 'ego', 'roads')
    if not all(key in grid for key in required):
        if grid['valid'] or any(key in grid for key in ('matrices', 'slots', 'ego', 'roads')) or grid.get('objects', []) != []:
            _error('八向结构不完整')
        base['layout'] = np.array(0, dtype=np.uint8)
        base['metadata'] = np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')))
        _validate_arrays(base)
        return base
    for name, (shape, kind, size) in _ARRAY_SHAPES.items():
        base[name] = np.zeros(shape, dtype=(np.float64 if kind == 'f' else np.bool_ if kind == 'b' else np.int8))
    base['kind'].fill(-1)
    base['object_index'].fill(-1)
    objects = grid['objects']
    if not isinstance(objects, list):
        _error('objects不是列表')
    by_id = {}
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get('id'), str) or not obj['id']:
            _error('对象ID必须是非空字符串')
        identity = obj['id']
        if obj.get('kind') not in KINDS or not isinstance(obj.get('slots'), list) or not obj['slots']:
            _error('对象类型或区域缺失')
        if len(set(obj['slots'])) != len(obj['slots']) or any(slot not in sum(SLOTS, ()) or slot == 'ego' for slot in obj['slots']):
            _error('对象区域非法或重复')
        state = _motion(obj)
        if identity in by_id:
            previous, previous_state = by_id[identity]
            if previous['kind'] != obj['kind'] or not np.array_equal(previous_state, state):
                _error('同ID跨格对象状态矛盾')
            previous['slots'] = sorted(set(previous['slots']) | set(obj['slots']))
        else:
            by_id[identity] = (dict(obj, slots=list(obj['slots'])), state)
    if len(by_id) > 8:
        _error('唯一周边对象超过8个')
    identities = sorted(by_id)
    base['object_ids'] = np.array(identities, dtype=np.str_)
    indices = {identity: index for index, identity in enumerate(identities)}
    ego = grid['ego']
    ego_id = grid['slots'].get('ego', {}).get('id', ego.get('id'))
    if not isinstance(ego_id, str) or not ego_id or (ego.get('id') is not None and ego['id'] != ego_id):
        _error('自车ID缺失或不一致')
    if ego_id in by_id:
        _error('自车不能同时列为周边对象')
    meta['ego_id'] = ego_id
    meta['slot_reasons'] = {}
    meta['road_reasons'] = {}
    base['ego_velocity'] = _vector(ego.get('velocity'), 2, '自车绝对速度')
    matrices = grid['matrices']
    for name, _ in _MOTION:
        matrix = matrices.get(name)
        if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
            _error('四矩阵必须是3×3')
    if set(grid['slots']) != set(sum(SLOTS, ())):
        _error('八向槽位名称不完整或多余')
    for i, row in enumerate(SLOTS):
        for j, name in enumerate(row):
            slot = grid['slots'][name]
            if slot.get('state') not in STATES:
                _error('区域状态非法：' + name)
            occupied = slot['state'] == 'occupied'
            base['state'][i, j] = STATES.index(slot['state'])
            base['slot_flags'][i, j] = [_flag(slot.get('applicable'), True, name),
                _flag(slot.get('clear'), False, name), _flag(slot.get('state_valid'), False, name)]
            reasons = {key: slot[key] for key in _SLOT_META if key in slot}
            if reasons:
                meta['slot_reasons'][name] = _json_copy(reasons)
            for k, key in enumerate(EVIDENCE_FIELDS):
                if key in slot:
                    base['evidence_present'][i, j, k] = True
                    if slot[key] is not None:
                        base['slot_evidence'][i, j, k] = _finite_number(slot[key], name + '.' + key)
                        base['evidence_valid'][i, j, k] = True
            if occupied:
                if slot.get('kind') not in KINDS:
                    _error('有对象格缺少有效类别')
                base['kind'][i, j] = KINDS.index(slot['kind'])
                if name == 'ego':
                    if slot.get('id') != ego_id:
                        _error('中心格自车ID不一致')
                    expected = np.concatenate((np.zeros(4), _vector(ego.get('acceleration'), 2, '自车加速度'),
                                               [_finite_number(ego.get('heading'), '自车朝向')]))
                    base['object_index'][i, j] = -2
                else:
                    identity = slot.get('id')
                    if identity not in by_id:
                        _error('格子引用不存在的对象')
                    obj, expected = by_id[identity]
                    if obj['kind'] != slot['kind'] or name not in obj['slots']:
                        _error('对象与格子的类别/成员不一致')
                    base['object_index'][i, j] = indices[identity]
                motion = {}
                for key, segment in _MOTION:
                    motion[key] = matrices[key][i][j]
                values = _motion(motion)
                if not np.array_equal(values, expected):
                    _error('矩阵与对象状态不一致')
                base['values'][i, j] = values
                base['value_valid'][i, j] = True
            else:
                if name == 'ego' or slot.get('id') is not None or slot.get('kind') is not None:
                    _error('无对象格不得携带对象身份')
                if any(matrices[key][i][j] is not None for key, _ in _MOTION):
                    _error('空/未知/不适用格必须保留空状态')
    for identity, (obj, _) in by_id.items():
        actual = {name for i, row in enumerate(SLOTS) for j, name in enumerate(row)
                  if base['object_index'][i, j] == indices[identity]}
        if actual != set(obj['slots']):
            _error('对象区域列表与格子索引不一致')
    for i, side in enumerate(('left', 'right')):
        road = grid['roads'].get(side)
        if not isinstance(road, dict):
            _error('缺少左右道路状态')
        base['roads'][i] = [_flag(road.get('known'), False, side),
                            _flag(road.get('exists'), True, side), _flag(road.get('crossable'), True, side)]
        if 'reason' in road:
            meta['road_reasons'][side] = _json_copy(road['reason'])
    base['metadata'] = np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')))
    _validate_arrays(base)
    return base


def _validate_arrays(arrays):
    if not isinstance(arrays, Mapping):
        _error('数组容器不是映射')
    if any(not isinstance(value, np.ndarray) or value.dtype.hasobject for value in arrays.values()):
        _error('每个成员必须是无pickle的NumPy数组')
    if not _BASE_KEYS.issubset(arrays):
        _error('缺少存储协议/元数据')
    if _scalar_text(arrays['storage_version'], 'storage_version') != GRID_STORAGE_VERSION:
        _error('不支持的存储版本')
    layout = np.asarray(arrays['layout'])
    if layout.shape != () or layout.dtype != np.dtype('uint8') or int(layout) not in (0, 1):
        _error('layout非法')
    try:
        meta = json.loads(_scalar_text(arrays['metadata'], 'metadata'), object_pairs_hook=_json_pairs,
                          parse_constant=lambda value: _error('元数据含非有限值'))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError('八向元数据JSON非法') from exc
    if not isinstance(meta, dict) or type(meta.get('valid')) is not bool:
        _error('元数据valid缺失')
    allowed = set(_META_FIELDS) | {'ego_id', 'slot_reasons', 'road_reasons'}
    if set(meta) - allowed:
        _error('元数据含未定义字段或重复数值状态')
    _validate_metadata(meta, bool(int(layout)))
    if int(layout) == 0:
        if set(arrays) != _BASE_KEYS or meta['valid'] or set(meta) - set(_META_FIELDS):
            _error('无效诊断布局不得冒充完整有效八向')
        return meta, False
    if set(arrays) != _BASE_KEYS | set(_ARRAY_SHAPES) | {'object_ids'}:
        _error('数组成员缺失或多余')
    for name, (shape, kind, size) in _ARRAY_SHAPES.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype.kind != kind or value.dtype.itemsize != size:
            _error(name + '形状或dtype错误')
        if kind == 'f' and not np.isfinite(value).all():
            _error(name + '含非有限数值')
    ids = np.asarray(arrays['object_ids'])
    if ids.ndim != 1 or ids.dtype.kind != 'U' or len(ids) > 8 or any(not v for v in ids.tolist()):
        _error('对象ID数组非法')
    if len(set(ids.tolist())) != len(ids) or ids.tolist() != sorted(ids.tolist()):
        _error('对象ID重复或顺序非法')
    if not isinstance(meta.get('ego_id'), str) or not meta['ego_id'] or meta['ego_id'] in ids:
        _error('自车ID非法')
    for key, legal_names, legal_fields in (('slot_reasons', set(sum(SLOTS, ())), set(_SLOT_META)),
                                           ('road_reasons', {'left', 'right'}, None)):
        data = meta.get(key)
        if not isinstance(data, dict) or set(data) - legal_names:
            _error(key + '非法')
        if legal_fields is not None and any(not isinstance(v, dict) or set(v) - legal_fields for v in data.values()):
            _error('槽位原因含未定义字段')
        values = [item for row in data.values() for item in row.values()] if legal_fields is not None else data.values()
        if any(not isinstance(value, str) for value in values):
            _error('道路/槽位原因只能为文本')
    state, kind, index, flags = (np.asarray(arrays[k]) for k in ('state', 'kind', 'object_index', 'slot_flags'))
    if np.any((state < 0) | (state >= len(STATES))) or np.any((kind < -1) | (kind >= len(KINDS))):
        _error('类型/状态码越界')
    if np.any((flags[..., 0] < -1) | (flags[..., 0] > 1)) or np.any((flags[..., 1:] < 0) | (flags[..., 1:] > 1)):
        _error('格子标志越界')
    occupied = state == 0
    if np.any(arrays['value_valid'] != occupied[..., None]):
        _error('连续数值掩码与区域状态不一致')
    if np.any(arrays['values'][~occupied] != 0) or np.any(kind[~occupied] != -1) or np.any(index[~occupied] != -1):
        _error('无对象格含隐藏状态')
    if np.any(kind[occupied] < 0) or not occupied[1, 1] or index[1, 1] != -2 or np.any(arrays['values'][1, 1, :4] != 0):
        _error('中心格或有对象格非法')
    if np.any(flags[occupied] != 1):
        _error('有对象格必须适用且明确有效')
    surrounding = occupied.copy(); surrounding[1, 1] = False
    if np.any((index[surrounding] < 0) | (index[surrounding] >= len(ids))):
        _error('格子对象索引越界')
    if set(index[surrounding].tolist()) != set(range(len(ids))):
        _error('存在没有占格的对象')
    for i in range(len(ids)):
        cells = index == i
        if not np.all(arrays['values'][cells] == arrays['values'][cells][0]) or not np.all(kind[cells] == kind[cells][0]):
            _error('同一对象跨格状态矛盾')
    present, valid, evidence = (np.asarray(arrays[k]) for k in ('evidence_present', 'evidence_valid', 'slot_evidence'))
    if np.any(valid & ~present) or np.any(evidence[~valid] != 0) or np.any(evidence[valid] < 0):
        _error('覆盖证据掩码或数值非法')
    if not present[..., 0].all() or not present[..., 2].all() or np.any(evidence[..., :2] > 1):
        _error('缺少覆盖证据或覆盖率越界')
    empty = state == 1; unknown = state == 2; absent = state == 3
    if np.any(flags[..., 2][~occupied] != 0) or np.any(flags[..., 1][unknown] != 0):
        _error('未知/无对象状态不能冒充有效观测')
    if np.any(flags[empty] != [1, 1, 0]) or np.any(flags[absent] != [0, 1, 0]):
        _error('空/不适用状态与标志矛盾')
    if np.any(~valid[..., 0][empty]) or np.any(evidence[..., 0][empty] < meta['observation_config']['empty_min_coverage']):
        _error('确认空格没有足够覆盖证据')
    if np.any(flags[0:3:2, 1, 0] != 1):
        _error('本车道前后格必须适用')
    roads = np.asarray(arrays['roads'])
    for r, col, side in ((0, 0, 'left'), (1, 2, 'right')):
        known, exists, crossable = roads[r].tolist()
        if known not in (0, 1) or exists not in (-1, 0, 1) or crossable not in (-1, 0, 1):
            _error('道路状态码越界')
        if known == 0:
            if (exists, crossable) != (-1, -1) or not np.all(state[:, col] == 2) or not np.all(flags[:, col, 0] == -1):
                _error('未知道路与区域证据矛盾')
        elif exists < 0 or crossable < 0 or (crossable and not exists):
            _error('已知道路状态组合非法')
        elif crossable:
            if not np.all(flags[:, col, 0] == 1) or np.any(state[:, col] == 3):
                _error('可跨越车道区域不适用')
        else:
            if not np.all(state[:, col] == 3):
                _error('无车道/禁跨区域不能包含对象')
            reason = 'noncrossable_boundary' if exists else 'no_adjacent_lane'
            if any(meta['slot_reasons'].get(SLOTS[i][col], {}).get('reason') != reason for i in range(3)):
                _error('禁跨/无车道缺少对应原因')
    return meta, True


def grid_from_arrays(arrays):
    """恢复现有调用接口需要的字典；原始物理量不作归一化或推断。"""
    meta, complete = _validate_arrays(arrays)
    grid = {key: copy.deepcopy(value) for key, value in meta.items() if key in _META_FIELDS}
    grid['objects'] = []
    if not complete:
        return grid
    grid['matrices'] = {key: [[None] * 3 for _ in range(3)] for key, _ in _MOTION}
    grid['slots'] = {}
    grid['roads'] = {}
    ids = arrays['object_ids'].tolist()
    objects = {}
    for i, row in enumerate(SLOTS):
        for j, name in enumerate(row):
            flags = arrays['slot_flags'][i, j]
            state = STATES[int(arrays['state'][i, j])]
            index = int(arrays['object_index'][i, j])
            slot = {'state': state, 'id': None, 'kind': None, 'applicable': None if flags[0] < 0 else bool(flags[0]),
                    'clear': bool(flags[1]), 'state_valid': bool(flags[2])}
            slot.update(meta['slot_reasons'].get(name, {}))
            for k, key in enumerate(EVIDENCE_FIELDS):
                if arrays['evidence_present'][i, j, k]:
                    slot[key] = float(arrays['slot_evidence'][i, j, k]) if arrays['evidence_valid'][i, j, k] else None
            if state == 'occupied':
                slot['id'] = meta['ego_id'] if index == -2 else ids[index]
                slot['kind'] = KINDS[int(arrays['kind'][i, j])]
                motion = {}
                for key, segment in _MOTION:
                    values = arrays['values'][i, j, segment]
                    motion[key] = float(values[0]) if key == 'heading' else values.tolist()
                    grid['matrices'][key][i][j] = motion[key]
                if index == -2:
                    grid['ego'] = {'id': meta['ego_id'], 'velocity': arrays['ego_velocity'].tolist(),
                                   'acceleration': motion['acceleration'], 'heading': motion['heading']}
                else:
                    obj = objects.setdefault(index, dict(id=slot['id'], kind=slot['kind'], **motion, slots=[], state_valid=True))
                    obj['slots'].append(name)
            grid['slots'][name] = slot
    grid['objects'] = [objects[index] for index in sorted(objects)]
    for i, side in enumerate(('left', 'right')):
        known, exists, crossable = arrays['roads'][i].tolist()
        grid['roads'][side] = dict(known=bool(known), exists=None if exists < 0 else bool(exists),
                                   crossable=None if crossable < 0 else bool(crossable))
        if side in meta['road_reasons']:
            grid['roads'][side]['reason'] = meta['road_reasons'][side]
    return grid


def save_grid(path, grid):
    """写入无pickle NPZ；文件路径和BytesIO都可用。由调用方负责发布事务。"""
    arrays = grid_to_arrays(grid)
    if hasattr(path, 'write'):
        np.savez_compressed(path, **arrays)
    else:
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, **arrays)


def load_grid(path):
    """只接受本协议的数组资产；损坏文件绝不静默回退旧JSON。"""
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            _error('NPZ包含重复成员')
        arrays = {key: archive[key] for key in archive.files}
    return grid_from_arrays(arrays)


def grid_tensor16(grid_or_arrays):
    """按需生成3×3×16张量及连续有效掩码，不保存归一化/展开副本。"""
    arrays = grid_to_arrays(grid_or_arrays) if isinstance(grid_or_arrays, dict) and 'storage_version' not in grid_or_arrays else grid_or_arrays
    _, complete = _validate_arrays(arrays)
    if not complete:
        _error('无效诊断布局没有可生成的张量')
    values = arrays['values']; occupied = arrays['state'] == 0
    result = np.zeros((3, 3, 16), dtype=np.float64)
    result[..., :6] = values[..., :6]
    result[..., 6] = np.where(occupied, np.sin(values[..., 6]), 0.)
    result[..., 7] = np.where(occupied, np.cos(values[..., 6]), 0.)
    for i in range(4):
        result[..., 8+i] = arrays['kind'] == i
        result[..., 12+i] = arrays['state'] == i
    valid = np.concatenate((arrays['value_valid'][..., :6], np.repeat(arrays['value_valid'][..., 6:7], 2, axis=2)), axis=2)
    return result, valid
