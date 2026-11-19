"""离线回顾经验的独立只读适配；不导入采集项目，不伪造在线答案或评分。"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np

from .configuration import (GRID_METRIC_VERSION, OCC_METRIC_VERSION,
    READABLE_RETRIEVAL_METRIC_VERSIONS, grid_metric_signature,
    occupancy_distance_config, occupancy_metric_signature)
from .contracts import VERSION, Representations
from .paths import filesystem_path
from .grid_storage import GRID_STORAGE_VERSION, grid_to_arrays, load_grid
from .representations import FEATURE_VERSION, GRID_REPRESENTATION_VERSION, OCC_REPRESENTATION_VERSION
from .retrieval import representation_channel_eligibility

READER_PROTOCOL = 'offline-retrospective-reader-1.1'
LIBRARY_PROTOCOL = 'retrospective-driving-experience-0.1'
ASSET_PROTOCOL = 'offline-current-feature-assets-1.0'
ANNOTATION_PROTOCOL = 'retrospective-driving-experience-1.0'
SEMANTIC_ANNOTATION_PROTOCOL = 'retrospective-driving-experience-2.0'
READABLE_ANNOTATION_PROTOCOLS = (ANNOTATION_PROTOCOL, SEMANTIC_ANNOTATION_PROTOCOL)
SECTIONS = ('当前情境', '后续过程', '驾驶行为及依据', '警惕对象与区域', '驾驶经验', '适用条件与边界')
SEMANTIC_SECTIONS = ('驾驶场景', '八向环境', '关键交互', '可能演化', '关注对象与区域',
                     '驾驶判断', '实际行为与结果', '驾驶经验')
DIRECTIONS = ('前方', '后方', '左侧', '右侧', '左前', '右前', '左后', '右后')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode('utf-8'))


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('离线资产JSON含重复字段')
        result[key] = value
    return result


def _json(value):
    result=json.loads(value.decode('utf-8'), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('离线资产含非有限数')))
    if not isinstance(result,dict):
        raise ValueError('离线JSON资产最外层必须为映射')
    return result


def _grid_asset(value, relative, storage_version=None):
    """仅解码哈希校验后的声明资产，不将坏 NPZ 静默替换为旧 JSON。"""
    if not isinstance(relative,str) or not relative:
        raise ValueError('离线八向资产缺少路径声明')
    suffix=Path(relative).suffix.lower()
    if suffix not in ('.npz','.json'):
        raise ValueError('离线八向资产必须声明 .npz 或旧 .json')
    if storage_version is not None and (storage_version!=GRID_STORAGE_VERSION or suffix!='.npz'):
        raise ValueError('离线八向存储版本与声明文件格式不符')
    return load_grid(BytesIO(value)) if suffix=='.npz' else _json(value)


def _same_grid_evidence(original, current, original_path, current_path):
    """BEV 派生仍不得改变八向证据；NPZ 只舍弃非检索诊断字段。"""
    if all(Path(name).suffix.lower()=='.json' for name in (original_path,current_path)):
        return {k:v for k,v in original.items() if k!='version'}=={k:v for k,v in current.items() if k!='version'}
    # 原守卫唯一允许变化的字段是表示 version；格式迁移后比较其余规范存储证据。
    left=grid_to_arrays(dict(original,version=FEATURE_VERSION))
    right=grid_to_arrays(dict(current,version=FEATURE_VERSION))
    return left.keys()==right.keys() and all(np.array_equal(left[k],right[k]) for k in left)


def _path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('离线资产路径必须是非空相对路径')
    target = filesystem_path(root / relative)
    if not target.is_relative_to(root):
        raise ValueError('离线资产路径越出冻结库')
    return target


def is_offline_library(root):
    """有发布清单即走严格读取；损坏清单不能静默退回在线库读取。"""
    root=filesystem_path(root)
    return (root/'manifest.json').exists() or (root/'freeze.json').exists()


def ensure_writable_online(root):
    if root is not None and is_offline_library(root):
        raise ValueError('离线发布库是只读冻结库；请指定独立在线写入目录，不能在原库写入或停用')


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _same_time(a,b):
    return _number(a) and _number(b) and math.isclose(a,b,abs_tol=1e-8,rel_tol=0)


def _experience_text(experience, protocol, maximum):
    """按教师协议校验并展示完整正文；旧记录保持原格式，不补段或自动转换。"""
    if protocol not in READABLE_ANNOTATION_PROTOCOLS:
        raise ValueError('离线教师经验协议不受支持')
    if (not isinstance(experience, dict) or experience.get('status') != 'accept'
        or experience.get('reject_reason') is not None):
        raise ValueError('离线经验必须经过接纳，且不能含拒收原因')
    if protocol == ANNOTATION_PROTOCOL:
        if (set(experience) != {'status', 'sections', 'uncertainty', 'reject_reason'}
            or not isinstance(experience.get('sections'), dict)
            or set(experience['sections']) != set(SECTIONS)
            or any(not isinstance(x, str) or not x.strip() for x in experience['sections'].values())
            or not isinstance(experience.get('uncertainty'), str)):
            raise ValueError('旧版离线经验必须保留完整六段及不确定项')
        leaves = list(experience['sections'].values()) + [experience['uncertainty']]
        text = '\n\n'.join(f'【{name}】\n{experience["sections"][name]}' for name in SECTIONS)
        if experience['uncertainty'].strip():
            text += '\n\n【不确定项】\n' + experience['uncertainty']
    else:
        if (set(experience) != {'status', 'experience', 'reject_reason'}
            or not isinstance(experience.get('experience'), dict)
            or set(experience['experience']) != set(SEMANTIC_SECTIONS)):
            raise ValueError('新版离线经验必须保留完整语义字段')
        body = experience['experience']
        directions = body['八向环境']
        if not isinstance(directions, dict) or set(directions) != set(DIRECTIONS):
            raise ValueError('新版离线经验须逐项描述八个方向')
        leaves = [body[name] for name in SEMANTIC_SECTIONS if name != '八向环境'] + list(directions.values())
        if any(not isinstance(value, str) or not value.strip() for value in leaves):
            raise ValueError('新版离线经验正文与八向描述必须为非空文本')
        blocks = []
        for name in SEMANTIC_SECTIONS:
            value = ('\n'.join(f'{direction}：{directions[direction]}' for direction in DIRECTIONS)
                     if name == '八向环境' else body[name])
            blocks.append(f'【{name}】\n{value}')
        text = '\n\n'.join(blocks)
    if type(maximum) is not int or maximum < 100 or sum(len(value) for value in leaves) > maximum:
        raise ValueError('离线经验超出标注长度协议，不截断经验')
    return text


def read_snapshot(root, check_dino_identity, *, allow_test_features=False, invalidated=()):
    """返回已加载内容和逐条拒绝审计；清单损坏则拒绝整库。"""
    root = filesystem_path(root)
    manifest_bytes = (root / 'manifest.json').read_bytes()
    manifest, freeze = _json(manifest_bytes), _json((root / 'freeze.json').read_bytes())
    if (not isinstance(manifest, dict) or manifest.get('protocol') != LIBRARY_PROTOCOL
        or manifest.get('frozen') is not True or not isinstance(manifest.get('records'), list)
        or not isinstance(freeze, dict) or freeze.get('manifest_sha256') != _sha(manifest_bytes)):
        raise ValueError('离线发布清单协议、冻结标志或哈希不符')
    if manifest.get('contains_test_records') is True and not allow_test_features:
        raise ValueError('正式入口拒绝包含测试替身经验的离线库')
    entries, seen = manifest['records'], set()
    # 同ID冲突属于清单损坏，不能靠读取先后来决定保留哪个。
    for item in entries:
        identity = item.get('id') if isinstance(item, dict) else None
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{32}', identity) or identity in seen:
            raise ValueError('离线发布清单ID非法或重复')
        if item.get('path') != f'records/{identity}/record.json':
            raise ValueError('离线记录路径与清单ID不一致')
        seen.add(identity)
    records, audit = [], []
    invalidated = set(invalidated)
    for item in entries:
        try:
            record = _read_record(root, item, check_dino_identity, allow_test_features)
            if (record['id'], record['revision']) in invalidated or (record['id'], None) in invalidated:
                audit.append({'id': record['id'], 'reason': '经验已停用'})
            else:
                records.append(record)
        except (ValueError, KeyError, TypeError, OSError, OverflowError, BadZipFile, EOFError) as error:
            audit.append({'path': str(root / item['path']), 'reason': str(error)})
    return records, audit


def _read_record(root, item, check_dino_identity, allow_test_features):
    folder = _path(root, item['path']).parent
    sums_raw = (folder / 'checksums.json').read_bytes()
    if _sha(sums_raw) != item.get('checksums_sha256'):
        raise ValueError('离线资产清单哈希不符')
    sums = _json(sums_raw)
    if not isinstance(sums, dict) or not sums:
        raise ValueError('离线资产校验映射缺失')

    def checked(relative):
        if relative not in sums or not isinstance(sums[relative], str):
            raise ValueError('离线资产缺少哈希：' + str(relative))
        data = _path(folder, relative).read_bytes()
        if _sha(data) != sums[relative]:
            raise ValueError('离线资产缺失或校验不符：' + relative)
        return data

    # 全部证据（含未来与审核）都检查；真正参与查询的数组只读取evidence/0。
    for name in sums:
        checked(name)
    metadata = checked('record.json')
    record = _json(metadata)
    if (record.get('protocol') != LIBRARY_PROTOCOL or record.get('record_origin') != 'offline_retrospective'
        or record.get('id') != item['id'] or record.get('status') != 'active'
        or type(record.get('revision')) is not int or record['revision'] < 1
        or not isinstance(record.get('frame_id'), str) or not record['frame_id']
        or not _number(record.get('time')) or record.get('feature_dir') != 'evidence/0'
        or record.get('text_path') != 'experience.txt'):
        raise ValueError('离线记录身份、协议、状态或当前资产目录非法')
    if record.get('test_only') is True and not allow_test_features:
        raise ValueError('正式入口拒绝测试教师替身经验')
    if any(record.get(key) is not False for key in ('annotation_is_ground_truth', 'mpc_tested', 'pdm_tested')):
        raise ValueError('离线文本不能冒充人工真值、MPC或PDM验收')
    task = _json(checked('audit/task.json'))
    draft = _json(checked('audit/experience.json'))
    review = _json(checked('audit/review.json'))
    if (task.get('id') != record['id'] or task.get('protocol') != LIBRARY_PROTOCOL
        or task.get('anchor_frame_id') != record['frame_id'] or not _same_time(task.get('anchor_time_s'),record['time'])
        or draft.get('task_id') != record['id'] or draft.get('protocol') != LIBRARY_PROTOCOL
        or review.get('decision') != 'approve' or not isinstance(review.get('reviewer'), str)
        or not review['reviewer'].strip() or review.get('draft_sha256') != _sha(checked('audit/experience.json'))):
        raise ValueError('离线人工审核或原帧绑定不符')
    timeline = task['timeline']
    if [row['offset_s'] for row in timeline] != [0., 1., 2., 3.]:
        raise ValueError('离线证据必须包含当前与未来1/2/3秒')
    tolerance=task.get('sampling',{}).get('time_tolerance_s')
    if not _number(tolerance) or tolerance<0:
        raise ValueError('离线证据缺少有效的时间采样容限')
    evidence_hashes = {'task': _digest(task), 'supervision': _sha(checked('audit/supervision.json'))}
    declared = record.get('evidence')
    if not isinstance(declared, list) or len(declared) != 4:
        raise ValueError('离线证据清单缺失')
    for i, row in enumerate(timeline):
        if (not _number(row['offset_s']) or not _number(row['time_s'])
            or not math.isclose(row['time_s'], record['time'] + row['offset_s'], abs_tol=tolerance, rel_tol=0)
            or declared[i] != {'offset_s': row['offset_s'], 'time_s': row['time_s'], 'path': f'evidence/{i}'}) :
            raise ValueError('离线证据时间或路径不一致')
        names = ['features.json', 'scene.json'] + ([row['native_image']] if row.get('native_image') else [])
        if not row.get('native_image'):
            raise ValueError('离线原生图像证据缺失')
        evidence_hashes[str(row['offset_s'])] = {name: _sha(checked(f'evidence/{i}/{name}')) for name in names}
        evidence_meta=_json(checked(f'evidence/{i}/features.json'))
        scene=_json(checked(f'evidence/{i}/scene.json'))
        if (evidence_meta.get('frame_id')!=row['frame_id'] or scene.get('frame_id')!=row['frame_id']
            or not _same_time(evidence_meta.get('time'),row['time_s']) or not _same_time(scene.get('time'),row['time_s'])):
            raise ValueError('离线证据的图像/状态原帧或时间不一致')
    evidence_digest = _digest(evidence_hashes)
    if review.get('evidence_digest') != evidence_digest or record.get('evidence_digest_at_review') != evidence_digest:
        raise ValueError('离线审核后的素材发生变化')
    derivation = record.get('feature_derivation')
    if record.get('feature_migration_id') is not None and derivation is None:
        raise ValueError('离线特征重建记录缺少原始教师证据来源')
    if derivation is not None:
        # 特征重建不是重新调用教师：同时验证旧模型所见证据与当前衍生检索资产。
        if (not isinstance(derivation,dict) or derivation.get('protocol') != 'offline-bev-rebuild-1.0'
            or derivation.get('source_version') != 'bev-occ-grid-2.0'
            or derivation.get('target_version') != FEATURE_VERSION
            or derivation.get('teacher_input_replaced') is not False
            or derivation.get('experience_changed') is not False
            or derivation.get('current_retrieval_regenerated') is not True):
            raise ValueError('离线特征重建来源契约不符')
        migration = derivation.get('migration_id')
        if (not isinstance(migration,str) or not re.fullmatch('bev21_[0-9a-f]{32}',migration)
            or record.get('feature_migration_id')!=migration):
            raise ValueError('离线特征重建身份非法')
        prefix = 'audit/feature_source/' + migration
        expected_paths = {'source_feature_dir':prefix+'/evidence0','source_review':prefix+'/review.json',
                          'source_record':prefix+'/record.json','source_draft':prefix+'/experience.json'}
        if any(derivation.get(k)!=v for k,v in expected_paths.items()):
            raise ValueError('离线重建原证据目录非法')
        original_review=_json(checked(derivation['source_review']))
        original_record=_json(checked(derivation['source_record']))
        original_draft_bytes=checked(derivation['source_draft'])
        original_draft=_json(original_draft_bytes)
        original_dir=derivation['source_feature_dir']
        original_meta=_json(checked(original_dir+'/features.json'))
        if (original_record.get('id')!=record['id'] or original_record.get('frame_id')!=record['frame_id']
            or original_draft.get('task_id')!=record['id'] or original_draft.get('annotation')!=draft.get('annotation')
            or original_review.get('decision')!='approve'
            or original_review.get('draft_sha256')!=_sha(original_draft_bytes)
            or original_record.get('experience')!=original_draft.get('experience')
            or original_meta.get('representation_version')!=derivation['source_version']
            or original_meta.get('frame_id')!=record['frame_id'] or not _same_time(original_meta.get('time'),record['time'])):
            raise ValueError('离线重建前的情境、答案或审核绑定不符')
        old_hashes = dict(evidence_hashes)
        old_hashes[str(timeline[0]['offset_s'])] = {
            name:_sha(checked(original_dir+'/'+name)) for name in ['features.json','scene.json',timeline[0]['native_image']]}
        old_digest=_digest(old_hashes)
        if (original_review.get('evidence_digest')!=old_digest
            or original_record.get('evidence_digest_at_review')!=old_digest
            or checked(original_dir+'/scene.json')!=checked('evidence/0/scene.json')
            or checked(original_dir+'/'+timeline[0]['native_image'])!=checked('evidence/0/'+timeline[0]['native_image'])):
            raise ValueError('离线重建改变原始教师场景或后续证据')
        if not isinstance(original_meta.get('paths'),dict) or not isinstance(original_meta.get('sha256'),dict):
            raise ValueError('离线重建原模型输入资产索引不合法')
        for key,relative in original_meta.get('paths',{}).items():
            _path(folder/original_dir,relative)
            if _sha(checked(original_dir+'/'+relative))!=original_meta.get('sha256',{}).get(key):
                raise ValueError('离线重建原模型输入资产校验失败：'+key)
    annotation_id = draft.get('annotation')
    if not isinstance(annotation_id, str) or not re.fullmatch('[0-9a-f]{64}', annotation_id):
        raise ValueError('离线标注请求指纹缺失')
    annotation = _json(checked(f'audit/annotations/{annotation_id}.json'))
    identity = annotation.get('identity', {})
    annotation_protocol = annotation.get('protocol_version')
    if (annotation.get('status') != 'completed' or annotation_protocol not in READABLE_ANNOTATION_PROTOCOLS
        or not isinstance(identity,dict)
        or identity.get('protocol_version') != annotation_protocol or identity.get('mode') not in ('api', 'server')
        or annotation.get('request_fingerprint') != annotation_id):
        raise ValueError('离线原始教师标注协议或请求绑定不符')
    if annotation.get('test_only') is True and not allow_test_features:
        raise ValueError('正式入口拒绝测试教师替身经验')
    experience = record.get('experience')
    if experience != draft.get('experience'):
        raise ValueError('离线经验与已审核草稿不一致')
    text = _experience_text(experience, annotation_protocol, identity.get('max_characters'))
    # Windows旧发布器使用平台换行；仅统一换行，不截断、重写或丢弃任一段经验。
    canonical=lambda value:value.replace('\r\n','\n').replace('\r','\n')
    if canonical(checked('experience.txt').decode('utf-8')) != canonical(text):
        raise ValueError('离线展示经验与已审核完整答案不一致')
    supervision = _json(checked('audit/supervision.json'))
    if (any(row.get('collision') or row.get('offroad') for row in supervision['audit'])
        and review.get('adverse_events_acknowledged') is not True):
        raise ValueError('事故或越界经验缺少人工反面教训确认')

    data = _json(checked('evidence/0/features.json'))
    versions = {'asset_protocol': ASSET_PROTOCOL, 'version': VERSION, 'representation_version': FEATURE_VERSION,
                'grid_representation_version': GRID_REPRESENTATION_VERSION,
                'occupancy_representation_version': OCC_REPRESENTATION_VERSION,
                'grid_metric_version': GRID_METRIC_VERSION, 'occupancy_metric_version': OCC_METRIC_VERSION}
    if (any(data.get(key) != value for key, value in versions.items())
        or data.get('retrieval_metric_version') not in READABLE_RETRIEVAL_METRIC_VERSIONS
        or data.get('status') not in ('ready', 'evidence_insufficient', 'invalid')
        or data.get('current_only') is not True
        or data.get('frame_id') != record['frame_id'] or not _same_time(data.get('time'),record['time'])):
        raise ValueError('离线当前特征状态、版本或原帧不符')
    check_dino_identity(data.get('dino_signature'))
    config = data.get('occupancy_config')
    if (not isinstance(config, dict) or config != occupancy_distance_config(config)
        or data.get('occupancy_metric_signature') != occupancy_metric_signature(config)):
        raise ValueError('离线OCC参数或指标签名不符')
    paths = data['paths']
    if not isinstance(paths,dict) or not isinstance(data.get('sha256'),dict):
        raise ValueError('离线特征路径或哈希映射缺失')
    for key in ('bev', 'feature_image', 'occupancy', 'grid', 'grid_preview', 'bev_feature'):
        if key not in paths:
            raise ValueError('离线当前特征资产缺失：' + key)
    def feature_bytes(key):
        _path(folder/'evidence/0',paths[key])
        value = checked(f'evidence/0/{paths[key]}')
        if _sha(value) != data['sha256'].get(key):
            raise ValueError('离线特征内部资产校验不符：' + key)
        return value
    # 校验所有特征资产，数组从这一次校验的字节读取，避免检查后又读取另一版本。
    assets = {key: feature_bytes(key) for key in paths}
    with np.load(BytesIO(assets['occupancy']), allow_pickle=False) as occupied:
        occ, known = occupied['occ'].copy(), occupied['known'].copy()
    vector = np.load(BytesIO(assets['bev_feature']), allow_pickle=False)
    if not isinstance(vector,np.ndarray):
        vector.close()
        raise ValueError('离线BEV向量必须是单一npy数组')
    grid = _grid_asset(assets['grid'],paths['grid'],data.get('grid_storage_version'))
    if derivation is not None:
        original_paths=original_meta.get('paths',{})
        if (not all(k in original_paths for k in ('bev','feature_image','occupancy','grid','grid_preview','bev_feature'))
            or original_meta.get('dino_signature')!=data.get('dino_signature')
            or checked(original_dir+'/'+original_paths['occupancy'])!=assets['occupancy']):
            raise ValueError('离线BEV重建不允许改变DINO身份或OCC占据')
        original_grid=_grid_asset(checked(original_dir+'/'+original_paths['grid']),
                                  original_paths['grid'],original_meta.get('grid_storage_version'))
        if not _same_grid_evidence(original_grid,grid,original_paths['grid'],paths['grid']):
            raise ValueError('离线BEV重建不允许改变八向状态或观测证据')
    if data.get('grid_metric_signature') != grid_metric_signature(grid.get('distance_config')):
        raise ValueError('离线八向距离参数签名不符')
    reps = Representations(record['frame_id'], record['time'],
        str(_path(folder, 'evidence/0/' + paths['bev'])), str(_path(folder, 'evidence/0/' + paths['feature_image'])),
        occ, known, grid, vector, version=data['version'], coordinates=data['coordinates'],
        dino_signature=data['dino_signature'], occupancy_config=config)
    reps.grid_path=str(_path(folder,'evidence/0/'+paths['grid']))
    # 原文件的旧准入结论是审计证据，不改写或伪装为新版指标。
    # 文件与审核链完整以后，各通道按当前规则独立重算，未知状态仅阻止对应通道。
    eligibility = representation_channel_eligibility(reps)
    if not eligibility['eligible']:
        reasons = [name + '：' + report['reason']
                   for name, report in eligibility['channels'].items()]
        raise ValueError('离线经验没有有效检索通道：' + '；'.join(reasons))
    record.update(text=text, offline_text=text, checksums=sums, _representations=reps,
                  _asset_root=str(folder), _metadata_sha256=_sha(metadata),
                  retrieval_metric_version=data['retrieval_metric_version'],
                  dino_signature=data['dino_signature'], reader_protocol=READER_PROTOCOL,
                  retrieval_eligibility=eligibility)
    return record
