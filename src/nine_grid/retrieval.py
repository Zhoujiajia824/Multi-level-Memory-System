"""三通道分别取Top-1与GOSPA启发的区域约束对象集合匹配。"""
from __future__ import annotations
import math
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contracts import Representations, RetrievalResult, plain
from .representations import (FEATURE_VERSION, GRID_REPRESENTATION_VERSION,
                              OCC_REPRESENTATION_VERSION, SLOTS, valid_dino_signature)
from .configuration import (GRID_METRIC_VERSION, OCC_METRIC_VERSION, RETRIEVAL_METRIC_VERSION,
    grid_distance_config, grid_metric_signature, grid_observation_config, grid_observation_signature,
    occupancy_distance_config, occupancy_metric_signature)

REGIONS = tuple(s for row in SLOTS for s in row if s != 'ego')
SCORE_CHANNELS = (0, 2, 3, 4, 5)
CLASS_NAMES = ('crosswalk','ego','vehicle','bicycle','pedestrian','static_obstacle')


def _invalid(reason, **details):
    return dict(valid=False, score=None, reason=reason, **details)


def _finite(value):
    return isinstance(value, (int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value)


def _vector(value, size=2):
    try:
        values=np.asarray(value,dtype=float)
        return values.shape==(size,) and bool(np.all(np.isfinite(values)))
    except (ValueError,TypeError,OverflowError):
        return False


def _unit_score(value):
    """只修复舍入误差；真正越界不能通过任意截断掩盖。"""
    if not math.isfinite(value) or value < -1e-10 or value > 1+1e-10:
        raise ValueError('相似度超出[0,1]')
    return float(min(1.,max(0.,value)))


def bev_similarity(a,b):
    if a is None or b is None or not _vector(a,768) or not _vector(b,768):
        return None
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    # 损坏的超大有限向量也不能让一次检索崩溃；真实预训练输出会先归一化。
    with np.errstate(over='ignore',invalid='ignore'):
        denominator=np.linalg.norm(a)*np.linalg.norm(b)
        numerator=float(np.dot(a,b))
    if not math.isfinite(denominator) or not math.isfinite(numerator) or denominator<=1e-12:
        return None
    return _unit_score(max(0.,numerator/denominator))


def occupancy_comparison(a,b,known_a,known_b,config=None):
    """面积软IoU、已知前景可比性、确认双空分别验收，不把0/0定义为1。"""
    cfg=occupancy_distance_config(config)
    try:
        a,b,ka,kb=(np.asarray(v,dtype=float) for v in (a,b,known_a,known_b))
    except (ValueError,TypeError,OverflowError):
        return _invalid('占据或观测支持不是数值数组')
    if any(v.shape!=(6,50,25) for v in (a,b,ka,kb)):
        return _invalid('占据或观测支持尺寸不符')
    if not all(np.all(np.isfinite(v)) and np.all((v>=0)&(v<=1)) for v in (a,b,ka,kb)):
        return _invalid('占据或观测支持非有限或超出[0,1]')
    common=np.minimum(ka,kb)
    details=dict(common_known_fraction={},foreground_comparable_fraction={},per_class={},empty_consistency=False)
    active=[]
    area=4.  # 检索单元2米见方，面积阈值统一用平方米。
    epsilon,tolerance=cfg['min_union_area_m2'],cfg['area_tolerance_m2']
    for k in SCORE_CHANNELS:
        name=CLASS_NAMES[k]
        support=float(common[k].mean())
        details['common_known_fraction'][name]=support
        retention=[]
        for values,own,side in ((a[k],ka[k],'query'),(b[k],kb[k],'memory')):
            total=float(values.sum()*area)
            if float(np.sum(values*(own<=0))*area)>tolerance:
                return _invalid(name+'前景缺少本方观测支持',**details)
            own_mass=float(np.sum(values*own)*area)
            compared=float(np.sum(values*common[k])*area)
            fraction=compared/own_mass if own_mass>epsilon else None
            retention.append(dict(side=side,total_area_m2=total,known_area_m2=own_mass,
                                  comparable_area_m2=compared,fraction=fraction))
        details['foreground_comparable_fraction'][name]=retention
        if support<cfg['min_common_known_fraction']:
            return _invalid(name+'共同观测覆盖不足',**details)
        for item in retention:
            if item['fraction'] is not None and item['comparable_area_m2']+tolerance<cfg['min_foreground_comparable_fraction']*item['known_area_m2']:
                return _invalid(name+'已知前景在共同观测区留存不足',**details)
        union=float(np.sum(common[k]*np.maximum(a[k],b[k]))*area)
        intersection=float(np.sum(common[k]*np.minimum(a[k],b[k]))*area)
        details['per_class'][name]=dict(intersection_area_m2=intersection,union_area_m2=union,
                                        score=intersection/union if union>epsilon else None)
        if union>epsilon:
            active.append(intersection/union)
    if active:
        return dict(valid=True,score=_unit_score(float(np.mean(active))),reason='有效非空占据比较',**details)
    # 逐类双空不加奖励；全空须验证本方完整前景和双方空观测覆盖。
    for k in SCORE_CHANNELS:
        if float(a[k].sum()*area)>epsilon or float(b[k].sum()*area)>epsilon:
            return _invalid('共同区为空但本方仍有不可比较前景',**details)
        if min(float(ka[k].mean()),float(kb[k].mean()))<cfg['empty_min_known_fraction']:
            return _invalid('没有足够观测证据确认双方为空',**details)
    details['empty_consistency']=True
    return dict(valid=True,score=1.,reason='双方确认空占据一致',**details)


def occupancy_similarity(a,b,known_a,known_b,config=None):
    return occupancy_comparison(a,b,known_a,known_b,config)['score']


def _distances(a,b,config,ego=False):
    scales=config['scales']
    groups=[('velocity','delta_V',('vs_mps','vd_mps')),('acceleration','delta_A',('as_mps2','ad_mps2'))]
    if not ego:
        groups.insert(0,('position','delta_P',('s_m','d_m')))
    result={}
    for key,name,axes in groups:
        result[name]=float(np.linalg.norm((np.asarray(a[key])-np.asarray(b[key]))/
            [scales[axes[0]],scales[axes[1]]])/math.sqrt(2))
    result['delta_H']=abs((float(a['heading'])-float(b['heading'])+math.pi)%(2*math.pi)-math.pi)/scales['heading_rad']
    weights=config['weights']
    cost=sum(weights[key]*result[name] for key,name in
             [('velocity','delta_V'),('acceleration','delta_A'),('heading','delta_H')])
    cost=cost/sum(weights[key] for key in ('velocity','acceleration','heading')) if ego else cost+weights['position']*result['delta_P']
    return float(cost),result


def _unique_objects(grid):
    """ID仅在一个场景内部合并，跨场景代价不比较ID。"""
    result={}
    for item in grid['objects']:
        identity=item['id']
        if identity in result:
            if any(not np.array_equal(result[identity][k],item[k]) for k in ('kind','position','velocity','acceleration','heading')):
                raise ValueError('同ID跨格对象状态不一致')
            result[identity]['slots'].update(item['slots'])
        else:
            result[identity]=dict(item,slots=set(item['slots']))
    return [result[k] for k in sorted(result)]


def validate_grid_schema(grid):
    """校验结构和原始证据。未知可保存，但不允许默认为已知空。"""
    try:
        if not isinstance(grid,dict) or grid.get('version')!=FEATURE_VERSION or grid.get('representation_version')!=GRID_REPRESENTATION_VERSION:
            return ['八向表示版本不匹配或缺失']
        if grid.get('valid') is not True:
            return ['八向投影无效']
        for key,validator,signature,signed in (
            ('distance_config',grid_distance_config,grid_metric_signature,'metric_signature'),
            ('observation_config',grid_observation_config,grid_observation_signature,'observation_signature')):
            if not isinstance(grid.get(key),dict) or grid[key]!=validator(grid[key]):
                return ['八向参数缺失或不完整：'+key]
            if grid.get(signed)!=signature(grid[key]):
                return ['八向参数签名不匹配：'+key]
        ego=grid['ego']
        if any(not _vector(ego.get(k)) for k in ('velocity','acceleration')) or not _finite(ego.get('heading')):
            return ['自车绝对Frenet运动状态缺失或非法']
        for side in ('left','right'):
            road=grid['roads'][side]
            if type(road['known']) is not bool:
                return ['道路已知掩码非法']
            if road['known']:
                if type(road['exists']) is not bool or type(road['crossable']) is not bool or (road['crossable'] and not road['exists']):
                    return ['道路存在与跨越属性组合非法']
            elif road['exists'] is not None or road['crossable'] is not None:
                return ['道路未知不能补成布尔属性']
            for name in (s for s in REGIONS if s.startswith(side)):
                slot=grid['slots'][name]
                if not road['known']:
                    if slot.get('state')!='unknown' or slot.get('applicable') is not None:
                        return ['道路未知时对应格必须保留未知']
                elif road['crossable']:
                    if slot.get('applicable') is not True or slot.get('state')=='not_applicable':
                        return ['可跨越邻道必须属于适用区域']
                elif (slot.get('applicable') is not False or slot.get('state')!='not_applicable' or
                    slot.get('reason')!=('noncrossable_boundary' if road['exists'] else 'no_adjacent_lane')):
                    return ['无邻道或禁跨道路与格子不适用证据不一致']
        if any(grid['slots'][s].get('applicable') is not True for s in ('front','rear')):
            return ['本车道前后格属于适用区域，缺少几何时必须保留未知']
        if not isinstance(grid['objects'],list):
            return ['八向对象必须为列表']
        for obj in grid['objects']:
            if (not isinstance(obj.get('id'),str) or not obj['id'] or obj.get('kind') not in ('vehicle','bicycle','pedestrian')
                or any(not _vector(obj.get(k)) for k in ('position','velocity','acceleration'))
                or not _finite(obj.get('heading')) or not isinstance(obj.get('slots'),list)
                or not obj['slots'] or not set(obj['slots']).issubset(REGIONS)):
                return ['八向对象类别、角色或运动状态不完整']
        objects=_unique_objects(grid)
        if len(objects)>8:
            return ['八向最多选择八个不同对象']
        by_id={o['id']:o for o in objects}
        matrices=grid['matrices']
        for key in ('position','velocity','acceleration','heading'):
            if (not isinstance(matrices[key],list) or len(matrices[key])!=3 or
                any(not isinstance(row,list) or len(row)!=3 for row in matrices[key])):
                return ['四矩阵必须分别为3×3当前状态']
            center=[0.,0.] if key in ('position','velocity') else ego[key]
            if not np.array_equal(matrices[key][1][1],center):
                return ['矩阵自车中心状态与绝对运动接口不一致']
        for name in REGIONS:
            slot=grid['slots'][name]
            state=slot['state']
            if state not in ('occupied','empty','unknown','not_applicable') or type(slot['clear']) is not bool or type(slot['state_valid']) is not bool:
                return ['格子状态或明确掩码缺失']
            if slot['applicable'] is not None and type(slot['applicable']) is not bool:
                return ['格子适用性非法']
            for field in ('coverage','unobserved_area_m2'):
                if field not in slot:
                    return ['格子观测覆盖证据缺失']
                if slot[field] is not None and (not _finite(slot[field]) or slot[field]<0 or (field=='coverage' and slot[field]>1)):
                    return ['格子覆盖率或未知面积非法']
            if state=='occupied':
                if slot['applicable'] is not True or not slot['clear'] or not slot['state_valid'] or slot.get('id') not in by_id or name not in by_id[slot['id']]['slots']:
                    return ['对象与格子证据不一致']
                if slot.get('kind')!=by_id[slot['id']]['kind']:
                    return ['格子类别与对象状态不一致']
            elif slot.get('id') is not None or slot['state_valid']:
                return ['空、未知或不适用格不能冒充有效对象']
            elif slot.get('kind') is not None:
                return ['无对象格子不能保留对象类别']
            elif state=='empty' and (slot['applicable'] is not True or not slot['clear'] or slot['coverage'] is None or slot['coverage']<grid['observation_config']['empty_min_coverage']):
                return ['已观测空格缺少覆盖证据']
            elif state=='unknown' and slot['clear']:
                return ['未知格不能标为明确']
            elif state=='not_applicable' and (slot['applicable'] is not False or not slot['clear']):
                return ['不适用格必须具有明确道路证据']
            i,j=next((i,j) for i,row in enumerate(SLOTS) for j,s in enumerate(row) if s==name)
            for key in ('position','velocity','acceleration','heading'):
                value=matrices[key][i][j]
                expected=by_id[slot['id']][key] if state=='occupied' else None
                if not np.array_equal(value,expected):
                    return ['四矩阵、空值和对象索引不一致']
        for obj in objects:
            if any(grid['slots'][s]['state']!='occupied' or grid['slots'][s].get('id')!=obj['id'] for s in obj['slots']):
                return ['对象跨格成员关系与格子索引不一致']
        return []
    except (KeyError,TypeError,ValueError,AttributeError,OverflowError):
        return ['八向道路、观测或运动状态字段缺失/非法']


def validate_representations(reps):
    if not isinstance(reps,Representations):
        return ['未加载有效表示']
    errors=validate_grid_schema(reps.grid)
    try:
        if not reps.occupancy_config or reps.occupancy_config!=occupancy_distance_config(reps.occupancy_config):
            errors.append('占据指标参数缺失或不完整')
    except (ValueError,TypeError):
        errors.append('占据指标参数非法')
    if (not _vector(reps.bev_feature,768) or np.linalg.norm(reps.bev_feature)<1e-12 or
        not valid_dino_signature(reps.dino_signature)):
        errors.append('BEV特征或DINO指纹无效')
    for name,value in (('占据',reps.occ),('观测掩码',reps.occ_known)):
        try:
            values=np.asarray(value)
            if values.shape!=(6,50,25) or not np.all(np.isfinite(values)) or not np.all((values>=0)&(values<=1)):
                errors.append(name+'尺寸或数值无效')
        except (TypeError,ValueError):
            errors.append(name+'非法')
    return errors


def grid_comparison(a,b):
    details=dict(matches=[],unmatched_query=[],unmatched_memory=[],gamma=None,D_T=None,D_E=None,
                 S_obj=None,C_star=None,timing_s=dict(assignment=0.))
    errors=validate_grid_schema(a)+validate_grid_schema(b)
    if errors:
        return _invalid('；'.join(dict.fromkeys(errors)),**details)
    if a['metric_signature']!=b['metric_signature'] or a['observation_signature']!=b['observation_signature']:
        return _invalid('八向指标或观测参数不兼容',**details)
    cfg=a['distance_config']
    if any(not g['roads'][s]['known'] for g in (a,b) for s in ('left','right')):
        return _invalid('缺少必要道路存在/跨越证据',**details)
    union={s for s in REGIONS if a['slots'][s]['applicable'] or b['slots'][s]['applicable']}
    joint={s for s in REGIONS if a['slots'][s]['clear'] and b['slots'][s]['clear']}
    if not union:
        return _invalid('没有可比较的适用区域',**details)
    details['gamma']=len(joint&union)/len(union)
    left,right=_unique_objects(a),_unique_objects(b)
    n,m=len(left),len(right)
    details.update(n=n,m=m,unmatched_query=[o['id'] for o in left],unmatched_memory=[o['id'] for o in right])
    if details['gamma']<cfg['min_joint_known_fraction']:
        return _invalid('八向共同明确比例低于阈值',**details)
    if any(not o['slots']&joint for o in left+right):
        return _invalid('已选可见对象没有共同明确角色，禁止静默忽略',**details)
    details['D_T']=sum(abs(int(a['roads'][s][k])-int(b['roads'][s][k])) for s in ('left','right') for k in ('exists','crossable'))/4
    details['D_E'],_=_distances(a['ego'],b['ego'],cfg,True)
    if not math.isfinite(details['D_E']):
        details['D_E']=None
        return _invalid('自车状态距离溢出',**details)
    if n+m==0:
        if any(g['slots'][s]['state']!='empty' for g in (a,b) for s in REGIONS if g['slots'][s]['applicable']):
            return _invalid('无对象但存在未知区域，不能当作确认双空',**details)
        details.update(S_obj=1.,C_star=0.)
    else:
        c=cfg['cutoff_c']
        forbidden=(n+m+1)*c+1
        normalization=(c/2)*(n+m)
        if not math.isfinite(forbidden) or not math.isfinite(normalization) or normalization<=0:
            return _invalid('部分分配参数超出有限数值范围',**details)
        costs=np.full((n+m,n+m),forbidden,dtype=float)
        pair_details={}
        for i,qa in enumerate(left):
            for j,mb in enumerate(right):
                shared=qa['slots']&mb['slots']&joint
                if qa['kind']!=mb['kind'] or not shared:
                    continue
                rho,components=_distances(qa,mb,cfg)
                if not math.isfinite(rho):
                    return _invalid('状态距离溢出',**details)
                if rho<c:  # 等于阈值明确不配对。
                    costs[i,j]=rho
                    pair_details[i,j]=dict(query_id=qa['id'],memory_id=mb['id'],shared_slots=sorted(shared),rho=rho,cost=rho,**components)
        for i in range(n):costs[i,m+i]=c/2
        for j in range(m):costs[n+j,j]=c/2
        costs[n:,m:]=0.
        start=time.perf_counter()
        rows,cols=linear_sum_assignment(costs)
        details['timing_s']['assignment']=time.perf_counter()-start
        if np.any(costs[rows,cols]>=forbidden):
            return _invalid('部分分配返回禁止位置',**details)
        total=float(costs[rows,cols].sum())
        details['C_star']=total
        details['S_obj']=_unit_score(1-total/normalization)
        details['matches']=[pair_details[i,j] for i,j in zip(rows,cols) if i<n and j<m]
        matched_q={v['query_id'] for v in details['matches']}
        matched_m={v['memory_id'] for v in details['matches']}
        details['unmatched_query']=[v['id'] for v in left if v['id'] not in matched_q]
        details['unmatched_memory']=[v['id'] for v in right if v['id'] not in matched_m]
    score=details['gamma']*details['S_obj']*math.exp(-cfg['lambda_T']*details['D_T']-cfg['lambda_E']*details['D_E'])
    return dict(valid=True,score=_unit_score(score),reason='有效八向部分匹配',**details)


def grid_similarity(a,b,return_matches=False):
    result=grid_comparison(a,b)
    return (result['score'],result['matches']) if return_matches else result['score']


def _same_record_evidence(a,b):
    """相同ID/修订的完全相同副本合并；矛盾证据不能靠遍历次序选一个。"""
    if plain({k:v for k,v in a.items() if not k.startswith('_')})!=plain({k:v for k,v in b.items() if not k.startswith('_')}):
        return False
    left,right=a.get('_representations'),b.get('_representations')
    if not isinstance(left,Representations) or not isinstance(right,Representations):
        return left is right
    fields=('frame_id','time','version','coordinates','dino_signature','grid','occupancy_config')
    return (all(plain(getattr(left,k))==plain(getattr(right,k)) for k in fields) and
        all(np.array_equal(getattr(left,k),getattr(right,k)) for k in ('bev_feature','occ','occ_known')))


def retrieve(query:Representations,records,snapshot_id='',config=None)->RetrievalResult:
    """合格经验池内各通道选一条；同ID只展示一次，不融合排序、不补齐。"""
    started=time.perf_counter()
    settings=(config or {}).get('retrieval',{})
    minimum_grid=settings.get('minimum_grid_score',.2)
    if not _finite(minimum_grid) or not 0<=minimum_grid<=1:
        raise ValueError('八向最低相关性阈值须位于[0,1]')
    if set(settings)&{'weights','minimum_score','final_k'}:
        raise ValueError('旧融合参数已取消，不能用于各通道Top-1检索')
    for key,value in (('per_channel_k',1),('selection','per_channel_top1'),
                      ('merge_same_record',True),('require_all_channels',True)):
        if key in settings and (type(settings[key]) is not type(value) or settings[key]!=value):
            raise ValueError('当前检索固定契约不匹配：'+key)
    diagnostics=dict(query_frame_id=query.frame_id,versions=dict(representation=FEATURE_VERSION,
        grid_representation=GRID_REPRESENTATION_VERSION,occupancy_representation=OCC_REPRESENTATION_VERSION,
        grid_metric=GRID_METRIC_VERSION,occupancy_metric=OCC_METRIC_VERSION,retrieval_metric=RETRIEVAL_METRIC_VERSION),
        dino_signature=query.dino_signature,timing_s=dict(bev=0.,occupancy=0.,grid=0.,assignment=0.,overall=0.),
        selection='per_channel_top1',minimum_grid_score=minimum_grid,
        channel_order=['bev','occupancy','grid'],merge_same_record=True,
        channel_recalls={c:[] for c in ('bev','occupancy','grid')},
        candidate_ids=[],candidates=[],all_records=[],reason='')
    result=RetrievalResult(query.frame_id,snapshot_id=snapshot_id,time=query.time,diagnostics=diagnostics)
    def finish(reason):
        diagnostics['reason']=reason
        diagnostics['timing_s']['overall']=time.perf_counter()-started
        return result
    if not valid_dino_signature(query.dino_signature):
        return finish('查询没有有效DINO权重指纹')
    # 同一经验只选择最高修订；冲突副本拒绝，不依赖遍历顺序。
    groups={}
    for record in records:
        identity,revision=record.get('id'),record.get('revision',1)
        if not isinstance(identity,str) or not identity or type(revision) is not int or revision<1:
            diagnostics['all_records'].append(dict(experience_id=identity,reason='经验ID或修订号非法'))
            continue
        previous=groups.get(identity)
        if previous is None or revision>previous.get('revision',1):
            groups[identity]=record
        elif revision==previous.get('revision',1) and record is not previous and not _same_record_evidence(record,previous):
            groups[identity]=dict(id=identity,revision=revision,status='duplicate_conflict')
    scored=[]
    for identity in sorted(groups):
        record=groups[identity]
        reps=record.get('_representations')
        diag=dict(query_frame_id=query.frame_id,experience_id=identity,revision=record.get('revision',1),
                  S_B=None,S_O=None,S_G=None,accepted=False,eligible=False,
                  matched_channels=[],reason='',channels={},
                  stored_retrieval_metric=record.get('retrieval_metric_version'))
        diagnostics['all_records'].append(diag)
        if record.get('status','active')!='active':
            diag['reason']='经验未启用或同一身份存在冲突副本';continue
        if not isinstance(reps,Representations) or reps.version!=query.version:
            diag['reason']='经验表示未加载或接口版本不兼容';continue
        if not valid_dino_signature(reps.dino_signature) or reps.dino_signature!=query.dino_signature:
            diag['reason']='DINO权重指纹不兼容或缺失';continue
        try:
            occ_signature=occupancy_metric_signature(reps.occupancy_config) if reps.occupancy_config else None
        except (ValueError,TypeError):
            occ_signature=None
        stored_grid=reps.grid if isinstance(reps.grid,dict) else {}
        diag['versions']=dict(diagnostics['versions'],dino_signature=reps.dino_signature,
            grid_metric_signature=stored_grid.get('metric_signature'),
            grid_observation_signature=stored_grid.get('observation_signature'),
            occupancy_metric_signature=occ_signature)
        for channel,callback in (
            ('bev',lambda:dict(score=bev_similarity(query.bev_feature,reps.bev_feature))),
            ('occupancy',lambda:occupancy_comparison(query.occ,reps.occ,query.occ_known,reps.occ_known,query.occupancy_config)),
            ('grid',lambda:grid_comparison(query.grid,reps.grid))):
            clock=time.perf_counter()
            if channel=='occupancy':
                try:
                    compatible=bool(query.occupancy_config and reps.occupancy_config and
                        query.occupancy_config==occupancy_distance_config(query.occupancy_config) and
                        reps.occupancy_config==occupancy_distance_config(reps.occupancy_config) and
                        occupancy_metric_signature(query.occupancy_config)==occupancy_metric_signature(reps.occupancy_config))
                except (ValueError,TypeError):
                    compatible=False
                report=callback() if compatible else _invalid('占据指标参数缺失或不兼容')
            else:
                report=callback()
            if channel=='bev':
                report.update(valid=report['score'] is not None,reason='有效BEV余弦' if report['score'] is not None else 'BEV证据无效')
            diagnostics['timing_s'][channel]+=time.perf_counter()-clock
            if channel=='grid':diagnostics['timing_s']['assignment']+=report.get('timing_s',{}).get('assignment',0.)
            diag['channels'][channel]=report
        diag.update(S_B=diag['channels']['bev']['score'],S_O=diag['channels']['occupancy']['score'],S_G=diag['channels']['grid']['score'])
        scored.append(dict(record=record,diag=diag))
    # 沿用现有证据与八向最低相关性准入；只移除融合分门槛。
    # 先验收再取第一名，损坏的高分记录不能挤掉有证据的合法候选。
    eligible=[]
    for item in scored:
        diag=item['diag']
        if any(not v['valid'] for v in diag['channels'].values()):
            diag['reason']='通道证据不足：'+'；'.join(v['reason'] for v in diag['channels'].values() if not v['valid'])
        elif diag['S_G']<minimum_grid:
            diag['reason']='八向相关性低于最低阈值'
        else:
            diag['eligible']=True
            diag['reason']='合格但未成为任一通道Top-1'
            eligible.append(item)
    selected=[]
    selected_ids=set()
    for channel,key in (('bev','S_B'),('occupancy','S_O'),('grid','S_G')):
        if not eligible:
            continue
        item=min(eligible,key=lambda v:(-v['diag'][key],v['record']['id']))
        identity=item['record']['id']
        diagnostics['channel_recalls'][channel]=[identity]
        item['diag']['matched_channels'].append(channel)
        item['diag']['accepted']=True
        item['diag']['reason']='接纳：成为'+','.join(item['diag']['matched_channels'])+'通道Top-1'
        if identity not in selected_ids:
            selected.append(item)
            selected_ids.add(identity)
    diagnostics['candidate_ids']=sorted(selected_ids)
    diagnostics['candidates']=[v['diag'] for v in selected]
    result.records=[v['record'] for v in selected]
    result.scores=[dict(id=v['record']['id'],revision=v['record'].get('revision',1),bev=v['diag']['S_B'],
        occupancy=v['diag']['S_O'],grid=v['diag']['S_G'],matched_channels=list(v['diag']['matched_channels']),
        object_matches=v['diag']['channels']['grid']['matches']) for v in selected]
    return finish('返回各通道Top-1并合并同ID' if selected else '空库' if not groups else '没有满足证据与相关性要求的候选')
