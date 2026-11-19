"""Compare batched eight-region retrieval with the pre-array scalar implementation.

Run from either checkout with: python scripts/check_grid_vector_retrieval.py
Uses deterministic synthetic evidence; does not load models or connect to a simulator.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid import retrieval as r
from nine_grid.configuration import (grid_distance_config, grid_metric_signature,
    grid_observation_config, grid_observation_signature, occupancy_distance_config)
from nine_grid.contracts import Representations
from nine_grid.representations import FEATURE_VERSION, GRID_REPRESENTATION_VERSION, SLOTS
from nine_grid.retrieval import REGIONS, _invalid, _distances, _unique_objects, _unit_score, validate_grid_schema

# Frozen scalar matching algorithm, retained only as an independent regression oracle.
def scalar_grid_comparison(a,b):
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


def make_grid(objects=(),unknown=(),roads=None,ego=None):
    ego=copy.deepcopy(ego or dict(velocity=[8.,0.],acceleration=[.1,0.],heading=0.))
    distance=grid_distance_config()
    observation=grid_observation_config()
    grid=dict(version=FEATURE_VERSION,representation_version=GRID_REPRESENTATION_VERSION,valid=True,
        distance_config=distance,metric_signature=grid_metric_signature(distance),
        observation_config=observation,observation_signature=grid_observation_signature(observation),
        ego=ego,roads=copy.deepcopy(roads or {side:dict(known=True,exists=True,crossable=True)
            for side in ('left','right')}),objects=copy.deepcopy(list(objects)),slots={})
    for name in REGIONS:
        state,applicable,reason='empty',True,'confirmed_empty'
        for side in ('left','right'):
            if name.startswith(side):
                road=grid['roads'][side]
                if not road['known']:state,applicable,reason='unknown',None,'unknown_road'
                elif not road['crossable']:
                    state,applicable,reason='not_applicable',False,'noncrossable_boundary' if road['exists'] else 'no_adjacent_lane'
        if name in unknown:state,reason='unknown','unobserved'
        grid['slots'][name]=dict(state=state,applicable=applicable,clear=state!='unknown',state_valid=False,
            id=None,kind=None,coverage=0. if state=='unknown' else 1.,unobserved_area_m2=1. if state=='unknown' else 0.,reason=reason)
    for obj in grid['objects']:
        for name in obj['slots']:
            grid['slots'][name].update(state='occupied',clear=True,state_valid=True,id=obj['id'],kind=obj['kind'])
    rebuild_matrices(grid)
    errors=validate_grid_schema(grid)
    assert not errors,errors
    return grid


def rebuild_matrices(grid):
    by_id={o['id']:o for o in grid['objects']}
    grid['matrices']={key:[[None]*3 for _ in range(3)] for key in ('position','velocity','acceleration','heading')}
    for key,matrix in grid['matrices'].items():
        for i,row in enumerate(SLOTS):
            for j,name in enumerate(row):
                if name=='ego':matrix[i][j]=copy.deepcopy([0.,0.] if key in ('position','velocity') else grid['ego'][key])
                elif grid['slots'][name]['state']=='occupied':
                    matrix[i][j]=copy.deepcopy(by_id[grid['slots'][name]['id']][key])


def obj(identity,slots,kind='vehicle',position=(15.,0.),velocity=(0.,0.),acceleration=(0.,0.),heading=0.):
    return dict(id=identity,slots=list(slots),kind=kind,position=list(position),velocity=list(velocity),
        acceleration=list(acceleration),heading=float(heading))


def random_grid(rng,prefix):
    objects=[]
    unknown=[]
    for name in REGIONS:
        draw=rng.random()
        if draw<.1:unknown.append(name)
        elif draw<.8:
            if objects and rng.random()<.2:
                objects[int(rng.integers(len(objects)))]['slots'].append(name)
            else:
                objects.append(obj(prefix+str(len(objects)),[name],kind=rng.choice(['vehicle','bicycle','pedestrian']),
                    position=rng.uniform([-30.,-7.],[70.,7.]),velocity=rng.uniform([-8.,-1.],[8.,1.]),
                    acceleration=rng.uniform(-1.,1.,2),heading=rng.uniform(-math.pi,math.pi)))
    return make_grid(objects,unknown,ego=dict(velocity=rng.uniform([0.,-.5],[20.,.5]).tolist(),
        acceleration=rng.uniform(-1.,1.,2).tolist(),heading=float(rng.uniform(-math.pi,math.pi))))


def similar_grid(grid,rng,prefix):
    other=copy.deepcopy(grid)
    remap={o['id']:prefix+str(i) for i,o in enumerate(other['objects'])}
    for o in other['objects']:
        o['id']=remap[o['id']]
        for key,scale in (('position',.5),('velocity',.15),('acceleration',.05)):
            o[key]=(np.asarray(o[key])+rng.normal(0.,scale,2)).tolist()
        o['heading']+=float(rng.normal(0.,.03))
    for slot in other['slots'].values():
        if slot['id'] is not None:slot['id']=remap[slot['id']]
    rebuild_matrices(other)
    return other


def assert_report_equal(actual,expected,path='report'):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys(),(path,actual.keys(),expected.keys())
        for key,value in expected.items():
            if key!='timing_s':assert_report_equal(actual[key],value,path+'.'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(path,len(actual),len(expected))
        for i,(a,b) in enumerate(zip(actual,expected)):assert_report_equal(a,b,path+'.'+str(i))
    elif isinstance(expected,float):
        assert math.isclose(actual,expected,rel_tol=2e-14,abs_tol=2e-14),(path,actual,expected)
    else:assert actual==expected,(path,actual,expected)


def special_pairs():
    empty=make_grid()
    occupied=make_grid([obj('q',['front'])])
    yield 'both_empty',empty,copy.deepcopy(empty)
    yield 'one_empty',empty,occupied
    yield 'both_empty_unknown',make_grid(unknown=['rear']),make_grid(unknown=['rear'])
    yield 'joint_below_threshold',make_grid(unknown=['rear','front','left_front']),empty
    yield 'occupied_without_joint_role',occupied,make_grid(unknown=['front'])
    missing={side:dict(known=True,exists=False,crossable=False) for side in ('left','right')}
    yield 'no_adjacent_lanes',make_grid(roads=missing),empty
    blocked={side:dict(known=True,exists=True,crossable=False) for side in ('left','right')}
    yield 'noncrossable',make_grid(roads=blocked),make_grid(roads=missing)
    unknown=copy.deepcopy(missing)
    unknown['left']=dict(known=False,exists=None,crossable=None)
    yield 'unknown_road',make_grid(roads=unknown),empty
    yield 'kind_mismatch',occupied,make_grid([obj('m',['front'],kind='bicycle')])
    yield 'different_roles',occupied,make_grid([obj('m',['rear'])])
    yield 'heading_wrap',make_grid([obj('q',['front'],heading=math.pi)]),make_grid([obj('m',['front'],heading=-math.pi)])
    cross=make_grid([obj('cross',['front','left_front'])])
    cross['objects'].append(copy.deepcopy(cross['objects'][0]))
    cross['objects'][0]['slots']=['front']
    cross['objects'][1]['slots']=['left_front']
    yield 'deduplicate_cross_region',cross,make_grid([obj('m',['front','left_front'])])
    # A cycle with identical costs exercises JV's stable object-ID ordering.
    qa=make_grid([obj('q1',['front','left_front']),obj('q0',['rear','left_rear'])])
    mb=make_grid([obj('m1',['front','rear']),obj('m0',['left_front','left_rear'])])
    yield 'assignment_tie',qa,mb
    yield 'assignment_tie_reordered',make_grid(list(reversed(qa['objects']))),make_grid(list(reversed(mb['objects'])))
    q=make_grid([obj('q',['front'],position=[0.,0.])])
    m=make_grid([obj('m',['front'],position=[14.,1.])])
    cutoff,_=_distances(q['objects'][0],m['objects'][0],q['distance_config'])
    for tag,c in (('below',np.nextafter(cutoff,-np.inf)),('equal',cutoff),('above',np.nextafter(cutoff,np.inf))):
        a,b=copy.deepcopy(q),copy.deepcopy(m)
        for grid in (a,b):
            grid['distance_config']['cutoff_c']=float(c)
            grid['metric_signature']=grid_metric_signature(grid['distance_config'])
        yield 'strict_cutoff_'+tag,a,b
    yield 'eligible_distance_overflow',make_grid([obj('q',['front'],position=[1e308,0.])]),make_grid([obj('m',['front'],position=[-1e308,0.])])
    yield 'ineligible_overflow_ignored',make_grid([obj('q',['front'],position=[1e308,0.])]),make_grid([obj('m',['rear'],position=[-1e308,0.])])
    nonfinite=copy.deepcopy(occupied)
    nonfinite['objects'][0]['heading']=float('inf')
    rebuild_matrices(nonfinite)
    yield 'nonfinite_rejected',occupied,nonfinite
    invalid=copy.deepcopy(cross)
    invalid['objects'][1]['heading']=.4
    yield 'inconsistent_duplicate',occupied,invalid
    overflow_ego=make_grid(ego=dict(velocity=[1e308,0.],acceleration=[0.,0.],heading=0.))
    yield 'ego_overflow',overflow_ego,empty
    yield 'not_a_grid',None,occupied


def representations(grid,index=0):
    feature=np.zeros(768,dtype=np.float64)
    feature[index%768]=1.
    occupancy=np.zeros((6,50,25),dtype=np.float32)
    occupancy[2,20,10]=float(index%2)
    return Representations(frame_id='test:'+str(index),time=0.,bev_path='',feature_image_path='',
        occ=occupancy,occ_known=np.ones_like(occupancy),grid=grid,bev_feature=feature,
        version=FEATURE_VERSION,dino_signature='test:grid-array-regression',occupancy_config=occupancy_distance_config())


def check_retrieval(rng):
    grid=make_grid([obj('q',['front','left_front']),obj('rear',['rear'],kind='pedestrian')])
    query=representations(grid)
    records=[dict(id='memory_%02d'%i,revision=1,_representations=representations(similar_grid(grid,rng,'m'),i)) for i in range(24)]
    # True score ties and an invalid grid channel must not alter independent BEV/OCC recall.
    records.extend([dict(id='a_tie',revision=1,_representations=representations(copy.deepcopy(grid),8)),
                    dict(id='z_tie',revision=1,_representations=representations(copy.deepcopy(grid),8)),
                    dict(id='bad_grid_bev',revision=1,_representations=representations(None))])
    with patch.object(r,'_grid_comparison_prepared',side_effect=lambda a,b:scalar_grid_comparison(a['grid'],b['grid'])):
        expected=r.retrieve(query,records)
    actual=r.retrieve(query,records)
    assert actual.diagnostics['channel_recalls']==expected.diagnostics['channel_recalls']
    assert [v['id'] for v in actual.records]==[v['id'] for v in expected.records]
    assert_report_equal(actual.scores,expected.scores,'top1_scores')
    for a,b in zip(actual.diagnostics['all_records'],expected.diagnostics['all_records']):
        assert_report_equal(a,b,'candidate_diagnostics')
    # Query preparation is per retrieval call, never a cache keyed by a mutable dict.
    query.grid['objects'][0]['position'][0]=44.
    rebuild_matrices(query.grid)
    with patch.object(r,'_grid_comparison_prepared',side_effect=lambda a,b:scalar_grid_comparison(a['grid'],b['grid'])):
        changed_expected=r.retrieve(query,records)
    changed=r.retrieve(query,records)
    assert_report_equal(changed.scores,changed_expected.scores,'mutated_query')
    assert changed.diagnostics['channel_recalls']==changed_expected.diagnostics['channel_recalls']
    assert actual.scores!=changed.scores,'mutated query must be revalidated and recomputed'
    return actual.diagnostics['channel_recalls']


def check_score_boundary_and_ranking(rng):
    """Exercise the 0.2 gate and nextafter-separated Top1 scores, not just pair cutoff."""
    query=representations(make_grid([obj('q',['front'],position=[0.,0.])]))
    base=20.*math.sqrt(2.)*.4/.35
    distances=[base]
    lower=upper=base
    for _ in range(16):
        lower=float(np.nextafter(lower,-np.inf))
        upper=float(np.nextafter(upper,np.inf))
        distances.extend([lower,upper])
    near=[]
    for distance in distances:
        grid=make_grid([obj('m',['front'],position=[distance,0.])])
        near.append((scalar_grid_comparison(query.grid,grid)['score'],grid))
    below=max((item for item in near if item[0]<.2),key=lambda item:item[0])
    above=min((item for item in near if item[0]>=.2),key=lambda item:item[0])
    assert 0.<above[0]-below[0]<1e-14
    # Better score has the lexically later ID; an inappropriate tolerance/tie rule flips it.
    records=[dict(id='a_below',_representations=representations(below[1])),
             dict(id='z_above',_representations=representations(above[1]))]
    pairs=[(query,records)]
    # Isolate ranking from the gate, plus an exact tie with stable ID ordering.
    pairs.append((query,records+[dict(id='y_above_tie',_representations=representations(copy.deepcopy(above[1])))]))
    nearby=make_grid([obj('m',['front'],position=[1.,0.])])
    worse=copy.deepcopy(nearby)
    worse['objects'][0]['position'][0]+=1e-13
    rebuild_matrices(worse)
    pairs.append((query,[dict(id='a_near_worse',_representations=representations(worse)),
                         dict(id='z_near_better',_representations=representations(nearby))]))
    for i in range(16):
        q=random_grid(rng,'query')
        candidates=[dict(id='random_%02d'%j,_representations=representations(
            similar_grid(q,rng,'memory') if j%2 else random_grid(rng,'memory'),j)) for j in range(12)]
        pairs.append((representations(q),candidates))
    maximum=0.
    changed_matches=changed_top1=changed_admission=0
    for q,candidates in pairs:
        with patch.object(r,'_grid_comparison_prepared',side_effect=lambda a,b:scalar_grid_comparison(a['grid'],b['grid'])):
            expected=r.retrieve(q,candidates)
        actual=r.retrieve(q,candidates)
        changed_top1+=int(actual.diagnostics['channel_recalls']!=expected.diagnostics['channel_recalls'])
        for a,b in zip(actual.diagnostics['all_records'],expected.diagnostics['all_records']):
            ga,gb=a['channels']['grid'],b['channels']['grid']
            changed_admission+=int(ga['eligible']!=gb['eligible'])
            changed_matches+=int([(p['query_id'],p['memory_id']) for p in ga['matches']]!=
                                 [(p['query_id'],p['memory_id']) for p in gb['matches']])
            if ga['score'] is not None:maximum=max(maximum,abs(ga['score']-gb['score']))
        assert_report_equal(actual.scores,expected.scores,'boundary_and_random_top1')
    assert changed_top1==changed_admission==changed_matches==0
    boundary=r.retrieve(*pairs[0])
    assert boundary.diagnostics['channel_recalls']['grid']==['z_above']
    assert r.retrieve(*pairs[1]).diagnostics['channel_recalls']['grid']==['y_above_tie']
    assert r.retrieve(*pairs[2]).diagnostics['channel_recalls']['grid']==['z_near_better']
    return dict(retrieval_cases=len(pairs),gate=.2,closest_scalar_score_below=below[0],closest_scalar_score_above=above[0],
        maximum_abs_score_difference=maximum,matching_changes=changed_matches,top1_changes=changed_top1,admission_changes=changed_admission)


def benchmark(rng,repeats):
    query=make_grid([obj('q'+str(i),[slot],position=[i*3.,i%3-1.]) for i,slot in enumerate(REGIONS)])
    memories=[similar_grid(query,rng,'m'+str(i)) for i in range(64)]
    samples={}
    def run(method):
        if method=='scalar':
            return [scalar_grid_comparison(query,g) for g in memories]
        if method=='batch':
            return [r.grid_comparison(query,g) for g in memories]
        prepared=r._prepare_grid(query)
        return [r._grid_comparison_prepared(prepared,r._prepare_grid(g)) for g in memories]
    for method in ('scalar','batch','batch_query_reused'):
        clock=time.perf_counter()
        cold=run(method)
        cold_ms=(time.perf_counter()-clock)*1000
        timings=[]
        for _ in range(repeats):
            clock=time.perf_counter()
            warm=run(method)
            timings.append((time.perf_counter()-clock)*1000)
        for a,b in zip(cold,warm):assert_report_equal(a,b)
        samples[method]=dict(first_pool_ms=round(cold_ms,3),warm_pool_median_ms=round(float(np.median(timings)),3),
            warm_pool_p95_ms=round(float(np.percentile(timings,95)),3))
    return dict(pool_size=len(memories),objects_per_grid=8,repeats=repeats,
        measurement='full schema validation + cost + assignment; prepared query reused only within one pool',times=samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-pairs',type=int,default=400)
    parser.add_argument('--benchmark-repeats',type=int,default=8)
    args=parser.parse_args()
    if args.random_pairs<0 or args.benchmark_repeats<1:parser.error('counts must be nonnegative; benchmark repeats must be positive')
    rng=np.random.default_rng(20261119)
    fixtures=list(special_pairs())
    for i in range(args.random_pairs):
        a=random_grid(rng,'q')
        b=similar_grid(a,rng,'m') if i%2==0 else random_grid(rng,'m')
        fixtures.append(('random_%d'%i,a,b))
    maximum=0.
    nonidentical_scores=changed_matches=0
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for name,a,b in fixtures:
            expected=scalar_grid_comparison(a,b)
            actual=r.grid_comparison(a,b)
            assert_report_equal(actual,expected,name)
            if expected['score'] is not None:
                maximum=max(maximum,abs(actual['score']-expected['score']))
                nonidentical_scores+=int(actual['score']!=expected['score'])
            changed_matches+=int([(p['query_id'],p['memory_id']) for p in actual['matches']]!=
                                 [(p['query_id'],p['memory_id']) for p in expected['matches']])
        recall=check_retrieval(rng)
        ranking=check_score_boundary_and_ranking(rng)
    timing=benchmark(rng,args.benchmark_repeats)
    print(json.dumps(dict(status='passed',comparison_pairs=len(fixtures),seed=20261119,
        comparison='same validity/reasons/matching IDs/Top1; floating values within 2e-14',
        maximum_abs_score_difference=maximum,nonidentical_scores=nonidentical_scores,matching_changes=changed_matches,
        channel_recalls=recall,ranking_and_admission=ranking,benchmark=timing),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
