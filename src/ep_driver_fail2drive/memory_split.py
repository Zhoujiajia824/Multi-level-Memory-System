"""Fixed route-pair partitions for same-family memory transfer experiments.

Only source files and existing captures are read. Scenario identities are used
for experiment isolation, never as a retrieval score or a driving-model input.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from nine_grid.paths import filesystem_path
from . import OFFICIAL_COMMIT
from .demo_catalog import FAMILIES

SCHEMA='fail2drive-memory-transfer-split-1.0'
DEFAULT=Path(__file__).resolve().parents[2]/'config/fail2drive_memory_transfer_split.json'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def load_transfer_split(path=None):
    value=json.loads(filesystem_path(path or DEFAULT).read_text(encoding='utf-8-sig'))
    if (not isinstance(value,dict) or value.get('schema')!=SCHEMA or value.get('official_commit')!=OFFICIAL_COMMIT
            or value.get('collection_route_count')!=34 or value.get('evaluation_route_count')!=17
            or value.get('evaluation_variant')!='Generalization'
            or any(type(value.get(k)) is not int or value[k]<1 for k in ('collection_seed','evaluation_seed'))):
        raise ValueError('Wrong memory-transfer split version or official identity')
    rows=value.get('rows',[])
    if (not isinstance(rows,list) or len(rows)!=17 or not all(isinstance(row,dict) for row in rows)
            or {row.get('family') for row in rows}!=set(FAMILIES)):
        raise ValueError('Split must contain the exact 17 distinct official families')
    collection=[];evaluation=[];reserved=[]
    for row in rows:
        pair=row.get('collection',[])
        if (not isinstance(pair,list) or len(pair)!=2 or not all(isinstance(r,dict) for r in pair)
                or {r.get('variant') for r in pair}!={'Base','Generalization'}):
            raise ValueError('Every family needs one Base and one Generalization demonstration')
        base=next(r for r in pair if r['variant']=='Base')
        gen=next(r for r in pair if r['variant']=='Generalization')
        test=row.get('evaluation',{});counterpart=row.get('reserved_counterpart',{})
        records=pair+[test,counterpart]
        for record in records:
            if not isinstance(record,dict):
                raise ValueError('Each split route must be a structured original route identity')
            rid=record.get('route_id');variant=record.get('variant')
            if (type(rid) is not int or variant not in ('Base','Generalization')
                    or not (0<=rid<100 if variant=='Base' else 1000<=rid<1100)
                    or record.get('routes')!='fail2drive_split/%s_%s_%04d.xml'%(variant,row['family'],rid)
                    or not isinstance(record.get('sha256'),str) or len(record['sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in record['sha256'])):
                raise ValueError('Split contains an invalid original route identity')
        if (gen['route_id']!=base['route_id']+1000 or test['variant']!='Generalization'
                or counterpart['variant']!='Base' or test['route_id']!=counterpart['route_id']+1000
                or test['route_id']==gen['route_id']):
            raise ValueError('Collection and evaluation must use different whole Base/Generalization pairs')
        collection.extend(pair);evaluation.append(test);reserved.extend([test,counterpart])
    cids=[r['route_id'] for r in collection];eids=[r['route_id'] for r in evaluation]
    reserved_ids=[r['route_id'] for r in reserved]
    if (len(set(cids))!=34 or len(set(eids))!=17 or len(set(reserved_ids))!=34
            or set(cids)&set(reserved_ids)):
        raise ValueError('Route identities or reserved pairs overlap across partitions')
    return value


def route_ids(split, partition):
    if partition=='collection':return [r['route_id'] for row in split['rows'] for r in row['collection']]
    if partition=='evaluation':return [row['evaluation']['route_id'] for row in split['rows']]
    raise ValueError('Choose collection or evaluation')


def _geometry(root,record):
    target=filesystem_path(root)/record['routes']
    if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError('Original route XML differs from the frozen split: '+record['routes'])
    try:
        routes=ET.parse(target).getroot().findall('route')
    except ET.ParseError as error:
        raise ValueError('Invalid original route XML: '+record['routes']) from error
    if len(routes)!=1 or routes[0].get('id')!=str(record['route_id']):
        raise ValueError('XML route ID differs from its partition')
    points=[[float(p.get(k)) for k in ('x','y','z')] for p in routes[0].findall('waypoints/position')]
    if not routes[0].get('town') or len(points)<2 or not all(math.isfinite(v) for p in points for v in p):
        raise ValueError('Original route lacks finite map/waypoint geometry')
    return _digest(dict(map=routes[0].get('town'),points=points))


def validate_transfer_source(split, official_root):
    collection_geometry=set();reserved_geometry=set()
    for row in split['rows']:
        pair=[_geometry(official_root,r) for r in row['collection']]
        held=[_geometry(official_root,row[k]) for k in ('evaluation','reserved_counterpart')]
        if pair[0]!=pair[1] or held[0]!=held[1]:
            raise ValueError('Pinned Base/Generalization partners no longer share their original sparse route')
        collection_geometry.update(pair);reserved_geometry.update(held)
    if collection_geometry&reserved_geometry:
        raise ValueError('Collection and evaluation share an exact sparse route, including across families')
    return dict(schema=SCHEMA,valid=True,split_sha256=_digest(split),
        collection_routes=route_ids(split,'collection'),evaluation_routes=route_ids(split,'evaluation'),
        collection_count=34,evaluation_count=17,families=17,
        source_check='original_XML_bytes_ID_map_and_sparse_waypoints',
        whole_pair_isolation=True,exact_sparse_geometry_overlap=False,
        limitation='same_Town13;partial_road_overlap_possible;not_unseen_map_or_family_generalization')


def validate_collection_partition(split, episodes, *, allow_partial=False):
    """Audit admitted actual captures before preparing any annotation tasks."""
    if type(allow_partial) is not bool:raise ValueError('Partial collection admission must be explicitly boolean')
    expected={r['route_id']:(row['family'],r) for row in split['rows'] for r in row['collection']}
    observed=Counter();errors=[];selected=[]
    for episode in episodes:
        rid=episode['official_route_id'];observed[rid]+=1
        if rid not in expected:
            errors.append('Route %s is outside the memory partition (held-out routes and partners are forbidden)'%rid)
            continue
        family,record=expected[rid]
        root=filesystem_path(episode['episode'])
        if (episode['scenario_family']!=family or episode['seed']!=split['collection_seed']
                or hashlib.sha256((root/'official_route.xml').read_bytes()).hexdigest()!=record['sha256']):
            errors.append('Capture %s differs from its frozen family/seed/XML'%rid)
            continue
        selected.append(dict(official_route_id=rid,scenario_family=family,seed=episode['seed'],
            official_route_sha256=record['sha256'],source_manifest_sha256=episode.get('source_manifest_sha256'),
            source_admission_policy=episode.get('admission_policy'),
            accepted_with_min_speed_penalty=bool(episode.get('accepted_with_min_speed_penalty',False))))
    missing=sorted(set(expected)-set(observed))
    duplicates=sorted(rid for rid,count in observed.items() if count!=1)
    if not selected:errors.append('Select at least one strictly admitted successful collection route')
    if missing and not allow_partial:errors.append('Missing successful collection routes: '+str(missing))
    if duplicates:errors.append('Select one successful capture per route; duplicates: '+str(duplicates))
    families=sorted({r['scenario_family'] for r in selected})
    selected_ids=sorted({r['official_route_id'] for r in selected})
    complete_pairs=[];partial_pairs=[]
    for row in split['rows']:
        present=set(selected_ids)&{r['route_id'] for r in row['collection']}
        if len(present)==2:complete_pairs.append(row['family'])
        elif present:partial_pairs.append(row['family'])
    evaluation=[row['evaluation']['route_id'] for row in split['rows'] if row['family'] in families]
    selection=dict(schema='fail2drive-successful-collection-selection-1.0',split_sha256=_digest(split),
        allow_partial_collection=allow_partial,
        selection_policy='explicit_input_captures_after_strict_admission_and_pending_human_review',
        captures=sorted(selected,key=lambda r:r['official_route_id']),
        evaluation_route_ids=evaluation,evaluation_seed=split['evaluation_seed'])
    return dict(schema=SCHEMA,valid=not errors,split_sha256=_digest(split),errors=errors,
        expected_routes=34,observed_unique_routes=len(observed),missing_route_ids=missing,
        collection_seed=split['collection_seed'],evaluation_route_ids=evaluation,
        evaluation_seed=split['evaluation_seed'],family_or_route_id_used_for_retrieval=False,
        allow_partial_collection=allow_partial,all_34_collected=not missing and not errors,
        selected_route_ids=selected_ids,selected_families=families,
        complete_pair_families=sorted(complete_pairs),partial_pair_families=sorted(partial_pairs),
        selection_snapshot=selection,selection_sha256=_digest(selection),
        quality_admission_unchanged=True,human_quality_approval_fabricated=False)
