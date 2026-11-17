"""Versioned, deterministic CARLA experiment instances (Python 3.7, no SDK)."""
import copy
import hashlib
import json
import math
import random

SCHEMA_VERSION = 'carla-experiment-2.0'
INSTANCE_VERSION = 'carla-instance-2.0'
COORDINATE_CONTRACT = {
    'version': 'carla-route-xy-bbox-1.0',
    'route_s': 'frozen_waypoint_polyline_xy_arc_m',
    'waypoint_s': 'OpenDRIVE_reference_s_m',
    'actor_progress_reference': 'world_bounding_box_center',
    'lateral_d': 'left_of_directed_route_positive_m',
    'world_axes': 'CARLA_world_xyz',
}
BEHAVIORS = ('follow', 'stop_resume', 'hard_brake', 'cut_in', 'cross', 'parked')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def finite(value, name, minimum=0., maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(name + ' is outside its finite numeric range')
    return float(value)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def route_spec(route):
    if not isinstance(route, dict) or set(route)-{'points','opendrive_sha256','task','server_validation','file','length_m','map','sampling_step_m','coordinate_contract','coordinate_migration'}:
        raise ValueError('route contains unsupported fields')
    if route.get('coordinate_contract') != COORDINATE_CONTRACT:
        raise ValueError('route must declare the XY arc / bounding-box-center coordinate contract')
    if 'coordinate_migration' in route and not isinstance(route['coordinate_migration'], dict):
        raise ValueError('route coordinate migration evidence must be a mapping')
    if 'file' in route and 'points' not in route:
        raise ValueError('route.file must be expanded by the configuration loader before validation')
    points = route.get('points')
    finite(route.get('sampling_step_m',2.), 'route.sampling_step_m', .1, 4.)
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError('route requires a locked sequence of at least two road points')
    fingerprint = route.get('opendrive_sha256', '')
    if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
        raise ValueError('route requires its actual OpenDRIVE SHA256')
    for point in points:
        if not isinstance(point, dict) or set(point)-{'road_id','section_id','lane_id','s_m','x_m','y_m','z_m','yaw_deg','junction'}:
            raise ValueError('unsupported locked route point')
        integer(point.get('road_id'), 'road_id')
        if type(point.get('lane_id')) is not int or point['lane_id'] == 0:
            raise ValueError('lane_id must be a nonzero integer')
        finite(point.get('s_m'), 's_m')
        if 'section_id' in point: integer(point['section_id'], 'section_id')
    return route


def _trigger(trigger):
    if not isinstance(trigger, dict) or set(trigger)-{'ego_s_m','earliest_tick','gap_m'}:
        raise ValueError('trigger only accepts ego_s_m, earliest_tick and gap_m')
    integer(trigger.get('earliest_tick', 0), 'earliest_tick')
    for key in ('ego_s_m','gap_m'):
        if key in trigger:
            bounds = trigger[key]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(key + ' requires [lower, upper]')
            low = finite(bounds[0], key, -1000. if key == 'gap_m' else 0.)
            high = finite(bounds[1], key, -1000. if key == 'gap_m' else 0.)
            if low > high: raise ValueError('inverted ' + key)


def validate_settings(settings):
    allowed = {'ego_blueprint','initial_speed_mps','weather','route','actors','signals','variations',
        'public_rules','evaluation','seed','name','scenario_name','family','map','instance_hash',
        'instance_manifest','physics_hz','status','split'}
    if not isinstance(settings, dict) or set(settings)-allowed:
        raise ValueError('experiment contains unsupported fields: ' + str(sorted(set(settings)-allowed)))
    finite(settings.get('initial_speed_mps'), 'initial_speed_mps', 0., 40.)
    if not isinstance(settings.get('ego_blueprint'), str) or not settings['ego_blueprint'].startswith('vehicle.'):
        raise ValueError('ego_blueprint must be an exact vehicle blueprint')
    route_spec(settings.get('route'))
    rows = settings.get('actors')
    if not isinstance(rows, list): raise ValueError('actors must be a list')
    roles = set()
    for row in rows:
        if not isinstance(row, dict) or set(row)-{'role','type','blueprint','placement','behavior','speed_mps',
                'trigger','duration_ticks','hold_ticks','deceleration_mps2','required','completion_s_m'}:
            raise ValueError('actor contains unsupported fields')
        role = row.get('role')
        if not isinstance(role, str) or not role or role == 'ego' or role in roles:
            raise ValueError('actor role must be unique and not ego')
        roles.add(role)
        if row.get('type') not in ('vehicle','pedestrian','prop'): raise ValueError('unsupported actor type')
        if not isinstance(row.get('blueprint'), str) or '*' in row['blueprint']:
            raise ValueError('blueprint must be exact, not a random wildcard')
        if row.get('behavior') not in BEHAVIORS: raise ValueError('unsupported actor behavior')
        if row['type'] == 'pedestrian' and row['behavior'] != 'cross': raise ValueError('pedestrians require cross')
        if row['type'] == 'prop' and row['behavior'] != 'parked': raise ValueError('props require parked')
        finite(row.get('speed_mps'), 'speed_mps', 0., 40.)
        place = row.get('placement')
        if not isinstance(place, dict): raise ValueError('actor placement is required')
        if 'route' in place:
            if set(place)-{'route','offset_m'}: raise ValueError('route placement cannot also be relative')
            route_spec(place['route'])
        else:
            if set(place)-{'s_m','lane','offset_m'}: raise ValueError('unsupported actor placement')
            finite(place.get('s_m'), 'placement.s_m')
            if place.get('lane') not in ('current','left','right','sidewalk_left','sidewalk_right'):
                raise ValueError('unsupported relative placement lane')
        finite(place.get('offset_m', 0.), 'offset_m', -10., 10.)
        _trigger(row.get('trigger', {}))
        for name in ('duration_ticks','hold_ticks'):
            integer(row.get(name, 200 if name == 'duration_ticks' else 150), name, 1)
        finite(row.get('deceleration_mps2', 4.), 'deceleration_mps2', .1, 10.)
        if 'required' in row and type(row['required']) is not bool: raise ValueError('required must be boolean')
        if row['behavior'] == 'cut_in' and place.get('lane') not in ('left','right'):
            raise ValueError('cut_in must start in a declared adjacent lane')
    signals = settings.get('signals', {})
    if not isinstance(signals, dict) or set(signals)-{'green_ticks','yellow_ticks','all_red_ticks','offset_ticks'}:
        raise ValueError('unsupported signal program')
    for key, default in (('green_ticks',500),('yellow_ticks',150),('all_red_ticks',50),('offset_ticks',0)):
        integer(signals.get(key, default), key, 0 if key == 'offset_ticks' else 1)
    rules = settings.get('public_rules', [])
    if not isinstance(rules, list): raise ValueError('public_rules must be a list')
    for rule in rules:
        if not isinstance(rule, dict) or set(rule)-{'id','kind','source','controlled_lane_ids','priority_lane_ids',
                'stop_route_s_m','conflict_route_s_m','description'}:
            raise ValueError('unsupported public rule')
        if rule.get('kind') != 'yield' or rule.get('source') not in ('map_sign','published_road_rule'):
            raise ValueError('public yield rules require an observable source')
        for key in ('controlled_lane_ids','priority_lane_ids'):
            if not isinstance(rule.get(key), list) or not rule[key] or any(
                    not isinstance(x,str) or len(x.split('|')) != 3 for x in rule[key]):
                raise ValueError('public rule requires lane identities')
        if set(rule['controlled_lane_ids']) & set(rule['priority_lane_ids']):
            raise ValueError('a lane cannot both yield and have priority in one rule')
        finite(rule.get('stop_route_s_m'), 'stop_route_s_m')
        bounds = rule.get('conflict_route_s_m')
        if not isinstance(bounds,list) or len(bounds)!=2 or bounds[0] > bounds[1]:
            raise ValueError('public rule requires ordered conflict_route_s_m')
        for value in bounds: finite(value, 'conflict_route_s_m')
    if not isinstance(settings.get('evaluation', {}), dict): raise ValueError('evaluation must be a mapping')
    return settings


def validate_spec(data):
    allowed = {'schema_version','name','family','description','map','desired_speed_mps','duration_s','experiment','status','split'}
    if not isinstance(data,dict) or set(data)-allowed or data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('only carla-experiment-2.0 scenarios are supported')
    for key in ('name','family','description','map'):
        if not isinstance(data.get(key),str) or not data[key].strip(): raise ValueError(key+' is required')
    if any(not (c.isascii() and (c.isalnum() or c in '_-')) for c in data['name']):
        raise ValueError('scenario name must be a safe ASCII identifier')
    finite(data.get('desired_speed_mps'), 'desired_speed_mps', .1, 40.)
    finite(data.get('duration_s'), 'duration_s', 1., 3600.)
    if data.get('status','candidate') not in ('candidate','validated'): raise ValueError('unsupported scenario status')
    if data.get('split','development') not in ('development','test','memory'): raise ValueError('unsupported split')
    validate_settings(data.get('experiment'))
    return data


def resolve_spec(data, seed):
    """Sample once, using independent parameter streams, and freeze the effective instance."""
    validate_spec(data)
    integer(seed, 'seed')
    settings = copy.deepcopy(data['experiment'])
    variations = settings.pop('variations', [])
    if not isinstance(variations,list): raise ValueError('variations must be a list')
    sampled = []
    seen = set()
    for row in variations:
        if not isinstance(row,dict) or set(row) != {'path','min','max'}: raise ValueError('invalid variation')
        path = row['path']
        if not isinstance(path,str) or path in seen: raise ValueError('duplicate/invalid variation path')
        seen.add(path)
        parts = path.split('.')
        permitted = (len(parts)>=3 and parts[0]=='actors' and parts[1].isdigit()
            and '.'.join(parts[2:]) in ('speed_mps','placement.s_m','duration_ticks','hold_ticks','trigger.earliest_tick'))
        if not permitted: raise ValueError('variation must affect an explicit actor parameter: '+path)
        low=finite(row['min'],path); high=finite(row['max'],path)
        if low>high: raise ValueError('inverted variation bounds')
        cursor=settings
        for key in parts[:-1]: cursor=cursor[int(key)] if isinstance(cursor,list) else cursor[key]
        key=parts[-1]
        if key not in cursor: raise ValueError('variation target must exist: '+path)
        rng=random.Random(int(digest({'seed':seed,'name':data['name'],'path':path}),16))
        is_tick=key.endswith('_ticks') or key=='earliest_tick'
        value=rng.randint(int(math.ceil(low)),int(math.floor(high))) if is_tick else round(rng.uniform(low,high),6)
        cursor[key]=value; sampled.append({'path':path,'value':value})
    validate_settings(settings)
    content={'version':INSTANCE_VERSION,'scenario_name':data['name'],'family':data['family'],
        'coordinate_contract':copy.deepcopy(COORDINATE_CONTRACT),
        'map':data['map'],'seed':seed,'spec_sha256':digest(data),'settings':copy.deepcopy(settings),
        'desired_speed_mps':data['desired_speed_mps'],'duration_s':data['duration_s'],
        'status':data.get('status','candidate'),'split':data.get('split','development'),'sampled':sampled}
    manifest=dict(content,instance_hash=digest(content))
    settings.update(name='experiment_v2',scenario_name=data['name'],family=data['family'],map=data['map'],
        seed=seed,instance_hash=manifest['instance_hash'],instance_manifest=manifest,
        status=manifest['status'],split=manifest['split'])
    return settings,manifest


def settings_from_manifest(manifest):
    value=copy.deepcopy(manifest)
    expected=value.pop('instance_hash',None)
    if value.get('version')!=INSTANCE_VERSION or digest(value)!=expected:
        raise ValueError('frozen instance checksum/version mismatch')
    if value.get('coordinate_contract') != COORDINATE_CONTRACT:
        raise ValueError('frozen instance coordinate contract mismatch')
    settings=copy.deepcopy(value['settings']);validate_settings(settings)
    settings.update(name='experiment_v2',scenario_name=value['scenario_name'],family=value['family'],
        map=value['map'],seed=value['seed'],instance_hash=expected,instance_manifest=copy.deepcopy(manifest),
        status=value['status'],split=value['split'])
    return settings
