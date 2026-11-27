"""Authored driving demonstrations and their official collection-route location anchors."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from nine_grid.paths import filesystem_path


SCHEMA_VERSION = "fail2drive-demonstrations-1.0"


def capture_implementation():
    directory = Path(__file__).parent
    files = ('demo_capture.py', 'demo_catalog.py', 'demo_events.py', 'demo_routes.py',
             'demo_teacher.py', 'public_state.py', 'traffic_rules.py', 'execution.py',
             'vehicle_geometry.py', 'camera_calibration.py', 'route.py', 'archive.py',
             'demo_export.py', 'demo_memory.py')
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}


FAMILIES = (
    "Wall", "FullyBlocked", "PedestriansOnRoad", "ConstructionPedestrian",
    "BadParking", "ConstructionPermutations", "CustomObstacles", "ObscuredStop",
    "HardBrake", "RightOfWay", "Animals", "PedestrianOtherBlocker",
    "RightConstruction", "OppositeConstruction", "ImageOnObject",
    "PassableObstacles", "PedestrianCrowd",
)
INTERACTION_KINDS = frozenset({
    "blocked_wait", "slow_lead", "hard_brake", "parked_bypass",
    "construction_crossing", "construction_bypass", "crossing",
    "roadside_clear", "stop_sign", "cross_traffic", "visual_proxy",
})
SUCCESS_REQUIREMENTS = frozenset({
    "actual_execution", "same_frame_capture", "collision_free", "on_road",
    "traffic_rules", "interaction_completed", "recovery_completed", "arrival",
})
CAPABILITIES = frozenset({
    "straight_route", "same_direction_left_lane", "crossable_left_boundary",
    "same_direction_right_lane", "opposite_left_lane", "right_sidewalk", "walkable_right_exit",
    "real_stop_sign", "conflicting_junction", "full_lane_blockage",
    "verified_passing_clearance",
})
STANDARD_BLUEPRINTS = frozenset({
    "vehicle.carlamotors.carlacola", "vehicle.tesla.model3",
    "vehicle.lincoln.mkz_2020", "vehicle.audi.a2", "static.prop.streetbarrier",
    "static.prop.trafficcone01", "static.prop.box03", "static.prop.barrel",
    "static.prop.vendingmachine", "static.prop.busstop",
    *("walker.pedestrian.%04d" % number for number in range(1, 10)),
})
# Extra assets from the pinned collection XML, only in the rebuilt Town13 pairs.
COLLECTION_BLUEPRINTS = frozenset({
    "vehicle.nissan.patrol_2021", "vehicle.kawasaki.ninja",
    "walker.pedestrian.0027", "walker.pedestrian.0040", "walker.pedestrian.0034",
    "walker.pedestrian.0015", "walker.animal.1009", "static.prop.kiosk_01", "static.prop.gnome",
    "static.prop.ampel", "static.prop.brickwall", "static.prop.snow", "vehicle.carlamotors.european_hgv",
    "static.prop.warningconstruction", "static.prop.trafficwarning",
    "static.prop.dirtdebris02", "static.prop.constructioncone",
})
AUTHORED_SCHEMA = 'fail2drive-authored-interaction-2.0'
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "config" / "fail2drive_demonstrations.json"
_FORBIDDEN_KEYS = frozenset({
    "official_route", "official_route_id", "official_seed", "route_id",
    "official_coordinates", "evaluation_result",
})
_FUTURE_INPUT_KEYS = frozenset({"future_controls","future_states","ground_truth_future","expert_controls"})


def _number(value, label, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + " must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(label + " is outside the allowed range")
    return value


def _strings(value, label, *, nonempty=True):
    if (not isinstance(value, list) or nonempty and not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
            or len(value) != len(set(value))):
        raise ValueError(label + " must be a list of distinct nonempty strings")
    return value


def _check_payload(value, label="catalog"):
    if isinstance(value, dict):
        for key, item in value.items():
            if (not isinstance(key,str) or key in _FORBIDDEN_KEYS
                    or '.teacher_policy' in label and key in _FUTURE_INPUT_KEYS):
                raise ValueError(label + " contains a forbidden evaluation/future field: " + str(key))
            _check_payload(item, label + "." + key)
    elif isinstance(value, list):
        for item in value:
            _check_payload(item, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(label + " contains non-finite data")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError(label + " contains non-JSON data")


def validate_catalog(catalog):
    """Return an isolated validated design; raise ValueError on unsafe metadata.

    Runtime topology/asset availability and actual episode outcomes remain separate
    checks. No missing asset or absent stop sign may be silently substituted.
    """
    if not isinstance(catalog, dict):
        raise ValueError("demonstration catalog must be an object")
    payload=deepcopy(catalog)
    # Route IDs are source provenance. The world scheduler may define NPC
    # trajectories. The ego teacher does not receive those future states.
    for item in payload.get('demonstrations',[]) if isinstance(payload.get('demonstrations'),list) else ():
        if isinstance(item,dict):item.pop('route_source',None)
    _check_payload(payload)
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported demonstration schema_version")
    if catalog.get("independent_design") is not True or catalog.get("official_evaluation") is not False:
        raise ValueError("demonstrations must be independent development designs")
    if catalog.get("publication_policy") != "executed_verified_and_reviewed_only":
        raise ValueError("unexecuted designs cannot be published as successful memories")
    specs = catalog.get("demonstrations")
    if not isinstance(specs, list) or len(specs) != len(FAMILIES):
        raise ValueError("catalog must contain exactly one independent design for each of 17 families")
    ids, families, seeds = set(), set(), set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("demonstration must be an object")
        identity = spec.get("id")
        if (not isinstance(identity, str) or not identity.startswith("demo_")
                or not identity.replace("_", "").isalnum() or identity in ids):
            raise ValueError("demonstration id must be unique and start with demo_")
        ids.add(identity)
        family = spec.get("family")
        if family not in FAMILIES or family in families:
            raise ValueError(identity + ": unknown or repeated family")
        families.add(family)
        seed = spec.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 or seed in seeds:
            raise ValueError(identity + ": seed must be a distinct unsigned 32-bit integer")
        seeds.add(seed)
        if spec.get("map") not in {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt", "Town13"}:
            raise ValueError(identity + ": use a standard independent development map")
        if spec.get("interaction_kind") not in INTERACTION_KINDS:
            raise ValueError(identity + ": unsupported interaction_kind")
        if spec.get("design_status") != "pending_execution":
            raise ValueError(identity + ": catalog is a design, never execution success evidence")
        validate_route_source(spec)
        _number(spec.get("approach_m"), identity + ".approach_m", minimum=20)
        _number(spec.get("interaction_m"), identity + ".interaction_m", minimum=5)
        _number(spec.get("recovery_m"), identity + ".recovery_m", minimum=15)
        _number(spec.get("cruise_speed_mps"), identity + ".cruise_speed_mps", minimum=0.1)
        wait = _number(spec.get("wait_s"), identity + ".wait_s", minimum=0)
        duration = _number(spec.get("max_duration_s"), identity + ".max_duration_s", minimum=30)
        travel = sum(spec[key] for key in ("approach_m", "interaction_m", "recovery_m"))
        if duration < wait + travel / spec["cruise_speed_mps"] + 15:
            raise ValueError(identity + ": duration does not cover approach, interaction and recovery")
        authored = spec.get('interaction_reconstruction',{}).get('schema') == AUTHORED_SCHEMA
        if family in {"FullyBlocked", "Wall"} and not authored and (
                spec["interaction_kind"] != "blocked_wait" or wait < 60 or duration < 150):
            raise ValueError(identity + ": preserve at least 60 seconds blockage and 150 seconds episode budget")
        capabilities = _strings(spec.get("required_capabilities"), identity + ".required_capabilities")
        if not set(capabilities).issubset(CAPABILITIES):
            raise ValueError(identity + ": unknown capability cannot be silently ignored")
        if family == "ObscuredStop" and "real_stop_sign" not in capabilities:
            raise ValueError(identity + ": a roadside prop cannot replace a real stop sign")
        if family == "RightOfWay" and "conflicting_junction" not in capabilities:
            raise ValueError(identity + ": cross traffic requires a verified conflicting branch")
        proxy = spec.get("proxy")
        if (not isinstance(proxy, dict) or not isinstance(proxy.get("description"), str)
                or not proxy["description"].strip() or not isinstance(proxy.get("limitations"), str)
                or not proxy["limitations"].strip()
                or proxy.get("official_visual_shift_reproduced") is not False):
            raise ValueError(identity + ": disclose independent mechanism proxy and visual limitations")
        if not isinstance(spec.get("supported_visual_proxy"), bool):
            raise ValueError(identity + ": supported_visual_proxy must be explicit")
        if family == "ImageOnObject" and not spec["supported_visual_proxy"]:
            raise ValueError(identity + ": ImageOnObject needs explicit visual-proxy labeling")
        actors = spec.get("actor_specs")
        if not isinstance(actors, list) or not actors:
            raise ValueError(identity + ": actor_specs must be nonempty")
        actor_ids = set()
        for actor in actors:
            if (not isinstance(actor, dict) or not isinstance(actor.get("id"), str)
                    or not actor["id"] or actor["id"] in actor_ids):
                raise ValueError(identity + ": actor ids must be nonempty and unique")
            actor_ids.add(actor["id"])
            if actor.get("actor_type") not in {"vehicle", "walker", "obstacle", "visual_proxy"}:
                raise ValueError(identity + ": unsupported actor_type")
            _strings(actor.get("blueprint_candidates"), identity + ".blueprint_candidates")
            available = STANDARD_BLUEPRINTS
            if spec.get("route_source") is not None:
                available |= COLLECTION_BLUEPRINTS
            if not set(actor["blueprint_candidates"]).issubset(available):
                raise ValueError(identity + ": blueprint is outside the explicit standard-asset allowlist")
            if actor.get("required") is not True:
                raise ValueError(identity + ": required mechanism actors cannot be silently omitted")
            for key in ("s_offset_m", "lateral_offset_m", "yaw_offset_deg", "speed_mps"):
                _number(actor.get(key), identity + ".actor." + key,
                        minimum=0 if key == "speed_mps" else None)
            lane = actor.get("lane_offset")
            if isinstance(lane, bool) or not isinstance(lane, int) or abs(lane) > 1:
                raise ValueError(identity + ": actor lane_offset must be -1, 0 or 1")
        triggers = spec.get("trigger_specs")
        if (not isinstance(triggers, list) or not triggers
                or any(not isinstance(trigger, dict) or trigger.get("source") != "current_observation"
                       or not isinstance(trigger.get("event"), str) or not trigger["event"]
                       for trigger in triggers)):
            raise ValueError(identity + ": triggers require current-observation provenance")
        policy = spec.get("teacher_policy")
        if (not isinstance(policy, dict) or policy.get("observation_only") is not True
                or not isinstance(policy.get("allow_opposite_lane"),bool)
                or policy['allow_opposite_lane'] and (
                    not authored or not {'opposite_left_lane','crossable_left_boundary'}.issubset(capabilities))
                or policy.get("kind") != spec["interaction_kind"]):
            raise ValueError(identity + ": teacher must use current observations and legal driving corridors")
        if not SUCCESS_REQUIREMENTS.issubset(_strings(
                spec.get("success_requirements"), identity + ".success_requirements")):
            raise ValueError(identity + ": incomplete executed-success requirements")
        if 'authored_variants' in spec:
            variants=spec['authored_variants']
            if not isinstance(variants,dict) or set(variants)!={'Base','Generalization'}:
                raise ValueError(identity + ': specify the two authored route variants')
            for name in variants:
                _authored_variant(spec,name)
    if families != set(FAMILIES):
        raise ValueError("catalog family coverage differs from the official 17-category taxonomy")
    return deepcopy(catalog)


def load_catalog(path=None):
    """Read a strict independent catalog without CARLA, network or model access."""
    source = filesystem_path(DEFAULT_CATALOG if path is None else path)
    try:
        catalog = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read demonstration catalog: " + str(source)) from exc
    return validate_catalog(catalog)


def get_demonstration(catalog, identity):
    """Return an isolated design matching its stable id or exact family name."""
    checked = validate_catalog(catalog)
    matches = [spec for spec in checked["demonstrations"]
               if identity in (spec["id"], spec["family"])]
    if len(matches) != 1:
        raise ValueError("unknown or ambiguous demonstration: " + str(identity))
    return _map_assets(deepcopy(matches[0]))


def _map_assets(spec):
    """Use the Fail2Drive assets on Town13 and keep the exact actor IDs in each capture."""
    asset = spec.get('town13_interaction_asset')
    if spec['map'] != 'Town13' or not asset or spec.get('interaction_reconstruction'):
        return spec
    roles = {'walker.animal.1009': 'crossing_actor', 'static.prop.gnome': 'blocking_obstacle',
             'static.prop.vendingmachine': 'passable_edge_obstacle'}
    if asset not in roles:
        raise ValueError('Unknown Town13 demonstration asset: '+str(asset))
    for actor in spec['actor_specs']:
        if actor['role'] == roles[asset]:
            actor['blueprint_candidates'] = [asset]
    if spec['family'] == 'PassableObstacles':
        # The reference scene has one vending machine facing across the road.
        spec['actor_specs'] = spec['actor_specs'][:1]
        spec['actor_specs'][0].update(lateral_offset_m=2.5, yaw_offset_deg=-90.)
    spec['proxy'] = dict(description=spec['town13_description'],
        limitations='使用 Fail2Drive 实体，位置、车流和事件时序按短片段设计。',
        official_visual_shift_reproduced=False)
    return spec


def _construction_work_zone(site):
    """Build the roadside warning and physical L-shaped closure from road stations."""
    if (not isinstance(site,dict) or set(site)!={'station_m','advance_warning_station_m','cone_left_m','lane_warning'}
            or type(site['lane_warning']) is not bool):
        raise ValueError('Invalid work zone layout or missing explicit lane-warning choice')
    station=_number(site.get('station_m'),'work-zone station',minimum=10.)
    advance=_number(site.get('advance_warning_station_m'),'advance warning station',minimum=0.,maximum=station-5.)
    left=_number(site.get('cone_left_m'),'construction cone row offset',minimum=0.3)

    def prop(name,blueprint,s,d,yaw,z,role='blocking_obstacle'):
        return dict(id=name,role=role,actor_type='obstacle',blueprint_candidates=['static.prop.'+blueprint],
                    lane_offset=0,s_offset_m=s,lateral_offset_m=d,yaw_offset_deg=yaw,
                    z_offset_m=z,speed_mps=0.,required=True)

    # Relative positions and asset pivots follow the pinned construction_crash_vehicle.py.
    props=[prop('advance_warning','warningconstruction',advance,-3.4,90.,.2,'construction_advance_warning')]
    if site['lane_warning']:
        props.append(prop('lane_warning','trafficwarning',station-5.,0.,270.,.03))
    props.append(prop('worksite_debris','dirtdebris02',station+2.,0.,90.,1.))
    for index in range(1,5):
        props.append(prop('side_cone_'+str(index),'constructioncone',station+2.*index,left,0.,.1))
    for index in range(1,4):
        props.append(prop('end_cone_'+str(index),'constructioncone',station+8.,left-index,90.,.1))
    return props


def _authored_variant(template, variant):
    """Resolve concrete actor placements, motion triggers and ego objectives."""
    result=deepcopy(template)
    variants=result.pop('authored_variants',{})
    if not isinstance(variants,dict) or variant not in variants:
        raise ValueError('Missing authored '+variant+' execution design for '+template['family'])
    design=variants[variant]
    allowed={'description','actors','cruise_speed_mps','activation_distance_m','wait_s','opposing_borrow','construction_site'}
    if (not isinstance(design,dict) or set(design)-allowed or not isinstance(design.get('description'),str)
        or not design['description'].strip() or not isinstance(design.get('actors'),dict)
        or type(design.get('opposing_borrow',False)) is not bool):
        raise ValueError('Invalid authored variant for '+template['family'])
    result['map']='Town13'
    if variant=='Generalization':
        _map_assets(result)
    result['cruise_speed_mps']=_number(design.get('cruise_speed_mps'),'authored cruise speed',minimum=.1)
    if 'activation_distance_m' in design:
        distance=_number(design['activation_distance_m'],'activation distance',minimum=.1)
        result['trigger_specs'][0]['distance_m']=distance
    if 'wait_s' in design:
        result['wait_s']=_number(design['wait_s'],'authored waiting time',minimum=0.)
        for trigger in result['trigger_specs']:
            if 'duration_s' in trigger:
                trigger['duration_s']=result['wait_s']
    actors={a['id']:a for a in result['actor_specs']}
    for name,changes in design['actors'].items():
        if changes is None:
            actors.pop(name,None)
        elif isinstance(changes,dict) and 'id' not in changes:
            actors.setdefault(name,{'id':name}).update(changes)
        else:
            raise ValueError('Invalid authored actor '+name)
    if 'construction_site' in design:
        site=design['construction_site']
        for actor in _construction_work_zone(site):
            if actor['id'] in actors:
                raise ValueError('Duplicate construction actor '+actor['id'])
            actors[actor['id']]=actor
        result['construction_site_station_m']=site['station_m']
    result['actor_specs']=list(actors.values())
    if not actors:
        raise ValueError('Authored scene requires a physical interaction')
    legal=STANDARD_BLUEPRINTS|COLLECTION_BLUEPRINTS
    ego_length=sum(result[k] for k in ('approach_m','interaction_m','recovery_m'))
    for actor in actors.values():
        choices=_strings(actor.get('blueprint_candidates'),actor['id']+' blueprints')
        if (not set(choices)<=legal or actor.get('required') is not True
                or actor.get('actor_type') not in {'obstacle','vehicle','walker','visual_proxy'}
                or type(actor.get('lane_offset')) is not int or abs(actor['lane_offset'])>1):
            raise ValueError('Invalid authored physical actor '+actor['id'])
        # Oncoming traffic can start upstream of the recorded ego segment.
        maximum=None if actor['role']=='oncoming_vehicle' else ego_length
        _number(actor['s_offset_m'],'actor station',minimum=0.,maximum=maximum)
        for k in ('lateral_offset_m','yaw_offset_deg','pitch_offset_deg','roll_offset_deg','z_offset_m'):
            _number(actor.get(k,0.),'actor '+k)
        _number(actor['speed_mps'],'actor speed',minimum=0.)
    if design.get('opposing_borrow'):
        vehicles=[a for a in actors.values() if a['role']=='oncoming_vehicle']
        if (len(vehicles)!=2 or vehicles[0].get('release_time_s')!=0. or vehicles[1].get('release_on_borrow') is not True
            or vehicles[0]['s_offset_m']>=vehicles[1]['s_offset_m']
            or any(a['lane_offset']!=1 or a['actor_type']!='vehicle' or a['speed_mps']<=0. for a in vehicles)):
            raise ValueError('Authored two-way borrowing needs the two ordered, independently released oncoming vehicles')
        result['teacher_policy']['allow_opposite_lane']=True
        result['required_capabilities']=['opposite_left_lane','crossable_left_boundary']
    if result['interaction_kind']=='slow_lead':
        result['required_capabilities'].append('walkable_right_exit')
    result['interaction_reconstruction']=dict(schema=AUTHORED_SCHEMA,variant=variant,description=design['description'],
        scope='authored_short_interaction;not_official_traffic_replay',official_scoring=False)
    result['proxy']=dict(description=design['description'],official_visual_shift_reproduced=False,
        limitations='基于该家族采集路线位置手工设置演员、交通和交互；正式评测使用其独立官方路线及交通。')
    return result


def catalog_digest(catalog):
    """Stable design identity to bind a real execution archive to this catalog."""
    checked = validate_catalog(catalog)
    return hashlib.sha256(json.dumps(checked, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _collection_route_source(official_root,route_id,split_path=None):
    """Derive the sole location-provenance record from frozen collection XML."""
    import xml.etree.ElementTree as ET
    from .memory_split import ALL_FAMILIES_SPLIT,load_transfer_split,_digest
    from . import OFFICIAL_COMMIT
    split=load_transfer_split(split_path or ALL_FAMILIES_SPLIT)
    if split not in (load_transfer_split(ALL_FAMILIES_SPLIT),load_transfer_split()):
        raise ValueError('Use the configured frozen collection/test partition')
    records={r['route_id']:(row,r) for row in split['rows'] for r in row['collection']}
    if type(route_id) is not int or route_id not in records:
        raise ValueError('Anchor route must belong to the frozen collection partition, never a held-out route or partner')
    row,record=records[route_id]
    root=filesystem_path(official_root)
    def read(rec):
        path=filesystem_path(root/rec['routes']);data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=rec['sha256']:
            raise ValueError('Collection XML differs from its frozen SHA256')
        route=ET.fromstring(data).find('route')
        if route is None or route.get('id')!=str(rec['route_id']) or route.get('town')!='Town13':
            raise ValueError('Collection XML route ID or map differs from the frozen selection')
        return path,route
    path,route=read(record)
    paired=next(r for r in row['collection'] if r['variant']=='Generalization')
    reference=route if paired['route_id']==route_id else read(paired)[1]
    geometry=[p.attrib for p in route.findall('waypoints/position')]
    if len(geometry)<2 or geometry!=[p.attrib for p in reference.findall('waypoints/position')]:
        raise ValueError('Collection location needs matching Base/Generalization sparse route geometry')
    events=reference.findall('scenarios/scenario')
    if row['family']=='PedestrianCrowd':
        events=[e for e in events if e.get('type')=='PedestrianCrowd']
    if len(events)!=1:
        raise ValueError('Collection-location hint needs one unambiguous original scenario anchor')
    event=events[0];trigger=event.find('trigger_point');distance=event.find('distance')
    if row['family']=='PedestrianCrowd':
        distance=event.find('pedestrian_center')
    if trigger is None:
        raise ValueError('Original collection scenario lacks a trigger anchor')
    xyz=[float(trigger.get(k)) for k in ('x','y','z')]
    yaw=float(trigger.get('yaw'))
    default_offsets={'DynamicObjectCrossing':12.,'ConstructionObstacleRightLane':100.}
    offset=float(distance.get('value')) if distance is not None else default_offsets.get(event.get('type'),0.)
    for value in [*xyz,yaw]:_number(value,'collection anchor coordinate or heading')
    _number(offset,'collection event distance',minimum=0)
    source=dict(schema='fail2drive-demo-route-source-1.0',mode='collection_route_anchor',
        official_route_id=route_id,variant=record['variant'],official_routes_path=str(path),
        routes_sha256=record['sha256'],collection_split_sha256=_digest(split),official_commit=OFFICIAL_COMMIT,
        original_scenario_types=[s.get('type') for s in route.findall('scenarios/scenario')],
        map='Town13',family=row['family'],trigger_xyz_carla_m=xyz,trigger_yaw_deg=yaw,event_offset_m=offset,
        official_scene_reproduced=False)
    if len(split['rows'])==len(FAMILIES):
        source.update(schema='fail2drive-demo-route-source-2.0',
            anchor_kind={'RightOfWay':'conflicting_junction_ahead','ObscuredStop':'controlled_stop_ahead'}.get(
                row['family'],'lane_station'),
            anchor_scenario_type=event.get('type'),anchor_reference_sha256=paired['sha256'],
            anchor_reference_variant='Generalization')
    return source


def validate_route_source(spec):
    """Validate a location hint by regenerating its original source record."""
    source=spec.get('route_source')
    if spec.get('route_policy')=='independent_topology_selection':
        if source is not None:
            raise ValueError('Independent topology selection must not claim a collection-route anchor')
        return None
    if (spec.get('route_policy')!='collection_route_anchored_topology' or spec.get('map')!='Town13'
            or not isinstance(source,dict) or not isinstance(source.get('official_routes_path'),str)):
        raise ValueError('Unsupported demonstration route provenance')
    path=filesystem_path(source['official_routes_path'])
    from .memory_split import DEFAULT,load_transfer_split,_digest
    old_split=DEFAULT if source.get('collection_split_sha256')==_digest(load_transfer_split(DEFAULT)) else None
    expected=_collection_route_source(path.parent.parent,source.get('official_route_id'),old_split)
    if 'original_scenario_types' not in source:  # Legacy location-only record.
        expected.pop('original_scenario_types')
    if spec.get('family')!=expected['family'] or source!=expected:
        raise ValueError('Collection anchor identity or geometry differs from its frozen original XML')
    return expected


def with_collection_route_anchor(spec,official_root,route_id,split_path=None):
    """Bind an authored short demo to a verified location, not the official NPC script."""
    source=_collection_route_source(official_root,route_id,split_path)
    if source['family']!=spec.get('family'):
        raise ValueError('Anchor route family differs from the selected demonstration')
    result=_authored_variant(spec,source['variant'])
    result.update(route_policy='collection_route_anchored_topology',route_source=source)
    if spec['family']=='PedestrianCrowd':
        result['interaction_reconstruction']['target_event']='PedestrianCrowd'
        result['proxy']['limitations']+=' 本片段聚焦路侧人群，官方同路线的开车门伴随事件另行区分。'
    return result
