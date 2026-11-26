"""Independent development demonstrations, never official evaluation answers.

The catalog describes mechanisms and relative placements only. Loading a design
does not certify a successful episode or authorize publishing it as memory.
"""
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
    "same_direction_right_lane", "opposite_left_lane", "right_sidewalk",
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
})
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "config" / "fail2drive_demonstrations.json"
_FORBIDDEN_KEYS = frozenset({
    "official_route", "official_route_id", "official_seed", "route_id",
    "official_coordinates", "world_coordinates", "future_controls",
    "future_states", "ground_truth_future", "expert_controls", "evaluation_result",
})


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
            if not isinstance(key, str) or key in _FORBIDDEN_KEYS:
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
    # Only this explicitly validated provenance block may carry a collection
    # route ID. Future/evaluation fields remain forbidden everywhere else.
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
        _number(spec.get("cruise_speed_mps"), identity + ".cruise_speed_mps", minimum=0.1, maximum=6)
        wait = _number(spec.get("wait_s"), identity + ".wait_s", minimum=0)
        duration = _number(spec.get("max_duration_s"), identity + ".max_duration_s", minimum=30)
        travel = sum(spec[key] for key in ("approach_m", "interaction_m", "recovery_m"))
        if duration < wait + travel / spec["cruise_speed_mps"] + 15:
            raise ValueError(identity + ": duration does not cover approach, interaction and recovery")
        if family in {"FullyBlocked", "Wall"} and (
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
                or policy.get("allow_opposite_lane") is not False
                or policy.get("kind") != spec["interaction_kind"]):
            raise ValueError(identity + ": teacher must use current observations and legal driving corridors")
        if not SUCCESS_REQUIREMENTS.issubset(_strings(
                spec.get("success_requirements"), identity + ".success_requirements")):
            raise ValueError(identity + ": incomplete executed-success requirements")
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


# Keys name actual actors in the authored catalog. Each tuple describes the
# physical reconstruction and changes real spawn or Walker/VehicleControl input.
# A None entry removes an actor for that variant. Unlisted actors are retained.
# Both short demos deliberately contain a relevant interaction, even where the
# official Base XML has an ordinary approach or an empty scenario element.
_COLLECTION_INTERACTIONS = {
    'BadParking': {
        'Base': ('轿车沿路侧平行停放，车身部分占道，减速后合法绕过。',
                 {'parked_car': dict(yaw_offset_deg=0., lateral_offset_m=-.8)}),
        'Generalization': ('斜停的日产巡逻车突出车道，减速后留出侧向净距。',
                           {'parked_car': dict(blueprint_candidates=['vehicle.nissan.patrol_2021'],
                                               yaw_offset_deg=15., lateral_offset_m=-.4)}),
    },
    'ConstructionPedestrian': {
        'Base': ('施工绕行时工人从右侧较远位置缓慢横穿，待其清空后通过。',
                 {'worker': dict(lateral_offset_m=-5., speed_mps=1.15)}),
        'Generalization': ('施工物后近侧工人横穿，绕行期间让行并确认净空。',
                           {'worker': dict(blueprint_candidates=['walker.pedestrian.0040'],
                                           lateral_offset_m=-3.2, speed_mps=1.5)}),
    },
    'ConstructionPermutations': {
        'Base': ('锥桶及路障纵向排列，减速后在允许的邻道绕行。',
                 {'cone_one': dict(lateral_offset_m=-.2), 'cone_two': dict(lateral_offset_m=-.2)}),
        'Generalization': ('散开的锥桶与斜置路障形成不规则施工边界，绕过最后一个锥桶后归道。',
                           {'barrier': dict(yaw_offset_deg=25.), 'cone_two': dict(s_offset_m=43.)}),
    },
    'CustomObstacles': {
        'Base': ('散落箱体与桶侵占车道，减速、留距绕行。',
                 {'road_debris': dict(blueprint_candidates=['static.prop.box03']),
                  'road_barrel': dict(blueprint_candidates=['static.prop.barrel'])}),
        'Generalization': ('园艺摆件占道，确认邻道可用后绕行。', {}),
    },
    'HardBrake': {
        'Base': ('较慢前车行驶后急停且刹车灯不亮，自车减速跟停，待前车恢复后再起步。',
                 {'lead_car': dict(s_offset_m=25., speed_mps=1.6)}),
        'Generalization': ('前车以更高速度行驶后无灯急刹，自车跟停并恢复。',
                           {'lead_car': dict(s_offset_m=27., speed_mps=2.2)}),
    },
    'RightOfWay': {
        'Base': ('轿车从交叉支路驶入冲突区，自车按实际占用让行。', {}),
        'Generalization': ('摩托车从交叉支路抢行，自车观察、让行并恢复。',
                           {'crossing_vehicle': dict(blueprint_candidates=['vehicle.kawasaki.ninja'],
                                                     speed_mps=4.)}),
    },
    'PassableObstacles': {
        'Base': ('路侧锥桶仍留有本车道净空，保持车道通过。',
                 {'edge_cone': dict(blueprint_candidates=['static.prop.trafficcone01'],
                                    lateral_offset_m=-2.2, yaw_offset_deg=0.), 'edge_cone_two': None}),
        'Generalization': ('路边自动售货机仍留有本车道净空，核实后保持车道通过。', {}),
    },
    'PedestrianOtherBlocker': {
        'Base': ('自动售货机旁出现横穿行人，减速避让。', {}),
        'Generalization': ('行人从路边售货亭后横穿，等待行人清空。',
                           {'roadside_blocker': dict(blueprint_candidates=['static.prop.kiosk_01'],
                                                     lateral_offset_m=-4.),
                            'crossing_pedestrian': dict(blueprint_candidates=['walker.pedestrian.0027'],
                                                        lateral_offset_m=-4.8)}),
    },
    'Animals': {
        'Base': ('真实动物从较远路侧缓慢横穿，观察并让行。',
                 {'crossing_proxy': dict(blueprint_candidates=['walker.animal.1009'],
                                         lateral_offset_m=-5., speed_mps=1.2)}),
        'Generalization': ('真实动物从较近路侧快速横穿，减速并确认其离开行驶区域。',
                           {'crossing_proxy': dict(lateral_offset_m=-3.3, speed_mps=2.)}),
    },
    'PedestriansOnRoad': {
        'Base': ('单个行人在机动车道内慢行，自车减速跟随。',
                 {'walker_two': None, 'walker_three': None}),
        'Generalization': ('三名行人在机动车道内错落行走，自车减速跟随直至道路恢复通行。',
                           {'walker_one': dict(blueprint_candidates=['walker.pedestrian.0034'], speed_mps=2.),
                            'walker_two': dict(blueprint_candidates=['walker.pedestrian.0007'], speed_mps=2.),
                            'walker_three': dict(blueprint_candidates=['walker.pedestrian.0015'], speed_mps=2.)}),
    },
}


def catalog_digest(catalog):
    """Stable design identity to bind a real execution archive to this catalog."""
    checked = validate_catalog(catalog)
    return hashlib.sha256(json.dumps(checked, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _collection_route_source(official_root,route_id,split_path=None):
    """Derive the sole location-provenance record from frozen collection XML."""
    import xml.etree.ElementTree as ET
    from .memory_split import load_transfer_split,_digest
    from . import OFFICIAL_COMMIT
    split=load_transfer_split(split_path)
    if _digest(split)!=_digest(load_transfer_split()):
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
    if len(events)!=1:
        raise ValueError('Collection-location hint needs one unambiguous original scenario anchor')
    event=events[0];trigger=event.find('trigger_point');distance=event.find('distance')
    if trigger is None:
        raise ValueError('Original collection scenario lacks a trigger anchor')
    xyz=[float(trigger.get(k)) for k in ('x','y','z')]
    yaw=float(trigger.get('yaw'))
    offset=(float(distance.get('value')) if distance is not None else
            (12. if event.get('type')=='DynamicObjectCrossing' else 0.))
    for value in [*xyz,yaw]:_number(value,'collection anchor coordinate or heading')
    _number(offset,'collection event distance',minimum=0)
    return dict(schema='fail2drive-demo-route-source-1.0',mode='collection_route_anchor',
        official_route_id=route_id,variant=record['variant'],official_routes_path=str(path),
        routes_sha256=record['sha256'],collection_split_sha256=_digest(split),official_commit=OFFICIAL_COMMIT,
        original_scenario_types=[s.get('type') for s in route.findall('scenarios/scenario')],
        map='Town13',family=row['family'],trigger_xyz_carla_m=xyz,trigger_yaw_deg=yaw,event_offset_m=offset,
        official_scene_reproduced=False)


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
    expected=_collection_route_source(path.parent.parent,source.get('official_route_id'))
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
    result=deepcopy(spec)
    result.update(map='Town13',route_policy='collection_route_anchored_topology',route_source=source)
    variant=source['variant']
    if variant=='Generalization':
        _map_assets(result)
    description,changes=_COLLECTION_INTERACTIONS[spec['family']][variant]
    actors={actor['id']:actor for actor in result['actor_specs']}
    for identity,settings in changes.items():
        if settings is None:
            # The standalone Town13 design may already have removed an actor.
            actors.pop(identity,None)
        else:
            actors[identity].update(settings)
    result['actor_specs']=list(actors.values())
    # Time the fast crossing to the approach: at cruising speed 20 m would
    # let the animal finish crossing the ego lane before the car encounters it.
    if variant=='Generalization' and spec['family'] in ('Animals','ConstructionPedestrian'):
        result['trigger_specs'][0]['distance_m']=12. if spec['family']=='Animals' else 14.
    result['interaction_reconstruction']=dict(variant=variant,description=description,
        scope='authored_short_interaction;not_official_traffic_replay')
    result['proxy']=dict(description=description,limitations=(
        '按该类别编写短交互，演员布置、动态交通和时序为重建设计；'
        '两个版本均包含可采集的交互，官方 Base 可能缺少此事件或使用不同演员。'),
        official_visual_shift_reproduced=False)
    return result
