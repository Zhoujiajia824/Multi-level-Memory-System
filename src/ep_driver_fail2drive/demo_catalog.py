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
    _check_payload(catalog)
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
        if spec.get("map") not in {"Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD_Opt"}:
            raise ValueError(identity + ": use a standard independent development map")
        if spec.get("interaction_kind") not in INTERACTION_KINDS:
            raise ValueError(identity + ": unsupported interaction_kind")
        if spec.get("design_status") != "pending_execution":
            raise ValueError(identity + ": catalog is a design, never execution success evidence")
        if spec.get("route_policy") != "independent_topology_selection":
            raise ValueError(identity + ": route must be selected independently by topology")
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
            if not set(actor["blueprint_candidates"]).issubset(STANDARD_BLUEPRINTS):
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
    return deepcopy(matches[0])


def catalog_digest(catalog):
    """Stable design identity to bind a real execution archive to this catalog."""
    checked = validate_catalog(catalog)
    return hashlib.sha256(json.dumps(checked, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
