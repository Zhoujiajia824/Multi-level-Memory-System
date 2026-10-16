"""Resolve a research suite on the actual server; never guess map coordinates.

list is offline. prepare changes maps and must run on an idle CARLA instance.
Prepared cases freeze routes and event geometry shared by all paired conditions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "carla_bridge/config/paper_suite.yaml"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def point(location):
    return {key: round(float(getattr(location, key)), 5) for key in ("x", "y", "z")}


def geometry_digest(cfg):
    return digest({key: cfg[key] for key in ("map", "ego", "route", "events", "reserved_event_sites")})


def read_catalog(path=CATALOG):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if set(data["splits"]["train"]) & set(data["splits"]["eval"]):
        raise ValueError("training and evaluation seeds must be disjoint")
    for name, case in data["cases"].items():
        profile = data["profiles"][case["profile"]]
        for event in case.get("events", []):
            if event not in profile["anchors"]:
                raise ValueError(f"unknown event {name}/{event}")
        control = case.get("paired_control")
        if control and data["cases"][control]["profile"] != case["profile"]:
            raise ValueError(f"paired routes differ: {name}")
    return data


def materialize(catalog, prepared, split, seeds=None):
    """Pure assembly, also used by offline tests. Does not declare driving success."""
    defaults = catalog["defaults"]
    seeds = list(seeds if seeds is not None else catalog["splits"][split])
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative 31-bit integers")
    other = "eval" if split == "train" else "train"
    if set(seeds) & set(catalog["splits"][other]):
        raise ValueError("requested seeds overlap the other split")
    cases = {}
    for name, template in catalog["cases"].items():
        profile = prepared[template["profile"]]
        for seed in seeds:
            cfg = {
                "scenario_name": f"{name}_{split}_s{seed}", "map": profile["map"],
                "duration_s": defaults["duration_s"], "seed": seed,
                "weather": copy.deepcopy(template.get("weather", defaults["weather"])),
                "traffic": copy.deepcopy(defaults["traffic"]),
                "ego": {"blueprint": defaults["ego_blueprint"], "spawn": copy.deepcopy(profile["spawn"])},
                "route": {"points": copy.deepcopy(profile["points"]),
                          "expected_length_m": profile["length_m"]},
                "events": [copy.deepcopy(profile["anchors"][e]) for e in template.get("events", [])],
                "reserved_event_sites": [copy.deepcopy(e.get("spawn", e.get("start")))
                                         for e in profile["anchors"].values()],
                "experiment": {"suite": catalog["suite"], "case_id": name, "split": split,
                               "category": template["category"], "paired_control": template.get("paired_control"),
                               "description": template["description"], "server_version": profile["server_version"],
                               "map_sha256": profile["map_sha256"], "route_sha256": digest(profile["points"]),
                               "prepared": True},
            }
            cfg["experiment"]["geometry_sha256"] = geometry_digest(cfg)
            cases[cfg["scenario_name"]] = cfg
    return cases


def validate_prepared(cfg):
    exp = cfg.get("experiment") or {}
    if exp.get("suite") != "paper_v1" or exp.get("prepared") is not True:
        raise ValueError("paper case must be produced by paper_suite prepare")
    pts = cfg.get("route", {}).get("points", [])
    if len(pts) < 2 or digest(pts) != exp.get("route_sha256"):
        raise ValueError("prepared route is missing or changed; prepare the suite again")
    if geometry_digest(cfg) != exp.get("geometry_sha256"):
        raise ValueError("prepared event/route geometry changed; prepare the suite again")
    if not math.isfinite(float(cfg["duration_s"])) or float(cfg["duration_s"]) <= 0:
        raise ValueError("duration_s must be positive and finite")
    if not 0 <= int(cfg["seed"]) < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    for key in ("num_vehicles", "num_walkers"):
        if type(cfg["traffic"][key]) is not int or cfg["traffic"][key] < 0:
            raise ValueError("traffic counts must be nonnegative integers")
    ids = [e["id"] for e in cfg.get("events", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("event IDs must be unique")
    for event in cfg.get("events", []):
        if event["type"] not in ("pedestrian_crossing", "stationary_vehicle"):
            raise ValueError("unsupported paper event")
        if not 0 <= event["trigger_distance_m"] <= event["anchor_distance_m"] < cfg["route"]["expected_length_m"] - 30:
            raise ValueError("event outside usable route interval")


def summarize_runs(runs_dir, output):
    """One row per attempted paper run, including failures; never drop bad runs."""
    fields = ["run_id", "case_id", "category", "split", "seed", "mode", "memory_access",
              "planner_backend", "semantic_influence", "planner_config_sha256",
              "retrieve", "write", "short_term", "memory_bank", "status", "success",
              "route_completion", "duration_s", "collision_count", "red_light_violations",
              "trajectory_rejections", "events_valid", "events_executed", "events_required",
              "route_sha256", "geometry_sha256", "case_config_sha256", "commit", "dirty"]
    rows = []
    for manifest_path in sorted(Path(runs_dir).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cfg = yaml.safe_load(manifest.get("scenario_yaml", "")) or {}
        exp = cfg.get("experiment") or {}
        if exp.get("suite") != "paper_v1":
            continue
        outcome = manifest.get("outcome") or {}
        events = outcome.get("experiment_events") or {}
        reports = sorted(manifest_path.parent.glob("carla_run_*.json"))
        metrics = json.loads(reports[0].read_text(encoding="utf-8")).get("metrics", {}) if reports else {}
        access = manifest.get("config", {}).get("memory_access", {})
        planning = manifest.get("config", {}).get("carla", {}).get("planning", {})
        row = {"run_id": manifest["run_id"], "seed": cfg["seed"],
               **{key: exp.get(key) for key in ("case_id", "category", "split", "route_sha256", "geometry_sha256")},
               **{key: manifest.get(key) for key in ("mode", "memory_access", "memory_bank", "status", "commit", "dirty")},
               **{key: access.get(key) for key in ("retrieve", "write", "short_term")},
               "planner_backend": planning.get("backend", "vlm_trajectory"),
               "semantic_influence": planning.get("semantic_influence", False),
               "planner_config_sha256": digest(planning),
               **{key: outcome.get(key) for key in ("success", "route_completion", "duration_s", "trajectory_rejections")},
               **{key: metrics.get(key) for key in ("collision_count", "red_light_violations")},
               "events_valid": events.get("valid"), "events_executed": events.get("executed"),
               "events_required": events.get("required"), "case_config_sha256": digest(cfg)}
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _sidewalk(carla, road_map, wp, side):
    """Find an actual sidewalk near the cross-section, without road projection."""
    tf = wp.transform
    right = tf.get_right_vector()
    for half_metres in range(4, 49):
        offset = side * half_metres * .5
        p = carla.Location(x=tf.location.x + right.x * offset,
                           y=tf.location.y + right.y * offset, z=tf.location.z)
        sidewalk = road_map.get_waypoint(p, project_to_road=False, lane_type=carla.LaneType.Sidewalk)
        if sidewalk and abs(sidewalk.transform.location.z - tf.location.z) < 1.5:
            # Keep the cross-section coordinate (projecting to a waypoint can
            # slide along the sidewalk into a different junction).
            p.z = sidewalk.transform.location.z + .3
            return p
    return None


def _resolve_anchor(carla, world, points, arc, spec, event_id):
    road_map = world.get_map()
    for i in range(5, len(points) - 6):
        if not spec["search_from_m"] <= arc[i] <= spec["search_to_m"]:
            continue
        if any(points[j].distance(points[i]) < 15 for j in range(len(points))
               if abs(arc[j] - arc[i]) > 60):
            continue
        wp = road_map.get_waypoint(points[i], project_to_road=True)
        if wp is None or wp.is_junction:
            continue
        # Exclude bends/junctions within roughly +/-10m of the event.
        nearby = [road_map.get_waypoint(points[j], project_to_road=True) for j in (i - 5, i + 5)]
        if any(p is None or p.is_junction for p in nearby):
            continue
        yaw = wp.transform.rotation.yaw
        if any(abs((p.transform.rotation.yaw - yaw + 180) % 360 - 180) > 12 for p in nearby):
            continue
        event = {"id": event_id, "type": spec["type"], "anchor_distance_m": round(arc[i], 4)}
        if spec["type"] == "pedestrian_crossing":
            right, left = _sidewalk(carla, road_map, wp, 1), _sidewalk(carla, road_map, wp, -1)
            if right is None or left is None or not 5 <= right.distance(left) <= 25:
                continue
            event.update(start=point(right), target=point(left), speed_mps=spec["speed_mps"],
                         trigger_distance_m=round(arc[i] - spec["lead_distance_m"], 4))
        else:
            # A same-direction adjacent driving lane must exist for bypassing.
            adjacent = [(wp.get_left_lane(), carla.LaneChange.Left), (wp.get_right_lane(), carla.LaneChange.Right)]
            if not any(p and p.lane_type == carla.LaneType.Driving and p.lane_id * wp.lane_id > 0
                       and p.road_id == wp.road_id and (wp.lane_change & permission)
                       for p, permission in adjacent):
                continue
            world.get_blueprint_library().find(spec["blueprint"])
            spawn = point(wp.transform.location)
            spawn.update(z=spawn["z"] + .3, yaw=float(yaw))
            event.update(spawn=spawn, blueprint=spec["blueprint"], trigger_distance_m=0.0)
        return event
    raise ValueError(f"no suitable geometry for {event_id} in requested route interval")


def prepare_profile(carla, client, profile, defaults):
    from carla_bridge.env.route_planner import RoutePlanner
    world = client.load_world(profile["map"])
    settings = world.get_settings()
    if settings.synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick(30.0)
    road_map = world.get_map()
    spawns = road_map.get_spawn_points()
    world.get_blueprint_library().find(defaults["ego_blueprint"])
    if not list(world.get_blueprint_library().filter("walker.pedestrian.*")):
        raise ValueError("server has no pedestrian blueprints")
    errors = []
    for index in profile["start_candidates"]:
        if index >= len(spawns):
            continue
        tf = spawns[index]
        planner = RoutePlanner(world, SimpleNamespace(get_location=lambda: tf.location))
        try:
            points = planner.plan(length_m=profile["length_m"], turns=profile["turns"],
                                  sampling_resolution=defaults["sampling_resolution"])
            arc = [0.0]
            for a, b in zip(points, points[1:]):
                arc.append(arc[-1] + a.distance(b))
            if not .98 * profile["length_m"] <= arc[-1] <= 1.05 * profile["length_m"]:
                raise ValueError("route ended early or has an unexpected length")
            if any(a.distance(b) > defaults["sampling_resolution"] * 3 for a, b in zip(points, points[1:])):
                raise ValueError("discontinuous route")
            anchors = {name: _resolve_anchor(carla, world, points, arc, spec, name)
                       for name, spec in profile["anchors"].items()}
            # Prevent a later loop from visiting an event's spatial location before
            # its scheduled route interval, which would invalidate paired timing.
            for event in anchors.values():
                anchor_i = min(range(len(arc)), key=lambda j: abs(arc[j] - event["anchor_distance_m"]))
                if any(points[j].distance(points[anchor_i]) < 15
                       for j in range(len(points)) if abs(arc[j] - arc[anchor_i]) > 60):
                    raise ValueError("event site is revisited by another part of the route")
            spawn = point(tf.location)
            spawn["yaw"] = float(tf.rotation.yaw)
            return {"map": profile["map"], "spawn_index": index, "spawn": spawn,
                    "points": [point(p) for p in points], "length_m": arc[-1], "anchors": anchors,
                    "server_version": client.get_server_version(),
                    "map_sha256": hashlib.sha256(road_map.to_opendrive().encode()).hexdigest()}
        except (ValueError, RuntimeError) as exc:
            errors.append(f"spawn {index}: {exc}")
    raise ValueError(f"{profile['map']}: no route passed preflight; " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "prepare", "summarize"])
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper_v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/carla_runs")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "outputs/paper_results.csv")
    args = parser.parse_args()
    catalog = read_catalog(args.catalog)
    if args.action == "summarize":
        count = summarize_runs(args.runs_dir, args.summary_csv)
        print(f"Wrote {count} attempted paper runs to {args.summary_csv}")
        return
    if args.action == "list":
        for name, case in catalog["cases"].items():
            print(f"{name:24} {catalog['profiles'][case['profile']]['map']:10} {case['category']:10} {case['description']}")
        return
    if args.output.exists():
        raise FileExistsError(f"use a new output directory; will not overwrite {args.output}")
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    available = {name.rsplit("/", 1)[-1] for name in client.get_available_maps()}
    required = {p["map"] for p in catalog["profiles"].values()}
    if required - available:
        raise ValueError(f"missing maps: {sorted(required - available)}")
    if client.get_client_version() != client.get_server_version():
        raise ValueError("CARLA client/server versions differ")
    print("Preparing on idle server: maps will be loaded; no VLM or traffic actors are started.", flush=True)
    prepared = {}
    for key, profile in catalog["profiles"].items():
        print(f"Checking {profile['map']} ...", flush=True)
        prepared[key] = prepare_profile(carla, client, profile, catalog["defaults"])
    # Write only after ALL profiles have passed, never a partly validated suite.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preflight.json").write_text(json.dumps({"catalog_sha256": digest(catalog),
        "profiles": prepared, "validation": "geometry_only_not_driving_test"}, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("train", "eval"):
        destination = args.output / split
        destination.mkdir()
        for name, cfg in materialize(catalog, prepared, split).items():
            validate_prepared(cfg)
            (destination / (name + ".yaml")).write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Prepared {len(catalog['cases'])} cases x 6 seeds: {args.output}", flush=True)
    print("Geometry preflight only. Spawn counts, event execution and 100% route completion still require driving runs.")


if __name__ == "__main__":
    main()
