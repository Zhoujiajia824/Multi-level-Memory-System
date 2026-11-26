"""Observe official scoring without changing SDK sources or numerical results.

Only the scoring process imports this module.  Exported futures never enter the
planning packet or numerical worker.  A human-penalty-filter pass is deliberately
excluded from the captured model rollout.
"""
import functools
import gzip
import inspect
import json
import logging
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from nine_grid.paths import filesystem_path

from .protocol import digest, write_json
from .profiles import get_profile

LOG = logging.getLogger(__name__)
VERSION = "navsim-official-replay-1.0"


def finite_json(value):
    """Detach arrays (including object-valued dataframe cells) before filtering."""
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric_values(frame):
    return finite_json(frame.to_dict(orient="records")[0])


def _actors(tracks):
    return [[dict(id=str(obj.track_token), type=obj.tracked_object_type.name.lower(),
                  center=[float(obj.center.x), float(obj.center.y), float(obj.center.heading)],
                  length_m=float(obj.box.length), width_m=float(obj.box.width))
             for obj in detection.tracked_objects] for detection in tracks]


class _PolicyObserver:
    def __init__(self, original, capture):
        self._original, self._capture, self._calls = original, capture, 0

    def __getattr__(self, key):
        return getattr(self._original, key)

    def simulate_environment(self, *args, **kwargs):
        result = self._original.simulate_environment(*args, **kwargs)
        self._calls += 1
        if self._calls == 1:
            try:
                self._capture["actors"] = _actors(result)
            except Exception as exc:
                self._capture["errors"].append("traffic capture: " + repr(exc))
        return result


class _SimulatorObserver:
    """Capture the original v1 model rollout without invoking another rollout."""
    def __init__(self, original, capture):
        self._original, self._capture, self._calls = original, capture, 0

    def __getattr__(self, key):
        return getattr(self._original, key)

    def simulate_proposals(self, *args, **kwargs):
        result = self._original.simulate_proposals(*args, **kwargs)
        self._calls += 1
        if self._calls == 1:
            try:
                states = np.asarray(result)
                if states.ndim != 3 or states.shape[0] != 2 or states.shape[2] != 11:
                    raise ValueError("Expected PDM reference and model in the official v1 rollout")
                self._capture["simulated_states"] = states[1].copy()
            except Exception as exc:
                self._capture["errors"].append("simulator capture: " + repr(exc))
        return result


def _observation_actors(observation, count):
    """Serialize the exact v1 scorer occupancy, never propagate actors ourselves.

    The official cache stores 10 Hz interpolated GT rectangles. Their corner
    order is nuPlan OrientedBox's FL, RL, RR, FR, so pose and dimensions below
    are a lossless conversion of the scored polygon, including future heading.
    The cache's unique_objects supplies only type/identity, not future poses.
    """
    frames = []
    for index in range(count):
        occupancy, actors = observation[index], []
        for token in occupancy.tokens:
            if str(token).startswith(observation.red_light_token):
                continue
            obj, polygon = observation.unique_objects[token], occupancy[token]
            corners = np.asarray(polygon.exterior.coords, dtype=float)
            if corners.shape != (5, 2) or not polygon.is_valid or polygon.is_empty:
                raise ValueError("Official v1 observation actor is not a rectangular box")
            length = float(np.linalg.norm(corners[0]-corners[1]))
            width = float(np.linalg.norm(corners[1]-corners[2]))
            if not (np.isclose(length, obj.box.length, atol=1e-5)
                    and np.isclose(width, obj.box.width, atol=1e-5)):
                raise ValueError("Official v1 observation corner order differs from OrientedBox")
            heading = math.atan2(*(corners[0]-corners[1])[::-1])
            actors.append(dict(id=str(token), type=obj.tracked_object_type.name.lower(),
                center=[float(polygon.centroid.x), float(polygon.centroid.y), heading],
                length_m=length, width_m=width, scored_polygon_wkt=polygon.wkt))
        frames.append(actors)
    return frames


class _ScorerObserver:
    def __init__(self, original, capture, v1=False):
        self._original, self._capture, self._calls, self._v1 = original, capture, 0, v1

    def __getattr__(self, key):
        return getattr(self._original, key)

    def score_proposals(self, *args, **kwargs):
        result = self._original.score_proposals(*args, **kwargs)
        self._calls += 1
        if self._calls == 1:
            try:
                # The pinned SDK scores PDM reference at index 0, model at 1.
                if len(result) != 2:
                    raise ValueError("Expected PDM reference and model in first scoring pass")
                scorer, capture = self._original, self._capture
                if not self._v1:
                    capture["raw"] = metric_values(result[1])
                states = np.asarray(scorer._states)
                if states.ndim != 3 or states.shape[0] != 2:
                    raise ValueError("Unexpected official simulated proposal shape")
                capture["ego_states"] = states[1].copy()
                if self._v1:
                    capture["actors"] = _observation_actors(scorer._observation, states.shape[1])
                capture["events"] = []
                for key, event_name in (("_collision_time_idcs", "raw_first_at_fault_collision"),
                                        ("_ttc_time_idcs", "raw_first_ttc_violation")):
                    index = float(getattr(scorer, key)[1])
                    if math.isfinite(index):
                        capture["events"].append(dict(frame_index=int(index), type=event_name))
                # This is the exact scorer mask, before the human filter resets it.
                offroad = np.asarray(scorer._ego_areas)[1, :, 1].astype(bool).copy()
                capture["raw_offroad_frames"] = np.flatnonzero(offroad).tolist()
                capture["events"].extend(dict(frame_index=int(i), type="raw_offroad")
                                         for i in np.flatnonzero(offroad))
                red_lights = []
                for index in range(states.shape[1]):
                    occupancy = scorer._observation[index]
                    reds = []
                    ego_polygon = scorer._ego_polygons[1, index]
                    for token in occupancy.tokens:
                        if str(token).startswith(scorer._observation.red_light_token):
                            polygon = occupancy[token]
                            reds.append(dict(token=str(token), wkt=polygon.wkt))
                            if ego_polygon.intersects(polygon):
                                capture["events"].append(dict(frame_index=index,
                                    type="raw_red_light_polygon_intersection", actor_id=str(token)))
                    red_lights.append(reds)
                capture["red_lights"] = red_lights
            except Exception as exc:
                self._capture["errors"].append("scorer capture: " + repr(exc))
        return result


def _token(cache, settings):
    # MetricCache has no token member; official cache processor's path layout is
    # cache/log/scenario_type/token/metric_cache.pkl, including on Windows.
    token = str(cache.file_path).replace("\\", "/").split("/")[-2]
    if token not in settings["evaluation_tokens"]:
        raise ValueError("Official replay token is outside the frozen split")
    return token


def _destination(cache, settings):
    token = _token(cache, settings)
    return token, filesystem_path(settings["output_dir"]) / "samples" / digest(token)[:24] / "official_replay"


def export_replay(arguments, result, capture, settings):
    """Serialize detached scoring-only data and render in the core environment."""
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array

    if capture["errors"]:
        raise RuntimeError("; ".join(capture["errors"]))
    from .score_record import current_attempt, score_payload
    payload = score_payload(arguments, result, settings, capture)
    profile = payload["profile"]
    if profile["is_v1"]:
        # v1 has no separate human-penalty-filter pass. These are the exact
        # returned scalar metrics, rather than a recalculated diagnostic score.
        capture["raw"] = finite_json(payload["metrics"])
    required = {"ego_states", "actors", "raw", "red_lights", "events"}
    if not required.issubset(capture):
        raise RuntimeError("Official score did not produce a complete replay capture")
    cache, sampling = arguments["metric_cache"], arguments["future_sampling"]
    token, destination = _destination(cache, settings)
    attempt_id = current_attempt(arguments, settings)
    actual = payload["states"]
    if (actual.shape != capture["ego_states"].shape
            or not np.array_equal(actual, capture["ego_states"])):
        raise ValueError("Captured model states differ from official returned states")
    count = int(sampling.num_poses) + 1
    if (actual.shape != (count, 11) or len(capture["actors"]) != count
            or len(capture["red_lights"]) != count or not np.isfinite(actual).all()):
        raise ValueError("Official replay arrays have inconsistent time grids")
    planned = get_trajectory_as_array(transform_trajectory(arguments["model_trajectory"], cache.ego_state),
                                      sampling, cache.ego_state.time_point)
    human = None
    if getattr(cache, "human_trajectory", None) is not None:
        human = get_trajectory_as_array(transform_trajectory(cache.human_trajectory, cache.ego_state),
                                        sampling, cache.ego_state.time_point)[:, :3]
    vehicle = cache.ego_state.car_footprint.vehicle_parameters
    drivable = cache.drivable_area_map
    # Preserve positional polygon/layer association. Map IDs can recur across
    # layers; a token lookup would silently choose the last matching geometry.
    if not (len(drivable.tokens) == len(drivable.map_types) == len(drivable._geometries)):
        raise ValueError("Official map polygon/layer arrays are inconsistent")
    map_polygons = [dict(token=str(token_), kind=kind.name, wkt=polygon.wkt)
                    for token_, kind, polygon in zip(drivable.tokens, drivable.map_types, drivable._geometries)]
    replay = finite_json(dict(version=VERSION, token=token, stage=payload["stage"],
        benchmark_profile=profile["id"], score_name=profile["score_name"],
        dt_s=float(sampling.interval_length),
        ego=dict(states=actual, length_m=float(vehicle.length), width_m=float(vehicle.width),
                 rear_axle_to_center_m=float(vehicle.rear_axle_to_center)),
        planned_poses_world=planned[:, :3], human_poses_world=human,
        map=dict(polygons=map_polygons, centerline=np.asarray(cache.centerline.linestring.coords)),
        actors=capture["actors"], red_lights=capture["red_lights"],
        metrics=dict(official_final=payload["metrics"], raw=capture["raw"]),
        events=capture["events"], provenance=dict(
            sample_attempt_id=attempt_id,
            official_commit=settings.get("official_commit"),
            benchmark_profile=profile["id"],
            official_source_sha256=settings.get("official_source_sha256"),
            input_condition=settings.get("input_condition"),
            score_unchanged=True,
            simulated_ego_source=("original_official_simulator_simulate_proposals_return_model_index_1"
                if profile["is_v1"] else "official_pdm_score_returned_states"),
            actor_source=("exact_official_scorer_observation_occupancy_interpolated_gt"
                if profile["is_v1"] else "first_official_policy_simulate_environment_return"),
            traffic_policy=("NAVSIM_v1_non_reactive_cached_GT" if profile["is_v1"]
                else type(arguments["traffic_agents_policy"]).__name__),
            raw_metrics_source=("official_v1_return_no_human_filter" if profile["is_v1"]
                else "first_official_scorer_pass_model_index_1_before_human_filter"),
            final_metrics_source=payload["source"],
            events_semantics="raw scorer diagnostics; red polygon intersections are not CARLA stop-line events",
            red_light_time_grid="exact scorer observation[index], including its native index mapping",
            coordinates="global_map_xy_m; ego/planned/human poses at rear axle; actor poses at box center",
            future_data_scope="scoring_only; never forwarded to planner or memory",
            log_name=payload["log_name"], time_s=payload["start_time_s"],
            human_reference_source=("official_cache_human_trajectory" if human is not None
                else "not_available_in_this_metric_cache; never replaced_with_PDM_reference"))))
    destination.mkdir(parents=True, exist_ok=True)
    replay_path = destination / "trace.json.gz"
    temp = destination / "trace.json.gz.tmp"
    with gzip.open(temp, "wt", encoding="utf-8") as stream:
        json.dump(replay, stream, allow_nan=False, separators=(",", ":"))
    temp.replace(replay_path)
    write_json(destination / "capture_status.json", dict(status="complete", token=token,
        sample_attempt_id=attempt_id,
        replay=str(replay_path), samples=count, dt_s=float(sampling.interval_length),
        physical_horizon_s=float(sampling.num_poses * sampling.interval_length), score_unchanged=True))
    started = time.perf_counter()
    command = [settings["core_python"], "-m", "ep_driver_navsim.replay_render",
               "--replay", str(replay_path), "--output-dir", str(destination)]
    with (destination / "render.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
            timeout=float(settings.get("bev_render_timeout_s", 180.0)),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if completed.returncode:
        raise RuntimeError("BEV renderer exited %d; see %s" % (completed.returncode, destination / "render.log"))
    # Read and validate the renderer's own decode verification rather than infer
    # video success from a zero process exit or a file merely being present.
    status = json.loads((destination / "render_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError("BEV renderer did not report successful video verification")
    status["render_wall_s"] = time.perf_counter() - started
    write_json(destination / "render_status.json", status)
    return replay_path


def _failure_status(cache, settings, exc):
    try:
        token, destination = _destination(cache, settings)
        name = "render_status.json" if (destination / "trace.json.gz").is_file() else "capture_status.json"
        write_json(destination / name, dict(status="failed", token=token, error=repr(exc),
            official_score_unchanged=True, failure_scope="visualization_only"))
    except Exception:
        LOG.exception("Unable to save official replay failure status")
    LOG.warning("Official BEV visualization failed; numerical score retained: %s", exc)


def wrap_score_function(original, settings):
    """Delegate the official score exactly once and return its exact result object."""
    if getattr(original, "_ep_official_replay_observer", False):
        return original
    video = settings.get("bev_video", True)
    signature = inspect.signature(original)
    profile = get_profile(settings)
    exported_count = 0

    @functools.wraps(original)
    def observed(*args, **kwargs):
        nonlocal exported_count
        bound = signature.bind(*args, **kwargs)
        arguments = dict(bound.arguments)
        capture = dict(errors=[])
        if profile["is_v1"]:
            if "traffic_agents_policy" in arguments:
                raise ValueError("NAVSIM v1 profile cannot use a v2 traffic-policy scorer")
            bound.arguments["simulator"] = _SimulatorObserver(arguments["simulator"], capture)
        elif "traffic_agents_policy" not in arguments:
            raise ValueError("NAVSIM v2 profile requires its official traffic-policy scorer")
        if video and not profile["is_v1"]:
            bound.arguments["traffic_agents_policy"] = _PolicyObserver(arguments["traffic_agents_policy"], capture)
        if video:
            bound.arguments["scorer"] = _ScorerObserver(arguments["scorer"], capture, v1=profile["is_v1"])
        # Exceptions from official scoring propagate exactly as before. Export
        # errors are separate and must never manufacture invalid score rows.
        result = original(*bound.args, **bound.kwargs)
        try:
            from .score_record import export_score_record
            export_score_record(arguments, result, settings, capture=capture)
        except Exception as exc:
            try:
                from .score_record import record_failure
                record_failure(arguments, settings, exc)
            except Exception:
                LOG.warning("Unable to save official score evidence failure status")
            LOG.warning("Official score evidence export failed; numerical result retained: %s", exc)
        if not video:
            return result
        try:
            export_replay(arguments, result, capture, settings)
        except Exception as exc:
            _failure_status(arguments["metric_cache"], settings, exc)
        else:
            exported_count += 1
            # Keep the first video immediately viewable, then amortize gallery
            # scans during large official splits. Launch also refreshes on exit.
            if exported_count == 1 or exported_count % 25 == 0:
                try:
                    from .replay_gallery import update_gallery
                    update_gallery(settings["output_dir"])
                except Exception as exc:
                    LOG.warning("BEV video saved but gallery refresh failed: %s", exc)
        return result

    observed._ep_official_replay_observer = True
    return observed


def install_score_replay(settings):
    """Install before the official entrypoint imports its scoring symbol."""
    import importlib
    module = importlib.import_module("navsim.evaluate.pdm_score")
    module.pdm_score = wrap_score_function(module.pdm_score, settings)
    return bool(settings.get("bev_video", True))
