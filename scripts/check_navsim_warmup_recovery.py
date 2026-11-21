"""Reproduce six failed warmup tokens and exercise full official aggregation.

Run with the pinned SDK Python 3.9. Existing samples are read-only. Only the six
previously failed plans are recomputed; the other 214 saved plans are rescored.
This is an integration regression, NOT a fresh benchmark or a paper result.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ep_driver_navsim.protocol import digest, write_json
from ep_driver_navsim.sdk_bootstrap import platform_setup

FAILED = ("6d8d91ad290f124b2", "e92394471671ed533", "c47263e93d93b6838",
          "c64e0d6194293f51e", "583a61a4880467bf7", "dd091b9e715ca6370")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, help="Fresh frozen settings matching current source")
    parser.add_argument("--previous-output", required=True, help="Read-only failed A_base_v1 run")
    parser.add_argument("--output-dir", required=True, help="NEW diagnostic directory")
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    old, output = Path(args.previous_output).resolve(), Path(args.output_dir).resolve()
    if output.exists() or output == old:
        raise ValueError("Choose a new diagnostic output directory")
    if settings["split"] != "warmup_two_stage" or len(settings["evaluation_tokens"]) != 220:
        raise ValueError("This regression requires the complete pinned warmup split")
    tokens = sorted(settings["evaluation_tokens"])
    for token in tokens:
        path = old / "samples" / digest(token)[:24] / "trajectory.json"
        if token not in FAILED and not path.is_file():
            raise FileNotFoundError("Missing read-only saved plan: " + token)
    sys.path.insert(0, settings["sdk_root"])
    os.environ.update(NUPLAN_MAPS_ROOT=settings["maps_runtime_root"], NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
        OPENSCENE_DATA_ROOT=settings["data_root"], NAVSIM_DEVKIT_ROOT=settings["sdk_root"],
        PYTHONUTF8="1", PYTHONIOENCODING="utf-8", NAVSIM_EXP_ROOT=str(output))
    os.environ["PYTHONPATH"] = (str(ROOT / "src") + os.pathsep + settings["sdk_root"]
                                + os.pathsep + os.environ.get("PYTHONPATH", ""))
    platform_setup()

    import cv2
    import numpy as np
    import pandas as pd
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataclasses import PDMResults, Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.script.run_pdm_score import (create_scene_aggregators, compute_final_scores,
                                                     calculate_individual_mapping_scores)
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.geometry.convert import relative_to_absolute_poses
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from ep_driver_navsim.agent import EPDriverPrivilegedAgent
    from ep_driver_navsim.launch import verify_sdk, verify_cache_contract, source_fingerprint
    from ep_driver_navsim.sdk_runtime import make_loader
    from ep_driver_navsim.score_replay import wrap_score_function
    from ep_driver_navsim.replay_gallery import update_gallery

    sdk_before = verify_sdk(settings["sdk_root"])
    if source_fingerprint(ROOT, "src")[1] != settings["numerical_source_sha256"]:
        raise ValueError("Freeze new settings after source edits before running this regression")
    verify_cache_contract(settings)
    settings.update(output_dir=str(output), bev_video=True, benchmark_result=False,
        validation_only="six new plans plus 214 old plans; not a fresh benchmark")
    settings["settings_path"] = str(output / "run_settings.json")
    output.mkdir(parents=True, exist_ok=False)
    write_json(settings["settings_path"], settings)
    loader = make_loader(settings)
    agent = EPDriverPrivilegedAgent(settings["settings_path"], settings["core_python"], timeout_s=settings["timeout_s"])
    loader._sensor_config = agent.get_sensor_config()
    fresh = {}
    try:
        for token in FAILED:
            print("Recomputing previously failed token:", token, flush=True)
            fresh[token] = agent.compute_trajectory(loader.get_agent_input_from_token(token),
                                                   loader.get_scene_from_token(token))
            print("Recovered executable plan:", token, flush=True)
    finally:
        agent.close()

    config_dir = Path(settings["sdk_root"]) / "navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=warmup_two_stage", "worker=sequential"])
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = instantiate(cfg.simulator), instantiate(cfg.scorer)
    policy = instantiate(cfg.traffic_agents_policy.reactive, sampling)
    cache_loader = MetricCacheLoader(Path(settings["metric_cache_path"]))
    observed_score = wrap_score_function(pdm_score, settings)
    rows = []
    for index, token in enumerate(tokens, 1):
        if token in fresh:
            trajectory = fresh[token]
        else:
            saved = json.loads((old / "samples" / digest(token)[:24] / "trajectory.json").read_text(encoding="utf-8"))
            if saved["frame_id"] != token or saved["sampling"] != dict(horizon_s=4.0, interval_s=0.5):
                raise ValueError("Saved plan identity/sampling mismatch: " + token)
            poses = np.asarray(saved["poses"], np.float32)
            if poses.shape != (8, 3) or not np.isfinite(poses).all():
                raise ValueError("Invalid saved trajectory: " + token)
            trajectory = Trajectory(poses, TrajectorySampling(time_horizon=4.0, interval_length=0.5))
        cache = cache_loader.get_from_token(token)  # Fresh cache: official scoring mutates its observation.
        function = observed_score if token in fresh else pdm_score
        row, states = function(metric_cache=cache, model_trajectory=trajectory, future_sampling=sampling,
            simulator=simulator, scorer=scorer, traffic_agents_policy=policy)
        endpoint = relative_to_absolute_poses(cache.ego_state.rear_axle, [StateSE2(*trajectory.poses[-1])])[0]
        row = row.assign(token=token, valid=True, log_name=cache.log_name, frame_type=cache.scene_type,
            start_time=cache.timepoint.time_s, endpoint_x=endpoint.x, endpoint_y=endpoint.y,
            start_point_x=cache.ego_state.rear_axle.x, start_point_y=cache.ego_state.rear_axle.y)
        row["ego_simulated_states"] = [states]
        rows.append(row)
        if index % 25 == 0 or index == len(tokens):
            print("Official diagnostic rescoring: %d/%d" % (index, len(tokens)), flush=True)
    raw = pd.concat(rows, ignore_index=True)
    if len(raw) != 220 or set(raw["token"]) != set(tokens) or not raw["valid"].all():
        raise AssertionError("Incomplete official diagnostic rows; no samples may be skipped")
    raw.to_pickle(output / "diagnostic_raw_rows.pkl")
    finish_check(raw, output, settings, cfg, sampling, sdk_before)


def finish_check(raw, output, settings, cfg, sampling, sdk_before):
    """Recheck saved diagnostic rows/videos without repeating inference/scoring."""
    import cv2
    import numpy as np
    from navsim.common.dataclasses import PDMResults
    from navsim.planning.script.run_pdm_score import (create_scene_aggregators, compute_final_scores,
                                                     calculate_individual_mapping_scores)
    from ep_driver_navsim.launch import verify_sdk, source_fingerprint
    from ep_driver_navsim.replay_gallery import update_gallery

    tokens = settings["evaluation_tokens"]
    if len(raw) != 220 or set(raw["token"]) != set(tokens) or not raw["valid"].all():
        raise AssertionError("Incomplete official diagnostic rows; no samples may be skipped")
    if source_fingerprint(ROOT, "src")[1] != settings["numerical_source_sha256"]:
        raise AssertionError("Numerical source differs from the saved diagnostic")
    mappings = {(now, prev): [tuple(pair) for pair in pairs]
                for now, prev, pairs in cfg.train_test_split.reactive_all_mapping}
    scored = compute_final_scores(create_scene_aggregators(mappings, raw.copy(), sampling))
    score_columns = [column for column in scored.columns if column != "pdm_score" and
        (any(field.name in column for field in fields(PDMResults)) or column in ("score", "two_frame_extended_comfort"))]
    combined, stage_one, stage_two = calculate_individual_mapping_scores(
        scored[score_columns + ["token", "weight"]], mappings)
    for group in (combined, stage_one, stage_two):
        if not np.isfinite(group.to_numpy(dtype=float)).all():
            raise AssertionError("Non-finite official aggregate after recovery")
    scored.to_csv(output / "diagnostic_aggregated_rows.csv", index=False)
    for token in FAILED:
        path = output / "samples" / digest(token)[:24] / "official_replay"
        status = json.loads((path / "render_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete" or status.get("decoded_frames") != 41:
            raise AssertionError("Recovered sample BEV export failed: " + token)
        video, decoded = cv2.VideoCapture(str(path / "official_bev.mp4")), 0
        try:
            while video.read()[0]:
                decoded += 1
        finally:
            video.release()
        if decoded != 41:
            raise AssertionError("Recovered sample BEV independent decode failed: " + token)
    if verify_sdk(settings["sdk_root"]) != sdk_before:
        raise AssertionError("Official SDK source was modified")
    result = dict(status="passed", benchmark_result=False, aggregate_epdms=None,
        recovered_tokens=list(FAILED), new_plans=6, reused_readonly_plans=214, official_scored_tokens=220,
        complete_two_stage_aggregation=True, official_aggregate_finite=True,
        recovered_bev_movies_independently_decoded=6, sdk_source_unchanged=True,
        reason="Mixed old/new trajectories verify the integration, not the current planner's benchmark performance")
    write_json(output / "recovery_check.json", result)
    print("Recovered sample replay gallery:", update_gallery(output), flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
