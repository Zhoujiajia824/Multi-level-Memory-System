"""Dataset checks and smoke scoring, executed by the official Python 3.9 SDK."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import lzma
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import yaml

from .protocol import digest, write_json
from .sdk_bootstrap import platform_setup


def sdk_versions():
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "scipy", "torch", "pytorch-lightning", "hydra-core", "nuplan-devkit", "shapely")}
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("Official SDK runtime must use Python 3.9")
    expected = {"numpy": "1.23.4", "torch": "2.0.1", "hydra-core": "1.2.0", "pytorch-lightning": "2.2.1", "nuplan-devkit": "1.2.0"}
    for package, version in expected.items():
        if versions[package].split("+")[0] != version:
            raise RuntimeError("SDK dependency version differs from v2.2: " + package)
    return dict(python=sys.version, packages=versions)


def make_loader(settings):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader

    sdk = Path(settings["sdk_root"])
    split_path = sdk / "navsim/planning/script/config/common/train_test_split/scene_filter" / (settings["split"] + ".yaml")
    scene_filter = instantiate(OmegaConf.load(split_path))
    data = Path(settings["data_root"])
    split_data = data / settings["split"]
    loader = SceneLoader(data_path=data / "navsim_logs/test", original_sensor_path=data / "sensor_blobs/test",
        synthetic_sensor_path=split_data / "sensor_blobs", synthetic_scenes_path=split_data / "synthetic_scene_pickles",
        scene_filter=scene_filter, sensor_config=SensorConfig.build_no_sensors())
    expected = set(settings["evaluation_tokens"])
    actual = set(loader.tokens_stage_one) | set(loader.reactive_tokens_stage_two)
    if expected != actual:
        raise RuntimeError("Official split/data coverage mismatch: missing=%d extra=%d" % (len(expected - actual), len(actual - expected)))
    return loader


def inspect_data(settings, loader):
    from .input_adapter import CAMERAS

    data = Path(settings["data_root"])
    missing = []
    refs = 0
    for token, rows in loader.scene_frames_dicts.items():
        # Check the current images actually requested by our Agent.
        row = rows[3]
        for camera in CAMERAS:
            path = data / "sensor_blobs/test" / row["cams"][camera.upper()]["data_path"]
            refs += 1
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    for token in loader.reactive_tokens_stage_two:
        path, _ = loader.synthetic_scenes[token]
        with Path(path).open("rb") as stream:
            value = pickle.load(stream)
        if value["scene_metadata"]["num_history_frames"] != 4:
            raise ValueError("Synthetic history count differs from official protocol")
        row = value["frames"][3]
        for camera in CAMERAS:
            reference = row["camera_dict"][camera]["data_path"]
            path = data / settings["split"] / "sensor_blobs" / reference
            refs += 1
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing %d current camera assets; first: %s" % (len(missing), missing[0]))
    return dict(original_count=len(loader.tokens_stage_one), reactive_synthetic_count=len(loader.reactive_tokens_stage_two),
                camera_references_checked=refs, asset_check="existence_and_nonzero_size; smoke_decodes_selected_inputs")


def check_cache(settings):
    from navsim.common.dataloader import MetricCacheLoader

    loader = MetricCacheLoader(Path(settings["metric_cache_path"]))
    expected, actual = set(settings["evaluation_tokens"]), set(loader.tokens)
    if expected - actual:
        raise RuntimeError("Missing metric caches for %d official tokens; refusing partial benchmark" % len(expected - actual))
    return dict(required_cache_count=len(expected), available_cache_count=len(actual))


def inspect_navigation(settings, loader):
    from .nav_preflight import inspect_navigation as checkpointed_inspection
    return checkpointed_inspection(settings, loader)


def smoke(settings, loader):
    """Use official cache/simulator/scorer on one original and one synthetic input.

    This is deliberately not an aggregate benchmark score: mapping pairs and all
    second-stage endpoints are required for an official two-stage aggregate.
    """
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from .score_replay import wrap_score_function
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from .agent import EPDriverPrivilegedAgent

    config_dir = Path(settings["sdk_root"]) / "navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=["train_test_split=" + settings["split"], "worker=sequential"])
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator, scorer = PDMSimulator(sampling), instantiate(cfg.scorer)
    policy = instantiate(cfg.traffic_agents_policy.reactive, sampling)
    cache_dir = Path(settings["output_dir"]) / "smoke_metric_cache"
    processor = MetricCacheProcessor(cache_path=str(cache_dir), force_feature_computation=False, proposal_sampling=sampling)
    agent = EPDriverPrivilegedAgent(settings["settings_path"], settings["core_python"], timeout_s=settings["timeout_s"])
    loader._sensor_config = agent.get_sensor_config()
    tokens = [sorted(loader.tokens_stage_one)[0], sorted(loader.reactive_tokens_stage_two)[0]]
    # A visualization smoke check should not rebuild already verified official
    # caches into a deeper output path (which can exceed Windows MAX_PATH).
    ready_cache = None
    if (Path(settings["metric_cache_path"]) / "metadata").is_dir():
        from .launch import verify_cache_contract
        verify_cache_contract(settings)
        candidate = MetricCacheLoader(Path(settings["metric_cache_path"]))
        if set(tokens).issubset(candidate.tokens):
            ready_cache = candidate
    pdm_score = wrap_score_function(pdm_score, settings)
    rows = []
    try:
        agent.initialize()
        for token in tokens:
            print("Official scorer smoke sample:", token, flush=True)
            scene = loader.get_scene_from_token(token)
            if ready_cache is not None:
                cache = ready_cache.get_from_token(token)
            else:
                scenario = NavSimScenario(scene, map_root=settings["maps_runtime_root"], map_version="nuplan-maps-v1.0")
                cache_entry = processor.compute_and_save_metric_cache(scenario)
                if cache_entry is None:
                    raise RuntimeError("Official cache generation failed for smoke token " + token)
                with lzma.open(cache_entry.file_name, "rb") as stream:
                    cache = pickle.load(stream)
            agent_input = loader.get_agent_input_from_token(token)
            trajectory = agent.compute_trajectory(agent_input, scene)
            score, _ = pdm_score(metric_cache=cache, model_trajectory=trajectory,
                future_sampling=sampling, simulator=simulator, scorer=scorer, traffic_agents_policy=policy)
            result = score.to_dict(orient="records")[0]
            # The official dataframe also contains NumPy arrays used by its
            # aggregator. Keep numeric arrays serializable without redefining scores.
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, np.generic):
                    result[key] = value.item()
            result.update(token=token, stage="original" if token in loader.tokens_stage_one else "synthetic")
            rows.append(result)
            print("Official scorer smoke completed:", token, flush=True)
    finally:
        agent.close()
        if settings.get("bev_video", True):
            from .replay_gallery import update_gallery
            print("Official scoring replay gallery:", update_gallery(settings["output_dir"]), flush=True)
    result = dict(status="passed", benchmark_result=False, aggregate_epdms=None, token_scores=rows,
                  reason="Only two integration samples; no full two-stage aggregation",
                  input_condition=settings["input_condition"])
    # Official metrics may use NaN for a quantity resolved only by two-stage
    # aggregation. Preserve its absence as null, never manufacture a score.
    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    write_json(Path(settings["output_dir"]) / "smoke_result.json", finite(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "check-cache", "smoke"])
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    compatibility = platform_setup()
    versions = sdk_versions()
    if args.action == "check-cache":
        report = check_cache(settings)
    else:
        loader = make_loader(settings)
        report = inspect_data(settings, loader)
        if args.action == "inspect":
            report["navigation"] = inspect_navigation(settings, loader)
        if args.action == "smoke":
            smoke(settings, loader)
    report.update(status="passed", versions=versions, platform_compatibility=compatibility,
                  split=settings["split"], input_condition=settings["input_condition"])
    write_json(Path(settings["output_dir"]) / (args.action + "_check.json"), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
