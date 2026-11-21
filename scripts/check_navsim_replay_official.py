"""Real-data parity check for the scoring observer, using two already saved plans.

Run with the pinned SDK Python 3.9.  This does not invoke the planner, a model,
CARLA, metric caching or two-stage benchmark aggregation.  The read-only caches
are loaded afresh for each score because the official scorer mutates observation.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ep_driver_navsim.protocol import digest, write_json
from ep_driver_navsim.sdk_bootstrap import platform_setup


TOKENS = ("0dc54a8c8203567b", "00caf7c273c7d5065")


def _snapshot_tracks(tracks):
    """Independent copy of the policy return, not cache.future_tracked_objects."""
    return [[dict(id=str(obj.track_token), type=obj.tracked_object_type.name.lower(),
                  center=[float(obj.center.x), float(obj.center.y), float(obj.center.heading)],
                  length_m=float(obj.box.length), width_m=float(obj.box.width))
             for obj in frame.tracked_objects] for frame in tracks]


class RecordingPolicy:
    def __init__(self, original):
        self.original = original
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def simulate_environment(self, *args, **kwargs):
        value = self.original.simulate_environment(*args, **kwargs)
        self.calls.append(_snapshot_tracks(value))
        return value


def assert_frame_exact(left, right):
    import numpy as np

    if list(left.columns) != list(right.columns) or not left.index.equals(right.index):
        raise AssertionError("Official score dataframe schema changed")
    for column in left.columns:
        for a, b in zip(left[column], right[column]):
            # assert_array_equal treats paired NaNs as equal, including the
            # extended-comfort placeholder resolved later by official aggregation.
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b),
                                          err_msg="Official score changed: " + column)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, help="Existing frozen run_settings.json")
    parser.add_argument("--output-dir", required=True, help="New verification directory")
    args = parser.parse_args()
    settings_path = Path(args.settings).resolve()
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    source_output = Path(settings["output_dir"])
    output = Path(args.output_dir).resolve()
    if output == source_output.resolve() or output.exists():
        raise ValueError("Choose a NEW output directory; existing experiment files are read-only")
    for token in TOKENS:
        if token not in settings["evaluation_tokens"]:
            raise ValueError("The saved two-sample check requires its original warmup split")
        if not (source_output / "samples" / digest(token)[:24] / "trajectory.json").is_file():
            raise FileNotFoundError("Missing already saved trajectory for " + token)

    sys.path.insert(0, settings["sdk_root"])
    os.environ.update(NUPLAN_MAPS_ROOT=settings["maps_runtime_root"],
                      NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
                      OPENSCENE_DATA_ROOT=settings["data_root"],
                      NAVSIM_DEVKIT_ROOT=settings["sdk_root"],
                      NAVSIM_EXP_ROOT=str(output), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    os.environ["PYTHONPATH"] = (str(ROOT / "src") + os.pathsep + settings["sdk_root"]
                                + os.pathsep + os.environ.get("PYTHONPATH", ""))
    compatibility = platform_setup()

    import cv2
    import numpy as np
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from ep_driver_navsim.launch import verify_sdk
    from ep_driver_navsim.score_replay import wrap_score_function
    from ep_driver_navsim.sdk_runtime import sdk_versions

    versions = sdk_versions()
    sdk_before = verify_sdk(settings["sdk_root"])
    config_dir = Path(settings["sdk_root"]) / "navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=" + settings["split"], "worker=sequential"])
    cache_loader = MetricCacheLoader(Path(settings["metric_cache_path"]))
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    export_settings = dict(settings, output_dir=str(output), bev_video=True)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_settings.json", export_settings)

    def score(token, trajectory, observed):
        simulator = instantiate(cfg.simulator)
        scorer = instantiate(cfg.scorer)
        policy = RecordingPolicy(instantiate(cfg.traffic_agents_policy.reactive, sampling))
        # Each call must unpickle a separate cache: do not reuse an observation
        # already overwritten by the prior human penalty filtering pass.
        cache = cache_loader.get_from_token(token)
        function = wrap_score_function(pdm_score, export_settings) if observed else pdm_score
        result = function(metric_cache=cache, model_trajectory=trajectory,
                          future_sampling=sampling, simulator=simulator,
                          scorer=scorer, traffic_agents_policy=policy)
        return result, policy.calls, cache.human_trajectory is not None

    rows = []
    for token in TOKENS:
        path = source_output / "samples" / digest(token)[:24] / "trajectory.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["frame_id"] != token or saved["sampling"] != dict(horizon_s=4.0, interval_s=0.5):
            raise ValueError("Saved trajectory source/sampling mismatch")
        poses = np.asarray(saved["poses"], dtype=np.float32)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError("Invalid saved plan")
        trajectory = Trajectory(poses, TrajectorySampling(time_horizon=4.0, interval_length=0.5))
        baseline, baseline_calls, has_human = score(token, trajectory, False)
        observed, observed_calls, _ = score(token, trajectory, True)
        assert_frame_exact(baseline[0], observed[0])
        np.testing.assert_array_equal(baseline[1], observed[1], err_msg="Official ego execution changed")
        if len(baseline_calls) != len(observed_calls):
            raise AssertionError("Official policy invocation count changed")
        if baseline_calls != observed_calls:
            raise AssertionError("Official reactive traffic outputs changed")
        destination = output / "samples" / digest(token)[:24] / "official_replay"
        with gzip.open(destination / "trace.json.gz", "rt", encoding="utf-8") as stream:
            replay = json.load(stream)
        if replay["version"] != "navsim-official-replay-1.0" or replay["token"] != token:
            raise AssertionError("Replay identity mismatch")
        np.testing.assert_array_equal(replay["ego"]["states"], observed[1])
        if replay["actors"] != observed_calls[0]:
            raise AssertionError("Replay traffic does not match model's first policy pass")
        if len(replay["actors"]) != 41 or len(replay["red_lights"]) != 41 or replay["dt_s"] != 0.1:
            raise AssertionError("Replay time grids are not aligned")
        status = json.loads((destination / "render_status.json").read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise AssertionError("Official scoring succeeded but replay export failed: " + repr(status))
        movie = destination / "official_bev.mp4"
        if not movie.is_file() or movie.stat().st_size <= 0:
            raise AssertionError("Verified replay movie is missing")
        video = cv2.VideoCapture(str(movie))
        decoded = 0
        try:
            if not video.isOpened() or not np.isclose(video.get(cv2.CAP_PROP_FPS), 10.0):
                raise AssertionError("Replay movie cannot be independently decoded at 10Hz")
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                if frame.shape[:2] != tuple(reversed(status["size"])):
                    raise AssertionError("Decoded replay geometry differs from render status")
                decoded += 1
        finally:
            video.release()
        if decoded != 41 or status.get("decoded_frames") != decoded or not status.get("playback_verified"):
            raise AssertionError("Replay MP4 does not contain exactly 41 verified frames")
        row = dict(token=token, stage=replay["stage"], exact_score_components=True,
                   exact_official_ego_execution=True, exact_reactive_traffic=True,
                   first_policy_pass_captured=True, policy_calls=len(observed_calls),
                   human_reference_available=has_human, replay_samples=41, independently_decoded_frames=decoded,
                   movie=str(movie), render_status=status)
        rows.append(row)
        print("Official scoring/replay exact parity passed:", token, flush=True)
    sdk_after = verify_sdk(settings["sdk_root"])
    if sdk_before != sdk_after:
        raise AssertionError("Official SDK changed during replay validation")
    result = dict(status="passed", benchmark_result=False, aggregate_epdms=None,
                  test="two saved real-data trajectories, unobserved vs observed official score",
                  sdk_source_unchanged=True, official_commit=settings["official_commit"],
                  input_condition=settings["input_condition"], versions=versions,
                  platform_compatibility=compatibility, samples=rows)
    write_json(output / "real_data_replay_check.json", result)
    from ep_driver_navsim.replay_gallery import update_gallery
    print("Replay gallery:", update_gallery(output), flush=True)
    print(json.dumps(dict(status="passed", samples=len(rows), benchmark_result=False), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
