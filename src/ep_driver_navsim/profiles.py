"""Versioned official evaluation contracts; no planner or score definitions."""
from copy import deepcopy


DEFAULT_PROFILE = "v2_two_stage"
V1_COMMIT = "3e8291bfa89ff247231e0227778840cd0a036896"
V2_COMMIT = "359c7f72304bfa8273e754224a213d3751bd2340"
PROFILES = {
    "v1_navtest": dict(id="v1_navtest", navsim_version="v1.1", sdk_commit=V1_COMMIT,
        lock_file="official_v1_1.json", metric="NAVSIM_v1.1_official_navtest_PDMS",
        splits=("navtest",), is_v1=True, two_stage=False, traffic_agents="non_reactive",
        scoring_script="run_pdm_score.py", score_name="PDMS"),
    "v2_navtest": dict(id="v2_navtest", navsim_version="v2.2", sdk_commit=V2_COMMIT,
        lock_file="official_v2_2.json", metric="NAVSIM_v2.2_official_navtest_single_stage_EPDMS",
        splits=("navtest",), is_v1=False, two_stage=False, traffic_agents="non_reactive",
        scoring_script="run_pdm_score_one_stage.py", score_name="EPDMS"),
    "v2_two_stage": dict(id="v2_two_stage", navsim_version="v2.2", sdk_commit=V2_COMMIT,
        lock_file="official_v2_2.json", metric="NAVSIM_v2.2_official_two_stage_reactive_EPDMS",
        splits=("warmup_two_stage", "navhard_two_stage"), is_v1=False, two_stage=True,
        traffic_agents="reactive", scoring_script="run_pdm_score.py", score_name="EPDMS"),
}


def get_profile(settings=None):
    settings = settings or {}
    name = settings if isinstance(settings, str) else settings.get("benchmark_profile", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError("Unknown NAVSIM benchmark_profile: " + str(name))
    return deepcopy(PROFILES[name])


def is_v1(settings):
    return get_profile(settings)["is_v1"]


def is_two_stage(settings):
    return get_profile(settings)["two_stage"]


def normalize_profile(settings):
    """Freeze an explicit profile and reject incompatible split/policy labels."""
    value = dict(settings)
    if value.get("split") == "navtest" and "benchmark_profile" not in value:
        raise ValueError("navtest requires benchmark_profile=v1_navtest or v2_navtest")
    profile = get_profile(value)
    split = value.get("split") or profile["splits"][0]
    if split not in profile["splits"]:
        raise ValueError("Split " + str(split) + " is incompatible with " + profile["id"])
    for key in ("traffic_agents", "traffic_policy"):
        if key in value and value[key] != profile["traffic_agents"]:
            raise ValueError("Traffic policy differs from the frozen " + profile["id"] + " protocol")
    value.update(benchmark_profile=profile["id"], split=split,
        navsim_version=profile["navsim_version"], traffic_policy=profile["traffic_agents"],
        traffic_agents=profile["traffic_agents"], official_metric=profile["metric"],
        official_commit=profile["sdk_commit"], score_name=profile["score_name"])
    return value
