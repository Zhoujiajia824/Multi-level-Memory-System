"""Owned launch/client settings for read-only use of shared model assets.

These profiles describe our private serving process, not the deployment state
of another user's public service. Reading a profile never starts a process,
loads weights, queries the server, or changes the shared deployment.
"""
from copy import deepcopy
import json
import math
from pathlib import Path, PurePosixPath


SERVICE_PROFILES = ("shared_qwen3_32b",)
PROFILE_ROOT = Path(__file__).resolve().parents[2] / "config" / "navsim"


def load_service_profile(name="shared_qwen3_32b"):
    """Load and validate the same owned configuration used by both entry points."""
    if name not in SERVICE_PROFILES:
        raise ValueError("Unknown owned model service profile: " + str(name))
    value = json.loads((PROFILE_ROOT / (name + ".json")).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema") != "ep-navsim-owned-model-service-1"
            or value.get("id") != name):
        raise ValueError("Owned model service profile has an invalid schema or identity")
    if value.get("host") != "127.0.0.1":
        raise ValueError("Owned model service must bind to 127.0.0.1")
    for key in ("port", "max_model_len", "max_new_tokens", "max_num_seqs", "image_limit",
                "max_pixels_per_image", "min_total_memory_gib"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError("Owned model service needs a positive integer: " + key)
    if value["port"] > 65535 or value["port"] in (8000, 8002):
        raise ValueError("Owned model service must use a separate valid port")
    if value["max_new_tokens"] >= value["max_model_len"]:
        raise ValueError("Owned model service must leave context for the complete nine-image input")
    if value["image_limit"] != 9:
        raise ValueError("NAVSIM owned service requires all eight RGB images and BEV")
    counts = value.get("gpu_counts")
    if (not isinstance(counts, list) or not counts
            or any(type(count) is not int or count <= 0 for count in counts)
            or len(set(counts)) != len(counts)):
        raise ValueError("Owned model service needs distinct positive GPU counts")
    root = value.get("public_root")
    if (not isinstance(root, str) or not PurePosixPath(root).is_absolute()
            or ".." in PurePosixPath(root).parts or "\\" in root):
        raise ValueError("Shared asset root must be an absolute POSIX path")
    path = value.get("model_path")
    if (not isinstance(path, str) or not path or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts or "\\" in path):
        raise ValueError("Shared model path must stay under its root")
    python = value.get("python_executable")
    if (not isinstance(python, str) or not PurePosixPath(python).is_absolute()
            or ".." in PurePosixPath(python).parts or "\\" in python
            or PurePosixPath(python).is_relative_to(PurePosixPath(root))):
        raise ValueError("Owned Python executable must be explicit, absolute and outside the shared asset root")
    for key in ("served_model_name", "vllm_version"):
        if not isinstance(value.get(key), str) or not value[key].strip() or "\n" in value[key]:
            raise ValueError("Owned model service needs an explicit " + key)
    structured = value.get("structured_outputs")
    if (not isinstance(structured, dict) or structured.get("backend") != "xgrammar"
            or structured.get("disable_any_whitespace") is not True):
        raise ValueError("Owned NAVSIM service requires compact xgrammar JSON output")
    if value.get("enable_thinking") is not False or value.get("dtype") != "bfloat16":
        raise ValueError("Owned 32B profile must use direct Instruct output and BF16")
    utilization = value.get("gpu_memory_utilization")
    if (isinstance(utilization, bool) or not isinstance(utilization, (int, float))
            or not math.isfinite(utilization) or not 0 < utilization < 1):
        raise ValueError("Owned GPU memory utilization must be between zero and one")
    return deepcopy(value)
