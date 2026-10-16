"""Lazy, version-pinned loading: never silently replace PDM with a heuristic."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import re
import subprocess
from importlib import import_module, metadata
from pathlib import Path
from types import SimpleNamespace

PDM_COMMIT = "b51d5d04fac1bd4389653b9ab2ff73ea88f435a3"
NUPLAN_COMMIT = "e9241677997dd86bfc0bcd44817ab04fe631405b"
PDM_ROOT = "tuplan_garage.planning.simulation.planner.pdm_planner"


def _verify_sources():
    hashes = json.loads(Path(__file__).with_name("upstream_hashes.json").read_text(encoding="utf-8"))
    roots = {name: Path(import_module(name).__file__).resolve().parent for name in ("tuplan_garage", "nuplan")}
    for relative, expected in hashes.items():
        package, path = relative.split("/", 1)
        source = roots[package] / path
        actual = hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError("Pinned upstream source mismatch: " + relative)
    result = {"upstream_source_files_verified": len(hashes),
              "upstream_source_manifest_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    for distribution, expected, package in (("tuplan-garage", PDM_COMMIT, "tuplan_garage"),
                                             ("nuplan-devkit", NUPLAN_COMMIT, "nuplan")):
        actual = None
        try:
            dist = metadata.distribution(distribution)
            direct = json.loads(dist.read_text("direct_url.json") or "{}")
            actual = direct.get("vcs_info", {}).get("commit_id")
            if actual is None:
                match = re.search(r"\b([0-9a-f]{40})\b", direct.get("url", ""))
                actual = match.group(1) if match else None
        except metadata.PackageNotFoundError:
            pass
        marker = roots[package].parent / "PINNED_COMMIT"
        if actual is None and marker.is_file():
            actual = marker.read_text(encoding="utf-8").strip()
        checkout = roots[package].parent
        if actual is None and (checkout / ".git").exists():
            actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                               text=True, timeout=10).strip()
        if actual is not None and actual != expected:
            raise RuntimeError(distribution + " source commit does not match the pinned version")
        result[distribution + "_actual_commit"] = actual
    return result


@lru_cache(maxsize=1)
def load_dependencies():
    try:
        import shapely
        if int(shapely.__version__.split(".")[0]) != 2:
            raise RuntimeError("PDM requires Shapely 2.x")
        modules = {
            "EgoState": "nuplan.common.actor_state.ego_state",
            "StateSE2": "nuplan.common.actor_state.state_representation",
            "StateVector2D": "nuplan.common.actor_state.state_representation",
            "TimePoint": "nuplan.common.actor_state.state_representation",
            "VehicleParameters": "nuplan.common.actor_state.vehicle_parameters",
            "Agent": "nuplan.common.actor_state.agent",
            "OrientedBox": "nuplan.common.actor_state.oriented_box",
            "SceneObjectMetadata": "nuplan.common.actor_state.scene_object",
            "TrackedObjects": "nuplan.common.actor_state.tracked_objects",
            "TrackedObjectType": "nuplan.common.actor_state.tracked_objects_types",
            "DetectionsTracks": "nuplan.planning.simulation.observation.observation_type",
            "TrajectorySampling": "nuplan.planning.simulation.trajectory.trajectory_sampling",
            "TrafficLightStatusData": "nuplan.common.maps.maps_datatypes",
            "TrafficLightStatusType": "nuplan.common.maps.maps_datatypes",
            "SemanticMapLayer": "nuplan.common.maps.maps_datatypes",
        }
        for name, suffix in {
            "BatchIDMPolicy": "proposal.batch_idm_policy",
            "PDMGenerator": "proposal.pdm_generator",
            "PDMProposalManager": "proposal.pdm_proposal",
            "PDMSimulator": "simulation.pdm_simulator",
            "PDMScorer": "scoring.pdm_scorer",
            "PDMObservation": "observation.pdm_observation",
            "PDMOccupancyMap": "observation.pdm_occupancy_map",
            "PDMEmergencyBrake": "utils.pdm_emergency_brake",
            "PDMPath": "utils.pdm_path",
            "MultiMetricIndex": "utils.pdm_enums",
            "WeightedMetricIndex": "utils.pdm_enums",
            "StateIndex": "utils.pdm_enums",
        }.items():
            modules[name] = PDM_ROOT + "." + suffix
        values = {name: getattr(import_module(module), name) for name, module in modules.items()}
        values["provenance"] = {
            "implementation": "upstream PDM core with CARLA map/actor adapter",
            "pdm_repository": "https://github.com/autonomousvision/tuplan_garage",
            "pdm_expected_commit": PDM_COMMIT,
            "nuplan_repository": "https://github.com/motional/nuplan-devkit",
            "nuplan_expected_commit": NUPLAN_COMMIT,
            "shapely_version": shapely.__version__,
            "adapter_version": 1,
        }
        values["provenance"].update(_verify_sources())
        for distribution, package in (("tuplan-garage", "tuplan_garage"), ("nuplan-devkit", "nuplan")):
            try:
                values["provenance"][distribution + "_version"] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                values["provenance"][distribution + "_version"] = "source_checkout"
            package_file = Path(import_module(package).__file__).resolve()
            values["provenance"][distribution + "_path"] = str(package_file.parent)
        return SimpleNamespace(**values)
    except Exception as exc:
        raise RuntimeError(
            "PDM-Closed dependencies could not be loaded. Follow carla_bridge/planning/README.md "
            "to install the pinned tuplan_garage and nuPlan sources; no substitute planner is used. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def validate_dependencies():
    return dict(load_dependencies().provenance)
