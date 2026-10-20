"""Outcome-grounded numeric planning experiences, separate from text memory.

An uneventful execution is evidence that parameters were executed in a context;
it is NOT evidence that semantic intervention was better than baseline PDM.
Only explicitly imported, independently evaluated matched execution pairs can
populate ``causal_advantage``. No model, embedding service or third-party index
is used here. Similarity and support counts are not confidence probabilities.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import uuid


SCHEMA_VERSION = 1
EXPERIENCE_FILENAME = "planning_experiences.jsonl"
_PARAMETERS = ("target_velocity_mps", "headway_time", "min_gap_to_lead_agent",
               "accel_max", "decel_max")
_TERMINAL_FAILURES = {"collision", "traffic_violation"}
_NORMAL_ENDINGS = {"route_complete", "success", "completed"}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _number(data, *names):
    for name in names:
        value = data.get(name)
        if _finite(value):
            return float(value)
    return None


def _clean(value):
    """Keep audit JSON strict; invalid measurements remain explicitly unknown."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    return str(value)


def _digest(value):
    return hashlib.sha256(json.dumps(_clean(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execution_planner_version(planning, controller=None, *, vehicle_model=None):
    """Identify numerical execution settings without splitting semantic arms.

    Older outcomes predate scored-rollout execution geometry, native-LQR
    execution, lane costs or persistent maneuvers. Versioned loading excludes them;
    constructing this identifier never reads, migrates or modifies a bank.
    Defaults match CarlaPDMPlanner and TrajectoryTracker, rather than the YAML
    preset, so omitted and explicitly supplied equivalent settings agree.
    MPC additionally identifies the measured vehicle model supplied by runtime
    setup. An unspecified model is distinct from measured physics, so earlier
    unqualified evidence cannot be mixed into a vehicle-specific bank.
    """
    planning, controller = dict(planning or {}), copy.deepcopy(dict(controller or {}))
    numeric = {key: float(planning.get(key, default)) for key, default in (
        ("lqr_lateral_horizon_s", 2.0), ("lqr_lateral_min_horizon_s", 1.0),
        ("sample_interval_s", .1), ("horizon_s", 4.0), ("max_speed_mps", 8.0),
        ("lane_change_length_m", 30.0))}
    numeric.update(lateral_control=planning.get("lateral_control", "pure_pursuit"),
                   lqr_lateral_preview=planning.get("lqr_lateral_preview", "fixed"))
    from .lane_quality import validate_lane_quality_config
    from .lane_maneuver import validate_maneuver_config
    numeric["lane_quality"] = validate_lane_quality_config(planning.get("lane_quality"))
    numeric["lane_maneuver"] = validate_maneuver_config(planning.get("lane_maneuver"))
    effective_controller = {"wheelbase_m": 2.8, "max_steer_rad": .6,
        "lookahead_min_m": 3.0, "lookahead_max_m": 12.0, "lookahead_k": .5, "steer_sign": -1,
        **{key: value for key, value in controller.items() if key != "pid"}}
    effective_controller["pid"] = {"kp": 1.0, "ki": .05, "kd": .05,
        "max_throttle": .75, "max_brake": .8, "deadband_mps": .15,
        **(controller.get("pid") or {})}

    def canonical(value):
        if type(value) in (int, float):
            if not math.isfinite(value):
                raise ValueError("execution planner version requires finite numerical settings")
            return float(value)
        if isinstance(value, dict):
            return {str(key): canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        return value

    if planning.get("backend") == "mpc":
        from dataclasses import asdict
        from .mpc_planner import effective_mpc_config
        numerical = {"backend": "mpc", "mpc": asdict(effective_mpc_config(planning)),
            "replan_interval_s": float(planning.get("replan_interval_s", .2)),
            "max_speed_mps": float(planning.get("max_speed_mps", 8.)),
            "lane_quality": numeric["lane_quality"], "lane_maneuver": numeric["lane_maneuver"]}
        return "carla_mpc_execution_v1:" + _digest(canonical({
            "planning": numerical, "controller": effective_controller,
            "vehicle_model": vehicle_model}))

    return "carla_pdm_execution_v5:" + _digest(canonical({"planning": numeric,
                                                         "controller": effective_controller}))


def _action(value):
    if isinstance(value, str):
        return {"longitudinal_decision": value.upper()}
    if not isinstance(value, dict):
        return None
    result = {key: str(value[key]).upper() for key in
              ("lateral_decision", "longitudinal_decision")
              if isinstance(value.get(key), str) and value[key]}
    return result or None


def _matches_action(record_action, query):
    return bool(record_action and query and
                all(record_action.get(k) == v for k, v in query.items()))


def _parameters(value):
    if not isinstance(value, dict):
        return {}
    # These are proposals only. The planner still applies its native limits and
    # resimulates against current actors. Do not import stop-policy parameters.
    result = {k: float(value[k]) for k in _PARAMETERS if _finite(value.get(k))}
    if set(result) != set(_PARAMETERS) or any(v <= 0 for v in result.values()):
        return {}
    return result


def _signature(context):
    """Relationships, not actor IDs, images or exact world coordinates."""
    signals = []
    for signal in context.get("signals", []):
        if signal.get("applicable") is not True:
            continue
        distance = _number(signal, "stop_line_distance_m")
        band = ("unknown" if distance is None else "at_line" if distance <= 5 else
                "approaching" if distance <= 20 else "distant")
        signals.append((str(signal.get("state", "unknown")).lower(), band))
    signals.sort()
    actor_relations = set()
    for actor in context.get("actors", []):
        if not isinstance(actor, dict):
            continue
        if actor.get("conflict") is True:
            relation = "conflict"
        elif actor.get("same_lane") is True and actor.get("ahead") is True:
            relation = "lead"
        else:
            continue
        distance = _number(actor, "distance_m")
        ttc = _number(actor, "ttc_s")
        actor_relations.add((relation, str(actor.get("type", "unknown")),
                             "near" if distance is not None and distance < 12 else "far",
                             "imminent" if ttc is not None and ttc < 3 else
                             "approaching" if ttc is not None and ttc < 8 else "unknown"))
    navigation = context.get("navigation")
    if isinstance(navigation, dict):
        navigation = navigation.get("maneuver", navigation.get("command", navigation.get("instruction")))
    return {"topology": context.get("topology_signature"),
            "navigation": navigation, "in_junction": context.get("in_junction"),
            "signals": signals, "actors": sorted(actor_relations),
            "blocked": context.get("blocked"),
            "special": sorted(str(v) for v in context.get("special_situations", [])),
            "baseline_stop_expected": context.get("baseline_stop_expected")}


class ExecutionExperienceMemory:
    """A JSONL execution bank with explicit access and observation contracts.

    ``bank_path`` is a memory-bank directory. A JSONL path is also accepted for
    isolated offline tools. Construction never creates a bank. ``memory_off``
    disables both reads and writes; ``read_only`` never changes the bank, even
    during begin/observe/finish. Those operations may write a *separate* run audit
    when ``run_dir`` is provided. Read and write flags are otherwise independent.

    Telemetry uses cumulative collision_count/violation_count, sim_time_s,
    route_progress_m, ego_speed_mps, acceleration_mps2 and jerk_mps3. ``begin``
    additionally accepts actual_action, selected_parameters,
    baseline_candidate_id and selected_candidate_id. An unexecuted requested
    action is never substituted for missing actual_action.
    """

    def __init__(self, bank_path, *, memory_mode="read_write", read_enabled=True,
                 write_enabled=True, run_id="", planner_version="pdm_execution_v1",
                 run_dir=None, observation_horizon_s=5.0, min_support=3,
                 max_score_residual=0.15, adjacent_exclusion_s=10.0,
                 max_speed_difference_mps=3.0, max_acceleration_mps2=5.0,
                 max_jerk_mps3=10.0):
        if memory_mode not in {"memory_off", "read_only", "read_write"}:
            raise ValueError("memory_mode must be memory_off, read_only or read_write")
        for name, value in (("observation_horizon_s", observation_horizon_s),
                            ("adjacent_exclusion_s", adjacent_exclusion_s),
                            ("max_speed_difference_mps", max_speed_difference_mps),
                            ("max_acceleration_mps2", max_acceleration_mps2),
                            ("max_jerk_mps3", max_jerk_mps3)):
            if not _finite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if type(min_support) is not int or min_support < 2:
            raise ValueError("min_support must be an integer >= 2 independent runs")
        if not _finite(max_score_residual) or not 0 <= max_score_residual <= 0.25:
            raise ValueError("max_score_residual must be within [0, 0.25]")
        if not isinstance(planner_version, str) or not planner_version:
            raise ValueError("planner_version is required")
        path = Path(bank_path) if bank_path is not None else None
        self.path = (path if path is not None and path.suffix == ".jsonl" else
                     path / EXPERIENCE_FILENAME if path is not None else None)
        self.read_enabled = bool(read_enabled) and memory_mode != "memory_off" and self.path is not None
        self.write_enabled = bool(write_enabled) and memory_mode == "read_write" and self.path is not None
        self.memory_mode = memory_mode
        self.run_id = str(run_id) or uuid.uuid4().hex
        self.planner_version = planner_version
        self.audit_path = Path(run_dir) / "planning_experience_audit.jsonl" if run_dir else None
        if self.audit_path is not None and self.path is not None:
            # Separate run audits must not be a back door around read_only.
            if self.audit_path.resolve() == self.path.resolve() or self.path.parent.resolve() in self.audit_path.resolve().parents:
                raise ValueError("run audit must be outside the experience bank")
        self.horizon = float(observation_horizon_s)
        self.min_support = min_support
        self.max_residual = float(max_score_residual)
        self.adjacent_exclusion_s = float(adjacent_exclusion_s)
        self.max_speed_difference = float(max_speed_difference_mps)
        self.max_acceleration = float(max_acceleration_mps2)
        self.max_jerk = float(max_jerk_mps3)
        self.pending = {}
        self.errors = []
        self._cache_token = None
        self._cache_records = []
        self._stats = {"loaded_records": 0, "malformed_lines": 0, "incompatible_records": 0,
                       "current_run_excluded": 0, "adjacent_excluded": 0,
                       "begun": 0, "observed": 0, "failed": 0, "censored": 0,
                       "persisted_updates": 0, "retrievals": 0, "imported_pairs": 0}

    def summarize(self):
        """Return counters and explicit ingestion errors, never a clean-bank claim."""
        return {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                "planner_version": self.planner_version, "memory_mode": self.memory_mode,
                "read_enabled": self.read_enabled, "write_enabled": self.write_enabled,
                "bank_path": str(self.path) if self.path else None,
                "pending_count": len(self.pending), **copy.deepcopy(self._stats),
                "errors": list(self.errors), "trained_model": False,
                "similarity_is_confidence": False}

    def _append(self, path, record):
        if path is None:
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(_clean(record), sort_keys=True, ensure_ascii=False,
                                        allow_nan=False) + "\n")
            return True
        except OSError as exc:
            self.errors.append("%s: %s" % (type(exc).__name__, exc))
            return False

    def _audit(self, event, record):
        self._append(self.audit_path, {"event": event, "run_id": self.run_id,
                                      "record": _clean(record)})

    def _persist(self, record):
        if self.write_enabled and self._append(self.path, record):
            self._stats["persisted_updates"] += 1
            self._cache_token = None

    def _load(self):
        if not self.read_enabled:
            return []
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return []
        except OSError as exc:
            self.errors.append("cannot stat experience bank: %s" % exc)
            return []
        token = (stat.st_mtime_ns, stat.st_size)
        if token == self._cache_token:
            return self._cache_records
        records = {}
        malformed = incompatible = 0
        try:
            with self.path.open(encoding="utf-8-sig") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    item = None
                    try:
                        item = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                            raise ValueError("record requires a string id")
                        if item.get("schema_version") != SCHEMA_VERSION or item.get("planner_version") != self.planner_version:
                            incompatible += 1
                            records.pop(item["id"], None)
                            continue
                        if item.get("kind") not in {"execution", "paired_execution"}:
                            raise ValueError("unknown record kind")
                        if not isinstance(item.get("context"), dict):
                            raise ValueError("record context must be an object")
                        if item["kind"] == "execution":
                            if (item.get("status") not in {"pending", "observed", "failed", "censored"}
                                    or not isinstance(item.get("outcome"), dict)
                                    or not isinstance(item.get("run_id"), str)
                                    or not _finite(item.get("started_at_s"))
                                    or not isinstance(item.get("initial_telemetry"), dict)
                                    or not isinstance(item.get("last_telemetry"), dict)):
                                raise ValueError("execution status/outcome/provenance is incomplete")
                        else:
                            payload = item.get("paired_evidence")
                            if (not isinstance(payload, dict) or not _finite(item.get("utility_difference"))
                                    or not -1 <= item["utility_difference"] <= 1
                                    or not isinstance(item.get("experimental_unit_id"), str)
                                    or not isinstance(item.get("evaluation_run_ids"), list)
                                    or len(item["evaluation_run_ids"]) != 2):
                                raise ValueError("paired execution record is incomplete")
                            validated = self._validated_pair_record(payload)
                            if any(item.get(k) != validated[k] for k in
                                   ("id", "experimental_unit_id", "context", "actual_action",
                                    "evaluation_run_ids", "utility_difference", "worse_safety")):
                                raise ValueError("paired evidence summary disagrees with its execution payload")
                        records[item["id"]] = item
                    except (ValueError, TypeError) as exc:
                        if isinstance(item, dict) and isinstance(item.get("id"), str):
                            records.pop(item["id"], None)
                        malformed += 1
                        message = "invalid experience line %d: %s" % (line_no, exc)
                        if message not in self.errors:
                            self.errors.append(message)
        except (OSError, UnicodeError) as exc:
            self.errors.append("cannot read experience bank: %s" % exc)
            return []
        self._stats.update(loaded_records=len(records), malformed_lines=malformed,
                           incompatible_records=incompatible)
        if incompatible:
            message = "%d incompatible schema/planner-version records excluded" % incompatible
            if message not in self.errors:
                self.errors.append(message)
        self._cache_token = token
        self._cache_records = list(records.values())
        return self._cache_records

    def _similar(self, stored, current):
        if not isinstance(stored, dict) or stored.get("schema_version") != SCHEMA_VERSION:
            return False
        if current.get("schema_version") != SCHEMA_VERSION:
            return False
        for candidate in (stored, current):
            if any(not isinstance(candidate.get(k), list) for k in
                   ("actors", "signals", "special_situations")):
                return False
            if any(not isinstance(v, dict) for v in candidate["actors"] + candidate["signals"]):
                return False
        # Unknown topology is not evidence of a compatible intersection.
        if not current.get("topology_signature") or not stored.get("topology_signature"):
            return False
        if _signature(stored) != _signature(current):
            return False
        speeds = [_number(c, "ego_speed_mps") for c in (stored, current)]
        if any(v is None for v in speeds) or abs(speeds[0] - speeds[1]) > self.max_speed_difference:
            return False
        limits = [_number(c, "speed_limit_mps") for c in (stored, current)]
        if all(v is not None for v in limits) and abs(limits[0] - limits[1]) > self.max_speed_difference:
            return False
        return True

    def retrieve(self, context, action=None):
        query_action = _action(action)
        result = {"status": "unknown", "support_count": 0, "independent_run_count": 0,
                  "record_ids": [], "score_residual": 0.0, "parameter_priors": [],
                  "action_feedback": [], "intervention_priors": [], "similarity_is_confidence": False,
                  "causal_advantage": {"status": "unknown", "estimate": None,
                                       "lower_bound": None, "support_count": 0},
                  "risk_feedback": {"known_failure": False, "verified": False,
                                    "support_count": 0, "negative_residual": 0.0,
                                    "scope": "action" if query_action else "scene"},
                  "reason": "no_compatible_observed_execution"}
        if not self.read_enabled:
            result["reason"] = "memory_read_disabled"
            result["bank_health"] = self.summarize()
            return result
        if not isinstance(context, dict):
            result["reason"] = "invalid_context"
            result["bank_health"] = self.summarize()
            return result
        self._stats["retrievals"] += 1
        executions, pairs = [], []
        for record in self._load():
            if record.get("run_id") == self.run_id:
                self._stats["current_run_excluded"] += 1
                continue
            if not self._similar(record.get("context"), context):
                continue
            if record["kind"] == "paired_execution":
                if self.run_id in record.get("evaluation_run_ids", []):
                    continue
                if query_action and not _matches_action(record.get("actual_action"), query_action):
                    continue
                pairs.append(record)
            elif record.get("status") in {"observed", "failed"}:
                if query_action and not _matches_action(record.get("actual_action"), query_action):
                    continue
                outcome = record.get("outcome", {})
                if outcome.get("complete") is True:
                    executions.append(record)
        # Dense sequential planner ticks are not independent experience support.
        deduped = []
        for record in sorted(executions, key=lambda r: (str(r.get("run_id")), r.get("started_at_s", 0))):
            stamp = record.get("started_at_s")
            nearby = any(r.get("run_id") == record.get("run_id") and
                         _finite(stamp) and _finite(r.get("started_at_s")) and
                         abs(stamp - r["started_at_s"]) < self.adjacent_exclusion_s
                         for r in deduped)
            if nearby:
                self._stats["adjacent_excluded"] += 1
            else:
                deduped.append(record)
        executions = deduped
        runs = {r.get("run_id") for r in executions if r.get("run_id")}
        result.update(support_count=len(executions), independent_run_count=len(runs),
                      record_ids=[r["id"] for r in executions])
        if len(runs) >= self.min_support:
            result.update(status="supported", reason="observed_executions_only_not_causal_gain")
        elif executions:
            result["reason"] = "insufficient_independent_runs"
        failures = [r for r in executions if r["outcome"].get("verified_failure") is True]
        failure_runs = {r["run_id"] for r in failures}
        negative = -self.max_residual if len(failure_runs) >= self.min_support else 0.0
        result["risk_feedback"].update(known_failure=len(failure_runs) >= self.min_support,
                                       verified=bool(failures), support_count=len(failure_runs),
                                       negative_residual=negative)
        # Scene-level failure evidence may request consultation, but must not
        # penalize every action or authorize stopping.
        if query_action:
            result["score_residual"] = negative
        groups = {}
        for record in executions:
            actual = record.get("actual_action")
            if actual:
                groups.setdefault(_digest(actual), {"action": actual, "records": []})["records"].append(record)
        for group in groups.values():
            bad = [r for r in group["records"] if r["outcome"].get("verified_failure") is True]
            n_bad = len({r["run_id"] for r in bad})
            result["action_feedback"].append({"action": group["action"],
                "known_failure": n_bad >= self.min_support, "verified": bool(bad),
                "support_count": n_bad, "score_residual": -self.max_residual if n_bad >= self.min_support else 0.0})
        parameter_groups = {}
        for record in executions:
            outcome = record["outcome"]
            actual = record.get("actual_action")
            parameters = _parameters(record.get("selected_parameters"))
            if (not parameters or not actual or actual.get("longitudinal_decision") in {"STOP", "YIELD"}
                    or outcome.get("parameter_execution_supported") is not True):
                continue
            key = _digest([actual, parameters])
            parameter_groups.setdefault(key, {"action": actual, "parameters": parameters,
                                              "records": []})["records"].append(record)
        for group in parameter_groups.values():
            n_runs = len({r["run_id"] for r in group["records"]})
            if n_runs >= self.min_support:
                result["parameter_priors"].append({"action": group["action"],
                    "parameters": group["parameters"], "support_count": n_runs,
                    "record_ids": [r["id"] for r in group["records"]],
                    "status": "execution_supported", "causal_gain": "unknown"})
        result["parameter_priors"] = sorted(result["parameter_priors"],
                                             key=lambda p: (-p["support_count"], _digest(p)))[:3]
        # At most one pair per independent experimental unit; no positive prior
        # from ordinary safe executions or from counterfactual PDM predictions.
        unique_pairs, used_runs, used_artifacts = {}, set(), set()
        for pair in pairs:
            runs = set(pair["evaluation_run_ids"])
            artifacts = {a.get("artifact_sha256") for a in
                         (pair["paired_evidence"].get("baseline", {}),
                          pair["paired_evidence"].get("intervention", {}))}
            if runs & used_runs or artifacts & used_artifacts:
                continue
            unit = pair["experimental_unit_id"]
            if unit in unique_pairs:
                continue
            unique_pairs[unit] = pair
            used_runs.update(runs)
            used_artifacts.update(artifacts)
        protocols = {}
        for pair in unique_pairs.values():
            payload = pair["paired_evidence"]
            key = (payload["protocol_id"], payload["utility_definition"], _digest(pair["actual_action"]))
            protocols.setdefault(key, []).append(pair)
        # Do not mix action meanings, experimental protocols or utility scales.
        group = max(protocols.values(), key=len) if protocols else []
        if group and query_action:
            result["causal_advantage"] = self._paired_advantage(group)
        elif protocols:
            # Consultation can benefit from knowing that one *specific* action
            # helped in independently evaluated matched executions. This never
            # makes an unqualified scene-level causal claim or authorizes an
            # arbitrary model action. The adoption path must query that action.
            # Choose the largest protocol per action before inspecting gain;
            # never combine different protocols or optimize their utility scale.
            action_groups = {}
            for protocol_group in protocols.values():
                key = _digest(protocol_group[0]["actual_action"])
                previous = action_groups.get(key)
                if previous is None or len(protocol_group) > len(previous):
                    action_groups[key] = protocol_group
            for action_group in action_groups.values():
                advantage = self._paired_advantage(action_group, family_size=len(action_groups))
                if advantage["status"] == "supported":
                    result["intervention_priors"].append({
                        "action": copy.deepcopy(action_group[0]["actual_action"]),
                        "causal_advantage": advantage,
                        "use": "consultation_only_recheck_requested_action_before_adoption"})
            result["intervention_priors"].sort(key=lambda p: (
                -p["causal_advantage"]["lower_bound"], _digest(p["action"])))
        result["bank_health"] = self.summarize()
        self._audit("retrieve", {"context": context, "action": query_action, "result": result})
        return copy.deepcopy(result)

    def _paired_advantage(self, group, *, family_size=1):
        """No simulated counterfactuals, action mixing or recursive retrieval.

        Bounds are conditional on the imported independent-pair protocol being
        valid. Scene queries use a Bonferroni bound across the actions inspected;
        a positive prior is evidence for requesting consultation, not permission
        to stop or a guarantee that a different intervention will help.
        """
        values = [p["utility_difference"] for p in group]
        estimate = statistics.mean(values)
        radius = math.sqrt(2 * math.log(20.0 * family_size) / len(values))
        lower = max(-1.0, estimate - radius)
        return {"status": "supported" if lower > 0 and len(values) >= self.min_support else "unknown",
            "estimate": estimate, "lower_bound": lower, "support_count": len(values),
            "method": "one_sided_95pct_hoeffding_independent_bounded_paired_utility",
            "assumption": "independently_sampled_matched_execution_pairs",
            "simultaneously_inspected_actions": family_size,
            "multiplicity_correction": "bonferroni" if family_size > 1 else "none",
            "record_ids": [p["id"] for p in group],
            "protocol_id": group[0]["paired_evidence"]["protocol_id"]}

    def begin(self, context, decision, telemetry):
        if self.memory_mode == "memory_off" or (not self.write_enabled and self.audit_path is None):
            return None
        context, decision, telemetry = _clean(context), _clean(decision or {}), _clean(telemetry or {})
        if not isinstance(context, dict) or context.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("experience context schema_version must be 1")
        if not isinstance(decision, dict) or not isinstance(telemetry, dict):
            raise ValueError("experience decision and telemetry must be objects")
        stamp = _number(telemetry, "sim_time_s")
        progress = _number(telemetry, "route_progress_m")
        if stamp is None or progress is None:
            raise ValueError("begin requires measured sim_time_s and route_progress_m")
        ident = uuid.uuid4().hex
        record = {"schema_version": SCHEMA_VERSION, "kind": "execution", "id": ident,
            "run_id": self.run_id, "planner_version": self.planner_version, "status": "pending",
            "context": context, "decision": decision, "started_at_s": stamp,
            "baseline_candidate_id": telemetry.get("baseline_candidate_id", decision.get("baseline_candidate_id")),
            "selected_candidate_id": telemetry.get("selected_candidate_id", decision.get("selected_candidate_id")),
            "actual_action": _action(telemetry.get("actual_action", decision.get("actual_action"))),
            "selected_parameters": _parameters(telemetry.get("selected_parameters", decision.get("selected_parameters"))),
            "initial_telemetry": telemetry, "last_telemetry": telemetry,
            "outcome": {"complete": False, "causal_gain": "unknown"}}
        self.pending[ident] = {"record": record, "stationary_s": 0.0, "samples": 0,
            "max_abs_acceleration": None, "max_abs_jerk": None, "measurement_gap": False,
            "counter_unknown": False, "last_time": stamp, "seen_collision": False,
            "seen_violation": False}
        self._stats["begun"] += 1
        self._persist(record)
        self._audit("begin", record)
        return ident

    def _observe_one(self, pending, telemetry):
        record = pending["record"]
        stamp = _number(telemetry, "sim_time_s")
        if stamp is None or stamp <= pending["last_time"]:
            return False
        dt = stamp - pending["last_time"]
        pending["measurement_gap"] |= dt > 0.5
        speed = _number(telemetry, "ego_speed_mps")
        if speed is not None and speed < 0.2:
            pending["stationary_s"] += min(dt, 0.5)
        for key, target in (("acceleration_mps2", "max_abs_acceleration"),
                            ("jerk_mps3", "max_abs_jerk")):
            value = _number(telemetry, key)
            if value is not None:
                pending[target] = max(pending[target] or 0.0, abs(value))
        for key, target in (("collision_count", "seen_collision"),
                            ("violation_count", "seen_violation")):
            initial = _number(record["initial_telemetry"], key)
            current = _number(telemetry, key)
            if current is None or initial is None or current < initial:
                pending["counter_unknown"] = True
            elif current > initial:
                pending[target] = True
        pending["last_time"] = stamp
        pending["samples"] += 1
        record["last_telemetry"] = _clean(telemetry)
        return True

    def _settle(self, ident, reason, *, complete):
        pending = self.pending.pop(ident)
        record = pending["record"]
        initial, last = record["initial_telemetry"], record["last_telemetry"]
        end_time = _number(last, "sim_time_s")
        end_progress = _number(last, "route_progress_m")
        delta = end_progress - initial["route_progress_m"] if end_progress is not None else None
        duration = end_time - record["started_at_s"] if end_time is not None else 0.0
        failure = pending["seen_collision"] or pending["seen_violation"] or reason in _TERMINAL_FAILURES
        coverage = pending["samples"] >= 2 and not pending["measurement_gap"] and not pending["counter_unknown"]
        complete = bool(complete and duration >= self.horizon and coverage and delta is not None)
        status = "failed" if failure else "observed" if complete else "censored"
        # Failed executions with a known event are useful negative evidence even
        # when a run ends before the nominal observation horizon.
        if failure:
            complete = True
        comfort_known = pending["max_abs_acceleration"] is not None and pending["max_abs_jerk"] is not None
        parameter_supported = bool(complete and coverage and comfort_known and not failure
            and delta is not None and delta >= 2.0 and duration >= min(2.0, self.horizon)
            and pending["max_abs_acceleration"] <= self.max_acceleration
            and pending["max_abs_jerk"] <= self.max_jerk)
        record.update(status=status, ended_at_s=end_time, outcome={
            "complete": complete, "termination_reason": reason, "duration_s": duration,
            "progress_m": delta, "stationary_s": pending["stationary_s"],
            "collision_observed": pending["seen_collision"], "violation_observed": pending["seen_violation"],
            "verified_failure": bool(failure), "measurement_coverage_complete": coverage,
            "max_abs_acceleration_mps2": pending["max_abs_acceleration"],
            "max_abs_jerk_mps3": pending["max_abs_jerk"],
            "parameter_execution_supported": parameter_supported,
            "causal_gain": "unknown", "counterfactual_executed": False})
        self._stats[status] += 1
        self._persist(record)
        self._audit("finish_experience", record)
        return copy.deepcopy(record)

    def observe(self, telemetry):
        if self.memory_mode == "memory_off":
            return []
        result = []
        for ident, pending in list(self.pending.items()):
            if telemetry.get("execution_policy_changed") is True:
                result.append(self._settle(ident, "execution_policy_changed", complete=False))
                continue
            if not self._observe_one(pending, telemetry):
                continue
            elapsed = pending["last_time"] - pending["record"]["started_at_s"]
            if pending["seen_collision"] or pending["seen_violation"]:
                result.append(self._settle(ident, "observed_adverse_event", complete=True))
            elif elapsed >= self.horizon:
                result.append(self._settle(ident, "observation_horizon", complete=True))
        return result

    def finish(self, status, telemetry=None):
        result = []
        if telemetry is not None:
            result.extend(self.observe(telemetry))
        for ident in list(self.pending):
            result.append(self._settle(ident, str(status), complete=status in _NORMAL_ENDINGS))
        self._audit("session_finish", {"status": str(status), "summary": self.summarize()})
        return result

    def import_paired_evidence(self, evidence):
        """Import audited actual-execution pairs; predictions are never pairs.

        Required provenance: independently sampled experimental_unit_id,
        protocol_id, matching_method='restored_full_state', full initial state
        (ego/actors/signals/map), two distinct evaluation run IDs and the exact
        same initial-state fingerprint/version/duration. Each arm must explicitly
        report executed_closed_loop evaluation, final measurements, complete
        observation and an externally evaluated utility in [0,1]. Import only
        checks that contract; it cannot independently certify an external lab's
        honesty or the simulator's restored-state fidelity.
        """
        if not self.write_enabled:
            raise PermissionError("paired evidence import requires memory write access")
        record = self._validated_pair_record(evidence)
        self._persist(record)
        self._stats["imported_pairs"] += 1
        self._audit("import_paired_evidence", record)
        return copy.deepcopy(record)

    def _validated_pair_record(self, evidence):
        value = _clean(evidence)
        required_strings = ("experimental_unit_id", "protocol_id", "state_fingerprint", "utility_definition")
        if (not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION
                or value.get("planner_version") != self.planner_version
                or value.get("matching_method") != "restored_full_state"
                or value.get("independently_sampled") is not True
                or value.get("protocol_fixed_before_evaluation") is not True
                or any(not isinstance(value.get(k), str) or not value[k] for k in required_strings)):
            raise ValueError("paired execution provenance/schema/version is incomplete")
        state = value.get("initial_state")
        if (not isinstance(state, dict) or not isinstance(state.get("ego"), dict)
                or any(not isinstance(state["ego"].get(k), (list, dict)) or not state["ego"][k]
                       for k in ("position", "velocity"))
                or not isinstance(state.get("actors"), list) or not isinstance(state.get("signals"), list)
                or not state.get("map")
                or _digest(state) != value["state_fingerprint"]):
            raise ValueError("full matched initial state fingerprint is required")
        context = value.get("context")
        if not isinstance(context, dict) or context.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("paired evidence requires a schema-1 retrieval context")
        arms = [value.get("baseline"), value.get("intervention")]
        for arm in arms:
            if (not isinstance(arm, dict) or arm.get("evaluation_kind") != "executed_closed_loop"
                    or arm.get("independent_evaluation") is not True or arm.get("complete") is not True
                    or arm.get("planner_version") != self.planner_version
                    or arm.get("state_fingerprint") != value["state_fingerprint"]
                    or _digest(arm.get("initial_state")) != value["state_fingerprint"]
                    or not isinstance(arm.get("run_id"), str) or not arm["run_id"]
                    or not isinstance(arm.get("artifact_sha256"), str) or len(arm["artifact_sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in arm["artifact_sha256"])
                    or not _finite(arm.get("utility")) or not 0 <= arm["utility"] <= 1
                    or not _finite(arm.get("duration_s")) or arm["duration_s"] <= 0):
                raise ValueError("both arms require independent executed observations and matching initial state/version")
            metrics = arm.get("metrics", {})
            if (not isinstance(metrics, dict) or any(not _finite(metrics.get(k)) or metrics[k] < 0 for k in
                   ("collision_count", "violation_count", "route_progress_m"))):
                raise ValueError("executed safety/progress metrics are required")
        baseline, intervention = arms
        if (baseline["run_id"] == intervention["run_id"] or baseline["duration_s"] != intervention["duration_s"]
                or baseline["artifact_sha256"] == intervention["artifact_sha256"]):
            raise ValueError("paired arms require distinct runs and equal evaluation duration")
        if self.run_id in {baseline["run_id"], intervention["run_id"]}:
            raise ValueError("current-run outcomes cannot be imported as independent evidence")
        actual = _action(intervention.get("actual_action"))
        if not actual:
            raise ValueError("paired intervention actual_action is required")
        # Utility cannot compensate for known worse safety outcomes.
        worse_safety = any(intervention["metrics"][k] > baseline["metrics"][k]
                           for k in ("collision_count", "violation_count"))
        difference = float(intervention["utility"] - baseline["utility"])
        if worse_safety:
            difference = min(difference, 0.0)
        record = {"schema_version": SCHEMA_VERSION, "kind": "paired_execution",
            "id": "pair_" + _digest([value["protocol_id"], value["experimental_unit_id"], actual]),
            "run_id": "offline_paired_evidence", "planner_version": self.planner_version,
            "experimental_unit_id": value["experimental_unit_id"], "context": context,
            "actual_action": actual, "evaluation_run_ids": [a["run_id"] for a in arms],
            "utility_difference": difference, "worse_safety": worse_safety,
            "paired_evidence": value}
        return copy.deepcopy(record)
