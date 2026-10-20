"""Query and adoption gates for optional, short-lived semantic preferences.

The governor never removes a finite PDM proposal or changes its native score.
Context conflict/TTC fields are current-state prediction proxies, not guarantees
or privileged future observations. A model request is neither an emergency
controller nor authority to stop merely because another actor is visible.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .selection import choose_candidate


DEFAULT_CONFIG = {
    "base_regret_budget": 0.08,
    "max_semantic_bonus": 0.08,
    "lateral_weight": 0.04,
    "longitudinal_weight": 0.04,
    "min_query_interval_s": 2.0,
    "low_quality_threshold": 0.4,
    "stall_duration_s": 5.0,
    "stall_progress_m": 0.5,
    "moving_option_progress_m": 2.0,
    "stop_distance_m": 6.0,
    "stop_time_window_s": 2.0,
    "max_target_distance_m": 60.0,
    "conflict_ttc_s": 5.0,
    "max_context_age_s": 1.0,
    "wait_review_interval_s": 15.0,  # accepted for old configs; normal waits never query on a timer
    "recovery_degraded_duration_s": 2.0,
    "recovery_min_observations": 3,
    "recovery_release_healthy_duration_s": 2.0,
    "recovery_release_progress_m": 2.0,
    "efficiency_degraded_duration_s": 4.0,
    "efficiency_min_observations": 3,
    "efficiency_speed_fraction": .35,
}
_LATERAL = {"FOLLOW_ROUTE", "KEEP_LANE", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}
_LONGITUDINAL = {"MAINTAIN", "ACCELERATE", "DECELERATE", "FOLLOW", "YIELD", "STOP"}
_REASONS = {"none", "path_blocked", "interaction", "traffic_signal", "visibility",
            "special_rule", "route_change", "planner_degraded"}
_ACTIVATIONS = {"immediate", "while_conflict", "approach_stop_line"}
_RELEASES = {"condition_cleared", "target_passed", "signal_changed", "maneuver_complete"}
_INTERVENTION_FIELDS = {"needed", "reason_code", "target_actor_ids", "target_stop_line_id",
                        "activation", "release", "valid_for_s"}
_PLANNER_REQUESTS = {"none", "expand_candidates"}
_CONTEXT_FIELDS = {"schema_version", "sim_time_s", "ego_speed_mps", "route_progress_m",
                   "in_junction", "lane_key", "navigation", "speed_limit_mps", "signals",
                   "actors", "blocked", "special_situations", "topology_signature",
                   "baseline_stop_expected"}


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _configuration(config):
    if config is not None and not isinstance(config, dict):
        raise ValueError("semantic governor config must be a dictionary")
    config = config or {}
    if any(not isinstance(name, str) for name in config):
        raise ValueError("semantic governor config keys must be strings")
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("unknown semantic governor config: " + ", ".join(sorted(unknown)))
    result = dict(DEFAULT_CONFIG, **config)
    for name, value in result.items():
        if not _number(value):
            raise ValueError("semantic governor %s must be a finite number" % name)
        if value < 0:
            raise ValueError("semantic governor %s must be non-negative" % name)
    for name in ("base_regret_budget", "max_semantic_bonus", "low_quality_threshold"):
        if result[name] > 1:
            raise ValueError("semantic governor %s must be within [0, 1]" % name)
    for name in ("lateral_weight", "longitudinal_weight"):
        if result[name] > 2:
            raise ValueError("semantic governor %s must be within [0, 2]" % name)
    for name in ("min_query_interval_s", "stall_duration_s", "stall_progress_m",
                 "moving_option_progress_m", "stop_distance_m", "stop_time_window_s",
                 "max_target_distance_m", "conflict_ttc_s", "max_context_age_s",
                 "wait_review_interval_s", "recovery_degraded_duration_s",
                 "recovery_release_healthy_duration_s", "recovery_release_progress_m",
                 "efficiency_degraded_duration_s"):
        if result[name] <= 0:
            raise ValueError("semantic governor %s must be positive" % name)
    if type(result["recovery_min_observations"]) is not int or result["recovery_min_observations"] < 2:
        raise ValueError("semantic governor recovery_min_observations must be an integer >= 2")
    if type(result["efficiency_min_observations"]) is not int or result["efficiency_min_observations"] < 2:
        raise ValueError("semantic governor efficiency_min_observations must be an integer >= 2")
    if not 0 < result["efficiency_speed_fraction"] < 1:
        raise ValueError("semantic governor efficiency_speed_fraction must be between zero and one")
    if result["max_target_distance_m"] < result["stop_distance_m"]:
        raise ValueError("max_target_distance_m must cover stop_distance_m")
    return result


def _context_error(context, now_s, max_age):
    if not isinstance(context, dict) or not _CONTEXT_FIELDS.issubset(context):
        return "missing_context_fields"
    if type(context["schema_version"]) is not int or context["schema_version"] != 1:
        return "unsupported_context_schema"
    for field in ("sim_time_s", "ego_speed_mps", "route_progress_m", "speed_limit_mps"):
        if not _number(context[field]):
            return "invalid_context_" + field
    if context["sim_time_s"] > now_s + 1e-6 or now_s - context["sim_time_s"] > max_age:
        return "stale_context"
    if context["ego_speed_mps"] < 0 or context["speed_limit_mps"] < 0:
        return "negative_context_speed"
    for field in ("in_junction", "blocked", "baseline_stop_expected"):
        if type(context[field]) is not bool:
            return "invalid_context_" + field
    for field in ("lane_key", "navigation", "topology_signature"):
        if not isinstance(context[field], str):
            return "invalid_context_" + field
    if (not isinstance(context["special_situations"], list)
            or any(not isinstance(item, str) for item in context["special_situations"])):
        return "invalid_context_special_situations"
    if not isinstance(context["actors"], list) or not isinstance(context["signals"], list):
        return "invalid_context_entities"
    seen = set()
    for actor in context["actors"]:
        if (not isinstance(actor, dict) or type(actor.get("id")) is not int
                or actor["id"] in seen):
            return "invalid_context_actor_identity"
        seen.add(actor["id"])
        if any(not _number(actor.get(field)) for field in
               ("distance_m", "relative_s_m", "relative_l_m")):
            return "invalid_context_actor_geometry"
        if actor["distance_m"] < 0:
            return "invalid_context_actor_geometry"
        if any(type(actor.get(field)) is not bool for field in ("same_lane", "conflict", "ahead")):
            return "invalid_context_actor_relation"
        if actor.get("ttc_s") is not None and (
                not _number(actor["ttc_s"]) or actor["ttc_s"] < 0):
            return "invalid_context_actor_ttc"
        if not isinstance(actor.get("type"), str):
            return "invalid_context_actor_type"
    seen = set()
    for signal in context["signals"]:
        if (not isinstance(signal, dict) or not isinstance(signal.get("id"), str)
                or not signal["id"] or signal["id"] in seen
                or not isinstance(signal.get("state"), str)
                or not _number(signal.get("stop_line_distance_m"))
                or type(signal.get("applicable")) is not bool):
            return "invalid_context_signal"
        seen.add(signal["id"])
    return None


def _memory_summary(memory):
    """Observed failures may invite a query; they never authorize an action."""
    memory = memory if isinstance(memory, dict) else {}
    risk = memory.get("risk_feedback", {})
    causal = memory.get("causal_advantage", {})
    risk = risk if isinstance(risk, dict) else {}
    causal = causal if isinstance(causal, dict) else {}
    known_failure = (risk.get("known_failure") is True and risk.get("verified") is True
                     and type(risk.get("support_count")) is int and risk["support_count"] > 0
                     and risk.get("scope") in ("scene", "action"))
    positive_prior = (causal.get("status") == "supported"
                      and _number(causal.get("lower_bound")) and causal["lower_bound"] > 0
                      and type(causal.get("support_count")) is int and causal["support_count"] > 0)
    supported_interventions = 0
    priors = memory.get("intervention_priors", [])
    for prior in priors if isinstance(priors, list) else []:
        if not isinstance(prior, dict):
            continue
        action, advantage = prior.get("action"), prior.get("causal_advantage")
        if (not isinstance(action, dict) or not isinstance(advantage, dict)
                or not isinstance(action.get("lateral_decision"), str)
                or action["lateral_decision"] not in _LATERAL
                or not isinstance(action.get("longitudinal_decision"), str)
                or action["longitudinal_decision"] not in _LONGITUDINAL):
            continue
        if (prior.get("use") == "consultation_only_recheck_requested_action_before_adoption"
                and advantage.get("status") == "supported"
                and _number(advantage.get("lower_bound")) and advantage["lower_bound"] > 0
                and type(advantage.get("support_count")) is int and advantage["support_count"] > 0):
            supported_interventions += 1
    return {"known_failure": known_failure, "positive_causal_prior": positive_prior,
            "supported_intervention_prior": supported_interventions > 0,
            "supported_intervention_prior_count": supported_interventions,
            "memory_authorizes_stop": False}


class SelectiveSemanticGovernor:
    """Keep PDM authoritative unless a fresh, grounded preference is admissible.

    ``query_decision`` consumes the query interval when it returns query=True;
    callers should dispatch that query immediately. ``submit`` starts a finite
    lease bounded by the originating snapshot; repeated ``select`` calls cannot
    renew it. Invalid or condition-cleared guidance is discarded permanently.
    None of these operations cancels the numerical planner's continuity state.
    """

    def __init__(self, config=None):
        self.config = _configuration(config)
        self._pending = None
        self._last_query_s = None
        self._last_observed_s = None
        self._progress_anchor = None
        self._wait_started_s = None
        self._last_submission = {"accepted": False, "reason": "no_submission"}
        self._recovery_anchor = None
        self._recovery_last_sample_s = None
        self._recovery_observations = 0
        self._recovery_state = {"degraded": False, "sustained": False,
                                "reason": "planner_health_unavailable"}
        self._efficiency_anchor = None
        self._efficiency_last_sample_s = None
        self._efficiency_observations = 0
        self._efficiency_state = {"degraded": False, "sustained": False,
                                  "reason": "no_measured_efficiency_constraint"}

    def _observe(self, context, now_s):
        if not _number(now_s) or now_s < 0:
            raise ValueError("now_s must be a finite non-negative number")
        if self._last_observed_s is not None and now_s < self._last_observed_s:
            # A restarted simulator cannot inherit another episode's leases.
            self._pending = None
            self._last_query_s = None
            self._progress_anchor = None
            self._wait_started_s = None
            self._reset_recovery()
            self._reset_efficiency()
        self._last_observed_s = float(now_s)
        error = _context_error(context, now_s, self.config["max_context_age_s"])
        if error:
            self._reset_recovery()
            self._reset_efficiency()
            return error
        self._observe_efficiency(context)
        self._observe_recovery(context, now_s)
        if context["baseline_stop_expected"] and context["ego_speed_mps"] <= 0.5:
            if self._wait_started_s is None:
                self._wait_started_s = float(now_s)
        else:
            self._wait_started_s = None
        progress = context["route_progress_m"]
        if (context["baseline_stop_expected"]
                or context["ego_speed_mps"] > 0.5 or self._progress_anchor is None
                or abs(progress - self._progress_anchor[1]) >= self.config["stall_progress_m"]):
            self._progress_anchor = (float(now_s), progress)
        return None

    def _reset_efficiency(self):
        self._efficiency_anchor = None
        self._efficiency_last_sample_s = None
        self._efficiency_observations = 0
        self._efficiency_state = {"degraded": False, "sustained": False,
                                  "reason": "no_measured_efficiency_constraint"}

    def _observe_efficiency(self, context):
        """A positive-scoring crawl can be unhealthy despite making progress.

        This invites semantic consultation only. A legal/currently clear target
        lane, sustained measured low speed and a specific slow route obstacle
        are all required. It never authorizes that lane change or a stop.
        """
        slow = context.get("slow_obstacle")
        good = (isinstance(slow, dict) and type(slow.get("actor_id")) is int
            and isinstance(slow.get("legal_clear_sides"), list)
            and bool(slow["legal_clear_sides"])
            and all(s in ("left", "right") for s in slow["legal_clear_sides"])
            and _number(slow.get("along_speed_mps")) and -.3 <= slow["along_speed_mps"]
            and _number(slow.get("clearance_m")) and slow["clearance_m"] > .5
            and any(a["id"] == slow["actor_id"] for a in context["actors"])
            and not context["baseline_stop_expected"] and not context["in_junction"])
        if good:
            threshold = max(1.2, self.config["efficiency_speed_fraction"]*context["speed_limit_mps"])
            good = (context["ego_speed_mps"] <= threshold
                    and context["ego_speed_mps"] <= slow["along_speed_mps"]+.8
                    and slow["along_speed_mps"] < threshold)
        if not good:
            self._reset_efficiency()
            return
        signature = (context["topology_signature"], context["lane_key"], slow["actor_id"],
                     tuple(sorted(slow["legal_clear_sides"])))
        sample = context["sim_time_s"]
        if (self._efficiency_anchor is None or self._efficiency_anchor[1] != signature
                or self._efficiency_last_sample_s is None
                or sample-self._efficiency_last_sample_s > self.config["max_context_age_s"]):
            self._efficiency_anchor = (sample, signature)
            self._efficiency_observations = 0
            self._efficiency_last_sample_s = None
        if self._efficiency_last_sample_s is None or sample > self._efficiency_last_sample_s+1e-9:
            self._efficiency_observations += 1
            self._efficiency_last_sample_s = sample
        elapsed = max(0., sample-self._efficiency_anchor[0])
        sustained = (elapsed+1e-9 >= self.config["efficiency_degraded_duration_s"]
                     and self._efficiency_observations >= self.config["efficiency_min_observations"])
        self._efficiency_state = {"degraded": True, "sustained": bool(sustained),
            "reason": "sustained_low_efficiency_following" if sustained else "observing_low_efficiency_following",
            "actor_id": slow["actor_id"], "duration_s": elapsed,
            "observation_count": self._efficiency_observations,
            "legal_clear_sides": list(slow["legal_clear_sides"]),
            "execution_authorized": False}

    def _reset_recovery(self):
        self._recovery_anchor = None
        self._recovery_last_sample_s = None
        self._recovery_observations = 0
        self._recovery_state = {"degraded": False, "sustained": False,
                                "reason": "planner_health_unavailable"}

    def _recovery_obstruction(self, context):
        """Current evidence that forbids treating a stop as free-road failure."""
        if context["baseline_stop_expected"]:
            return "expected_signal_or_queue_wait"
        efficiency_id = (self._efficiency_state.get("actor_id")
                         if self._efficiency_state["sustained"] else None)
        if context.get("routine_following") is True and efficiency_id is None:
            return "ordinary_same_lane_following"
        stop_range = max(12.0, context["ego_speed_mps"] * self.config["stop_time_window_s"])
        if any(s["applicable"] and s["state"].upper() in ("RED", "YELLOW")
               and 0 <= s["stop_line_distance_m"] <= stop_range for s in context["signals"]):
            return "current_stop_signal"
        blockers = [a for a in context["actors"] if self._path_blocking(a)]
        if ((context["blocked"] and efficiency_id is None)
                or any(a["id"] != efficiency_id for a in blockers)):
            return "current_path_blockage"
        if any(a["id"] != efficiency_id and a["conflict"] and a["ttc_s"] is not None
               and a["ttc_s"] <= self.config["conflict_ttc_s"] for a in context["actors"]):
            return "current_interaction_conflict"
        return None

    def _observe_recovery(self, context, now_s):
        """Observe native health without consuming query cooldown or new samples twice."""
        health = context.get("planner_health")
        counts = ("candidate_count", "finite_candidate_count", "moving_candidate_count",
                  "positive_progressing_count", "drivable_area_failure_count",
                  "no_collision_failure_count")
        valid = (isinstance(health, dict) and health.get("source") in (
                    "native_pdm_baseline", "nonlinear_mpc_baseline")
                 and all(type(health.get(k)) is int and health[k] >= 0 for k in counts)
                 and type(health.get("baseline_stop")) is bool
                 and type(health.get("no_positive_progressing_candidate")) is bool
                 and health["positive_progressing_count"] <= health["moving_candidate_count"]
                 <= health["finite_candidate_count"] <= health["candidate_count"]
                 and health["no_positive_progressing_candidate"] ==
                     (health["positive_progressing_count"] == 0))
        if not valid:
            self._reset_recovery()
            return
        obstruction = self._recovery_obstruction(context)
        efficiency_degraded = self._efficiency_state["sustained"]
        degraded = (efficiency_degraded or (health["positive_progressing_count"] == 0
                    and (health["baseline_stop"] or health["finite_candidate_count"] == 0
                         or context["ego_speed_mps"] <= 0.5)))
        if obstruction or not degraded:
            self._reset_recovery()
            self._recovery_state.update(reason=obstruction or "native_progressing_candidates_available",
                                        native_health=deepcopy(health))
            return
        signature = (context["topology_signature"], context["navigation"], context["lane_key"])
        sample_s = context["sim_time_s"]
        # Repeated select/query/recovery checks of one physics snapshot never
        # count as several independent observations. Gaps also cannot create a
        # fabricated continuous failure interval from only two distant samples.
        if (self._recovery_anchor is None or self._recovery_anchor[1] != signature
                or self._recovery_last_sample_s is None
                or sample_s - self._recovery_last_sample_s > self.config["max_context_age_s"]):
            self._recovery_anchor = (float(sample_s), signature)
            self._recovery_observations = 0
            self._recovery_last_sample_s = None
        if self._recovery_last_sample_s is None or sample_s > self._recovery_last_sample_s + 1e-9:
            self._recovery_observations += 1
            self._recovery_last_sample_s = float(sample_s)
        elapsed = max(0.0, float(sample_s)-self._recovery_anchor[0])
        sustained = (efficiency_degraded or (elapsed + 1e-9 >= self.config["recovery_degraded_duration_s"]
                     and self._recovery_observations >= self.config["recovery_min_observations"]))
        self._recovery_state = {"degraded": True, "sustained": bool(sustained),
                                "reason": ("sustained_low_efficiency_following" if efficiency_degraded else
                                    "sustained_native_planner_degradation" if sustained
                                    else "observing_native_planner_degradation"),
                                "duration_s": max(elapsed, self._efficiency_state.get("duration_s", 0.)),
                                "observation_count": max(self._recovery_observations,
                                    self._efficiency_state.get("observation_count", 0)),
                                "inefficient_following": bool(efficiency_degraded),
                                "efficiency": deepcopy(self._efficiency_state),
                                "native_health": deepcopy(health)}

    def observe_planner_health(self, context, now_s):
        """Record an ORIGINAL pool observation; do not replace it after expansion."""
        error = self._observe(context, now_s)
        result = deepcopy(self._recovery_state)
        if error:
            result.update(reason=error, context_error=error)
        return result

    def recovery_request(self, context, now_s):
        """Authorize candidate generation only, never override native PDM scores."""
        error = self._observe(context, now_s)
        result = {"authorized": False, "request": "expand_candidates",
                  "reason": error or "no_recovery_request",
                  "planner_health": deepcopy(self._recovery_state),
                  "native_score_override": False}
        if error:
            self._pending = None
            return result
        if self._pending is None:
            return result
        intervention = self._pending["semantic"]["intervention"]
        if intervention.get("planner_request", "none") != "expand_candidates":
            return result
        rejection, evidence = self._authorization(context, now_s)
        result.update(reason=rejection or "grounded_candidate_expansion",
                      authorized=rejection is None, evidence=evidence,
                      expires_at_s=self._pending["expires_at_s"])
        if rejection:
            self._pending = None
        return result

    @staticmethod
    def _path_blocking(actor):
        """Require evidence belonging to this actor, not another scene object.

        ``path_blocked`` is produced from the ordered route corridor. A same-lane
        label or a scene-level blocked flag cannot substitute for that evidence.
        Missing fields fail closed only for semantic blockage authorization;
        native PDM proposals remain available.
        """
        return (actor.get("path_blocked") is True and actor.get("stationary") is True
                and actor.get("routine_wait") is not True
                and actor.get("blocking_kind") != "moving_lead"
                and _number(actor.get("route_distance_m"))
                and 1 < actor["route_distance_m"] < 30)

    def _conditions(self, context, intervention):
        """Discrete conditions for query reuse; continuous positions may evolve.

        An unchanged lease still gets revalidated on every numerical selection.
        New conflicts, blockers, stop-line activation bands and topology changes
        invalidate consultation reuse without pretending to predict future GT.
        """
        stop_range = max(self.config["stop_distance_m"],
                         context["ego_speed_mps"] * self.config["stop_time_window_s"])
        targets = set(intervention["target_actor_ids"])
        def conflict(actor):
            return (actor["conflict"] and actor["ttc_s"] is not None
                    and actor["ttc_s"] <= self.config["conflict_ttc_s"])
        actor_conditions = tuple(sorted((a["id"], a["same_lane"], a["ahead"],
            conflict(a), self._path_blocking(a), a["distance_m"] <= stop_range,
            a["ttc_s"] is not None and a["ttc_s"] <= self.config["stop_time_window_s"])
            for a in context["actors"] if a["id"] in targets))
        # Existing, unmentioned actors can become urgent without acquiring a
        # new ID. Include activation bands for ALL conflicts, not just targets.
        active_conflicts = tuple(sorted((a["id"], a["distance_m"] <= stop_range,
            a["ttc_s"] <= self.config["stop_time_window_s"],
            a.get("occupied_ego_conflict") is True)
            for a in context["actors"] if conflict(a)))
        blockers = tuple(sorted(a["id"] for a in context["actors"] if self._path_blocking(a)))
        signals = tuple(sorted((s["id"], s["state"].upper(),
            "passed" if s["stop_line_distance_m"] < 0 else
            "near" if s["stop_line_distance_m"] <= stop_range else "far")
            for s in context["signals"] if s["applicable"] and
            (s["id"] == intervention["target_stop_line_id"]
             or 0 <= s["stop_line_distance_m"] <= stop_range)))
        summary = context.get("candidates_summary")
        quality = None
        if isinstance(summary, dict):
            score = summary.get("baseline_planning_score", summary.get("baseline_score"))
            progress = summary.get("baseline_progress_m")
            quality = (_number(score) and score < self.config["low_quality_threshold"],
                       _number(progress) and progress >= self.config["moving_option_progress_m"])
        return (context["topology_signature"], context["navigation"], context["lane_key"],
                context["blocked"], context["baseline_stop_expected"],
                tuple(sorted(context["special_situations"])), actor_conditions,
                active_conflicts, blockers, signals, quality)

    @staticmethod
    def _memory_discount(candidate, memory):
        """Verified action failures can only subtract from a semantic bonus."""
        feedback = memory.get("action_feedback", []) if isinstance(memory, dict) else []
        discount = 0.0
        for item in feedback if isinstance(feedback, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("action"), dict):
                continue
            action = item["action"]
            lateral, longitudinal = action.get("lateral_decision"), action.get("longitudinal_decision")
            if (not isinstance(lateral, str) or lateral not in _LATERAL
                    or not isinstance(longitudinal, str) or longitudinal not in _LONGITUDINAL
                    or lateral not in candidate.get("lateral_semantics", [])
                    or longitudinal != candidate.get("longitudinal")):
                continue
            residual = item.get("score_residual")
            if (item.get("known_failure") is True and item.get("verified") is True
                    and type(item.get("support_count")) is int and item["support_count"] >= 3
                    and _number(residual) and -0.15 <= residual <= 0):
                discount = max(discount, -float(residual))
        return discount

    def query_decision(self, context, candidates, now_s, memory_evidence=None):
        error = self._observe(context, now_s)
        memory = _memory_summary(memory_evidence)
        result = {"query": False, "eligible": False, "reason": "simple_cruising",
                  "context_error": error, "memory": memory,
                  "next_query_time_s": (self._last_query_s + self.config["min_query_interval_s"]
                                        if self._last_query_s is not None else float(now_s))}
        if error:
            self._pending = None
            result["reason"] = error
            return result
        baseline, audit = choose_candidate(candidates or [], {}, False)
        valid = [item for item in audit["candidates"] if item["eligible"]]
        moving = [item for item in valid
                  if _number(item.get("forward_progress_m"))
                  and item["forward_progress_m"] >= self.config["moving_option_progress_m"]
                  and item["base_score"] > 0
                  and item.get("longitudinal") != "STOP"]
        # Red lights and causally grounded signal queues are numerical driving
        # tasks. Their duration, model lease expiry, fog tag or a memory prior
        # never creates a reason to consult the VLM again by itself.
        legitimate_wait = context["baseline_stop_expected"]
        routine_following = context.get("routine_following") is True
        wait_review_due = False  # retained audit/config compatibility; no timer-driven query
        # A first-time crossing interaction can deserve consultation even when
        # native scores are high. Mere visibility, adjacent opposing traffic
        # and ordinary same-lane following are insufficient. These conflicts
        # remain current-state forecast proxies, never future ground truth.
        crossing = [a for a in context["actors"]
                    if a["type"].startswith(("vehicle.", "walker."))
                    and not a["same_lane"] and a["conflict"] and a["ttc_s"] is not None
                    and a["ttc_s"] <= self.config["conflict_ttc_s"]]
        urgent_crossing = [a for a in crossing
                           if a["ttc_s"] <= self.config["stop_time_window_s"]]
        # While waiting, a route forecast assumes a hypothetical 2 m/s launch.
        # Cross traffic meeting that hypothetical launch is ordinary at red.
        # Only a forecast invading the MEASURED occupied ego body escalates.
        waiting_hazards = [a for a in context["actors"]
            if a["type"].startswith(("vehicle.", "walker.")) and not a["same_lane"]
            and ((a.get("occupied_ego_conflict") is True
                  and (_number(a.get("occupied_ego_conflict_ttc_s"))
                       and a["occupied_ego_conflict_ttc_s"] <= self.config["stop_time_window_s"]))
                 or ("occupied_ego_conflict" not in a and a in urgent_crossing
                     and a["distance_m"] <= 3.0))]
        waiting_obstacles = [a for a in context["actors"]
            if self._path_blocking(a) and not a["type"].startswith("vehicle.")
            and a["distance_m"] <= self.config["stop_distance_m"]]
        lease_changed = False
        if self._pending is not None:
            drop_reason, evidence = self._authorization(context, now_s)
            if drop_reason:
                self._pending = None
                result["lease_drop_reason"] = drop_reason
            else:
                intervention = self._pending["semantic"]["intervention"]
                lease_changed = (self._conditions(context, intervention)
                                 != self._conditions(self._pending["context"], intervention))
                recovery_escalation = (self._recovery_state["sustained"]
                    and intervention.get("planner_request", "none") != "expand_candidates")
                continuing_search = intervention.get("planner_request", "none") == "expand_candidates"
                if ((not lease_changed or continuing_search) and not recovery_escalation
                        and not waiting_hazards
                        and (not legitimate_wait or waiting_obstacles) and not routine_following):
                    result.update(reason="active_guidance_cached", active_guidance_cached=True,
                                  expires_at_s=self._pending["expires_at_s"],
                                  legitimate_wait=bool(legitimate_wait),
                                  unexpected_stall=False, evidence=evidence,
                                  baseline_selected_id=audit["baseline_selected_id"],
                                  moving_proposal_count=len(moving),
                                  candidate_pool_sha256=audit["candidate_pool_sha256"])
                    return result
        stalled = (self._progress_anchor is not None
                   and now_s - self._progress_anchor[0] >= self.config["stall_duration_s"])
        if legitimate_wait and waiting_hazards:
            reason = "occupied_ego_crossing_conflict"
        elif legitimate_wait and waiting_obstacles:
            reason = "path_blocked"
        elif legitimate_wait:
            reason = "expected_signal_or_queue_wait"
        elif self._efficiency_state["sustained"]:
            reason = "sustained_low_efficiency_following"
        elif routine_following and not urgent_crossing and not context["blocked"]:
            reason = "ordinary_same_lane_following"
        elif self._recovery_state["sustained"]:
            reason = "sustained_native_planner_degradation"
        elif context["special_situations"]:
            reason = "special_situation"
        elif memory["known_failure"]:
            reason = "verified_memory_failure"
        elif memory["supported_intervention_prior"]:
            reason = "memory_supported_intervention"
        elif lease_changed:
            reason = "advisory_conditions_changed"
        elif urgent_crossing:
            reason = "urgent_crossing_conflict"
        elif len(crossing) >= 2:
            reason = "complex_interaction"
        elif context["blocked"]:
            reason = "path_blocked"
        elif baseline is None:
            reason = "no_finite_pdm_proposal"
        elif baseline.get("planning_score", baseline["base_score"]) < self.config["low_quality_threshold"]:
            reason = "low_pdm_quality"
        elif not moving and self._recovery_state["degraded"]:
            reason = "observing_native_planner_degradation"
        elif not moving:
            reason = "no_progressing_proposal"
        elif stalled:
            reason = "unexplained_stall"
        else:
            reason = "simple_cruising"
        eligible = reason not in ("simple_cruising", "expected_signal_or_queue_wait",
                                  "ordinary_same_lane_following", "observing_native_planner_degradation")
        result.update(eligible=eligible, reason=reason, legitimate_wait=bool(legitimate_wait),
                      routine_following=bool(routine_following),
                      waiting_hazard_actor_ids=sorted(a["id"] for a in waiting_hazards),
                      wait_review_due=bool(wait_review_due),
                      wait_elapsed_s=(now_s-self._wait_started_s if self._wait_started_s is not None else 0.0),
                      crossing_conflict_actor_ids=sorted(a["id"] for a in crossing),
                      urgent_crossing_actor_ids=sorted(a["id"] for a in urgent_crossing),
                      baseline_selected_id=audit["baseline_selected_id"],
                      moving_proposal_count=len(moving),
                      candidate_pool_sha256=audit["candidate_pool_sha256"],
                      planner_recovery=deepcopy(self._recovery_state),
                      planner_efficiency=deepcopy(self._efficiency_state),
                      unexpected_stall=bool(stalled and not legitimate_wait))
        if eligible:
            if self._last_query_s is None or now_s + 1e-9 >= result["next_query_time_s"]:
                result["query"] = True
                self._last_query_s = float(now_s)
                result["next_query_time_s"] = now_s + self.config["min_query_interval_s"]
            else:
                result["trigger_reason"] = reason
                result["reason"] = "query_cooldown"
        return result

    def submit(self, semantic, context, now_s):
        error = self._observe(context, now_s)
        self._pending = None
        reason = error or self._semantic_error(semantic)
        if reason:
            self._last_submission = {"accepted": False, "reason": reason}
            return dict(self._last_submission)
        intervention = semantic["intervention"]
        if not intervention["needed"]:
            self._last_submission = {"accepted": True, "active": False, "reason": "model_abstained"}
            return dict(self._last_submission)
        snapshot = deepcopy(context)
        self._pending = {"semantic": deepcopy(semantic), "context": snapshot,
                         "submitted_at_s": float(now_s),
                         "expires_at_s": min(float(now_s), context["sim_time_s"])
                                          + intervention["valid_for_s"]}
        if intervention.get("planner_request", "none") == "expand_candidates":
            # A finite maneuver lease is distinct from the current native
            # health observation. One improving numerical snapshot must not
            # discard the search before the measured car has actually moved.
            self._pending["recovery"] = {"initial_degradation": deepcopy(self._recovery_state),
                "start_route_progress_m": context["route_progress_m"],
                "healthy_since_s": None, "last_healthy_sample_s": None}
        reason, evidence = self._authorization(context, now_s)
        if reason:
            self._pending = None
        self._last_submission = {"accepted": reason is None, "active": reason is None,
                                 "reason": reason or "grounded_submission", "evidence": evidence}
        return deepcopy(self._last_submission)

    @staticmethod
    def _semantic_error(semantic):
        if not isinstance(semantic, dict):
            return "invalid_semantic_object"
        if not isinstance(semantic.get("lateral_decision"), str) or semantic["lateral_decision"] not in _LATERAL:
            return "invalid_lateral_decision"
        if (not isinstance(semantic.get("longitudinal_decision"), str)
                or semantic["longitudinal_decision"] not in _LONGITUDINAL):
            return "invalid_longitudinal_decision"
        intervention = semantic.get("intervention")
        if not isinstance(intervention, dict):
            return "legacy_schema_has_no_intervention_authority"
        if (not _INTERVENTION_FIELDS.issubset(intervention)
                or set(intervention) - _INTERVENTION_FIELDS - {"planner_request"}):
            return "invalid_intervention_fields"
        if type(intervention["needed"]) is not bool:
            return "invalid_intervention_needed"
        if (not isinstance(intervention["reason_code"], str)
                or intervention["reason_code"] not in _REASONS
                or not isinstance(intervention["activation"], str)
                or intervention["activation"] not in _ACTIVATIONS
                or not isinstance(intervention["release"], str)
                or intervention["release"] not in _RELEASES):
            return "invalid_intervention_enum"
        targets = intervention["target_actor_ids"]
        if (not isinstance(targets, list) or len(targets) > 8
                or any(type(item) is not int for item in targets)
                or len(set(targets)) != len(targets)):
            return "invalid_target_actor_ids"
        if (intervention["target_stop_line_id"] is not None
                and (not isinstance(intervention["target_stop_line_id"], str)
                     or not intervention["target_stop_line_id"].strip())):
            return "invalid_target_stop_line_id"
        if not _number(intervention["valid_for_s"]) or not 0.2 <= intervention["valid_for_s"] <= 6:
            return "invalid_intervention_duration"
        if intervention["needed"] == (intervention["reason_code"] == "none"):
            return "inconsistent_intervention_need"
        if not intervention["needed"] and (targets or intervention["target_stop_line_id"] is not None):
            return "unneeded_intervention_has_targets"
        request = intervention.get("planner_request", "none")
        if not isinstance(request, str) or request not in _PLANNER_REQUESTS:
            return "invalid_planner_request"
        if (request == "expand_candidates") != (intervention["reason_code"] == "planner_degraded"):
            return "planner_request_reason_mismatch"
        if request == "expand_candidates":
            if (not intervention["needed"] or targets
                    or intervention["target_stop_line_id"] is not None):
                return "planner_recovery_cannot_target_traffic_entities"
            if (semantic["lateral_decision"] not in ("FOLLOW_ROUTE", "KEEP_LANE")
                    or semantic["longitudinal_decision"] not in ("MAINTAIN", "ACCELERATE", "FOLLOW")
                    or intervention["activation"] != "immediate"
                    or intervention["release"] != "condition_cleared"):
                return "invalid_planner_recovery_semantics"
        return None

    def _authorization(self, context, now_s):
        if self._pending is None:
            return "no_active_intervention", {}
        pending = self._pending
        if now_s >= pending["expires_at_s"]:
            return "intervention_expired", {}
        original = pending["context"]
        intervention = pending["semantic"]["intervention"]
        reason = intervention["reason_code"]
        evidence = {"reason_code": reason, "actor_ids": [], "signal_id": None,
                    "conflict_is_prediction_proxy": True}
        if (context["topology_signature"] != original["topology_signature"]
                or context["navigation"] != original["navigation"]):
            return "navigation_context_changed", evidence
        if (intervention["release"] == "maneuver_complete"
                and context["lane_key"] != original["lane_key"]):
            return "maneuver_completed", evidence
        actors = {actor["id"]: actor for actor in context["actors"]}
        targets = []
        for identity in intervention["target_actor_ids"]:
            actor = actors.get(identity)
            if actor is None:
                return "target_actor_missing", evidence
            if actor["distance_m"] > self.config["max_target_distance_m"]:
                return "target_actor_outside_scope", evidence
            if intervention["release"] == "target_passed" and actor["relative_s_m"] < 0:
                return "target_passed", evidence
            targets.append(actor)
        def conflicts(actor):
            return (actor["conflict"] is True and actor["ttc_s"] is not None
                    and actor["ttc_s"] <= self.config["conflict_ttc_s"])
        relevant = [actor for actor in targets if conflicts(actor) or self._path_blocking(actor)
                    or (actor["same_lane"] and actor["ahead"] and actor["relative_s_m"] >= 0)]
        evidence["actor_ids"] = [actor["id"] for actor in relevant]
        signal = None
        signal_id = intervention["target_stop_line_id"]
        if signal_id is not None:
            signal = next((item for item in context["signals"] if item["id"] == signal_id), None)
            if signal is None or not signal["applicable"]:
                return "signal_not_applicable", evidence
            evidence["signal_id"] = signal_id
            if intervention["release"] == "signal_changed":
                previous = next((item for item in original["signals"] if item["id"] == signal_id), None)
                if previous is None or signal["state"].upper() != previous["state"].upper():
                    return "signal_changed", evidence
        special = {item.lower() for item in context["special_situations"]}
        if reason == "planner_degraded":
            evidence["planner_recovery"] = deepcopy(self._recovery_state)
            recovery = pending.get("recovery") or {}
            if not recovery.get("initial_degradation", {}).get("sustained"):
                return "planner_degradation_not_sustained", evidence
            obstruction = self._recovery_obstruction(context)
            if obstruction:
                return obstruction, evidence
            if context["lane_key"] != original["lane_key"]:
                return "recovery_lane_context_changed", evidence
            health = self._recovery_state.get("native_health")
            if not isinstance(health, dict):
                return "planner_health_unavailable", evidence
            native_healthy = (health["positive_progressing_count"] > 0
                              and health["baseline_stop"] is False
                              and not self._efficiency_state["degraded"])
            sample_s = context["sim_time_s"]
            if native_healthy:
                last = recovery["last_healthy_sample_s"]
                if (recovery["healthy_since_s"] is None or last is None
                        or sample_s-last > self.config["max_context_age_s"]):
                    recovery["healthy_since_s"] = sample_s
                recovery["last_healthy_sample_s"] = sample_s
            else:
                recovery["healthy_since_s"] = None
                recovery["last_healthy_sample_s"] = None
            healthy_duration = (max(0.0, sample_s-recovery["healthy_since_s"])
                                if recovery["healthy_since_s"] is not None else 0.0)
            progress = max(0.0, context["route_progress_m"]-recovery["start_route_progress_m"])
            evidence.update(recovery_authorization=("initial_sustained_degradation"
                if now_s == pending["submitted_at_s"] else "continuing_recovery"),
                native_planner_currently_healthy=bool(native_healthy),
                native_healthy_duration_s=healthy_duration, recovery_progress_m=progress,
                recovery_release_progress_m=self.config["recovery_release_progress_m"],
                recovery_release_healthy_duration_s=self.config["recovery_release_healthy_duration_s"])
            if (native_healthy and healthy_duration + 1e-9 >= self.config["recovery_release_healthy_duration_s"]
                    and progress + 1e-9 >= self.config["recovery_release_progress_m"]):
                return "native_planner_recovery_confirmed", evidence
        elif reason == "interaction":
            if not relevant or not any(conflicts(actor) for actor in relevant):
                return "no_current_interaction_conflict", evidence
        elif reason == "path_blocked":
            slow_target = (self._efficiency_state["sustained"] and any(
                actor["id"] == self._efficiency_state.get("actor_id") for actor in targets))
            if not slow_target and (not context["blocked"] or not any(self._path_blocking(actor) for actor in targets)):
                return "path_blockage_not_grounded", evidence
        elif reason == "traffic_signal":
            if signal is None or signal["state"].upper() not in ("RED", "YELLOW"):
                return "no_current_stop_signal", evidence
            if signal["stop_line_distance_m"] < 0:
                return "stop_line_already_passed", evidence
        elif reason == "visibility":
            if not any(any(word in value for word in ("visibility", "occlu", "fog")) for value in special):
                return "visibility_condition_not_grounded", evidence
        elif reason == "special_rule":
            if not special:
                return "special_rule_not_grounded", evidence
        elif reason == "route_change":
            navigation = context["navigation"].upper()
            if not special and not any(word in navigation for word in ("LEFT", "RIGHT", "TURN", "CHANGE")):
                return "route_change_not_grounded", evidence
        if intervention["activation"] == "while_conflict" and not any(conflicts(actor) for actor in relevant):
            return "conflict_cleared", evidence
        stop_range = max(self.config["stop_distance_m"],
                         context["ego_speed_mps"] * self.config["stop_time_window_s"])
        evidence["stop_activation_distance_m"] = stop_range
        if intervention["activation"] == "approach_stop_line":
            if signal is None or not 0 <= signal["stop_line_distance_m"] <= stop_range:
                return "stop_line_not_within_activation_range", evidence
        near_conflict = [actor for actor in relevant if conflicts(actor)
                         and (actor["distance_m"] <= stop_range
                              or actor["ttc_s"] <= self.config["stop_time_window_s"])]
        near_lead = [actor for actor in relevant
                     if actor["same_lane"] and actor["ahead"] and self._path_blocking(actor)
                     and actor["distance_m"] <= stop_range]
        near_signal = (signal is not None and signal["state"].upper() in ("RED", "YELLOW")
                       and 0 <= signal["stop_line_distance_m"] <= stop_range)
        evidence["stop_or_yield_authorized"] = bool(near_conflict or near_lead or near_signal)
        if pending["semantic"]["longitudinal_decision"] in ("STOP", "YIELD"):
            if not evidence["stop_or_yield_authorized"]:
                return "stop_or_yield_without_local_evidence", evidence
        return None, evidence

    def select(self, candidates, context, now_s, enabled=True, memory_evidence=None):
        error = self._observe(context, now_s)
        baseline, baseline_audit = choose_candidate(candidates, {}, False)
        semantic = deepcopy(self._pending["semantic"]) if self._pending else {}
        legacy, legacy_audit = choose_candidate(candidates, semantic, bool(enabled))
        rejection, evidence = (error, {}) if error else self._authorization(context, now_s)
        if not enabled:
            rejection = "semantic_influence_disabled"
        if rejection:
            self._pending = None
        authorized = rejection is None and baseline is not None
        generation_request = (semantic.get("intervention", {}).get("planner_request") == "expand_candidates")
        bonuses = {}
        regrets = {}
        memory_discounts = {}
        for item in baseline_audit["candidates"]:
            memory_discounts[item["id"]] = 0.0
            regret = (max(0.0, baseline.get("planning_score", baseline["base_score"])
                          - item.get("planning_score", item["base_score"]))
                      if baseline is not None and item["eligible"] else None)
            regrets[item["id"]] = regret
            stop_bonus_allowed = (item.get("longitudinal") not in ("STOP", "YIELD")
                                  or evidence.get("stop_or_yield_authorized") is True)
            if (authorized and not generation_request and stop_bonus_allowed and item["eligible"]
                    and regret <= self.config["base_regret_budget"] + 1e-12):
                raw = (self.config["lateral_weight"]
                       * (semantic.get("lateral_decision") in item.get("lateral_semantics", []))
                       + self.config["longitudinal_weight"]
                       * (semantic.get("longitudinal_decision") == item.get("longitudinal")))
                bounded = min(self.config["max_semantic_bonus"], raw)
                bonuses[item["id"]] = max(0.0, bounded-self._memory_discount(item, memory_evidence))
                memory_discounts[item["id"]] = bounded-bonuses[item["id"]]
            else:
                bonuses[item["id"]] = 0.0
        selected, diagnostics = choose_candidate(candidates, semantic, bool(authorized),
                                                  semantic_bonus_by_id=bonuses)
        for item in diagnostics["candidates"]:
            item["memory_bonus_discount"] = memory_discounts[item["id"]]
            item["semantic_base_regret"] = regrets[item["id"]]
            item["semantic_bonus_withheld_by_regret"] = bool(
                authorized and regrets[item["id"]] is not None
                and regrets[item["id"]] > self.config["base_regret_budget"] + 1e-12)
            item["semantic_stop_bonus_withheld"] = bool(
                authorized and item.get("longitudinal") in ("STOP", "YIELD")
                and evidence.get("stop_or_yield_authorized") is not True)
        changed = bool(selected and baseline and selected["id"] != baseline["id"])
        diagnostics["governor"] = {
            "policy": "selective", "authorized": bool(authorized), "adopted": changed,
            "reason": rejection or ("bounded_preference_adopted" if changed else "baseline_retained"),
            "evidence": evidence, "last_submission": deepcopy(self._last_submission),
            "expires_at_s": self._pending["expires_at_s"] if self._pending else None,
            "base_regret_budget": self.config["base_regret_budget"],
            "max_semantic_bonus": self.config["max_semantic_bonus"],
            "base_regret": regrets[selected["id"]] if selected else None,
            "legacy_selected_id": legacy["id"] if legacy else None,
            "legacy_selected_bonus": legacy["semantic_bonus"] if legacy else None,
            "legacy_candidate_pool_sha256": legacy_audit["candidate_pool_sha256"],
            "memory": _memory_summary(memory_evidence),
            "global_pdm_candidate_gate_applied": False,
            "numerical_planner_continuity_reset": False,
            "planner_recovery": deepcopy(self._recovery_state),
            "candidate_generation_only": bool(generation_request and authorized),
        }
        return selected, diagnostics
