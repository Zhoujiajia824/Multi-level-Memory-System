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
    "wait_review_interval_s": 15.0,
}
_LATERAL = {"FOLLOW_ROUTE", "KEEP_LANE", "CHANGE_LANE_LEFT", "CHANGE_LANE_RIGHT"}
_LONGITUDINAL = {"MAINTAIN", "ACCELERATE", "DECELERATE", "FOLLOW", "YIELD", "STOP"}
_REASONS = {"none", "path_blocked", "interaction", "traffic_signal", "visibility",
            "special_rule", "route_change"}
_ACTIVATIONS = {"immediate", "while_conflict", "approach_stop_line"}
_RELEASES = {"condition_cleared", "target_passed", "signal_changed", "maneuver_complete"}
_INTERVENTION_FIELDS = {"needed", "reason_code", "target_actor_ids", "target_stop_line_id",
                        "activation", "release", "valid_for_s"}
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
                 "wait_review_interval_s"):
        if result[name] <= 0:
            raise ValueError("semantic governor %s must be positive" % name)
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
        self._last_wait_review_s = None
        self._last_submission = {"accepted": False, "reason": "no_submission"}

    def _observe(self, context, now_s):
        if not _number(now_s) or now_s < 0:
            raise ValueError("now_s must be a finite non-negative number")
        if self._last_observed_s is not None and now_s < self._last_observed_s:
            # A restarted simulator cannot inherit another episode's leases.
            self._pending = None
            self._last_query_s = None
            self._progress_anchor = None
            self._wait_started_s = None
            self._last_wait_review_s = None
        self._last_observed_s = float(now_s)
        error = _context_error(context, now_s, self.config["max_context_age_s"])
        if error:
            return error
        if context["baseline_stop_expected"] and context["ego_speed_mps"] <= 0.5:
            if self._wait_started_s is None:
                self._wait_started_s = float(now_s)
        else:
            self._wait_started_s = None
            self._last_wait_review_s = None
        progress = context["route_progress_m"]
        if (context["baseline_stop_expected"]
                or context["ego_speed_mps"] > 0.5 or self._progress_anchor is None
                or abs(progress - self._progress_anchor[1]) >= self.config["stall_progress_m"]):
            self._progress_anchor = (float(now_s), progress)
        return None

    @staticmethod
    def _path_blocking(actor):
        """Require evidence belonging to this actor, not another scene object.

        ``path_blocked`` is produced from the ordered route corridor. A same-lane
        label or a scene-level blocked flag cannot substitute for that evidence.
        Missing fields fail closed only for semantic blockage authorization;
        native PDM proposals remain available.
        """
        return (actor.get("path_blocked") is True and actor.get("stationary") is True
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
            a["ttc_s"] <= self.config["stop_time_window_s"])
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
            score, progress = summary.get("baseline_score"), summary.get("baseline_progress_m")
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
                  and item.get("longitudinal") != "STOP"]
        # blocked may denote an unexpected stationary obstacle, not a queue.
        # Only the separately explained signal/lead-queue flag excuses waiting.
        legitimate_wait = context["baseline_stop_expected"]
        wait_reference = (self._last_wait_review_s if self._last_wait_review_s is not None
                          else self._wait_started_s)
        wait_review_due = (wait_reference is not None and
                           now_s - wait_reference >= self.config["wait_review_interval_s"])
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
                if not lease_changed and not wait_review_due:
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
        if context["special_situations"]:
            reason = "special_situation"
        elif memory["known_failure"]:
            reason = "verified_memory_failure"
        elif memory["supported_intervention_prior"]:
            reason = "memory_supported_intervention"
        elif wait_review_due:
            # A review is consultation only. It neither clears the red light nor
            # labels a legitimate traffic queue as a planner failure.
            reason = "expected_wait_review"
        elif legitimate_wait:
            reason = "expected_signal_or_queue_wait"
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
        elif baseline["base_score"] < self.config["low_quality_threshold"]:
            reason = "low_pdm_quality"
        elif not moving:
            reason = "no_progressing_proposal"
        elif stalled:
            reason = "unexplained_stall"
        else:
            reason = "simple_cruising"
        eligible = reason not in ("simple_cruising", "expected_signal_or_queue_wait")
        result.update(eligible=eligible, reason=reason, legitimate_wait=bool(legitimate_wait),
                      wait_review_due=bool(wait_review_due),
                      wait_elapsed_s=(now_s-self._wait_started_s if self._wait_started_s is not None else 0.0),
                      crossing_conflict_actor_ids=sorted(a["id"] for a in crossing),
                      urgent_crossing_actor_ids=sorted(a["id"] for a in urgent_crossing),
                      baseline_selected_id=audit["baseline_selected_id"],
                      moving_proposal_count=len(moving),
                      candidate_pool_sha256=audit["candidate_pool_sha256"],
                      unexpected_stall=bool(stalled and not legitimate_wait))
        if eligible:
            if self._last_query_s is None or now_s + 1e-9 >= result["next_query_time_s"]:
                result["query"] = True
                self._last_query_s = float(now_s)
                if self._wait_started_s is not None:
                    self._last_wait_review_s = float(now_s)
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
        if set(intervention) != _INTERVENTION_FIELDS:
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
        if reason == "interaction":
            if not relevant or not any(conflicts(actor) for actor in relevant):
                return "no_current_interaction_conflict", evidence
        elif reason == "path_blocked":
            if not context["blocked"] or not any(self._path_blocking(actor) for actor in targets):
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
        bonuses = {}
        regrets = {}
        memory_discounts = {}
        for item in baseline_audit["candidates"]:
            memory_discounts[item["id"]] = 0.0
            regret = (max(0.0, baseline["base_score"] - item["base_score"])
                      if baseline is not None and item["eligible"] else None)
            regrets[item["id"]] = regret
            stop_bonus_allowed = (item.get("longitudinal") not in ("STOP", "YIELD")
                                  or evidence.get("stop_or_yield_authorized") is True)
            if (authorized and stop_bonus_allowed and item["eligible"]
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
        }
        return selected, diagnostics
