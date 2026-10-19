"""Executed outcome support and access boundaries for numeric planning memory."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from carla_bridge.planning.experience_memory import ExecutionExperienceMemory, EXPERIENCE_FILENAME


VERSION = "test_pdm_execution_v1"
ACTION = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "MAINTAIN"}
PARAMETERS = {"target_velocity_mps": 8.0, "headway_time": 1.5,
              "min_gap_to_lead_agent": 2.0, "accel_max": 2.0, "decel_max": 4.0}


def context(**changes):
    value = {"schema_version": 1, "sim_time_s": 10.0, "ego_speed_mps": 3.0,
             "route_progress_m": 40.0, "in_junction": False,
             "lane_key": "24:0:2", "navigation": "right_turn", "speed_limit_mps": 12,
             "topology_signature": "two_lane_to_right_exit_v1", "signals": [],
             "actors": [], "blocked": False, "special_situations": [],
             "baseline_stop_expected": False}
    value.update(changes)
    return value


def telemetry(t=0.0, **changes):
    value = {"sim_time_s": t, "route_progress_m": 40 + 3 * t,
             "ego_speed_mps": 3.0, "collision_count": 0, "violation_count": 0,
             "acceleration_mps2": 0.1, "jerk_mps3": 0.2,
             "baseline_candidate_id": "base", "selected_candidate_id": "actual",
             "actual_action": copy.deepcopy(ACTION), "selected_parameters": copy.deepcopy(PARAMETERS)}
    value.update(changes)
    return value


def memory(bank, run_id="query", **kwargs):
    return ExecutionExperienceMemory(bank, run_id=run_id, planner_version=VERSION,
                                     observation_horizon_s=1.0, **kwargs)


def executed(bank, run_id, *, ctx=None, final=None, initial=None, action=ACTION,
             decision=None, start=0.0):
    m = memory(bank, run_id)
    first = telemetry(start)
    first.update(initial or {})
    first["actual_action"] = copy.deepcopy(action)
    ident = m.begin(ctx or context(), decision or {}, first)
    records = []
    for tick in range(1, 11):
        last = telemetry(start + tick / 10)
        if tick == 10:
            last.update(final or {})
        records.extend(m.observe(last))
    assert len(records) == 1 and records[0]["id"] == ident
    return m, records[0]


def test_memory_off_never_reads_or_creates_bank(tmp_path, monkeypatch):
    bank = tmp_path / "must_not_exist"
    m = memory(bank, memory_mode="memory_off")
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("memory_off performed filesystem read"))
    assert m.begin(context(), {}, telemetry()) is None
    assert m.observe(telemetry(1)) == []
    assert m.finish("interrupted") == []
    result = m.retrieve(context())
    assert result["reason"] == "memory_read_disabled"
    assert not result["bank_health"]["read_enabled"]
    assert not result["bank_health"]["write_enabled"]


def test_read_only_missing_bank_only_writes_separate_run_audit(tmp_path):
    bank = tmp_path / "bank"
    m = memory(bank, memory_mode="read_only", run_dir=tmp_path / "run")
    m.begin(context(), {}, telemetry())
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        m.observe(telemetry(t))
    m.finish("route_complete")
    assert m.retrieve(context())["status"] == "unknown"
    assert not bank.exists()
    assert (tmp_path / "run" / "planning_experience_audit.jsonl").is_file()
    assert m.summarize()["persisted_updates"] == 0


def test_read_only_existing_bank_is_byte_identical(tmp_path):
    executed(tmp_path / "bank", "previous")
    path = tmp_path / "bank" / EXPERIENCE_FILENAME
    before = path.read_bytes()
    m = memory(tmp_path / "bank", memory_mode="read_only", run_dir=tmp_path / "audit")
    m.retrieve(context())
    m.begin(context(), {}, telemetry())
    m.observe(telemetry(0.2))
    m.finish("interrupted")
    assert path.read_bytes() == before
    with pytest.raises(PermissionError):
        m.import_paired_evidence({})


def test_audit_must_not_modify_bank_through_a_read_only_back_door(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        memory(tmp_path / "bank", memory_mode="read_only", run_dir=tmp_path / "bank" / "run")


def test_read_and_write_switches_are_independent(tmp_path):
    m = memory(tmp_path / "write_only", read_enabled=False)
    m.begin(context(), {}, telemetry())
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        m.observe(telemetry(t))
    assert (tmp_path / "write_only" / EXPERIENCE_FILENAME).is_file()
    assert m.retrieve(context())["reason"] == "memory_read_disabled"
    no_write = memory(tmp_path / "no_write", write_enabled=False)
    assert no_write.begin(context(), {}, telemetry()) is None
    assert not (tmp_path / "no_write").exists()


def test_safe_execution_never_claims_causal_semantic_benefit(tmp_path):
    for run in ("a", "b", "c"):
        _, record = executed(tmp_path, run)
        assert record["status"] == "observed"
        assert record["outcome"]["counterfactual_executed"] is False
    result = memory(tmp_path).retrieve(context())
    assert result["status"] == "supported"
    assert result["independent_run_count"] == 3
    assert result["score_residual"] == 0
    assert result["causal_advantage"]["status"] == "unknown"
    assert result["causal_advantage"]["estimate"] is None
    assert result["risk_feedback"]["known_failure"] is False
    prior = result["parameter_priors"][0]
    assert prior["parameters"] == PARAMETERS and prior["action"] == ACTION
    assert prior["support_count"] == 3 and prior["causal_gain"] == "unknown"


def test_no_support_from_current_run_or_dense_adjacent_windows(tmp_path):
    for start in (0, 2, 4):
        executed(tmp_path, "same", start=start)
    result = memory(tmp_path, "same").retrieve(context())
    assert result["support_count"] == 0
    assert result["bank_health"]["current_run_excluded"] == 3
    result = memory(tmp_path).retrieve(context())
    assert result["status"] == "unknown"
    assert result["independent_run_count"] == 1
    assert result["bank_health"]["adjacent_excluded"] == 2
    assert result["parameter_priors"] == []


def test_even_distant_windows_from_one_run_are_not_independent_evidence(tmp_path):
    for start in (0, 20, 40):
        executed(tmp_path, "same", start=start)
    result = memory(tmp_path).retrieve(context())
    assert result["support_count"] == 3 and result["independent_run_count"] == 1
    assert result["status"] == "unknown" and result["parameter_priors"] == []


def test_actor_relationships_not_ids_and_unrelated_opposing_traffic(tmp_path):
    actor = {"id": 12, "same_lane": True, "ahead": True, "distance_m": 20,
             "ttc_s": 6, "type": "vehicle", "conflict": False}
    for run in ("a", "b", "c"):
        executed(tmp_path, run, ctx=context(actors=[actor]))
    query_actor = dict(actor, id=999)
    irrelevant = {"id": 11, "same_lane": False, "ahead": True,
                  "distance_m": 3, "type": "vehicle", "conflict": False}
    result = memory(tmp_path).retrieve(context(actors=[query_actor, irrelevant]))
    assert result["status"] == "supported"
    assert memory(tmp_path).retrieve(context(actors=[dict(query_actor, conflict=True)]))["support_count"] == 0


def test_stop_line_distance_prevents_at_line_experience_matching_distant_red_light(tmp_path):
    red = {"id": 1, "state": "Red", "applicable": True, "stop_line_distance_m": 2}
    for run in ("a", "b", "c"):
        executed(tmp_path, run, ctx=context(signals=[red]))
    m = memory(tmp_path)
    assert m.retrieve(context(signals=[dict(red, id=999)]))["status"] == "supported"
    result = m.retrieve(context(signals=[dict(red, stop_line_distance_m=45)]))
    assert result["support_count"] == 0 and result["parameter_priors"] == []


@pytest.mark.parametrize("changes", [
    {"topology_signature": "different_exit"}, {"topology_signature": None},
    {"navigation": "left_turn"}, {"ego_speed_mps": 10}, {"in_junction": True},
    {"blocked": True}, {"baseline_stop_expected": True},
    {"signals": [{"id": 1, "state": "Red", "applicable": True}]},
    {"schema_version": 2},
])
def test_incompatible_context_returns_unknown_not_a_stop_policy(tmp_path, changes):
    for run in ("a", "b", "c"):
        executed(tmp_path, run)
    result = memory(tmp_path).retrieve(context(**changes), "STOP")
    assert result["status"] == "unknown" and result["score_residual"] == 0
    assert result["parameter_priors"] == []


def test_actual_failure_produces_only_bounded_action_specific_negative_feedback(tmp_path):
    stop = {"lateral_decision": "FOLLOW_ROUTE", "longitudinal_decision": "STOP"}
    for run in ("a", "b", "c"):
        _, record = executed(tmp_path, run, action=stop, final={"collision_count": 1})
        assert record["status"] == "failed"
    m = memory(tmp_path)
    stop_result = m.retrieve(context(), "STOP")
    assert stop_result["score_residual"] == -0.15
    assert stop_result["risk_feedback"]["known_failure"] is True
    assert stop_result["risk_feedback"]["verified"] is True
    assert m.retrieve(context(), ACTION)["score_residual"] == 0
    scene_result = m.retrieve(context())
    assert scene_result["score_residual"] == 0
    assert scene_result["risk_feedback"]["scope"] == "scene"
    assert scene_result["parameter_priors"] == []
    assert stop_result["causal_advantage"]["status"] == "unknown"


def test_rejected_stop_request_is_not_mislabelled_as_executed_stop(tmp_path):
    for run in ("a", "b", "c"):
        executed(tmp_path, run, decision={"requested_action": {"longitudinal_decision": "STOP"},
                 "adopted": False, "rejection_reason": "unsupported_stop"},
                 final={"collision_count": 1})
    assert memory(tmp_path).retrieve(context(), "STOP")["support_count"] == 0
    assert memory(tmp_path).retrieve(context(), "MAINTAIN")["risk_feedback"]["known_failure"] is True


def test_missing_actual_action_never_falls_back_to_the_requested_action(tmp_path):
    for run in ("a", "b", "c"):
        executed(tmp_path, run, action=None, decision={"requested_action": ACTION})
    result = memory(tmp_path).retrieve(context())
    assert result["support_count"] == 3 and result["parameter_priors"] == []
    assert result["action_feedback"] == []


@pytest.mark.parametrize("final", [
    {"acceleration_mps2": 27}, {"jerk_mps3": 299},
    {"collision_count": 1}, {"violation_count": 1},
])
def test_bad_executed_outcome_cannot_become_successful_parameter_prior(tmp_path, final):
    for run in ("a", "b", "c"):
        executed(tmp_path, run, final=final)
    assert memory(tmp_path).retrieve(context())["parameter_priors"] == []


def test_counter_reset_or_measurement_gap_censors_instead_of_inventing_safety(tmp_path):
    m = memory(tmp_path, "record")
    m.begin(context(), {}, telemetry(collision_count=4))
    m.observe(telemetry(0.1, collision_count=4))
    records = m.observe(telemetry(1.0, collision_count=0))
    assert records[0]["status"] == "censored"
    assert records[0]["outcome"]["measurement_coverage_complete"] is False
    assert memory(tmp_path).retrieve(context())["support_count"] == 0


def test_missing_counter_or_comfort_measurement_does_not_validate_parameters(tmp_path):
    for run in ("a", "b", "c"):
        m = memory(tmp_path, run)
        m.begin(context(), {}, telemetry())
        for tick in range(1, 11):
            m.observe(telemetry(tick / 10, collision_count=None, jerk_mps3=None))
    assert memory(tmp_path).retrieve(context())["parameter_priors"] == []


def test_interruption_and_short_success_are_censored(tmp_path):
    for status in ("interrupted", "route_complete", "timeout", "error"):
        m = memory(tmp_path / status, "record")
        m.begin(context(), {}, telemetry())
        m.observe(telemetry(0.2))
        records = m.finish(status, telemetry(0.4))
        assert records[0]["status"] == "censored"
        assert records[0]["outcome"]["causal_gain"] == "unknown"


def test_policy_change_censors_before_new_policy_measurements_are_added(tmp_path):
    m = memory(tmp_path)
    m.begin(context(), {}, telemetry())
    m.observe(telemetry(0.2))
    records = m.observe(telemetry(0.3, collision_count=1, execution_policy_changed=True))
    assert records[0]["status"] == "censored"
    assert records[0]["outcome"]["duration_s"] == pytest.approx(0.2)
    assert records[0]["outcome"]["collision_observed"] is False
    assert not m.pending


def test_pending_crash_record_is_not_retrieved_as_a_completed_experience(tmp_path):
    m = memory(tmp_path, "crash")
    m.begin(context(), {}, telemetry())
    result = memory(tmp_path).retrieve(context())
    assert result["support_count"] == 0
    assert result["bank_health"]["loaded_records"] == 1


def test_rejected_trajectory_is_not_recorded_as_an_executed_collision_failure(tmp_path):
    m = memory(tmp_path)
    m.begin(context(), {}, telemetry())
    m.observe(telemetry(0.2))
    record = m.finish("invalid_trajectory")[0]
    assert record["status"] == "censored"
    assert record["outcome"]["verified_failure"] is False


def test_malformed_unknown_version_and_truncated_rows_are_explicitly_reported(tmp_path):
    executed(tmp_path, "good")
    path = tmp_path / EXPERIENCE_FILENAME
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"id":"unknown","schema_version":2,"planner_version":"other"}\n')
        stream.write('{"id":"incomplete","schema_version":1,"planner_version":"' + VERSION + '","kind":"execution","context":{}}\n')
        stream.write('{"truncated":\n')
    result = memory(tmp_path).retrieve(context())
    assert result["support_count"] == 1
    health = result["bank_health"]
    assert health["malformed_lines"] == 2 and health["incompatible_records"] == 1
    assert len(health["errors"]) == 3
    assert health["trained_model"] is False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pair(unit="one", gain=1.0):
    initial = {"ego": {"position": [1, 2, 0], "velocity": [3, 0, 0]},
               "actors": [], "signals": [], "map": "Town05_opendrive_hash"}
    state_hash = digest(initial)
    arm = {"evaluation_kind": "executed_closed_loop", "independent_evaluation": True,
           "complete": True, "planner_version": VERSION, "state_fingerprint": state_hash,
           "initial_state": initial, "duration_s": 10.0,
           "metrics": {"collision_count": 0, "violation_count": 0, "route_progress_m": 20},
           "actual_action": copy.deepcopy(ACTION)}
    return {"schema_version": 1, "planner_version": VERSION,
            "experimental_unit_id": unit, "protocol_id": "heldout_controlled_protocol_v1",
            "state_fingerprint": state_hash, "matching_method": "restored_full_state",
            "initial_state": initial, "independently_sampled": True,
            "protocol_fixed_before_evaluation": True, "utility_definition": "predeclared progress utility in [0,1]",
            "context": context(),
            "baseline": dict(copy.deepcopy(arm), run_id="base_" + unit, utility=0.0,
                             artifact_sha256=digest([unit, "base"])),
            "intervention": dict(copy.deepcopy(arm), run_id="intervention_" + unit, utility=gain,
                                 artifact_sha256=digest([unit, "intervention"]))}


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(planner_version="different"),
    lambda p: p.update(independently_sampled="true"),
    lambda p: p.update(protocol_fixed_before_evaluation=False),
    lambda p: p["intervention"].update(evaluation_kind="pdm_simulated_counterfactual"),
    lambda p: p["intervention"].update(independent_evaluation=False),
    lambda p: p["intervention"].update(utility=1.1),
    lambda p: p["intervention"].update(complete=False),
    lambda p: p["intervention"].update(state_fingerprint="different"),
    lambda p: p["intervention"]["initial_state"]["ego"].update(position=[100, 200, 0]),
    lambda p: p["intervention"].update(run_id=p["baseline"]["run_id"]),
    lambda p: p["intervention"].update(artifact_sha256=p["baseline"]["artifact_sha256"]),
    lambda p: p["intervention"].update(duration_s=9),
    lambda p: p["intervention"].update(actual_action=None),
])
def test_strict_pair_import_rejects_unsupported_counterfactual_or_matching_claims(tmp_path, mutate):
    evidence = pair()
    mutate(evidence)
    with pytest.raises(ValueError):
        memory(tmp_path).import_paired_evidence(evidence)
    assert not (tmp_path / EXPERIENCE_FILENAME).exists()


def test_paired_positive_gain_requires_a_positive_bound_not_only_one_good_trial(tmp_path):
    m = memory(tmp_path)
    m.import_paired_evidence(pair("one"))
    advantage = m.retrieve(context(), ACTION)["causal_advantage"]
    assert advantage["estimate"] == 1
    assert advantage["status"] == "unknown" and advantage["lower_bound"] < 0
    for i in range(1, 10):
        m.import_paired_evidence(pair(str(i)))
    advantage = m.retrieve(context(), ACTION)["causal_advantage"]
    assert advantage["status"] == "supported" and 0 < advantage["lower_bound"] < 1
    assert advantage["support_count"] == 10
    assert m.retrieve(context(), "STOP")["causal_advantage"]["status"] == "unknown"
    assert m.retrieve(context())["causal_advantage"]["status"] == "unknown"


def test_scene_consultation_priors_require_action_specific_paired_execution_benefit(tmp_path):
    m = memory(tmp_path)
    for i in range(10):
        m.import_paired_evidence(pair(str(i)))
    result = m.retrieve(context())
    assert result["causal_advantage"]["status"] == "unknown"
    assert len(result["intervention_priors"]) == 1
    prior = result["intervention_priors"][0]
    assert prior["action"] == ACTION
    assert prior["causal_advantage"]["status"] == "supported"
    assert prior["causal_advantage"]["lower_bound"] > 0
    assert prior["causal_advantage"]["support_count"] == 10
    assert prior["use"] == "consultation_only_recheck_requested_action_before_adoption"
    # Consultation evidence for MAINTAIN does not provide a STOP benefit.
    assert m.retrieve(context(), "STOP")["causal_advantage"]["status"] == "unknown"


def test_ordinary_safe_execution_cannot_create_consultation_benefit_priors(tmp_path):
    for run in ("a", "b", "c"):
        executed(tmp_path, run)
    result = memory(tmp_path).retrieve(context())
    assert result["parameter_priors"]
    assert result["intervention_priors"] == []
    assert result["causal_advantage"]["status"] == "unknown"


def test_one_successful_pair_cannot_create_scene_consultation_prior(tmp_path):
    m = memory(tmp_path)
    m.import_paired_evidence(pair())
    assert m.retrieve(context())["intervention_priors"] == []


def test_consultation_priors_do_not_mix_actions_and_correct_for_inspected_actions(tmp_path):
    m = memory(tmp_path)
    for action in ("MAINTAIN", "DECELERATE"):
        for i in range(10):
            evidence = pair(action + str(i))
            evidence["intervention"]["actual_action"]["longitudinal_decision"] = action
            m.import_paired_evidence(evidence)
    result = m.retrieve(context())
    assert len(result["intervention_priors"]) == 2
    for prior in result["intervention_priors"]:
        advantage = prior["causal_advantage"]
        assert advantage["support_count"] == 10
        assert advantage["simultaneously_inspected_actions"] == 2
        assert advantage["multiplicity_correction"] == "bonferroni"
        action_specific = m.retrieve(context(), prior["action"])["causal_advantage"]
        assert 0 < advantage["lower_bound"] < action_specific["lower_bound"]


def test_consultation_prior_never_combines_separate_protocols_to_reach_support(tmp_path):
    m = memory(tmp_path)
    for i in range(10):
        evidence = pair(str(i))
        evidence["protocol_id"] = "protocol_" + str(i)
        m.import_paired_evidence(evidence)
    assert m.retrieve(context())["intervention_priors"] == []


def test_consultation_prior_excludes_current_evaluation_run_and_different_topology(tmp_path):
    writer = memory(tmp_path)
    for i in range(6):
        writer.import_paired_evidence(pair(str(i)))
    # Six perfect pairs support a positive bound; exclusion reduces this to
    # five, where the conservative bound remains below zero.
    assert writer.retrieve(context())["intervention_priors"]
    assert memory(tmp_path, run_id="base_0").retrieve(context())["intervention_priors"] == []
    assert writer.retrieve(context(topology_signature="different"))["intervention_priors"] == []


def test_mutating_consultation_prior_does_not_change_cached_evidence(tmp_path):
    m = memory(tmp_path)
    for i in range(10):
        m.import_paired_evidence(pair(str(i)))
    m.retrieve(context())["intervention_priors"][0]["action"]["longitudinal_decision"] = "STOP"
    assert m.retrieve(context())["intervention_priors"][0]["action"] == ACTION


def test_duplicate_units_and_reused_evaluation_runs_do_not_inflate_causal_evidence(tmp_path):
    m = memory(tmp_path)
    original = pair()
    for i in range(10):
        evidence = copy.deepcopy(original)
        evidence["experimental_unit_id"] = str(i)
        m.import_paired_evidence(evidence)
    advantage = m.retrieve(context(), ACTION)["causal_advantage"]
    assert advantage["support_count"] == 1 and advantage["status"] == "unknown"


def test_more_progress_cannot_offset_worse_executed_collision_outcomes(tmp_path):
    m = memory(tmp_path)
    for i in range(10):
        evidence = pair(str(i))
        evidence["intervention"]["metrics"]["collision_count"] = 1
        m.import_paired_evidence(evidence)
    advantage = m.retrieve(context(), ACTION)["causal_advantage"]
    assert advantage["estimate"] == 0 and advantage["status"] == "unknown"


def test_tampered_pair_summary_cannot_bypass_strict_import_checks(tmp_path):
    m = memory(tmp_path)
    record = m.import_paired_evidence(pair())
    record["utility_difference"] = 0.5
    with (tmp_path / EXPERIENCE_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    result = memory(tmp_path).retrieve(context(), ACTION)
    # An invalid later update invalidates that ID instead of reviving old data.
    assert result["bank_health"]["malformed_lines"] == 1
    assert "disagrees" in result["bank_health"]["errors"][0]
    assert result["causal_advantage"]["support_count"] == 0


def test_tampered_pair_metrics_type_is_reported_as_a_bad_line(tmp_path):
    m = memory(tmp_path)
    record = m.import_paired_evidence(pair())
    record["paired_evidence"]["baseline"]["metrics"] = "invalid"
    with (tmp_path / EXPERIENCE_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    result = m.retrieve(context(), ACTION)
    assert result["bank_health"]["malformed_lines"] == 1
    assert result["causal_advantage"]["support_count"] == 0


def test_causal_bounds_do_not_mix_different_protocols_or_utility_definitions(tmp_path):
    m = memory(tmp_path)
    for i in range(10):
        evidence = pair(str(i))
        evidence["protocol_id"] = "different_protocol_" + str(i)
        m.import_paired_evidence(evidence)
    advantage = m.retrieve(context(), ACTION)["causal_advantage"]
    assert advantage["support_count"] == 1 and advantage["status"] == "unknown"


def test_mutating_retrieval_output_cannot_change_the_cached_bank(tmp_path):
    for run in ("a", "b", "c"):
        executed(tmp_path, run)
    m = memory(tmp_path)
    result = m.retrieve(context())
    result["parameter_priors"][0]["action"]["longitudinal_decision"] = "STOP"
    assert m.retrieve(context())["parameter_priors"][0]["action"] == ACTION


@pytest.mark.parametrize("changes", [{"actors": None}, {"signals": ["bad"]},
                                      {"special_situations": None}])
def test_broken_context_relationship_arrays_do_not_crash_or_match_empty_scene(tmp_path, changes):
    executed(tmp_path, "previous")
    assert memory(tmp_path).retrieve(context(**changes))["status"] == "unknown"


def test_cache_refreshes_after_another_run_appends(tmp_path):
    m = memory(tmp_path)
    assert m.retrieve(context())["support_count"] == 0
    executed(tmp_path, "a")
    assert m.retrieve(context())["support_count"] == 1
    executed(tmp_path, "b")
    assert m.retrieve(context())["support_count"] == 2


@pytest.mark.parametrize("kwargs", [
    {"memory_mode": "unknown"}, {"min_support": 1}, {"min_support": True},
    {"max_score_residual": 10}, {"observation_horizon_s": float("nan")},
    {"planner_version": ""},
])
def test_invalid_configuration_is_rejected(tmp_path, kwargs):
    options = {"planner_version": VERSION, **kwargs}
    with pytest.raises(ValueError):
        ExecutionExperienceMemory(tmp_path, **options)
