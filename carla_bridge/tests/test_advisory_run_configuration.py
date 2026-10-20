"""Run access, audit labels and experiment reporting for selective advice.

No simulator, model API or model cache is needed for these integration checks.
"""
import csv
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from carla_bridge.inference_mode import model_inference_enabled
from carla_bridge.metrics.run_reporter import write_report
from carla_bridge.run_session import BankLock, RunSession
from src.vla_memory.common.config import Config


def planning(**overrides):
    return {"backend": "pdm_closed", "semantic_influence": False,
            "model_inference": False, "semantic_policy": "selective",
            "planning_memory": "full", **overrides}


def session_at(tmp_path, *, plan=None, mode="memory_on", **kwargs):
    scenario = tmp_path / "R05_multilane_eval_s101.yaml"
    scenario.write_text("scenario_name: R05_multilane_eval_s101\n", encoding="utf-8")
    cfg = Config(overrides={"output_dir": str(tmp_path / "o"),
                            "carla": {"planning": planning() if plan is None else plan}})
    return RunSession(cfg, scenario, mode, **kwargs)


@pytest.mark.parametrize("policy", ["selective", "always", "shadow"])
def test_no_model_full_numeric_memory_is_explicitly_supported(policy):
    assert model_inference_enabled(planning(semantic_policy=policy), "memory_on") is False


@pytest.mark.parametrize("memory", [None, "off", "gate"])
def test_no_model_memory_on_requires_explicit_full(memory):
    cfg = planning()
    if memory is None:
        cfg.pop("planning_memory")
    else:
        cfg["planning_memory"] = memory
    with pytest.raises(ValueError, match="planning-memory full"):
        model_inference_enabled(cfg, "memory_on")


@pytest.mark.parametrize("key,value", [
    ("semantic_policy", None), ("semantic_policy", "learned"),
    ("semantic_policy", True), ("semantic_policy", []),
    ("planning_memory", None), ("planning_memory", "read_only"),
    ("planning_memory", True), ("planning_memory", []),
])
def test_new_controls_reject_unknown_or_mistyped_values(key, value):
    with pytest.raises(ValueError, match=key):
        model_inference_enabled(planning(**{key: value}), "memory_off")


def test_readonly_numeric_only_bank_is_accepted_locked_and_unchanged(tmp_path):
    bank = tmp_path / "b"
    bank.mkdir()
    experience = bank / "planning_experiences.jsonl"
    experience.write_text('{"experience_id":"fixture"}\n', encoding="utf-8")
    before = experience.read_bytes()
    session = session_at(tmp_path, memory_bank=str(bank), memory_access="read_only")
    try:
        session.prepare()
        with pytest.raises(RuntimeError, match="其他运行"):
            BankLock(bank).acquire()
        cfg = session.config
        assert cfg.get("memory_access") == {"retrieve": True, "write": False, "short_term": False}
        assert cfg.get_nested("mid_term", "persistence") == {
            "enabled": False, "auto_load_on_init": False, "save_on_close": False,
            "strict_load": True}
        manifest = json.loads((session.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["semantic_policy"] == "selective"
        assert manifest["planning_memory"] == manifest["planning_memory_effective"] == "full"
        assert manifest["semantic_schema_version"] == 2
        assert manifest["semantic_prompt_version"] == 8
        assert not (bank / "assets").exists()
    finally:
        session.close()
    assert experience.read_bytes() == before
    assert sorted(path.name for path in bank.iterdir()) == ["planning_experiences.jsonl"]
    BankLock(bank).acquire().close()


def test_numeric_only_seed_can_be_copied_without_text_metadata(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    source = seed / "planning_experiences.jsonl"
    source.write_text('{"experience_id":"fixture"}\n', encoding="utf-8")
    session = session_at(tmp_path, memory_seed=str(seed))
    try:
        session.prepare()
        copied = session.config.get("memory_db_dir") / source.name
        assert copied != source and copied.read_bytes() == source.read_bytes()
        assert not (copied.parent / "mid_term_meta.json").exists()
        assert session.config.get("memory_access")["write"] is True
    finally:
        session.close()
    assert list(seed.iterdir()) == [source]


def test_model_enabled_still_requires_text_metadata_for_readonly_bank(tmp_path):
    bank = tmp_path / "b"
    bank.mkdir()
    (bank / "planning_experiences.jsonl").write_text("{}\n", encoding="utf-8")
    session = session_at(tmp_path, plan=planning(model_inference=True),
                         memory_bank=str(bank), memory_access="read_only")
    with pytest.raises(ValueError, match="不完整"):
        session.prepare()
    BankLock(bank).acquire().close()


def test_memory_off_overrides_full_and_does_not_create_bank(tmp_path):
    session = session_at(tmp_path, mode="memory_off")
    try:
        session.prepare()
        assert session.manifest["planning_memory"] == "full"
        assert session.manifest["planning_memory_effective"] == "off"
        assert session.config.get("memory_access") == {
            "retrieve": False, "write": False, "short_term": False}
        assert session.lock is None
        assert not session.config.get("memory_db_dir").exists()
    finally:
        session.close()


def test_configs_without_new_controls_retain_old_labels_and_schema(tmp_path):
    session = session_at(tmp_path, mode="memory_off", plan={"backend": "pdm_closed",
                         "semantic_influence": False, "model_inference": False})
    try:
        session.prepare()
        assert "memory_off__pdm_closed__sem_off__model_off" in session.run_dir.name
        assert "policy_" not in session.run_dir.name and "__pm_" not in session.run_dir.name
        assert session.manifest["semantic_policy"] == "always"
        assert session.manifest["planning_memory"] == "off"
        assert session.manifest["semantic_schema_version"] == 1
    finally:
        session.close()


@pytest.mark.skipif(os.name != "nt", reason="native Windows MAX_PATH compatibility")
def test_windows_new_mode_labels_reserve_space_for_actual_report_names(tmp_path):
    scenario_name = "R05_multilane_eval_s101"
    # Deep enough to require compaction even with a short pytest fixture root.
    target = tmp_path / ("deep" * 9)
    target.mkdir()
    session = session_at(target, mode="memory_off")
    try:
        session.prepare()
        longest = session.run_dir / ("decisions_carla_" + scenario_name + "_memory_off.jsonl")
        assert len(str(longest)) < 260
        assert session.run_id[:8] in session.run_dir.name
        paths = write_report({}, {"scenario": scenario_name, "mode": "memory_off"},
                             str(session.run_dir))
        assert all(Path(path).is_file() for path in paths)
        assert "R05_multilane_eval_s101" in session.manifest["scenario_path"]
    finally:
        session.close()


@pytest.mark.parametrize("mode,control,expected", [
    ("memory_on", None, "native_lqr"),
    ("memory_on", "pure_pursuit", "pure_pursuit"),
    ("memory_off", "native_lqr", "native_lqr"),
    ("memory_off", "pure_pursuit", "pure_pursuit"),
])
@pytest.mark.parametrize("preview", [None, "fixed", "speed_scaled"])
@pytest.mark.parametrize("quality,maneuver", [(None, None), ("on", "on"), ("off", "off"), ("on", "off"), ("off", "on")])
def test_cli_carries_shadow_and_numeric_memory_without_model_api_keys(tmp_path, monkeypatch,
                                                                     mode, control, expected, preview,
                                                                     quality, maneuver):
    from carla_bridge import run_carla_demo as entry

    scenario = tmp_path / "s.yaml"
    scenario.write_text("scenario_name: numeric_cli\n", encoding="utf-8")
    configs = []

    def load_config(*, overrides):
        cfg = Config(overrides={**overrides, "output_dir": str(tmp_path / "o")})
        for block in ("scene_understanding", "decision"):
            cfg.data.setdefault(block, {})["api_key_env"] = "ADVISORY_TEST_ABSENT_KEY"
        configs.append(cfg)
        return cfg

    monkeypatch.delenv("ADVISORY_TEST_ABSENT_KEY", raising=False)
    monkeypatch.setattr(entry, "load_config", load_config)
    monkeypatch.setattr(entry, "set_log_directory", Mock())
    monkeypatch.setattr(entry, "setup_logger", Mock())
    loop = Mock()
    loop.setup.side_effect = RuntimeError("reached numeric setup")
    loop.outcome.return_value = {}
    monkeypatch.setattr(entry, "ClosedLoop", Mock(return_value=loop))
    argv = ["run_carla_demo", "--scenario", str(scenario),
        "--planner", "pdm_closed", "--mode", mode, "--semantic-influence", "off",
        "--model-inference", "off", "--semantic-policy", "shadow", "--planning-memory", "full",
        "--planner-recovery", "off"]
    if control is not None:
        argv.extend(["--pdm-lateral-control", control])
    if preview is not None:
        argv.extend(["--pdm-lateral-preview", preview])
    if quality is not None:
        argv.extend(["--lane-quality", quality, "--lane-maneuver", maneuver])
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(RuntimeError, match="reached numeric setup"):
        entry.main()
    cfg = configs[0]
    assert cfg.get_nested("carla", "planning", "semantic_policy") == "shadow"
    assert cfg.get_nested("carla", "planning", "planning_memory") == "full"
    assert cfg.get_nested("carla", "planning", "recovery", "enabled") is False
    assert cfg.get_nested("carla", "planning", "lateral_control") == expected
    default_planning = yaml.safe_load((Path(entry.__file__).parent / "config" / "carla.yaml").read_text(encoding="utf-8"))["carla"]["planning"]
    assert cfg.get_nested("carla", "planning", "lqr_lateral_preview", default="fixed") == (preview or default_planning.get("lqr_lateral_preview", "fixed"))
    assert cfg.get_nested("carla", "planning", "lane_quality", "enabled") is (quality != "off")
    assert cfg.get_nested("carla", "planning", "lane_maneuver", "enabled") is (maneuver != "off")
    assert cfg.get_nested("carla", "planning", "model_inference") is False
    assert cfg.get_nested("carla", "planning", "semantic_influence") is False
    assert cfg.get_nested("mid_term", "persistence", "enabled") is False
    loop.close.assert_called_once()


def test_repository_default_uses_the_same_native_lateral_controller_for_all_semantic_modes():
    path = Path(__file__).resolve().parents[1] / "config" / "carla.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["carla"]["planning"]["lateral_control"] == "native_lqr"


@pytest.mark.parametrize("control,preview", [(None, None), ("native_lqr", "speed_scaled"), ("pure_pursuit", "fixed")])
def test_manifest_records_lateral_controller_including_legacy_default(tmp_path, control, preview):
    cfg = planning()
    if control is not None:
        cfg["lateral_control"] = control
        cfg["lqr_lateral_preview"] = preview
    session = session_at(tmp_path, plan=cfg, mode="memory_off")
    try:
        session.prepare()
        manifest = json.loads((session.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["pdm_lateral_control"] == (control or "pure_pursuit")
        assert manifest["pdm_lateral_preview"] == (preview or "fixed")
        assert manifest["model_inference"] is False
    finally:
        session.close()


def test_csv_keeps_controller_configuration_and_leaves_unknown_legacy_runs_blank(tmp_path):
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases

    scenario = yaml.safe_dump(cases("eval")["R05_multilane_eval_s101"])
    for name, configured, recorded in (
        ("native", "native_lqr", None), ("pp", "pure_pursuit", None),
        ("manifest_only", None, "native_lqr"), ("legacy_unknown", None, None),
        ("config_precedence", "pure_pursuit", "native_lqr"),
    ):
        directory = tmp_path / "runs" / name
        directory.mkdir(parents=True)
        cfg = planning()
        if configured is not None:
            cfg["lateral_control"] = configured
            cfg["lqr_lateral_preview"] = "speed_scaled" if configured == "native_lqr" else "fixed"
        manifest = {"run_id": name, "status": "finished", "mode": "memory_off",
                    "scenario_yaml": scenario, "config": {"carla": {"planning": cfg}}}
        if recorded is not None:
            manifest["pdm_lateral_control"] = recorded
            manifest["pdm_lateral_preview"] = "speed_scaled"
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "lateral.csv"
    assert summarize_runs(tmp_path / "runs", output) == 5
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert rows["native"]["pdm_lateral_control"] == "native_lqr"
    assert rows["pp"]["pdm_lateral_control"] == "pure_pursuit"
    assert rows["manifest_only"]["pdm_lateral_control"] == "native_lqr"
    assert rows["legacy_unknown"]["pdm_lateral_control"] == ""
    assert rows["config_precedence"]["pdm_lateral_control"] == "pure_pursuit"
    assert rows["native"]["pdm_lateral_preview"] == "speed_scaled"
    assert rows["pp"]["pdm_lateral_preview"] == "fixed"
    assert rows["manifest_only"]["pdm_lateral_preview"] == "speed_scaled"
    assert rows["legacy_unknown"]["pdm_lateral_preview"] == ""
    assert rows["config_precedence"]["pdm_lateral_preview"] == "fixed"


def test_report_labels_shadow_and_stop_attribution_as_a_proxy(tmp_path):
    advisory = {"model_queries": 2, "query_checks_skipped": 8, "model_failures": 1,
        "inference_wall_time_s": 20.5, "wall_time_s": 60,
        "semantic_changed_plans": 0, "semantic_rejected_plans": 4,
        "shadow_changed_plans": 3, "semantic_stop_proxy_s": .4}
    info = {"scenario": "test", "mode": "memory_off", "semantic_policy": "shadow",
        "planning_memory": "full", "model_inference": False, "duration_s": 10,
        "pdm_replan_interval_s": .2, "advisory": advisory,
        "lane_quality_enabled": True, "lane_maneuver_enabled": False}
    md, report = write_report({"route_completion": 1.0}, info, str(tmp_path))
    text = Path(md).read_text(encoding="utf-8")
    assert "实际执行基础数值规划" in text and "不是已执行结果" in text
    assert "额外停车归因代理: 0.40 s" in text
    assert "不是无必要停车真值" in text
    assert "模型同步查询期间仿真冻结" in text
    assert "20.50 s" in text and "60.00 s" in text and "None s" not in text
    assert "车道位置软成本: True" in text and "持续绕行候选: False" in text
    assert json.loads(Path(report).read_text(encoding="utf-8"))["run_info"]["advisory"] == advisory


@pytest.mark.parametrize("quality,maneuver", [(True, False), (False, True)])
def test_current_manifest_and_csv_preserve_numerical_ablation_controls(tmp_path, quality, maneuver):
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases
    session = session_at(tmp_path, mode="memory_off", plan=planning(
        lane_quality={"enabled": quality}, lane_maneuver={"enabled": maneuver}))
    session.scenario_path.write_text(yaml.safe_dump(cases("eval")["R05_multilane_eval_s101"]), encoding="utf-8")
    try:
        session.prepare()
        assert session.manifest["lane_quality_enabled"] is quality
        assert session.manifest["lane_maneuver_enabled"] is maneuver
        version = "carla_pdm_execution_v3:" + "a" * 16
        session.set_status("route_completed", outcome={"execution_planner_version": version})
        output = tmp_path / "numerical.csv"
        assert summarize_runs(session.run_dir.parent, output) == 1
        with output.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert row["lane_quality_enabled"] == str(quality)
        assert row["lane_maneuver_enabled"] == str(maneuver)
        assert row["execution_planner_version"] == version
    finally:
        session.close()


def test_paper_summary_separates_policy_memory_and_advisory_cost(tmp_path):
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases

    scenario = cases("eval")["R05_multilane_eval_s101"]
    for policy, memory in (("always", "off"), ("selective", "gate"), ("shadow", "full")):
        directory = tmp_path / "runs" / policy
        directory.mkdir(parents=True)
        manifest = {"run_id": policy, "status": "finished", "mode": "memory_on",
            "scenario_yaml": yaml.safe_dump(scenario),
            "config": {"carla": {"planning": planning(semantic_policy=policy, planning_memory=memory)}}}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "carla_run_test.json").write_text(json.dumps({
            "run_info": {"advisory": {"model_queries": 2, "semantic_stop_proxy_s": .4},
                         "wall_time_s": 31},
            "metrics": {"route_completion": 1.0}}), encoding="utf-8")
    output = tmp_path / "results.csv"
    assert summarize_runs(tmp_path / "runs", output) == 3
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert [rows[p]["planning_memory"] for p in ("always", "selective", "shadow")] == ["off", "gate", "full"]
    assert all(row["semantic_policy"] == policy for policy, row in rows.items())
    assert all(row["model_queries"] == "2" and row["semantic_stop_proxy_s"] == "0.4"
               for row in rows.values())
    assert all(row["wall_time_s"] == "31" for row in rows.values())


def test_paper_summary_retains_advisory_counts_if_only_failure_manifest_exists(tmp_path):
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases

    directory = tmp_path / "runs" / "failed"
    directory.mkdir(parents=True)
    manifest = {"run_id": "failed", "status": "failed", "mode": "memory_off",
        "scenario_yaml": yaml.safe_dump(cases("eval")["R05_multilane_eval_s101"]),
        "config": {}, "outcome": {"advisory": {"model_queries": 2}, "wall_time_s": 15}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "results.csv"
    assert summarize_runs(tmp_path / "runs", output) == 1
    with output.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["status"] == "failed" and row["model_queries"] == "2"
    assert row["wall_time_s"] == "15"


@pytest.mark.parametrize("recovery", [None, {"enabled": False},
    {"enabled": True, "max_paths": 3, "transition_lengths_m": [2.0]}])
def test_manifest_records_recovery_configuration_without_claiming_it_was_executed(tmp_path, recovery):
    cfg = planning()
    if recovery is not None:
        cfg["recovery"] = recovery
    session = session_at(tmp_path, plan=cfg, mode="memory_off")
    try:
        session.prepare()
        manifest = json.loads((session.run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["planner_recovery"] == ({"enabled": True} if recovery is None else recovery)
        assert "recovery_selected_plans" not in manifest["planner_recovery"]
    finally:
        session.close()


def test_csv_distinguishes_recovery_switch_and_actual_expansion_execution_counts(tmp_path):
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases

    scenario = yaml.safe_dump(cases("eval")["R05_multilane_eval_s101"])
    for name, recovery in (("enabled", {"enabled": True}), ("disabled", {"enabled": False}),
                            ("historical", None), ("manifest_default", None),
                            ("partial_configuration", {"max_paths": 3})):
        directory = tmp_path / "runs" / name
        directory.mkdir(parents=True)
        cfg = planning()
        if recovery is not None:
            cfg["recovery"] = recovery
        manifest = {"run_id": name, "status": "finished", "mode": "memory_off",
                    "scenario_yaml": scenario, "config": {"carla": {"planning": cfg}}}
        if name in ("manifest_default", "partial_configuration"):
            manifest["planner_recovery"] = {"enabled": True}
        if name != "historical":
            manifest["outcome"] = {"advisory": {"recovery_expanded_plans": 5,
                "recovery_selected_plans": 3, "recovery_no_solution_plans": 2}}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "recovery.csv"
    assert summarize_runs(tmp_path / "runs", output) == 5
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert rows["enabled"]["planner_recovery"] == rows["manifest_default"]["planner_recovery"] == "True"
    assert rows["partial_configuration"]["planner_recovery"] == "True"
    assert rows["disabled"]["planner_recovery"] == "False"
    assert rows["historical"]["planner_recovery"] == rows["historical"]["recovery_expanded_plans"] == ""
    assert rows["enabled"]["recovery_expanded_plans"] == "5"
    assert rows["enabled"]["recovery_selected_plans"] == "3"
    assert rows["enabled"]["recovery_no_solution_plans"] == "2"


def test_csv_preserves_report_recovery_counts_when_outcome_contains_older_partial_statistics(tmp_path):
    from carla_bridge.paper_suite import summarize_runs
    from carla_bridge.tests.test_paper_suite import cases

    directory = tmp_path / "runs" / "reported"
    directory.mkdir(parents=True)
    manifest = {"run_id": "reported", "status": "finished", "mode": "memory_off",
        "scenario_yaml": yaml.safe_dump(cases("eval")["R05_multilane_eval_s101"]),
        "config": {"carla": {"planning": {"recovery": {"enabled": False}}}},
        "outcome": {"advisory": {"model_queries": 1}}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "carla_run_test.json").write_text(json.dumps({"run_info": {
        "advisory": {"recovery_expanded_plans": 9, "recovery_selected_plans": 7,
                     "recovery_no_solution_plans": 2}}}), encoding="utf-8")
    output = tmp_path / "reported.csv"
    summarize_runs(tmp_path / "runs", output)
    with output.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["planner_recovery"] == "False"
    assert row["model_queries"] == "1"
    assert [row[key] for key in ("recovery_expanded_plans", "recovery_selected_plans",
                               "recovery_no_solution_plans")] == ["9", "7", "2"]
