"""Exercise the real CLI/config/preflight boundary without CARLA or model weights."""
from unittest.mock import Mock

import pytest

from src.vla_memory.common.config import Config
from src.vla_memory.perception.local_qwen_client import LocalQwenRuntime


@pytest.fixture
def entry_harness(tmp_path, monkeypatch):
    from carla_bridge import run_carla_demo as entry

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scenario_name: local_entry\n", encoding="utf-8")
    configs = []

    def load_config(*, overrides):
        config = Config(overrides={**overrides, "output_dir": str(tmp_path / "runs"),
                                   "log_dir": str(tmp_path / "logs")})
        for stage in ("scene_understanding", "decision"):
            key = "LOCAL_ENTRY_ABSENT_" + stage.upper()
            monkeypatch.delenv(key, raising=False)
            config.data.setdefault(stage, {})["api_key_env"] = key
        configs.append(config)
        return config

    monkeypatch.setattr(entry, "load_config", load_config)
    monkeypatch.setattr(entry, "set_log_directory", Mock())
    monkeypatch.setattr(entry, "setup_logger", Mock())
    loop = Mock()
    loop.setup.side_effect = RuntimeError("test reached CARLA setup")
    loop.outcome.return_value = {}
    factory = Mock(return_value=loop)
    monkeypatch.setattr(entry, "ClosedLoop", factory)
    forbidden_load = Mock(side_effect=AssertionError("CLI preflight loaded real model weights"))
    monkeypatch.setattr(LocalQwenRuntime, "load", forbidden_load)
    return entry, scenario, configs, factory, loop, forbidden_load


def test_local_cli_without_api_keys_passes_local_preflight_and_forwards_runtime_options(
        entry_harness, tmp_path, monkeypatch):
    entry, scenario, configs, factory, loop, forbidden_load = entry_harness
    checkpoint = tmp_path / "local_checkpoint"
    preflight = []

    def check_environment(runtime):
        preflight.append((runtime.model_path, runtime.device, runtime.dtype))

    def construct_loop(**kwargs):
        assert preflight == [(checkpoint.resolve(), "cuda:2", "bfloat16")]
        return loop

    monkeypatch.setattr(LocalQwenRuntime, "check_environment", check_environment)
    factory.side_effect = construct_loop
    monkeypatch.setattr("sys.argv", [
        "run_carla_demo", "--scenario", str(scenario), "--mode", "memory_off",
        "--planner", "pdm_closed", "--semantic-influence", "on", "--model-inference", "on",
        "--vlm-backend", "local", "--local-vlm-model-path", str(checkpoint),
        "--local-vlm-device", "cuda:2", "--local-vlm-dtype", "bfloat16",
    ])
    with pytest.raises(RuntimeError, match="test reached CARLA setup"):
        entry.main()

    assert preflight == [(checkpoint.resolve(), "cuda:2", "bfloat16")]
    assert len(configs) == 1
    config = configs[0]
    assert config.get_nested("vlm", "backend") == "local"
    assert config.get_nested("vlm", "local", "model_path") == str(checkpoint)
    assert config.get_nested("vlm", "local", "device") == "cuda:2"
    assert config.get_nested("vlm", "local", "dtype") == "bfloat16"
    assert config.get_nested("carla", "planning", "model_inference") is True
    factory.assert_called_once_with(config=config, scenario_yaml=str(scenario), mode="memory_off")
    loop.setup.assert_called_once()
    loop.run.assert_not_called()
    loop.close.assert_called_once()
    forbidden_load.assert_not_called()


def test_model_off_cli_ignores_invalid_local_checkpoint_and_skips_preflight(
        entry_harness, tmp_path, monkeypatch):
    from src.vla_memory.perception import factory as backend_factory

    entry, scenario, configs, factory, loop, forbidden_load = entry_harness
    invalid_checkpoint = tmp_path / "does_not_exist"
    assert not invalid_checkpoint.exists()
    forbidden_preflight = Mock(side_effect=AssertionError("model-off checked local model environment"))
    monkeypatch.setattr(backend_factory, "validate_vlm_setup", forbidden_preflight)
    monkeypatch.setattr(LocalQwenRuntime, "check_environment", forbidden_preflight)
    monkeypatch.setattr("sys.argv", [
        "run_carla_demo", "--scenario", str(scenario), "--mode", "memory_off",
        "--planner", "mpc", "--semantic-influence", "off", "--model-inference", "off",
        "--planning-memory", "off", "--vlm-backend", "local",
        "--local-vlm-model-path", str(invalid_checkpoint), "--local-vlm-device", "cuda:999",
        "--local-vlm-dtype", "bfloat16",
    ])
    with pytest.raises(RuntimeError, match="test reached CARLA setup"):
        entry.main()

    assert configs[0].get_nested("carla", "planning", "model_inference") is False
    assert configs[0].get_nested("vlm", "backend") == "local"
    assert configs[0].get_nested("vlm", "local", "model_path") == str(invalid_checkpoint)
    factory.assert_called_once()
    loop.setup.assert_called_once()
    loop.close.assert_called_once()
    forbidden_preflight.assert_not_called()
    forbidden_load.assert_not_called()
