"""Offline model loading must never silently retry against the Hub."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.vla_memory.perception.dinov2_extractor import DINOv2Extractor


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def model_loaders(monkeypatch):
    model = Mock()
    model.config.hidden_size = 768
    processor_loader = Mock(return_value=object())
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoImageProcessor=SimpleNamespace(from_pretrained=processor_loader),
        AutoModel=SimpleNamespace(from_pretrained=model_loader),
    ))
    return processor_loader, model_loader, model


@pytest.mark.parametrize("nested_hub", [False, True])
def test_offline_cache_never_allows_remote_lookup(tmp_path, model_loaders, nested_hub):
    cache = tmp_path / "hub" if nested_hub else tmp_path
    (cache / "models--facebook--dinov2-base").mkdir(parents=True)
    extractor = DINOv2Extractor(cache_dir=tmp_path, device="cpu", local_files_only=True)
    extractor.load_model()
    for loader in model_loaders[:2]:
        loader.assert_called_once_with(
            "facebook/dinov2-base", cache_dir=str(cache), local_files_only=True,
        )
    assert extractor.get_feature_dim() == 768


def test_explicit_local_model_is_used_without_cache_or_network(tmp_path, model_loaders):
    model_dir = tmp_path / "snapshot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    extractor = DINOv2Extractor(model_dir=model_dir, device="cpu", local_files_only=False)
    extractor.load_model()
    for loader in model_loaders[:2]:
        loader.assert_called_once_with(str(model_dir), local_files_only=True)


def test_missing_local_model_does_not_fall_back_to_hub(tmp_path, model_loaders):
    extractor = DINOv2Extractor(model_dir=tmp_path / "missing", device="cpu")
    with pytest.raises(FileNotFoundError, match="config.json"):
        extractor.load_model()
    for loader in model_loaders[:2]:
        loader.assert_not_called()


def test_incomplete_cache_fails_once_with_offline_recovery_hint(tmp_path, model_loaders):
    processor_loader, model_loader, _ = model_loaders
    model_loader.side_effect = OSError("model.safetensors not found")
    extractor = DINOv2Extractor(cache_dir=tmp_path, device="cpu", local_files_only=True)
    with pytest.raises(RuntimeError, match="未尝试联网下载"):
        extractor.load_model()
    assert processor_loader.call_count == model_loader.call_count == 1
    assert model_loader.call_args.kwargs["local_files_only"] is True
    assert extractor._model is None
    assert extractor._processor is None


def test_failed_device_transfer_leaves_model_unloaded(tmp_path, model_loaders):
    model_loaders[2].to.side_effect = RuntimeError("CUDA unavailable")
    extractor = DINOv2Extractor(cache_dir=tmp_path, device="cpu", local_files_only=True)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        extractor.load_model()
    assert extractor._model is None


def test_explicit_online_mode_remains_available(tmp_path, model_loaders):
    extractor = DINOv2Extractor(cache_dir=tmp_path, device="cpu", local_files_only=False)
    extractor.load_model()
    assert all(loader.call_args.kwargs["local_files_only"] is False for loader in model_loaders[:2])


def test_prepare_utility_has_offline_verification_and_explicit_download():
    spec = importlib.util.spec_from_file_location("prepare_models", ROOT / "scripts" / "00_prepare_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._parse_args(["--offline"]).local_files_only is True
    assert module._parse_args(["--download"]).local_files_only is False
    assert module._parse_args([]).local_files_only is None


def test_entry_enables_crash_trace_and_flush_without_shell_prefix():
    env = dict(os.environ)
    env.pop("PYTHONFAULTHANDLER", None)
    env.pop("PYTHONUNBUFFERED", None)
    result = subprocess.run([
        sys.executable, "-c",
        "import carla_bridge.run_carla_demo; import faulthandler, sys; "
        "assert faulthandler.is_enabled(); "
        "assert sys.stdout.line_buffering and sys.stdout.write_through; "
        "assert sys.stderr.line_buffering and sys.stderr.write_through",
    ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
