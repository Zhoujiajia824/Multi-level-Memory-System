"""Local VLM contracts, exercised with explicit test doubles, never model weights.

These tests verify offline loading, multimodal ordering and replay evidence.  They
do not assert driving quality or claim a real Qwen checkpoint was executed.
"""
from contextlib import contextmanager, nullcontext
import gc
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock
import weakref

import pytest
from PIL import Image

from src.vla_memory.common.config import Config
from src.vla_memory.common.model_call_audit import observe_model_requests
from src.vla_memory.perception.local_qwen_client import (
    LocalQwenRuntime,
    LocalQwenVLMClient,
    validate_local_model_directory,
)


@pytest.fixture
def model_directory(tmp_path):
    directory = tmp_path / "Qwen2.5-VL-3B-Instruct"
    directory.mkdir()
    for name, value in {
        "config.json": {"model_type": "qwen2_5_vl"},
        "preprocessor_config.json": {"image_processor_type": "Qwen2VLImageProcessor"},
        "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
        "tokenizer.json": {},
        "chat_template.json": {"chat_template": "{{ messages }}"},
        "model.safetensors.index.json": {"weight_map": {
            "visual.weight": "model-00001-of-00002.safetensors",
            "language.weight": "model-00002-of-00002.safetensors"}},
    }.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    for number in (1, 2):
        (directory / f"model-{number:05d}-of-00002.safetensors").write_bytes(b"weight-fixture")
    return directory


@pytest.fixture
def images(tmp_path):
    paths = []
    for number in range(6):
        path = tmp_path / f"frame_{number}.png"
        Image.new("RGB", (14, 14), (number * 20, 10, 30)).save(path)
        paths.append(str(path))
    return paths


class FakeRuntime:
    def __init__(self, output='{"ok":true}'):
        self.output = output
        self.generate = Mock(side_effect=self._generate)
        self.model_path = Path("local-checkpoint")

    def _generate(self, *args, **kwargs):
        if isinstance(self.output, BaseException):
            raise self.output
        return self.output


def text_content(message):
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(
        block["text"] for block in content if block.get("type") == "text")


def test_complete_local_model_directory_is_resolved(model_directory):
    assert validate_local_model_directory(str(model_directory)) == model_directory.resolve()


@pytest.mark.parametrize("missing", [
    "config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
    "chat_template.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
])
def test_incomplete_local_download_fails_before_runtime(model_directory, missing):
    (model_directory / missing).unlink()
    with pytest.raises((FileNotFoundError, ValueError), match=".+"):
        validate_local_model_directory(model_directory)


def test_wrong_model_family_is_rejected(model_directory):
    (model_directory / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    with pytest.raises(ValueError, match="qwen2_5_vl"):
        validate_local_model_directory(model_directory)


def test_tokenizer_vocab_and_merges_are_supported(model_directory):
    (model_directory / "tokenizer.json").unlink()
    (model_directory / "vocab.json").write_text("{}", encoding="utf-8")
    (model_directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    assert validate_local_model_directory(model_directory) == model_directory.resolve()
    (model_directory / "merges.txt").unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_local_model_directory(model_directory)


@pytest.mark.parametrize("source", ["jinja", "tokenizer_config"])
def test_local_model_accepts_supported_alternative_chat_template(model_directory, source):
    (model_directory / "chat_template.json").unlink()
    if source == "jinja":
        (model_directory / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    else:
        (model_directory / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "Qwen2Tokenizer", "chat_template": "{{ messages }}"}), encoding="utf-8")
    assert validate_local_model_directory(model_directory) == model_directory.resolve()


def test_empty_chat_template_is_rejected_before_model_loading(model_directory):
    (model_directory / "chat_template.json").write_text('{"chat_template":""}', encoding="utf-8")
    with pytest.raises(ValueError, match="chat_template"):
        validate_local_model_directory(model_directory)


def test_unsharded_safetensors_download_is_supported(model_directory):
    (model_directory / "model.safetensors.index.json").unlink()
    (model_directory / "model.safetensors").write_bytes(b"weight-fixture")
    assert validate_local_model_directory(model_directory) == model_directory.resolve()


@pytest.mark.parametrize("name", ["config.json", "model-00001-of-00002.safetensors"])
def test_git_lfs_pointer_is_not_a_downloaded_model(model_directory, name):
    (model_directory / name).write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:fixture\nsize 123\n",
        encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match=".+"):
        validate_local_model_directory(model_directory)


def test_weight_index_cannot_escape_model_directory(model_directory, tmp_path):
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside-weight")
    (model_directory / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"visual.weight": "../outside.safetensors"}}), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_local_model_directory(model_directory)


def test_scene_passes_exact_prompt_context_and_system_message(images):
    runtime = FakeRuntime()
    client = LocalQwenVLMClient(runtime, system_prompt="只输出 JSON", max_tokens=123,
                                temperature=0.0, timeout=9)
    assert client.understand_scene(images[0], "分析驾驶场景", {"nav": "left"}) == '{"ok":true}'
    messages, paths = runtime.generate.call_args.args
    assert paths == images[:1]
    assert messages[0] == {"role": "system", "content": "只输出 JSON"}
    assert messages[-1]["role"] == "user"
    assert "分析驾驶场景" in text_content(messages[-1])
    assert "nav: left" in text_content(messages[-1])
    assert runtime.generate.call_args.kwargs == {"max_tokens": 123, "temperature": 0.0,
                                                 "timeout": 9}


def test_decision_keeps_recent_four_in_original_time_order(images):
    runtime = FakeRuntime()
    client = LocalQwenVLMClient(runtime, system_prompt="驾驶语义决策")
    assert client.decide("当前帧在末尾", image_paths=images) == '{"ok":true}'
    messages, paths = runtime.generate.call_args.args
    assert paths == images[-4:]
    user = messages[-1]
    assert text_content(user) == "当前帧在末尾"
    blocks = [block for block in user["content"] if block["type"] == "image_url"]
    assert len(blocks) == 4
    assert [block["image_url"]["url"] for block in blocks] == [
        Path(path).resolve().as_uri() for path in images[-4:]]


def test_legacy_single_image_and_explicit_empty_list_have_distinct_meanings(images):
    runtime = FakeRuntime()
    client = LocalQwenVLMClient(runtime)
    client.decide("兼容单图", optional_image_path=images[0])
    assert runtime.generate.call_args.args[1] == images[:1]
    client.decide("明确无图", image_paths=[], optional_image_path=images[0])
    assert runtime.generate.call_args.args[1] == []
    assert text_content(runtime.generate.call_args.args[0][-1]) == "明确无图"


def test_missing_image_fails_without_inference_or_api(tmp_path):
    runtime = FakeRuntime()
    client = LocalQwenVLMClient(runtime)
    with pytest.raises(FileNotFoundError):
        client.decide("input", image_paths=[str(tmp_path / "missing.png")])
    runtime.generate.assert_not_called()


def test_local_failure_never_falls_back_to_remote_api(images, monkeypatch):
    from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient
    forbidden = Mock(side_effect=AssertionError("local failure attempted API fallback"))
    monkeypatch.setattr(OpenAICompatibleVLMClient, "_call_with_retry", forbidden)
    runtime = FakeRuntime(RuntimeError("local CUDA OOM"))
    with pytest.raises(RuntimeError, match="local CUDA OOM"):
        LocalQwenVLMClient(runtime).decide("input", image_paths=images[:1])
    assert runtime.generate.call_count == 1
    forbidden.assert_not_called()


@pytest.mark.parametrize("empty", ["", " ", "\n\t"])
def test_empty_generation_is_not_a_successful_decision(empty, images):
    with pytest.raises(RuntimeError):
        LocalQwenVLMClient(FakeRuntime(empty)).decide("input", image_paths=images[:1])


@pytest.mark.parametrize("field,value", [
    ("temperature", True), ("temperature", False), ("timeout", True), ("timeout", False),
])
def test_boolean_is_not_accepted_as_generation_temperature_or_timeout(field, value):
    with pytest.raises(ValueError, match=field):
        LocalQwenVLMClient(FakeRuntime(), **{field: value})


@pytest.mark.parametrize("failure", [False, True])
def test_local_dispatch_audit_records_success_or_failure_once(images, failure, injected_runtime):
    events = []
    runtime, processor, model, _ = injected_runtime
    processor.batch_decode.return_value = ['{"decision":1}']
    if failure:
        model.generate.side_effect = RuntimeError("failed generation")
    client = LocalQwenVLMClient(runtime, system_prompt="系统约束")
    with observe_model_requests(events.append):
        if failure:
            with pytest.raises(RuntimeError):
                client.decide("完整输入", image_paths=images[:2])
        else:
            assert client.decide("完整输入", image_paths=images[:2]) == '{"decision":1}'
    assert len(events) == 2
    start, end = events
    assert start["event"] == "request_start" and end["event"] == "request_end"
    assert start["request_id"] == end["request_id"]
    assert all(row["stage"] == "decision" and row["backend"] == "local"
               and row["provider"] == "local_qwen" for row in events)
    assert start["image_paths"] == images[:2]
    assert start["messages"][0]["content"] == "系统约束"
    assert end["status"] == ("failed" if failure else "success")
    assert end["wall_duration_s"] >= 0
    if failure:
        assert end["error_type"] == "RuntimeError"
    else:
        assert end["response"] == '{"decision":1}'


def test_failed_audit_callback_does_not_repeat_successful_local_generation(images, injected_runtime):
    runtime, processor, model, _ = injected_runtime
    processor.batch_decode.return_value = ['{"ok":true}']
    with observe_model_requests(Mock(side_effect=OSError("audit disk full"))):
        assert LocalQwenVLMClient(runtime).decide("input", image_paths=images[:1]) == '{"ok":true}'
    model.generate.assert_called_once()


def test_local_model_io_stays_portable_in_existing_replay_audit(tmp_path, images, injected_runtime):
    from carla_bridge.model_call_audit import ModelCallAudit
    audit = ModelCallAudit(tmp_path / "run")
    client = LocalQwenVLMClient(injected_runtime[0], system_prompt="精确系统提示词")
    with audit.consultation({"image_path": images[0], "carla_frame_id": 123,
                             "sim_time_s": 5.5}, {"reason": "blocked_lane"}, {}) as evidence:
        client.understand_scene(images[0], "精确场景提示词", {"nav": "left"})
        evidence["record"] = {"raw_response": client.decide("精确决策提示词", image_paths=images[:2])}
    requests = [json.loads(line) for line in (audit.run_dir / "model_requests.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [row["stage"] for row in requests] == ["scene", "scene", "decision", "decision"]
    assert all(row["carla_frame_id"] == 123 and row["backend"] == "local" for row in requests)
    assert requests[0]["system_prompt"] == "精确系统提示词"
    assert "nav: left" in requests[0]["prompt"]
    archived = requests[2]["image_paths"]
    assert [Image.open(audit.run_dir / name).getpixel((0, 0)) for name in archived] == [
        Image.open(name).getpixel((0, 0)) for name in images[:2]]
    image_blocks = [block for block in requests[2]["messages"][-1]["content"]
                    if block.get("type") == "image_url"]
    assert [block["image_url"]["url"] for block in image_blocks] == archived
    assert not any("file:" in str(block) for block in image_blocks)


@pytest.mark.parametrize("backend", [None, "api"])
def test_factory_default_api_uses_existing_clients_without_loading_local_weights(monkeypatch, backend):
    from src.vla_memory.perception.factory import build_vlm_clients
    from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient
    forbidden = Mock(side_effect=AssertionError("API backend loaded local weights"))
    monkeypatch.setattr(LocalQwenRuntime, "load", forbidden)
    config = Config(overrides={"vlm": {"backend": "api"}})
    if backend is None:
        config.data.pop("vlm", None)
    scene, decision = build_vlm_clients(config)
    assert isinstance(scene, OpenAICompatibleVLMClient)
    assert isinstance(decision, OpenAICompatibleVLMClient)
    forbidden.assert_not_called()


def test_factory_local_stages_share_one_lazy_runtime(model_directory, monkeypatch):
    from src.vla_memory.perception.factory import build_vlm_clients
    load = Mock(side_effect=AssertionError("factory eagerly loaded checkpoint"))
    monkeypatch.setattr(LocalQwenRuntime, "load", load)
    cfg = Config(overrides={"vlm": {"backend": "local", "local": {
        "model_path": str(model_directory), "device": "cuda:1"}}})
    scene, decision = build_vlm_clients(cfg)
    assert isinstance(scene, LocalQwenVLMClient) and isinstance(decision, LocalQwenVLMClient)
    assert scene.runtime is decision.runtime
    assert Path(scene.runtime.model_path) == model_directory.resolve()
    assert scene.max_tokens == cfg.get_nested("scene_understanding", "max_tokens")
    assert decision.max_tokens == cfg.get_nested("decision", "max_tokens")
    from src.vla_memory.decision.config_access import get_decision_system_prompt
    assert decision.system_prompt == get_decision_system_prompt(cfg)
    load.assert_not_called()


def test_unknown_backend_is_rejected_instead_of_silently_using_api():
    from src.vla_memory.perception.factory import build_vlm_clients
    with pytest.raises(ValueError):
        build_vlm_clients(Config(overrides={"vlm": {"backend": "typo_local"}}))


def test_local_config_does_not_import_transformers_when_models_are_disabled(model_directory):
    code = r'''
import builtins, json, sys
sys.path[:] = json.loads(sys.argv[1])
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(("torch", "transformers", "src.vla_memory.perception.local_qwen_client")):
        raise AssertionError("pure planner imported local inference: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from carla_bridge.closed_loop import ClosedLoop
from src.vla_memory.common.config import Config
cfg = Config(overrides={"vlm": {"backend": "local", "local": {"model_path": sys.argv[2]}},
    "carla": {"planning": {"backend": "pdm_closed", "semantic_influence": False,
                           "model_inference": False}}})
loop = ClosedLoop(cfg, "unused.yaml", mode="memory_off")
assert loop.model_inference is False and loop.loop is None
'''
    result = subprocess.run([sys.executable, "-c", code, json.dumps(sys.path), str(model_directory)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


class FakeBatch(dict):
    def __init__(self, tensor):
        super().__init__(input_ids=tensor, attention_mask=tensor.new_ones(tensor.shape))
        self.destination = None

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, destination):
        self.destination = destination
        return self


class FakeProcessor:
    def __init__(self, torch, decoded='{"decision":"FOLLOW_ROUTE"}'):
        self.tokenizer = NS(eos_token_id=99)
        self.chat_template = "{{ messages }}"
        self.apply_chat_template = Mock(return_value="formatted multimodal prompt")
        self.batch_decode = Mock(return_value=[decoded])
        self.batch = FakeBatch(torch.tensor([[11, 12, 13]], dtype=torch.long))
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.image_pixels = [(image.mode, image.getpixel((0, 0)))
                             for image in (kwargs.get("images") or [])]
        return self.batch


@pytest.fixture
def injected_runtime(model_directory):
    # CPU tensors only: this exercises generation plumbing without a checkpoint.
    torch = pytest.importorskip("torch")
    runtime = LocalQwenRuntime(model_directory, device="cpu", dtype="float32")
    processor = FakeProcessor(torch)
    model = NS(generate=Mock(return_value=torch.tensor([[11, 12, 13, 40, 41, 99]])),
               config=NS(max_position_embeddings=32768),
               generation_config=NS(eos_token_id=99), device="cpu")
    runtime._processor = processor
    runtime._model = model
    runtime._torch = torch
    runtime._device = torch.device("cpu")
    return runtime, processor, model, torch


def test_runtime_uses_captured_images_in_order_and_trims_input_tokens(injected_runtime, images):
    runtime, processor, model, _ = injected_runtime
    messages = [{"role": "system", "content": "只输出语义 JSON"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": Path(path).as_uri()}}
        for path in images[:2]] + [{"type": "text", "text": "当前图在末尾"}]}]
    result = runtime.generate(messages, images[:2], max_tokens=7, temperature=0.0, timeout=20)
    assert result == '{"decision":"FOLLOW_ROUTE"}'
    chat_args = processor.apply_chat_template.call_args
    assert chat_args.kwargs == {"tokenize": False, "add_generation_prompt": True}
    chat = chat_args.args[0]
    assert chat[0] == messages[0]
    assert [block["type"] for block in chat[1]["content"]] == ["image", "image", "text"]
    batch_call = processor.calls[0]
    assert batch_call["text"] == ["formatted multimodal prompt"]
    assert batch_call["padding"] is True and batch_call["return_tensors"] == "pt"
    assert processor.image_pixels == [("RGB", (0, 10, 30)), ("RGB", (20, 10, 30))]
    decoded_tokens = processor.batch_decode.call_args.args[0]
    assert [token.tolist() for token in decoded_tokens] == [[40, 41, 99]]
    assert processor.batch_decode.call_args.kwargs["skip_special_tokens"] is True
    assert model.generate.call_args.kwargs["max_new_tokens"] == 7
    assert model.generate.call_args.kwargs["do_sample"] is False
    assert "temperature" not in model.generate.call_args.kwargs
    assert str(processor.batch.destination) == "cpu"


def test_runtime_text_only_decision_never_invents_an_image(injected_runtime):
    runtime, processor, _, _ = injected_runtime
    runtime.generate([{"role": "user", "content": "文本状态决策"}], [],
                     max_tokens=10, temperature=0.0, timeout=20)
    assert processor.calls[0].get("images") is None
    assert "文本状态决策" in str(processor.apply_chat_template.call_args.args[0])


def test_runtime_rejects_token_budget_exhaustion_without_eos(injected_runtime):
    runtime, processor, model, torch = injected_runtime
    model.generate.return_value = torch.tensor([[11, 12, 13, 40, 41, 42]])
    with pytest.raises(RuntimeError, match="(?i)truncat|token|截断"):
        runtime.generate([{"role": "user", "content": "input"}], [],
                         max_tokens=3, temperature=0.0, timeout=20)


def test_runtime_accepts_eos_exactly_at_token_budget(injected_runtime):
    runtime, _, _, _ = injected_runtime
    assert runtime.generate([{"role": "user", "content": "input"}], [],
                            max_tokens=3, temperature=0.0, timeout=20).startswith("{")


def test_runtime_context_overflow_does_not_dispatch_or_claim_a_model_call(injected_runtime):
    runtime, _, model, _ = injected_runtime
    model.config.max_position_embeddings = 4
    events = []
    with observe_model_requests(events.append):
        with pytest.raises(ValueError, match="上下文"):
            LocalQwenVLMClient(runtime, max_tokens=3).decide("input")
    model.generate.assert_not_called()
    assert events == []


def test_runtime_timeout_without_eos_is_recorded_as_failed_not_partial_success(
        injected_runtime, monkeypatch):
    from src.vla_memory.perception import local_qwen_client as module
    runtime, processor, model, torch = injected_runtime
    model.generate.return_value = torch.tensor([[11, 12, 13, 40, 41]])
    monkeypatch.setattr(module.time, "perf_counter", Mock(side_effect=[0.0, 2.0, 2.0]))
    events = []
    with observe_model_requests(events.append):
        with pytest.raises(TimeoutError):
            LocalQwenVLMClient(runtime, max_tokens=10, timeout=1).decide("input")
    assert [event["event"] for event in events] == ["request_start", "request_end"]
    assert events[-1]["status"] == "failed" and events[-1]["error_type"] == "TimeoutError"
    assert "response" not in events[-1]
    processor.batch_decode.assert_not_called()


def test_runtime_empty_decoded_generation_is_a_failed_dispatch(injected_runtime):
    runtime, processor, _, _ = injected_runtime
    processor.batch_decode.return_value = [" \n"]
    events = []
    with observe_model_requests(events.append):
        with pytest.raises(RuntimeError, match="空输出"):
            LocalQwenVLMClient(runtime).decide("input")
    assert events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "RuntimeError"


def test_shared_runtime_is_closed_once_even_if_both_stages_are_registered():
    from src.vla_memory.perception.factory import close_vlm_clients
    runtime = NS(close=Mock())
    close_vlm_clients([NS(runtime=runtime), NS(runtime=runtime), NS()])
    runtime.close.assert_called_once()


def test_runtime_load_is_local_only_safe_and_idempotent(model_directory, monkeypatch):
    torch = pytest.importorskip("torch")
    processor = FakeProcessor(torch)
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    processor_loader = Mock(return_value=processor)
    model_loader = Mock(return_value=model)
    transformers = NS(AutoProcessor=NS(from_pretrained=processor_loader),
        Qwen2_5_VLForConditionalGeneration=NS(from_pretrained=model_loader),
        StoppingCriteria=object, StoppingCriteriaList=list)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    runtime = LocalQwenRuntime(model_directory, device="cpu", dtype="float32",
                               min_pixels=196, max_pixels=784)
    assert runtime._model is None
    runtime.load()
    runtime.load()
    processor_loader.assert_called_once()
    model_loader.assert_called_once()
    for loader in (processor_loader, model_loader):
        assert Path(loader.call_args.args[0]) == model_directory.resolve()
        assert loader.call_args.kwargs["local_files_only"] is True
        assert loader.call_args.kwargs["trust_remote_code"] is False
    assert processor_loader.call_args.kwargs["min_pixels"] == 196
    assert processor_loader.call_args.kwargs["max_pixels"] == 784
    assert model_loader.call_args.kwargs["attn_implementation"] == "sdpa"
    assert model_loader.call_args.kwargs["use_safetensors"] is True
    model.eval.assert_called_once()


@pytest.mark.parametrize("tokenizer_template", [None, "{{ tokenizer_messages }}"])
def test_load_explicitly_falls_back_to_tokenizer_template_or_fails_before_weights(
        model_directory, monkeypatch, tokenizer_template):
    torch = pytest.importorskip("torch")
    processor = FakeProcessor(torch)
    processor.chat_template = None
    processor.tokenizer.chat_template = tokenizer_template
    model = Mock()
    model.to.return_value = model
    model_loader = Mock(return_value=model)
    runtime = LocalQwenRuntime(model_directory, device="cpu")
    monkeypatch.setattr(runtime, "check_environment", Mock(return_value=(
        torch, NS(from_pretrained=Mock(return_value=processor)),
        NS(from_pretrained=model_loader), torch.device("cpu"))))
    if tokenizer_template:
        runtime.load()
        assert processor.chat_template == tokenizer_template
        assert runtime._processor is processor and runtime._model is model
        model_loader.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="ValueError.*聊天模板"):
            runtime.load()
        model_loader.assert_not_called()
        assert runtime._model is None and runtime._processor is None


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_close_clears_runtime_and_only_releases_its_own_cuda_device(model_directory, device_type):
    runtime = LocalQwenRuntime(model_directory, device="cuda:2" if device_type == "cuda" else "cpu")
    device = NS(type=device_type, index=2 if device_type == "cuda" else None)
    empty_cache = Mock()
    context = Mock(return_value=nullcontext())
    runtime._torch = NS(cuda=NS(device=context, empty_cache=empty_cache))
    runtime._device = device
    runtime._model, runtime._processor = Mock(), Mock()
    runtime.close()
    runtime.close()
    assert runtime._model is runtime._processor is runtime._torch is runtime._device is None
    if device_type == "cuda":
        context.assert_called_once_with(device)
        empty_cache.assert_called_once()
    else:
        context.assert_not_called()
        empty_cache.assert_not_called()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_failed_to_device_load_releases_partial_model_traceback_and_can_retry(
        model_directory, monkeypatch, device_type):
    device = NS(type=device_type, index=2 if device_type == "cuda" else None)
    state = NS(fail=True, active_device=None)
    created = []
    contexts = []
    cache_releases = []

    class PartialModel:
        def __init__(self):
            self.evaluated = False

        def to(self, target):
            assert target is device
            if state.fail:
                raise MemoryError("fixture device transfer OOM")
            return self

        def eval(self):
            self.evaluated = True

    def from_pretrained(*args, **kwargs):
        model = PartialModel()
        created.append(weakref.ref(model))
        return model

    @contextmanager
    def select_device(target):
        assert target is device
        contexts.append(target)
        state.active_device = target
        try:
            yield
        finally:
            state.active_device = None

    def empty_cache():
        assert state.active_device is device
        cache_releases.append(device)

    torch = NS(float32=object(), cuda=NS(device=select_device, empty_cache=empty_cache))
    processor_class = NS(from_pretrained=Mock(side_effect=lambda *args, **kwargs: NS(
        chat_template="{{ messages }}", tokenizer=NS())))
    model_class = NS(from_pretrained=from_pretrained)
    runtime = LocalQwenRuntime(model_directory, device="cuda:2" if device_type == "cuda" else "cpu",
                               dtype="float32")
    monkeypatch.setattr(runtime, "check_environment", Mock(return_value=(
        torch, processor_class, model_class, device)))

    with pytest.raises(RuntimeError, match="MemoryError.*fixture device transfer OOM") as captured:
        runtime.load()
    # Retain the returned exception and its traceback while checking the model.
    # The original .to() traceback must no longer retain its self/partial weights.
    gc.collect()
    assert captured.value.__cause__ is None and captured.value.__context__ is None
    assert created[0]() is None
    assert runtime._model is runtime._processor is runtime._torch is runtime._device is None
    assert cache_releases == ([device] if device_type == "cuda" else [])
    assert contexts == ([device] if device_type == "cuda" else [])

    state.fail = False
    runtime.load()
    assert len(created) == 2
    assert created[1]() is runtime._model and runtime._model.evaluated is True
    assert runtime._processor.chat_template == "{{ messages }}"
    assert runtime._device is device
    assert cache_releases == ([device] if device_type == "cuda" else [])


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("failure_stage", ["processor", "generate", "decode"])
def test_failed_inference_releases_temporary_tensors_without_unloading_shared_model(
        model_directory, monkeypatch, device_type, failure_stage):
    # All tensors are CPU tensors. The CUDA stub only checks cache ownership;
    # this test never allocates real GPU memory or requires a CUDA driver.
    torch = pytest.importorskip("torch")
    device = NS(type=device_type, index=2 if device_type == "cuda" else None)
    state = NS(fail=True, generate_calls=0, active_device=None)
    tensor_references = []
    contexts = []
    cache_releases = []

    class TransientProcessor:
        chat_template = "{{ messages }}"
        tokenizer = NS(eos_token_id=99)

        def apply_chat_template(self, *args, **kwargs):
            return "input prompt"

        def __call__(self, **kwargs):
            input_ids = torch.tensor([[11, 12, 13]])
            tensor_references.append(weakref.ref(input_ids))
            batch = FakeBatch(input_ids)
            if state.fail and failure_stage == "processor":
                raise ValueError("fixture processor preparation failed")
            return batch

        def batch_decode(self, tokens, **kwargs):
            if state.fail and failure_stage == "decode":
                raise ValueError("fixture generation decode failed")
            return ['{"ok":true}']

    def generate(**kwargs):
        state.generate_calls += 1
        kv_cache = torch.tensor([17.0, 18.0, 19.0])
        tensor_references.append(weakref.ref(kv_cache))
        if state.fail and failure_stage == "generate":
            raise RuntimeError("fixture generation OOM")
        output = torch.tensor([[11, 12, 13, 40, 99]])
        tensor_references.append(weakref.ref(output))
        return output

    @contextmanager
    def select_device(target):
        assert target is device
        contexts.append(target)
        state.active_device = target
        try:
            yield
        finally:
            state.active_device = None

    def empty_cache():
        assert state.active_device is device
        cache_releases.append(device)

    processor = TransientProcessor()
    model = NS(generate=generate, config=NS(max_position_embeddings=32768),
               generation_config=NS(eos_token_id=99))
    runtime = LocalQwenRuntime(model_directory, device="cpu")
    runtime._processor, runtime._model = processor, model
    runtime._device = device
    runtime._torch = NS(inference_mode=torch.inference_mode,
                        cuda=NS(device=select_device, empty_cache=empty_cache))
    forbidden_load = Mock(side_effect=AssertionError("failed inference reloaded shared weights"))
    monkeypatch.setattr(runtime, "check_environment", forbidden_load)
    client = LocalQwenVLMClient(runtime)
    events = []
    expected_error = RuntimeError if failure_stage == "generate" else ValueError
    with observe_model_requests(events.append):
        with pytest.raises(expected_error, match="fixture") as captured:
            client.decide("input")
    # Keep the returned exception alive: its traceback must not retain inputs,
    # the generated sequence, sliced tokens or the failing generate's KV cache.
    gc.collect()
    assert captured.value.__cause__ is captured.value.__context__ is None
    assert tensor_references and all(reference() is None for reference in tensor_references)
    assert runtime._model is model and runtime._processor is processor
    assert runtime._device is device
    assert cache_releases == ([device] if device_type == "cuda" else [])
    assert contexts == ([device] if device_type == "cuda" else [])
    if failure_stage == "processor":
        assert events == [] and state.generate_calls == 0
    else:
        assert [event["event"] for event in events] == ["request_start", "request_end"]
        assert events[-1]["status"] == "failed" and state.generate_calls == 1

    state.fail = False
    with observe_model_requests(events.append):
        assert client.decide("retry input") == '{"ok":true}'
    assert events[-1]["status"] == "success"
    assert runtime._model is model and runtime._processor is processor
    assert cache_releases == ([device] if device_type == "cuda" else [])
    forbidden_load.assert_not_called()
