"""Lightweight backend selection; importing this module never loads a model."""
import os
from pathlib import Path


def vlm_backend(config):
    backend = config.get_nested("vlm", "backend", default="api")
    if backend not in ("api", "local"):
        raise ValueError("vlm.backend 必须为 api 或 local")
    return backend


def local_runtime_options(config):
    values = dict(config.get_nested("vlm", "local", default={}) or {})
    source = values.get("model_path")
    if not source:
        raise ValueError("本地 VLM 未设置 vlm.local.model_path / --local-vlm-model-path")
    source = Path(source).expanduser()
    values["model_path"] = source if source.is_absolute() else config.root / source
    return values


def validate_vlm_setup(config):
    """Run before CARLA startup, only when model inference is enabled."""
    if vlm_backend(config) == "local":
        from src.vla_memory.perception.local_qwen_client import LocalQwenRuntime
        LocalQwenRuntime(**local_runtime_options(config)).check_environment()
        return
    names = {config.get_nested(stage, "api_key_env", default="DASHSCOPE_API_KEY")
             for stage in ("scene_understanding", "decision")}
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("API后端 api_key_env 必须为非空环境变量名")
    missing = sorted(name for name in names if not os.environ.get(name))
    if missing:
        raise EnvironmentError("VLM API Key 未设置：%s；请在当前终端设置API环境变量，"
                               "或使用 --vlm-backend local 运行本地模型" % missing)


def build_vlm_clients(config):
    from src.vla_memory.decision.config_access import get_decision_system_prompt
    scene = config.get("scene_understanding", {}) or {}
    decision = config.get("decision", {}) or {}
    if vlm_backend(config) == "local":
        from src.vla_memory.perception.local_qwen_client import LocalQwenRuntime, LocalQwenVLMClient
        runtime = LocalQwenRuntime(**local_runtime_options(config))

        def make(block, prompt, timeout, max_tokens):
            return LocalQwenVLMClient(runtime, max_tokens=block.get("max_tokens", max_tokens),
                temperature=block.get("temperature", 0.0), system_prompt=prompt,
                timeout=block.get("timeout", timeout))
    else:
        from src.vla_memory.perception.openai_compatible_client import OpenAICompatibleVLMClient

        def make(block, prompt, timeout, max_tokens):
            return OpenAICompatibleVLMClient(provider=block.get("provider", "qwen"),
                api_key_env=block.get("api_key_env", "DASHSCOPE_API_KEY"),
                base_url=block.get("base_url", ""), model_name=block.get("model_name", "qwen-vl-max"),
                timeout=block.get("timeout", timeout), max_tokens=block.get("max_tokens", max_tokens),
                temperature=block.get("temperature", 0.1), retry_times=block.get("retry_times", 3),
                retry_interval_seconds=block.get("retry_interval_seconds", 5), system_prompt=prompt)
    return (make(scene, scene.get("system_prompt", ""), 60, 2048),
            make(decision, get_decision_system_prompt(config), 120, 4096))


def close_vlm_clients(clients):
    """Release each shared local model once, even after a partial setup."""
    seen = set()
    for client in clients:
        runtime = getattr(client, "runtime", None)
        if runtime is not None and id(runtime) not in seen:
            seen.add(id(runtime))
            runtime.close()
