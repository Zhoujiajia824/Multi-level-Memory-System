"""Offline, in-process Qwen2.5-VL inference; no HTTP service or API fallback."""
from __future__ import annotations

import argparse
import gc
import json
import math
import threading
import time
import warnings
from pathlib import Path
from typing import List, Optional

from src.vla_memory.common.logging_utils import get_logger
from src.vla_memory.common.model_call_audit import describe_model_request, emit_model_request
from src.vla_memory.perception.vlm_client import VLMClient

logger = get_logger("local_qwen_vlm")


def validate_local_model_directory(model_path):
    """Check a local export, including every referenced weight shard, offline."""
    if not isinstance(model_path, (str, Path)) or not str(model_path).strip():
        raise ValueError("请设置本地 VLM 的 model_path，不接受远程模型 ID")
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("本地 Qwen 模型目录不存在: %s" % root)

    def required(name):
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError("模型文件路径越界: %s" % name)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError("本地模型文件缺失或为空: %s" % path)
        with path.open("rb") as stream:
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError("模型文件仍是 Git LFS 指针，未下载真实文件: %s" % path)
        return path

    config = json.loads(required("config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("本地后端要求 Qwen2.5-VL（model_type=qwen2_5_vl）")
    required("preprocessor_config.json")
    tokenizer_config = json.loads(required("tokenizer_config.json").read_text(encoding="utf-8"))
    if (root / "chat_template.jinja").is_file():
        required("chat_template.jinja")
    elif (root / "chat_template.json").is_file():
        template = json.loads(required("chat_template.json").read_text(encoding="utf-8"))
        if not isinstance(template, dict) or not template.get("chat_template"):
            raise ValueError("chat_template.json 中缺少非空 chat_template")
    elif not tokenizer_config.get("chat_template"):
        raise FileNotFoundError("本地 VLM 聊天模板缺失：需要 chat_template.json/jinja 或tokenizer_config中的模板")
    if (root / "tokenizer.json").is_file():
        required("tokenizer.json")
    else:
        required("vocab.json")
        required("merges.txt")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        contents = json.loads(required(index.name).read_text(encoding="utf-8"))
        weight_map = contents.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("模型分片索引 weight_map 为空或无效")
        shards = list(weight_map.values())
        if any(not isinstance(shard, str) or not shard.endswith(".safetensors") for shard in shards):
            raise ValueError("模型分片索引包含无效文件名")
        for shard in set(shards):
            required(shard)
    else:
        required("model.safetensors")
    return root


class LocalQwenRuntime:
    """One lazily loaded model shared by scene understanding and decision."""

    def __init__(self, model_path, device="cuda:0", dtype="auto", attn_implementation="sdpa",
                 min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = str(device)
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
            if type(value) is not int or value <= 0:
                raise ValueError("%s 必须是正整数" % key)
        if min_pixels > max_pixels:
            raise ValueError("min_pixels 不能超过 max_pixels")
        if dtype not in ("auto", "bfloat16", "float16", "float32"):
            raise ValueError("dtype 必须为 auto/bfloat16/float16/float32")
        if attn_implementation not in ("sdpa", "eager"):
            raise ValueError("本地 VLM 仅支持 sdpa/eager，无需安装 flash-attn")
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self._model = self._processor = self._torch = self._device = None
        self._lock = threading.RLock()

    def check_environment(self):
        validate_local_model_directory(self.model_path)
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("本地 Qwen 依赖不可用，请在 mulmem_carla 安装 "
                               "carla_bridge/requirements-local-vlm.txt: %s" % exc) from exc
        device = torch.device(self.device)
        if device.type not in ("cuda", "cpu"):
            raise ValueError("本地 VLM device 必须为 cpu 或 cuda:N")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("本地 VLM 要求 CUDA，但当前 PyTorch 无可用 GPU；检查CUDA版torch或改用cpu")
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise ValueError("本地 VLM GPU %s 不存在；注意 CUDA_VISIBLE_DEVICES 会重新编号" % device)
            device = torch.device("cuda:%d" % index)
            if self.dtype == "bfloat16":
                with torch.cuda.device(device):
                    if not torch.cuda.is_bf16_supported():
                        raise ValueError("所选GPU不支持bfloat16，请改为float16或auto")
        elif self.dtype == "float16":
            raise ValueError("CPU本地推理请使用dtype=float32或auto")
        return torch, AutoProcessor, Qwen2_5_VLForConditionalGeneration, device

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            torch, processor_class, model_class, device = self.check_environment()
            dtype = self.dtype
            if dtype == "auto":
                if device.type == "cpu":
                    dtype = "float32"
                else:
                    with torch.cuda.device(device):
                        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            options = {"local_files_only": True, "trust_remote_code": False}
            logger.info("本地VLM加载: model=%s device=%s dtype=%s offline=True",
                        self.model_path, device, dtype)
            processor = model = None
            failure = None
            try:
                processor = processor_class.from_pretrained(str(self.model_path),
                    min_pixels=self.min_pixels, max_pixels=self.max_pixels, **options)
                if not getattr(processor, "chat_template", None):
                    template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
                    if not template:
                        raise ValueError("本地 VLM processor 未加载聊天模板，请核对 chat_template 文件")
                    processor.chat_template = template
                model = model_class.from_pretrained(str(self.model_path),
                    torch_dtype=getattr(torch, dtype), attn_implementation=self.attn_implementation,
                    low_cpu_mem_usage=True, use_safetensors=True, **options)
                model = model.to(device)
                model.eval()
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
                processor = model = None
            if failure is not None:
                # Leave the except scope first: its traceback can retain a
                # partially moved model, even after our own locals are cleared.
                gc.collect()
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                raise RuntimeError("本地 VLM 加载失败（已清理未完成的模型）: " + failure)
            # Publish only a completely loaded pair; failed loads remain retryable.
            self._torch, self._device = torch, device
            self._processor, self._model = processor, model

    def generate(self, messages, image_paths, *, max_tokens, temperature, timeout):
        from PIL import Image, ImageOps
        with self._lock:
            self.load()
            images = []
            inputs = generated = tokens = None
            failure, failure_kind = None, RuntimeError
            dispatched = False
            try:
                for path in image_paths:
                    with Image.open(path) as source:
                        images.append(ImageOps.exif_transpose(source).convert("RGB"))
                converted, image_index = [], 0
                for message in messages:
                    content = message["content"]
                    if isinstance(content, list):
                        blocks = []
                        for block in content:
                            if block.get("type") == "image_url":
                                blocks.append({"type": "image", "image": images[image_index]})
                                image_index += 1
                            else:
                                blocks.append(dict(block))
                        content = blocks
                    converted.append({"role": message["role"], "content": content})
                if image_index != len(images):
                    raise ValueError("本地 VLM 图像列表与消息中的占位数不匹配")
                text = self._processor.apply_chat_template(converted, tokenize=False,
                                                          add_generation_prompt=True)
                inputs = self._processor(text=[text], images=images or None,
                                         padding=True, return_tensors="pt").to(self._device)
                input_length = inputs["input_ids"].shape[-1]
                text_config = getattr(self._model.config, "text_config", self._model.config)
                context_limit = getattr(text_config, "max_position_embeddings", None)
                if isinstance(context_limit, int) and input_length + max_tokens > context_limit:
                    raise ValueError("本地 VLM 上下文超限：输入%d + 输出预算%d > %d；"
                                     "请减少记忆/图片像素或max_tokens" %
                                     (input_length, max_tokens, context_limit))
                options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0,
                           "max_time": float(timeout), "use_cache": True}
                if temperature > 0:
                    options["temperature"] = temperature
                started = time.perf_counter()
                emit_model_request("request_start", attempt=1, dispatched=True, backend="local",
                    provider="local_qwen", model=str(self.model_path), device=str(self._device),
                    max_tokens=max_tokens, temperature=temperature, input_token_count=input_length)
                dispatched = True
                with self._torch.inference_mode():
                    generated = self._model.generate(**inputs, **options)
                tokens = generated[0][input_length:]
                eos = getattr(self._model.generation_config, "eos_token_id", None)
                if eos is None:
                    eos = getattr(self._processor.tokenizer, "eos_token_id", None)
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                ended = len(tokens) > 0 and int(tokens[-1]) in eos_ids
                duration = time.perf_counter() - started
                if len(tokens) >= max_tokens and not ended:
                    raise RuntimeError("本地 VLM 输出达到max_tokens且未结束，拒绝截断结果；请增加输出预算")
                if duration >= timeout and not ended:
                    raise TimeoutError("本地 VLM 生成达到时间限制，拒绝未结束的结果")
                result = self._processor.batch_decode([tokens], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)[0]
                if not isinstance(result, str) or not result.strip():
                    raise RuntimeError("本地 VLM 返回空输出")
                emit_model_request("request_end", attempt=1, dispatched=True, backend="local",
                    provider="local_qwen", status="success", response=result,
                    wall_duration_s=duration, output_token_count=len(tokens))
                return result
            except Exception as exc:
                failure = "%s: %s" % (type(exc).__name__, exc)
                failure_kind = (TimeoutError if isinstance(exc, TimeoutError) else
                                ValueError if isinstance(exc, ValueError) else RuntimeError)
                if dispatched:
                    emit_model_request("request_end", attempt=1, dispatched=True, backend="local",
                        provider="local_qwen", status="failed", error_type=type(exc).__name__,
                        error=str(exc), wall_duration_s=time.perf_counter()-started)
                inputs = generated = tokens = None
            finally:
                for image in images:
                    image.close()
            # A caller may retain this exception for a bounded retry. Do not
            # attach the original generate traceback (and its KV cache) to it.
            gc.collect()
            if self._device.type == "cuda":
                with self._torch.cuda.device(self._device):
                    self._torch.cuda.empty_cache()
            raise failure_kind("本地 VLM 推理失败（已清理本次临时张量）: " + failure)

    def close(self):
        with self._lock:
            torch, device = self._torch, self._device
            self._model = self._processor = None
            self._torch = self._device = None
            gc.collect()
            if torch is not None and device is not None and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()


class LocalQwenVLMClient(VLMClient):
    def __init__(self, runtime, max_tokens=2048, temperature=0.0, system_prompt=None, timeout=120):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens 必须为正整数")
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须为非负有限数")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须为正有限数")
        self.runtime, self.max_tokens, self.temperature = runtime, max_tokens, float(temperature)
        self.timeout, self.system_prompt = float(timeout), system_prompt
        self.provider, self.model_name = "local_qwen", str(runtime.model_path)

    def _call(self, stage, prompt, image_paths, extra_context=None):
        paths = [str(Path(path).expanduser().resolve()) for path in image_paths]
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError("本地 VLM 图像文件不存在: %s" % path)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        content = [{"type": "image_url", "image_url": {"url": Path(path).as_uri()}} for path in paths]
        content.append({"type": "text", "text": prompt})
        if extra_context:
            content.append({"type": "text", "text": "\n额外上下文:\n" +
                            "\n".join("%s: %s" % pair for pair in extra_context.items())})
        messages.append({"role": "user", "content": content})
        with describe_model_request(stage, messages, paths):
            result = self.runtime.generate(messages, paths, max_tokens=self.max_tokens,
                                           temperature=self.temperature, timeout=self.timeout)
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("本地 VLM 返回空输出")
        return result

    def understand_scene(self, image_path, prompt, extra_context=None):
        return self._call("scene", prompt, [image_path], extra_context)

    def decide(self, prompt, image_paths: Optional[List[str]] = None,
               optional_image_path: Optional[str] = None, max_images=4):
        if type(max_images) is not int or max_images <= 0:
            raise ValueError("max_images 必须为正整数")
        if image_paths is None:
            image_paths = [optional_image_path] if optional_image_path else []
            if optional_image_path:
                warnings.warn("optional_image_path 已弃用，请使用image_paths", DeprecationWarning, stacklevel=2)
        if len(image_paths) > max_images:
            logger.warning("本地 VLM 图片%d张超过%d，仅保留最近图片", len(image_paths), max_images)
            image_paths = image_paths[-max_images:]
        return self._call("decision", prompt, image_paths)


def main():
    parser = argparse.ArgumentParser(description="离线检查/真实推理本地 Qwen2.5-VL；不连接 CARLA")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--check", action="store_true", help="检查模型文件、依赖与设备；不加载权重")
    parser.add_argument("--image", help="使用一张真实本地图片执行推理")
    parser.add_argument("--prompt", default="请描述这张驾驶场景图片。")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    runtime = LocalQwenRuntime(args.model_path, args.device, args.dtype)
    try:
        runtime.check_environment()
        if args.check:
            print(json.dumps({"status": "ok", "model_path": str(runtime.model_path),
                              "device": args.device, "offline": True, "weights_loaded": False}, ensure_ascii=False))
        if args.image:
            client = LocalQwenVLMClient(runtime, max_tokens=args.max_new_tokens)
            print(client.understand_scene(args.image, args.prompt))
        elif not args.check:
            parser.error("请指定 --check 或 --image")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
