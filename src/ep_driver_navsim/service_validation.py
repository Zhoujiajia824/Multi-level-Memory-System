"""Read-only model-service attestation before any official evaluation work.

GET endpoints never perform inference. A localhost vLLM whose optional developer
configuration endpoint is disabled can be attested through its same-user
listening process. No developer routes are enabled; raw configuration, command
lines, environment, credentials, PID and uptime are never archived.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

import yaml


PROTOCOL = "navsim-model-service-validation-1"
_FIELDS = ("max_model_len", "max_num_seqs", "image_limit", "max_pixels_per_image", "structured_outputs")


class ModelServiceValidationError(ValueError):
    pass


class ModelServiceVerificationUnsupported(ModelServiceValidationError):
    pass


def _get_json(url, headers, timeout):
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(2*1024*1024+1)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code in (404, 501):
            raise ModelServiceVerificationUnsupported("Service configuration endpoint is unavailable (HTTP %s)" % code) from None
        raise ModelServiceValidationError("Model service GET was rejected (HTTP %s)" % code) from None
    except (OSError, urllib.error.URLError) as exc:
        raise ModelServiceValidationError("Model service GET failed (%s); no evaluation was started" % type(exc).__name__) from None
    if len(raw) > 2*1024*1024:
        raise ModelServiceVerificationUnsupported("Model service metadata exceeds the safe read limit")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise ModelServiceVerificationUnsupported("Model service metadata is not JSON") from None


def _integer(value, label):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value) and int(value) > 0:
        return int(value)
    raise ModelServiceVerificationUnsupported("Cannot attest service integer field: " + label)


def _structured(value):
    if (not isinstance(value, dict) or not isinstance(value.get("backend"), str)
            or type(value.get("disable_any_whitespace")) is not bool):
        raise ModelServiceVerificationUnsupported("Cannot attest explicit structured backend/whitespace policy")
    return {key: value[key] for key in ("backend", "disable_any_whitespace")}


def _info_config(value):
    """Keep an allowlist only; the endpoint also contains sensitive environment."""
    config = value.get("vllm_config") if isinstance(value, dict) else None
    if not isinstance(config, dict):
        raise ModelServiceVerificationUnsupported("Service configuration endpoint did not return structured configuration")
    model = config.get("model_config", {})
    scheduler = config.get("scheduler_config", {})
    if not isinstance(model, dict) or not isinstance(scheduler, dict):
        raise ModelServiceVerificationUnsupported("Unrecognized model/scheduler configuration")
    multimodal = model.get("multimodal_config") or config.get("multimodal_config") or {}
    if not isinstance(multimodal, dict):
        raise ModelServiceVerificationUnsupported("Unrecognized multimodal configuration")
    limits = multimodal.get("limit_per_prompt") or multimodal.get("limit_per_prompt_dict") or {}
    if not isinstance(limits, dict):
        raise ModelServiceVerificationUnsupported("Unrecognized multimodal prompt limits")
    image = limits.get("image")
    if isinstance(image, dict):
        image = image.get("count")
    processor = multimodal.get("mm_processor_kwargs") or {}
    output = model.get("override_generation_config") or {}
    if not isinstance(processor,dict) or not isinstance(output,dict):
        raise ModelServiceVerificationUnsupported("Unrecognized processor/output configuration")
    return dict(max_model_len=_integer(model.get("max_model_len"), "max_model_len"),
                max_num_seqs=_integer(scheduler.get("max_num_seqs"), "max_num_seqs"),
                image_limit=_integer(image, "image_limit"),
                max_pixels_per_image=_integer(processor.get("max_pixels"), "max_pixels_per_image"),
                structured_outputs=_structured(config.get("structured_outputs_config")),
                output_token_cap=_integer(output.get("max_new_tokens"), "explicit output_token_cap"))


def _option(argv, name):
    values = []
    for index, word in enumerate(argv):
        if word == name:
            if index+1 >= len(argv) or argv[index+1].startswith("--"):
                raise ModelServiceVerificationUnsupported("Missing explicit vLLM option: " + name)
            values.append(argv[index+1])
        elif word.startswith(name+"="):
            values.append(word[len(name)+1:])
    if len(values) != 1:
        raise ModelServiceVerificationUnsupported("Require exactly one explicit vLLM option: " + name)
    return values[0]


def _argv_config(argv, port, model_id):
    try:
        index = argv.index("serve")
    except ValueError:
        return None
    if not any(Path(value).name == "vllm" or value.startswith("vllm.") for value in argv[:index]):
        return None
    if _integer(_option(argv, "--port"), "port") != port:
        raise ModelServiceVerificationUnsupported("Listener process port differs from the model endpoint")
    if _option(argv, "--served-model-name") != model_id:
        raise ModelServiceValidationError("Listener process does not serve the requested model ID")
    try:
        structured = json.loads(_option(argv, "--structured-outputs-config"))
        limit = json.loads(_option(argv, "--limit-mm-per-prompt"))
        processor = json.loads(_option(argv, "--mm-processor-kwargs"))
        output = json.loads(_option(argv, "--override-generation-config"))
    except ModelServiceVerificationUnsupported:
        raise
    except (ValueError, TypeError):
        raise ModelServiceVerificationUnsupported("Explicit vLLM deployment options are not JSON") from None
    return dict(max_model_len=_integer(_option(argv, "--max-model-len"), "max_model_len"),
                max_num_seqs=_integer(_option(argv, "--max-num-seqs"), "max_num_seqs"),
                image_limit=_integer(limit.get("image"), "image_limit"),
                max_pixels_per_image=_integer(processor.get("max_pixels"), "max_pixels_per_image"),
                structured_outputs=_structured(structured),
                output_token_cap=_integer(output.get("max_new_tokens"), "output_token_cap"))


def _local_process_config(parsed, model_id, proc_root=Path("/proc")):
    """Tie /proc argv to the HTTP port's socket, never merely a matching name."""
    if os.name != "posix" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ModelServiceVerificationUnsupported("Configuration endpoint unavailable; local listener inspection is unsupported")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    uid = os.getuid()
    inodes = set()
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc_root/"net"/name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            local = fields[1].rsplit(":", 1)
            loopbacks = {"00000000", "0100007F", "00000000000000000000000000000000", "00000000000000000000000001000000"}
            if (len(local) == 2 and local[0].upper() in loopbacks and int(local[1],16) == port
                    and fields[3] == "0A" and fields[7] == str(uid)):
                inodes.add(fields[9])
    if not inodes:
        raise ModelServiceVerificationUnsupported("No same-user localhost listener found for model endpoint")
    holders = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            for fd in (directory/"fd").iterdir():
                try:
                    link = os.readlink(fd)
                except OSError:
                    continue
                if link in {"socket:["+inode+"]" for inode in inodes}:
                    holders.append(directory)
                    break
        except OSError:
            continue
    configs = []
    errors = []
    for holder in holders:
        directory, visited = holder, set()
        while directory.name not in visited and directory.name.isdigit():
            visited.add(directory.name)
            try:
                if directory.stat().st_uid != uid:
                    break
                argv = [word.decode("utf-8") for word in (directory/"cmdline").read_bytes().split(b"\0") if word]
                config = _argv_config(argv, port, model_id)
                if config is not None:
                    configs.append(config)
                    break
                status = (directory/"status").read_text()
                parent = re.search(r"^PPid:\s*(\d+)", status, re.MULTILINE)
                if parent is None or parent[1] == "0":
                    break
                directory = proc_root/parent[1]
            except ModelServiceVerificationUnsupported as exc:
                errors.append(str(exc))
                break
            except (OSError, UnicodeError):
                break
    if not configs:
        detail = errors[0] if errors else "Listener is not tied to an explicit same-user vLLM serve command"
        raise ModelServiceVerificationUnsupported(detail)
    if errors or len(configs) != len(holders):
        raise ModelServiceVerificationUnsupported("Not every localhost listener could be tied to the same explicit vLLM deployment")
    if any(value != configs[0] for value in configs[1:]):
        raise ModelServiceVerificationUnsupported("Ambiguous localhost model listeners have different deployment settings")
    return configs[0]


def validate_model_service(settings, *, fetch_json=None, process_inspector=None):
    """Return only stable safe metadata; strict is the formal-evaluation default.

    ``manual`` is an explicit unsupported-endpoint escape hatch. It always
    records verified=False and cannot masquerade as actual compact attestation.
    """
    if settings.get("fast") != "server" or not settings.get("structured_model_output", False):
        return None
    mode = settings.get("model_service_validation_mode", "strict")
    if mode not in ("strict", "manual"):
        raise ModelServiceValidationError("model_service_validation_mode must be strict or manual")
    model = yaml.safe_load(Path(settings["models_config"]).read_text(encoding="utf-8"))["fast"]
    parsed = urllib.parse.urlsplit(model["base_url"])
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ModelServiceValidationError("Model endpoint must not contain embedded credentials or query parameters")
    base = model["base_url"].rstrip("/")
    if not parsed.path.rstrip("/").endswith("/v1"):
        raise ModelServiceVerificationUnsupported("Model service validation requires an OpenAI /v1 endpoint")
    server = base[:-3]
    key = os.environ.get(model.get("api_key_env", ""), "")
    headers = {"Authorization": "Bearer "+key} if key else {}
    fetch = fetch_json or _get_json
    models = fetch(base+"/models", headers, 8.)
    if not isinstance(models,dict) or not isinstance(models.get("data"),list):
        raise ModelServiceVerificationUnsupported("Model service did not advertise a model list")
    matches = [row for row in models.get("data", []) if isinstance(row,dict) and row.get("id") == model["model"]]
    if len(matches) != 1:
        raise ModelServiceValidationError("Requested model ID is not uniquely advertised by the configured service")
    version = fetch(server+"/version", headers, 8.)
    version = version.get("version") if isinstance(version,dict) else None
    if not isinstance(version,str) or not version.strip() or "\n" in version:
        raise ModelServiceVerificationUnsupported("Actual model service version is unavailable")
    observed = dict(model_id=model["model"], vllm_version=version,
                    max_model_len=_integer(matches[0].get("max_model_len"), "advertised max_model_len"))
    declared = settings.get("model_service", {})
    if not isinstance(declared,dict):
        raise ModelServiceValidationError("Declared model service settings must be an object")
    if declared.get("max_model_len") != observed["max_model_len"]:
        raise ModelServiceValidationError("Model service max_model_len differs from its declared configuration")
    if declared.get("vllm_version") not in (None, version):
        raise ModelServiceValidationError("Model service version differs from its declared configuration")
    if mode == "manual":
        return dict(schema=PROTOCOL, verified=False, verification_method="manual_declaration",
                    effective_service=observed,
                    unverified_fields=[field for field in _FIELDS if field != "max_model_len"] + ["output_token_cap"])
    try:
        actual = _info_config(fetch(server+"/server_info?config_format=json", headers, 8.))
        method = "service_configuration_api"
    except ModelServiceVerificationUnsupported:
        inspector = process_inspector or _local_process_config
        actual = inspector(parsed, model["model"])
        method = "same_user_local_listener_and_api"
    if actual["max_model_len"] != observed["max_model_len"]:
        raise ModelServiceValidationError("Model API and attested service context limits disagree")
    for field in _FIELDS:
        if field != "structured_outputs" and type(declared.get(field)) is not int:
            raise ModelServiceValidationError("Declared model service integer is missing or invalid: " + field)
        if declared.get(field) != actual[field]:
            raise ModelServiceValidationError("Actual model service differs in " + field + "; choose matching configuration before evaluation")
    output = _integer(model.get("max_tokens", 4096), "requested max_tokens")
    if actual["output_token_cap"] < output:
        raise ModelServiceValidationError("Model service output cap is lower than the requested completion budget")
    observed.update(actual)
    identity = hashlib.sha256(json.dumps(observed,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return dict(schema=PROTOCOL, verified=True, verification_method=method,
                effective_service=observed, service_identity_sha256=identity,
                verified_fields=["model_id", "vllm_version", *_FIELDS, "output_token_cap"])
