"""Run isolation and explicit persistent-memory access (no CARLA dependency)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

from carla_bridge.inference_mode import model_inference_enabled


class BankLock:
    """OS-released exclusive lock, outside the bank; also protects read snapshots.

    Readers are serialized too, so they cannot observe a partially saved index.
    The lock file must never be unlinked: other processes may hold its inode.
    """

    def __init__(self, bank: Path):
        self.path = bank.parent / ("." + bank.name + ".lock")
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"记忆库正在被其他运行使用: {self.path}") from exc
        self.handle = handle
        return self

    def close(self):
        if self.handle is not None:
            self.handle.close()  # OS releases the lock, including after process exit.
            self.handle = None


def _safe_config(value):
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(s in k.lower() for s in
                ("api_key", "password", "secret", "authorization")) else _safe_config(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_config(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class RunSession:
    def __init__(self, config, scenario_path, mode, memory_bank=None,
                 memory_access="read_write", memory_seed=None, short_term=True, retrieve=True):
        self.config = config
        self.mode = mode
        self.run_id = uuid.uuid4().hex[:16]
        self.lock = None
        self.manifest = {}
        self.scenario_path = Path(scenario_path).resolve()
        self.memory_bank = memory_bank
        self.memory_seed = memory_seed
        self.access = "off" if mode == "memory_off" else memory_access
        self.short_term = short_term and mode != "memory_off"
        self.retrieve = retrieve and mode != "memory_off"
        self.started_at = datetime.now().astimezone()
        # Use the case filename (including split/seed) and portable path characters.
        scenario_label = re.sub(r"[^\w-]+", "_", self.scenario_path.stem).strip("_")[:64] or "scenario"
        planning = config.data.get("carla", {}).get("planning", {})
        self.model_inference = model_inference_enabled(planning, mode)
        self.semantic_policy = planning.get("semantic_policy", "always")
        self.planning_memory = planning.get("planning_memory", "off")
        self.numeric_memory_only = not self.model_inference and self.planning_memory == "full"
        if not self.model_inference:
            self.short_term = False
        backend = planning.get("backend", "vlm_trajectory")
        planner_label = "pdm_closed" if backend == "pdm_closed" else "vlm_trajectory"
        semantic = ("sem_on" if planning.get("semantic_influence", True) else "sem_off") if backend == "pdm_closed" else "sem_na"
        memory_label = "memory_off" if self.access == "off" else "memory_" + self.access
        if mode != "memory_off":
            if not self.retrieve:
                memory_label += "_no_retrieval"
            if not self.short_term:
                memory_label += "_no_short_term"
        labels = [self.started_at.strftime("%Y%m%d_%H%M%S"), scenario_label,
                  memory_label, planner_label, semantic]
        if not self.model_inference:
            labels.append("model_off")
        # Configurations predating these controls retain their legacy labels.
        if "semantic_policy" in planning:
            labels.append("policy_" + self.semantic_policy)
        if "planning_memory" in planning:
            labels.append("pm_" + self.planning_memory)
        dirname = "__".join(labels + [self.run_id[:8]])
        output_root = Path(config.get("output_dir")) / "carla_runs"
        # Native Windows libraries can still enforce MAX_PATH despite Python's
        # long-path support. Reserve room for decisions/reports and asset names;
        # keep full experimental labels in the manifest instead of dropping IDs.
        if os.name == "nt" and ("semantic_policy" in planning or "planning_memory" in planning):
            reserve = max(96, len(f"decisions_carla_{self.scenario_path.stem}_{mode}.jsonl"))
            budget = 258 - len(str(output_root.resolve())) - 1 - reserve
            if len(dirname) > budget:
                compact = "_".join((
                    "pdm" if backend == "pdm_closed" else "vlm",
                    "s1" if planning.get("semantic_influence", True) else "s0",
                    "i1" if self.model_inference else "i0",
                    {"always": "a", "selective": "s", "shadow": "h"}[self.semantic_policy],
                    {"off": "o", "gate": "g", "full": "f"}[self.planning_memory],
                    {"off": "mo", "read_write": "rw", "read_only": "ro"}.get(self.access, "mx"),
                    "r" if self.retrieve else "nr", "st" if self.short_term else "nst"))
                suffix = f"__{compact}__{self.run_id[:8]}"
                prefix = self.started_at.strftime("%Y%m%d_%H%M%S")
                room = budget - len(prefix) - len(suffix) - 2
                dirname = prefix + ("__" + scenario_label[:room] if room > 0 else "") + suffix
                if len(dirname) > budget:
                    dirname = self.run_id
                if len(dirname) > budget:
                    raise ValueError("Windows 输出路径过长，请将 output_dir 配置为更短的目录")
        self.run_dir = output_root / dirname

    def _path(self, value):
        p = Path(value)
        return (p if p.is_absolute() else self.config.root / p).resolve()

    def _bank_exists(self, bank):
        return ((bank / "mid_term_meta.json").is_file()
                or (self.numeric_memory_only and (bank / "planning_experiences.jsonl").is_file()))

    def prepare(self):
        if self.access not in ("read_write", "read_only", "off"):
            raise ValueError("无效的 memory_access")
        if self.mode == "memory_off" and (self.memory_bank or self.memory_seed):
            raise ValueError("memory_off 不使用记忆库，请移除 --memory-bank/--memory-seed")
        if self.access == "read_only" and (not self.memory_bank or self.memory_seed):
            raise ValueError("只读模式必须指定 --memory-bank，且不能指定 --memory-seed")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        bank = self._path(self.memory_bank) if self.memory_bank else self.run_dir / "memory_db"
        seed = self._path(self.memory_seed) if self.memory_seed else None
        if seed and (bank == seed or seed in bank.parents or bank in seed.parents):
            raise ValueError("快照源和目标库不能相同或相互包含")
        try:
            if self.access != "off":
                self.lock = BankLock(bank).acquire()
                if seed:
                    source_lock = BankLock(seed).acquire()
                    try:
                        if not self._bank_exists(seed):
                            raise ValueError(f"历史记忆库不存在或不完整: {seed}")
                        if bank.exists():
                            raise ValueError("复制历史记忆需要一个尚不存在的目标库")
                        shutil.copytree(seed, bank)
                        # Absolute asset references must point to the independent copy.
                        meta = bank / "mid_term_meta.json"
                        asset_keys = {"image_path", "image_feature_path", "feature_path",
                                      "anchor_image_path", "key_image_paths"}
                        def relocate(v, key=""):
                            if isinstance(v, dict):
                                return {k: relocate(x, k) for k, x in v.items()}
                            if isinstance(v, list):
                                return [relocate(x, key) for x in v]
                            if isinstance(v, str) and v and key in asset_keys:
                                original = self._path(v)
                                if not original.is_file():
                                    raise FileNotFoundError(f"历史库引用的资料不存在: {original}")
                                try:
                                    return str(bank / original.relative_to(seed))
                                except ValueError:
                                    # Older banks kept their images outside the database.
                                    # Bring those references into the snapshot as well.
                                    digest = hashlib.sha256(str(original).encode()).hexdigest()[:16]
                                    target = bank / "assets" / "imported" / (digest + original.suffix)
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(original, target)
                                    return str(target)
                            return v
                        if meta.is_file():
                            meta.write_text(json.dumps(relocate(json.loads(meta.read_text(encoding="utf-8"))),
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
                    finally:
                        source_lock.close()
                if self.access == "read_only" and not self._bank_exists(bank):
                    raise ValueError(f"只读记忆库不存在或不完整: {bank}")
                if self.access == "read_write":
                    bank.mkdir(parents=True, exist_ok=True)
            writable = self.access == "read_write"
            # Assets referenced by persistent memories live with the bank, not disposable logs.
            assets = bank / "assets" if writable else self.run_dir / "assets"
            data = self.config.data
            data.update(output_dir=self.run_dir, log_dir=self.run_dir / "logs",
                        feature_dir=assets / "features", memory_db_dir=bank)
            data["memory_access"] = {"retrieve": self.retrieve, "write": writable,
                                     "short_term": self.short_term}
            data.setdefault("mid_term", {})["persistence"] = {
                "enabled": self.access != "off" and self.model_inference,
                "auto_load_on_init": self.access != "off" and self.model_inference,
                "save_on_close": writable and self.model_inference, "strict_load": True,
            }
            data.setdefault("persistence", {})["save_dir"] = bank
            data["carla"]["output"] = {"image_dir": str(assets / "images"),
                                         "mosaic_dir": str(assets / "mosaic")}
            data["run_id"] = self.run_id
            data["timestamp_origin_us"] = time.time_ns() // 1000
            # Do not call ensure_output_dirs(): it also creates shared/readonly bank paths.
            for path in (data["log_dir"], data["feature_dir"], assets / "images", assets / "mosaic"):
                path.mkdir(parents=True, exist_ok=True)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=self.config.root,
                                               text=True, stderr=subprocess.DEVNULL).strip()
            try:
                revision, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
            except (OSError, subprocess.CalledProcessError):
                revision, dirty = "unknown", None
            self.manifest = {"run_id": self.run_id, "status": "prepared", "mode": self.mode,
                             "run_directory": self.run_dir.name,
                             "started_at": self.started_at.isoformat(),
                             "model_inference": self.model_inference,
                             "semantic_policy": self.semantic_policy,
                             "planning_memory": self.planning_memory,
                             "planning_memory_effective": self.planning_memory if self.access != "off" else "off",
                             "semantic_schema_version": data.get("semantic", {}).get(
                                 "selective_schema_version" if self.semantic_policy in ("selective", "shadow") else "schema_version",
                                 2 if self.semantic_policy in ("selective", "shadow") else 1),
                             "semantic_prompt_version": data.get("semantic", {}).get(
                                 "selective_prompt_version" if self.semantic_policy in ("selective", "shadow") else "prompt_version"),
                             "memory_access": self.access, "memory_bank": str(bank),
                             "memory_seed": str(seed) if seed else None,
                             "commit": revision, "dirty": dirty,
                             "scenario_path": str(self.scenario_path),
                             "scenario_yaml": self.scenario_path.read_text(encoding="utf-8"),
                             "config": _safe_config(data)}
            self.set_status("prepared")
            return self
        except BaseException:
            self.close()
            raise

    def set_status(self, status, error=None, outcome=None):
        self.manifest["status"] = status
        if outcome is not None:
            self.manifest["outcome"] = outcome
        if error is not None:
            self.manifest["error_type"] = type(error).__name__
        path = self.run_dir / "manifest.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)

    def close(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None
