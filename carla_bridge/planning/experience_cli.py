"""Inspect execution memory and import externally evaluated execution pairs.

No simulator, model, embedding index or network access is needed. Import does
not manufacture counterfactuals: input must satisfy the execution-pair contract
in experience_memory. Local artifact hashes check integrity, not experimental
truth or completeness of a simulator state restoration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

from carla_bridge.run_session import BankLock
from .experience_memory import ExecutionExperienceMemory, EXPERIENCE_FILENAME


DEFAULT_PLANNER_VERSION = "carla_pdm_selective_v1"
MAX_INPUT_BYTES = 32 * 1024 * 1024


class EvidenceImportError(ValueError):
    """A complete batch failed validation; no bank update should be applied."""


def _reject_constant(_value):
    raise EvidenceImportError("non-finite JSON numbers are forbidden")


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise EvidenceImportError("duplicate JSON object keys are forbidden")
        result[key] = value
    return result


def _loads(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _bank_directory(bank):
    path = Path(bank).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise EvidenceImportError("--bank must name a memory-bank directory")
    return path


def _memory(bank, planner_version, *, mode="read_only"):
    return ExecutionExperienceMemory(bank, memory_mode=mode,
        planner_version=planner_version, run_id="offline_pair_import_cli")


def _checked_experience_path(bank):
    path = bank / EXPERIENCE_FILENAME
    if path.is_symlink():
        raise EvidenceImportError("experience file symlinks are not supported; use a regular bank file")
    return path


def _inspect_locked(bank, planner_version):
    path = _checked_experience_path(bank)
    memory = _memory(bank, planner_version)
    records = memory._load()  # Same strict validation as every online retrieval.
    versions, schemas, raw_json_errors = Counter(), Counter(), []
    if path.is_file():
        with path.open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = _loads(line)
                    if isinstance(item, dict):
                        versions[str(item.get("planner_version", "<missing>"))] += 1
                        schemas[str(item.get("schema_version", "<missing>"))] += 1
                except (ValueError, TypeError):
                    raw_json_errors.append(line_no)
    health = memory.summarize()
    return {"bank_exists": bank.is_dir(), "experience_file_exists": path.is_file(),
        "planner_version": planner_version, "detected_planner_versions": dict(versions),
        "detected_schema_versions": dict(schemas),
        "raw_json_invalid_lines": raw_json_errors,
        "compatible_record_kinds": dict(Counter(r["kind"] for r in records)),
        "compatible_record_states": dict(Counter(r.get("status", "paired_evidence") for r in records)),
        "health": health, "trained_model": False,
        "evidence_limit": "integrity_and_contract_checks_do_not_certify_external_execution_truth"}


def inspect_bank(bank, *, planner_version=DEFAULT_PLANNER_VERSION):
    """Read a consistent snapshot; do not create a missing bank or its lock."""
    bank = _bank_directory(bank)
    _checked_experience_path(bank)
    if not bank.exists():
        return _inspect_locked(bank, planner_version)
    lock = BankLock(bank).acquire()
    try:
        return _inspect_locked(bank, planner_version)
    finally:
        lock.close()


def _load_batch(source):
    if not source.is_file():
        raise EvidenceImportError("--input must name a regular JSON or JSONL file")
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise EvidenceImportError("input exceeds the 32 MiB batch limit; split the batch")
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".jsonl":
        batch = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                batch.append(_loads(line))
            except ValueError as exc:
                raise EvidenceImportError("invalid JSONL at line %d" % line_no) from exc
    else:
        try:
            value = _loads(text)
        except ValueError as exc:
            raise EvidenceImportError("invalid JSON input") from exc
        batch = value if isinstance(value, list) else [value]
    if not batch or any(not isinstance(item, dict) for item in batch):
        raise EvidenceImportError("input must contain one or more pair objects")
    return batch


def _verify_local_artifacts(evidence, root):
    """Only inspect explicitly referenced regular files within an allowed tree.

    Evidence content cannot request HTTP/UNC retrieval or access another local
    directory. Resolve symlinks before containment checks. Never echo artifact
    bytes or evidence payloads to stdout/stderr.
    """
    verified = missing = 0
    for name in ("baseline", "intervention"):
        arm = evidence[name]
        local = arm.get("artifact_path")
        if local is None:
            missing += 1
            continue
        if not isinstance(local, str) or not local or "\x00" in local or "://" in local or local.startswith(("\\\\", "//")):
            raise EvidenceImportError(name + " artifact_path must name a permitted local file")
        try:
            candidate = Path(local).expanduser()
            candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
            candidate.relative_to(root)
        except (ValueError, OSError) as exc:
            raise EvidenceImportError(name + " artifact_path escapes the permitted artifact root") from exc
        try:
            if not stat.S_ISREG(candidate.stat().st_mode):
                raise EvidenceImportError(name + " artifact_path must name a regular file")
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise EvidenceImportError(name + " local artifact is unavailable") from exc
        if digest.hexdigest() != arm["artifact_sha256"]:
            raise EvidenceImportError(name + " local artifact SHA256 mismatch")
        verified += 1
    return verified, missing


def import_pairs(bank, source, *, planner_version=DEFAULT_PLANNER_VERSION, artifact_root=None):
    """Validate the entire batch before acquiring the bank lock or changing it.

    Under the same exclusive lock as RunSession, preserve existing bytes and
    atomically replace the file with the complete appended batch. A validation,
    lock, staging or replacement failure cannot leave a partially imported batch.
    Reimporting identical pair IDs is idempotent; changing an existing ID fails.
    """
    bank = _bank_directory(bank)
    _checked_experience_path(bank)
    source = Path(source).expanduser().resolve()
    root = Path(artifact_root).expanduser().resolve() if artifact_root else source.parent
    if not root.is_dir():
        raise EvidenceImportError("artifact root must be an existing directory")
    batch = _load_batch(source)
    validator = _memory(bank, planner_version)
    validated, unique, verified, unverified = [], {}, 0, 0
    for index, evidence in enumerate(batch, 1):
        try:
            for arm in ("baseline", "intervention"):
                if isinstance(evidence.get(arm), dict) and not isinstance(evidence[arm].get("metrics"), dict):
                    raise EvidenceImportError(arm + " metrics must be an object")
            record = validator._validated_pair_record(evidence)
            present, missing = _verify_local_artifacts(evidence, root)
        except (ValueError, TypeError, KeyError) as exc:
            raise EvidenceImportError("pair %d rejected: %s" % (index, exc)) from exc
        if record["id"] in unique and record != unique[record["id"]]:
            raise EvidenceImportError("batch contains conflicting evidence for the same pair ID")
        if record["id"] not in unique:
            unique[record["id"]] = record
            validated.append(record)
            verified += present
            unverified += missing

    lock, staged = BankLock(bank).acquire(), None
    try:
        _checked_experience_path(bank)
        existing = validator._load()
        if validator.summarize()["malformed_lines"]:
            raise EvidenceImportError("existing bank has malformed records; inspect it before importing")
        if any(message.startswith(("cannot read", "cannot stat")) for message in validator.errors):
            raise EvidenceImportError("existing bank could not be read; import refused")
        if _inspect_locked(bank, planner_version)["raw_json_invalid_lines"]:
            raise EvidenceImportError("existing bank has invalid or duplicate-key JSON; import refused")
        by_id = {record["id"]: record for record in existing}
        additions = []
        for record in validated:
            prior = by_id.get(record["id"])
            if prior is not None and prior != record:
                raise EvidenceImportError("existing pair ID has different evidence; import refused")
            if prior is None:
                additions.append(record)
        if additions:
            bank.mkdir(parents=True, exist_ok=True)
            target = bank / EXPERIENCE_FILENAME
            descriptor, name = tempfile.mkstemp(prefix=".planning-import-", suffix=".jsonl", dir=bank)
            staged = Path(name)
            with os.fdopen(descriptor, "wb") as output:
                if target.exists():
                    with target.open("rb") as current:
                        shutil.copyfileobj(current, output)
                        if current.tell():
                            current.seek(-1, os.SEEK_END)
                            if current.read(1) != b"\n":
                                output.write(b"\n")
                for record in additions:
                    output.write((json.dumps(record, sort_keys=True, ensure_ascii=False,
                                             allow_nan=False) + "\n").encode("utf-8"))
                output.flush()
                os.fsync(output.fileno())
            if target.exists():
                os.chmod(staged, stat.S_IMODE(target.stat().st_mode))
            os.replace(staged, target)
            staged = None
        return {"planner_version": planner_version, "input_pairs": len(batch),
            "unique_input_pairs": len(validated), "imported_pairs": len(additions),
            "already_present_pairs": len(validated) - len(additions),
            "locally_verified_artifacts": verified, "artifacts_without_local_verification": unverified,
            "evidence_limit": "external_execution_truth_and_state_restoration_remain_unverified",
            "health": inspect_after_import(bank, planner_version)}
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
        lock.close()


def inspect_after_import(bank, planner_version):
    # Caller already owns BankLock: do not recursively acquire an exclusive lock.
    return _inspect_locked(bank, planner_version)["health"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "import-pairs"):
        sub = commands.add_parser(command)
        sub.add_argument("--bank", required=True, help="same bank directory used by --memory-bank")
        sub.add_argument("--planner-version", default=DEFAULT_PLANNER_VERSION)
        if command == "import-pairs":
            sub.add_argument("--input", required=True, help="JSON pair/object array or .jsonl batch")
            sub.add_argument("--artifact-root", help="explicitly allowed local artifact directory tree")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_bank(args.bank, planner_version=args.planner_version)
        else:
            result = import_pairs(args.bank, args.input, planner_version=args.planner_version,
                                  artifact_root=args.artifact_root)
    except (ValueError, RuntimeError, OSError, UnicodeError) as exc:
        # Validation messages describe fields, never dump evidence or artifacts.
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
