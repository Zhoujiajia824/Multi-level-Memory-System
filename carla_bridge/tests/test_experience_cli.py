"""Offline experience tooling: no partial batches or unrequested artifact reads."""
import hashlib
import json
from pathlib import Path

import pytest

from carla_bridge.planning.experience_cli import (
    DEFAULT_PLANNER_VERSION, EvidenceImportError, import_pairs, inspect_bank, main)
from carla_bridge.planning.experience_memory import EXPERIENCE_FILENAME
from carla_bridge.run_session import BankLock
from carla_bridge.tests.test_execution_experience_memory import VERSION, executed, pair


def source(tmp_path, values, name="pairs.json"):
    path = tmp_path / name
    if path.suffix == ".jsonl":
        path.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")
    else:
        path.write_text(json.dumps(values), encoding="utf-8")
    return path


def local_pair(tmp_path):
    evidence = pair()
    for arm in ("baseline", "intervention"):
        path = tmp_path / (arm + ".json")
        content = json.dumps({"test_fixture": arm}).encode()
        path.write_bytes(content)
        evidence[arm]["artifact_path"] = path.name
        evidence[arm]["artifact_sha256"] = hashlib.sha256(content).hexdigest()
    return evidence


def test_inspect_missing_bank_creates_no_bank_parent_or_lock(tmp_path):
    bank = tmp_path / "absent_parent" / "bank"
    result = inspect_bank(bank)
    assert not result["bank_exists"]
    assert result["planner_version"] == DEFAULT_PLANNER_VERSION
    assert not (tmp_path / "absent_parent").exists()


def test_inspect_reports_versions_states_and_malformed_rows_without_changing_bank(tmp_path):
    bank = tmp_path / "bank"
    executed(bank, "executed")
    path = bank / EXPERIENCE_FILENAME
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "unknown", "schema_version": 8, "planner_version": "other"}) + "\n")
        stream.write("{broken\n")
    before = path.read_bytes()
    result = inspect_bank(bank, planner_version=VERSION)
    assert path.read_bytes() == before
    assert sorted(p.name for p in bank.iterdir()) == [EXPERIENCE_FILENAME]
    assert result["compatible_record_states"] == {"observed": 1}
    assert result["detected_planner_versions"]["other"] == 1
    assert result["health"]["malformed_lines"] == 1
    assert result["health"]["incompatible_records"] == 1
    assert not result["trained_model"]


@pytest.mark.parametrize("name", ["pairs.json", "pairs.jsonl"])
def test_complete_json_and_jsonl_batches_import_atomically_and_idempotently(tmp_path, name):
    data = source(tmp_path, [pair("a"), pair("b")], name)
    bank = tmp_path / "bank"
    result = import_pairs(bank, data, planner_version=VERSION)
    assert result["imported_pairs"] == 2
    assert result["artifacts_without_local_verification"] == 4
    assert result["health"]["loaded_records"] == 2
    before = (bank / EXPERIENCE_FILENAME).read_bytes()
    repeated = import_pairs(bank, data, planner_version=VERSION)
    assert repeated["imported_pairs"] == 0
    assert repeated["already_present_pairs"] == 2
    assert (bank / EXPERIENCE_FILENAME).read_bytes() == before
    assert sorted(p.name for p in bank.iterdir()) == [EXPERIENCE_FILENAME]


def test_default_version_matches_closed_loop_and_cli_returns_json(tmp_path, capsys):
    evidence = pair()
    evidence["planner_version"] = DEFAULT_PLANNER_VERSION
    for arm in ("baseline", "intervention"):
        evidence[arm]["planner_version"] = DEFAULT_PLANNER_VERSION
    data = source(tmp_path, evidence)
    assert main(["import-pairs", "--bank", str(tmp_path / "bank"), "--input", str(data)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["planner_version"] == DEFAULT_PLANNER_VERSION
    assert result["imported_pairs"] == 1


@pytest.mark.parametrize("change", ["invalid_pair", "invalid_jsonl", "wrong_version", "conflicting_id"])
def test_entire_batch_validated_before_creating_bank_or_lock(tmp_path, change):
    evidence = pair("bad")
    if change == "invalid_pair":
        evidence["intervention"]["complete"] = False
    elif change == "wrong_version":
        evidence["planner_version"] = "wrong"
    elif change == "conflicting_id":
        evidence = pair("good")
        evidence["intervention"]["utility"] = .5
    data = source(tmp_path, [pair("good"), evidence], "pairs.jsonl")
    if change == "invalid_jsonl":
        with data.open("a", encoding="utf-8") as stream:
            stream.write("{broken\n")
    with pytest.raises(EvidenceImportError):
        import_pairs(tmp_path / "bank", data, planner_version=VERSION)
    assert not (tmp_path / "bank").exists()
    assert not (tmp_path / ".bank.lock").exists()


def test_existing_pair_id_cannot_be_silently_replaced_by_different_evidence(tmp_path):
    bank = tmp_path / "bank"
    data = source(tmp_path, pair())
    import_pairs(bank, data, planner_version=VERSION)
    before = (bank / EXPERIENCE_FILENAME).read_bytes()
    data = source(tmp_path, pair(gain=.5))
    with pytest.raises(EvidenceImportError, match="different evidence"):
        import_pairs(bank, data, planner_version=VERSION)
    assert (bank / EXPERIENCE_FILENAME).read_bytes() == before


def test_local_artifact_paths_are_hashed_and_mismatch_rejects_entire_batch(tmp_path):
    evidence = local_pair(tmp_path)
    data = source(tmp_path, evidence)
    result = import_pairs(tmp_path / "verified", data, planner_version=VERSION)
    assert result["locally_verified_artifacts"] == 2
    assert result["artifacts_without_local_verification"] == 0
    (tmp_path / "baseline.json").write_bytes(b"changed")
    with pytest.raises(EvidenceImportError, match="SHA256 mismatch"):
        import_pairs(tmp_path / "rejected", data, planner_version=VERSION)
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("reference", ["../outside.txt", "https://example.test/artifact", "//server/share/artifact"])
def test_evidence_cannot_request_external_files_or_network_access(tmp_path, reference, capsys):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    secret = "DO_NOT_ECHO_ARTIFACT_CONTENT"
    (tmp_path / "outside.txt").write_text(secret, encoding="utf-8")
    evidence = pair()
    evidence["baseline"]["artifact_path"] = reference
    evidence["baseline"]["artifact_sha256"] = hashlib.sha256(secret.encode()).hexdigest()
    data = source(inputs, evidence)
    assert main(["import-pairs", "--bank", str(tmp_path / "bank"), "--input", str(data),
                 "--planner-version", VERSION]) == 2
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert not (tmp_path / "bank").exists()


def test_artifacts_elsewhere_require_an_explicit_allowed_root(tmp_path):
    inputs, artifacts = tmp_path / "inputs", tmp_path / "artifacts"
    inputs.mkdir()
    artifacts.mkdir()
    evidence = local_pair(artifacts)
    data = source(inputs, evidence)
    result = import_pairs(tmp_path / "bank", data, planner_version=VERSION, artifact_root=artifacts)
    assert result["locally_verified_artifacts"] == 2


def test_symlink_escape_is_checked_before_opening_artifact(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"not permitted")
    try:
        (inputs / "link").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is not enabled on this host")
    evidence = pair()
    evidence["baseline"]["artifact_path"] = "link"
    data = source(inputs, evidence)
    with pytest.raises(EvidenceImportError, match="escapes"):
        import_pairs(tmp_path / "bank", data, planner_version=VERSION)


def test_busy_bank_refuses_import_and_inspect_without_bypassing_the_session_lock(tmp_path):
    bank = tmp_path / "bank"
    data = source(tmp_path, pair())
    import_pairs(bank, data, planner_version=VERSION)
    before = (bank / EXPERIENCE_FILENAME).read_bytes()
    lock = BankLock(bank).acquire()
    try:
        with pytest.raises(RuntimeError, match="lock"):
            import_pairs(bank, data, planner_version=VERSION)
        with pytest.raises(RuntimeError, match="lock"):
            inspect_bank(bank, planner_version=VERSION)
    finally:
        lock.close()
    assert (bank / EXPERIENCE_FILENAME).read_bytes() == before


def test_atomic_replace_failure_leaves_bank_identical_and_cleans_own_stage(tmp_path, monkeypatch):
    from carla_bridge.planning import experience_cli
    bank = tmp_path / "bank"
    data = source(tmp_path, pair("one"))
    import_pairs(bank, data, planner_version=VERSION)
    before = (bank / EXPERIENCE_FILENAME).read_bytes()
    data = source(tmp_path, [pair("two"), pair("three")])
    def fail(*_args):
        raise OSError("test disk failure")
    monkeypatch.setattr(experience_cli.os, "replace", fail)
    with pytest.raises(OSError, match="disk failure"):
        import_pairs(bank, data, planner_version=VERSION)
    assert (bank / EXPERIENCE_FILENAME).read_bytes() == before
    assert sorted(p.name for p in bank.iterdir()) == [EXPERIENCE_FILENAME]


def test_import_refuses_a_malformed_existing_bank(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    path = bank / EXPERIENCE_FILENAME
    path.write_bytes(b"{broken\n")
    before = path.read_bytes()
    with pytest.raises(EvidenceImportError, match="malformed"):
        import_pairs(bank, source(tmp_path, pair()), planner_version=VERSION)
    assert path.read_bytes() == before


def test_wrong_metrics_type_is_a_reported_validation_failure(tmp_path, capsys):
    evidence = pair()
    evidence["baseline"]["metrics"] = "malformed"
    data = source(tmp_path, evidence)
    assert main(["import-pairs", "--bank", str(tmp_path / "bank"), "--input", str(data),
                 "--planner-version", VERSION]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "EvidenceImportError"
    assert not (tmp_path / "bank").exists()


def test_duplicate_key_bank_rows_are_reported_and_block_import(tmp_path):
    bank = tmp_path / "bank"
    executed(bank, "executed")
    path = bank / EXPERIENCE_FILENAME
    data = path.read_text(encoding="utf-8").splitlines()
    data[-1] = data[-1].replace('"status": "observed"', '"status": "observed", "status": "observed"')
    path.write_text("\n".join(data) + "\n", encoding="utf-8")
    before = path.read_bytes()
    assert inspect_bank(bank, planner_version=VERSION)["raw_json_invalid_lines"]
    with pytest.raises(EvidenceImportError, match="duplicate-key"):
        import_pairs(bank, source(tmp_path, pair()), planner_version=VERSION)
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload", ['{"schema_version":1,"schema_version":2}', '{"value":NaN}'])
def test_nonfinite_json_and_duplicate_keys_are_rejected_before_writing(tmp_path, payload):
    data = tmp_path / "pairs.json"
    data.write_text(payload, encoding="utf-8")
    with pytest.raises(EvidenceImportError):
        import_pairs(tmp_path / "bank", data, planner_version=VERSION)
    assert not (tmp_path / "bank").exists()
