"""Plan a minimal original-test transfer using the pinned official scene filter.

Run with NAVSIM's Python 3.9. Default output is a checksum manifest only. The
optional uncompressed tar preserves complete selected log pickles and current
eight-camera JPEGs; it never rewrites logs, reads models, scores, or uploads data.
Maps and already separate synthetic packages are outside this transfer.

Examples:
  SDK_PYTHON scripts/export_navsim_test_assets.py --settings RUN/run_settings.json
  SDK_PYTHON scripts/export_navsim_test_assets.py --settings RUN/run_settings.json --tar transfer.tar
  python scripts/export_navsim_test_assets.py --verify-manifest MANIFEST --verify-root DATA_ROOT
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
SPLITS = ("warmup_two_stage", "navhard_two_stage")
FORMAT = "navsim-original-test-assets-1.0"


def safe_relative(value):
    value = str(value)
    if not value or "\x00" in value or PureWindowsPath(value).drive:
        raise ValueError("Unsafe data-relative path: " + repr(value))
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Unsafe data-relative path: " + repr(value))
    return path.as_posix()


def source_file(root, relative):
    relative = safe_relative(relative)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("Asset resolves outside data root: " + relative) from None
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Required regular asset missing: " + str(path))
    return path


def file_evidence(path):
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or before.st_size == 0:
        raise RuntimeError("Source changed or empty during content verification: " + str(path))
    return dict(size_bytes=after.st_size, sha256=checksum.hexdigest())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
            prefix=".asset_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_sdk(sdk_root):
    identity = json.loads((ROOT / "config/navsim/official_v2_2.json").read_text(encoding="utf-8"))
    entries = []
    for path in sorted((sdk_root / "navsim").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml"):
            checksum = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            entries.append(path.relative_to(sdk_root).as_posix() + ":" + checksum)
    actual = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    if actual != identity["source_sha256"] or len(entries) != identity["source_files"]:
        raise RuntimeError("SDK source does not match the pinned official NAVSIM v2.2 identity")
    return dict(commit=identity["commit"], source_sha256=actual, source_files=len(entries))


def collect(data_root, sdk_root, splits):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import filter_scenes

    sensors = SensorConfig(**dict({camera: [3] for camera in CAMERAS}, lidar_pc=False))
    assert set(sensors.get_sensors_at_iteration(3)) == set(CAMERAS)
    assert all(not sensors.get_sensors_at_iteration(index) for index in range(12) if index != 3)
    assets, coverage, anchors = {}, {}, {}
    for split in splits:
        config = sdk_root / "navsim/planning/script/config/common/train_test_split/scene_filter" / (split + ".yaml")
        scene_filter = instantiate(OmegaConf.load(config))
        if not isinstance(scene_filter, SceneFilter) or scene_filter.num_history_frames != 4:
            raise ValueError("Unsupported official scene/sensor alignment")
        requested = set(scene_filter.tokens or [])
        if not requested:
            raise ValueError("Official split must enumerate original anchor tokens")
        scenes, _ = filter_scenes(data_root / "navsim_logs/test", scene_filter)
        found = set(scenes)
        missing, extra = sorted(requested - found), sorted(found - requested)
        coverage[split] = dict(requested_original_tokens=len(requested), found_original_tokens=len(found),
            complete=not missing and not extra, missing_tokens=missing, unexpected_tokens=extra,
            tokens=sorted(found), scene_filter_path=str(config), scene_filter_sha256=file_evidence(config)["sha256"])
        if missing or extra:
            raise ValueError("Official original-token coverage failed: " + json.dumps(coverage[split]))
        for token, frames in sorted(scenes.items()):
            row = frames[scene_filter.num_history_frames - 1]
            if str(row["token"]) != token:
                raise ValueError("Official filter returned a misaligned source anchor")
            log_name = safe_relative(row["log_name"])
            if "/" in log_name:
                raise ValueError("Invalid original log name")
            log = safe_relative("navsim_logs/test/" + log_name + ".pkl")
            assets.setdefault(log, dict(kind="complete_original_log", anchor_tokens=set(), splits=set()))
            assets[log]["anchor_tokens"].add(token)
            assets[log]["splits"].add(split)
            camera_files = {}
            for camera in sensors.get_sensors_at_iteration(3):
                raw = safe_relative(row["cams"][camera.upper()]["data_path"])
                relative = safe_relative("sensor_blobs/test/" + raw)
                camera_files[camera] = relative
                assets.setdefault(relative, dict(kind="current_camera", anchor_tokens=set(), splits=set(), camera=camera))
                assets[relative]["anchor_tokens"].add(token)
                assets[relative]["splits"].add(split)
            existing = anchors.get(token)
            anchor = dict(token=token, log=log, cameras=camera_files, timestamp_us=int(row["timestamp"]))
            if existing is not None and existing != anchor:
                raise ValueError("Conflicting source assets for a token shared between official splits")
            anchors[token] = anchor
        print("Official original coverage:", split, len(found), "/", len(requested), flush=True)
    entries, missing_assets = [], []
    for index, (relative, row) in enumerate(sorted(assets.items()), 1):
        try:
            evidence = file_evidence(source_file(data_root, relative))
        except (OSError, RuntimeError) as exc:
            missing_assets.append(dict(path=relative, error=str(exc)))
            continue
        entries.append(dict(path=relative, **evidence, **{key: sorted(value) if isinstance(value, set) else value for key, value in row.items()}))
        if index % 500 == 0:
            print("Content verified:", index, "/", len(assets), flush=True)
    if missing_assets:
        raise ValueError("Required original assets unavailable: " + json.dumps(missing_assets[:20]) + "; count=" + str(len(missing_assets)))
    return entries, coverage, list(anchors.values())


def verify_manifest(manifest_path, root):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported transfer manifest")
    for index, row in enumerate(manifest["files"], 1):
        actual = file_evidence(source_file(root, row["path"]))
        if actual != {key: row[key] for key in ("size_bytes", "sha256")}:
            raise RuntimeError("Transferred asset content differs: " + row["path"])
        if index % 500 == 0:
            print("Verified transfer:", index, "/", len(manifest["files"]), flush=True)
    return dict(status="passed", files=len(manifest["files"]), bytes=sum(row["size_bytes"] for row in manifest["files"]))


def make_tar(path, data_root, manifest, manifest_path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError("Choose a new tar path: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    required = manifest["summary"]["estimated_uncompressed_tar_bytes"]
    free = shutil.disk_usage(path.parent).free
    if free < required + 64 * 1024 * 1024:
        raise OSError("Insufficient free space for uncompressed tar: need %d, free %d" % (required, free))
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError("Existing partial tar will not be overwritten: " + str(temporary))
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
            for row in manifest["files"]:
                source = source_file(data_root, row["path"])
                # Content is checked immediately before each untouched file is
                # archived; tar stores raw JPEG/log bytes without compression.
                if file_evidence(source) != {key: row[key] for key in ("size_bytes", "sha256")}:
                    raise RuntimeError("Source changed before archive: " + row["path"])
                archive.add(source, arcname=row["path"], recursive=False)
            archive.add(manifest_path, arcname="navsim_original_test_manifest.json", recursive=False)
        # Validate archive member content independently before publishing it.
        expected = {row["path"]: row for row in manifest["files"]}
        with tarfile.open(temporary, "r") as archive:
            for member in archive:
                if member.name == "navsim_original_test_manifest.json":
                    continue
                relative = safe_relative(member.name)
                row = expected.pop(relative)
                if not member.isfile() or member.size != row["size_bytes"]:
                    raise RuntimeError("Invalid archived asset: " + relative)
                checksum = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        checksum.update(chunk)
                if checksum.hexdigest() != row["sha256"]:
                    raise RuntimeError("Archive content differs: " + relative)
            if expected:
                raise RuntimeError("Archive omitted required source files")
        os.replace(temporary, path)
        return dict(path=str(path), **file_evidence(path), compression="none", content_verified=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings", help="Existing frozen run_settings.json")
    parser.add_argument("--sdk-root")
    parser.add_argument("--data-root")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/navsim/transfer"))
    parser.add_argument("--tar", help="Optional NEW uncompressed tar; omitted by default")
    parser.add_argument("--verify-manifest")
    parser.add_argument("--verify-root", help="Data root after transfer; read-only size/SHA256 verification")
    args = parser.parse_args()
    if args.verify_manifest:
        if not args.verify_root:
            parser.error("--verify-manifest requires --verify-root")
        print(json.dumps(verify_manifest(args.verify_manifest, args.verify_root)), flush=True)
        return 0
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError("Manifest generation must use the pinned SDK Python 3.9")
    if not args.settings:
        parser.error("Manifest generation requires --settings")
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    sdk_root = Path(args.sdk_root or os.environ.get("EP_NAVSIM_SDK_ROOT") or settings["sdk_root"]).resolve()
    data_root = Path(args.data_root or os.environ.get("EP_NAVSIM_DATA_ROOT") or settings["data_root"]).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / "src"), str(sdk_root)]
    from ep_driver_navsim.sdk_bootstrap import platform_setup
    platform_setup()
    identity = verify_sdk(sdk_root)
    started = time.perf_counter()
    entries, coverage, anchors = collect(data_root, sdk_root, list(dict.fromkeys(args.splits)))
    categories = {}
    for row in entries:
        category = categories.setdefault(row["kind"], dict(files=0, size_bytes=0))
        category["files"] += 1
        category["size_bytes"] += row["size_bytes"]
    size = sum(row["size_bytes"] for row in entries)
    # Reserve conservative PAX/header overhead plus manifest, without claiming
    # JPEG compression savings. Actual tar is optional and content verified.
    tar_estimate = sum(math.ceil(row["size_bytes"] / 512) * 512 + 4096 for row in entries) + 8 * 1024 * 1024
    summary = dict(files=len(entries), unique_original_anchors=len(anchors), size_bytes=size,
        size_gib=size / 1024 ** 3, categories=categories, estimated_uncompressed_tar_bytes=tar_estimate,
        estimated_uncompressed_tar_gib=tar_estimate / 1024 ** 3, local_free_bytes=shutil.disk_usage(output).free,
        manifest_only=args.tar is None, all_source_files_content_hashed=True, original_logs_unchanged=True,
        maps_included=False, synthetic_assets_included=False, lidar_included=False,
        camera_history_indices=[3], cameras=list(CAMERAS), covered_splits=list(coverage),
        filter_scenes_calls=len(coverage), wall_s=time.perf_counter() - started)
    manifest = dict(format=FORMAT, status="complete", generated_utc=datetime.now(timezone.utc).isoformat(),
        source_data_root=str(data_root), sdk=identity,
        official_selector="navsim.common.dataclasses.SceneFilter + navsim.common.dataloader.filter_scenes",
        coverage=coverage, summary=summary, anchors=anchors, files=entries,
        transfer_scope="Original test assets required by selected official anchors and current eight cameras only; synthetic packages/maps must already exist separately",
        target_inventory_checked=False, note="No server inventory supplied; manifest enumerates the required original-test supplement, without claiming remote absence by content.")
    manifest_path = output / "original_test_assets_manifest.json"
    write_json(manifest_path, manifest)
    write_json(output / "original_test_assets_summary.json", summary)
    if verify_sdk(sdk_root) != identity:
        raise RuntimeError("Pinned SDK changed during selection")
    if args.tar:
        result = make_tar(args.tar, data_root, manifest, manifest_path)
        write_json(output / "original_test_assets_tar_status.json", result)
    print(json.dumps(dict(status="complete", manifest=str(manifest_path), **summary), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
