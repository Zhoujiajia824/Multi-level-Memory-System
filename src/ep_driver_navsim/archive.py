"""Small replay archives, independent of retrieval and model-input construction."""
import gzip
import json

from nine_grid.contracts import plain
from nine_grid.paths import filesystem_path
from .protocol import write_bytes


def write_json_gz(path, value):
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    # Deterministic header, so archive integrity hashes remain reproducible.
    write_bytes(path, gzip.compress(payload, compresslevel=6, mtime=0))


def read_json(path):
    path = filesystem_path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(raw.decode("utf-8"))


def read_input_packet(path):
    """Accept a sample directory, new .json.gz, or a legacy .json path."""
    path = filesystem_path(path)
    if path.is_dir():
        path = path / "input_packet.json"
    if path.name == "input_packet.json" and path.with_suffix(".json.gz").is_file():
        path = path.with_suffix(".json.gz")
    return read_json(path)


def write_input_packet(root, packet):
    root = filesystem_path(root)
    path = root / "input_packet.json.gz"
    write_json_gz(path, packet)
    # Only retire the exact legacy packet after its replacement is published.
    (root / "input_packet.json").unlink(missing_ok=True)
    return path


def _record_for_archive(record):
    # Never call plain/asdict on the internal Representations object: it expands
    # occupancy tensors and duplicates the complete grid for every retrieved row.
    result = {key: plain(value) for key, value in record.items()
              if key not in ("_representation", "_representations")}
    reps = record.get("_representations")
    if reps is not None:
        result["representation_assets"] = {
            key: plain(getattr(reps, key, None))
            for key in ("bev_path", "feature_image_path", "grid_path", "dino_signature",
                        "frame_id", "time", "version", "coordinates")}
    return result


def serialize_retrieval(retrieved, root):
    """Preserve prose, source/checksums, scores and full compressed diagnostics.

    This is called only after the original RetrievalResult has been consumed by
    the model. Neither the record objects nor their diagnostic dict is mutated.
    """
    diagnostics = retrieved.diagnostics
    detail_path = filesystem_path(root) / "retrieval_diagnostics.json.gz"
    write_json_gz(detail_path, plain(diagnostics))
    summary = {key: plain(value) for key, value in diagnostics.items()
               if key not in ("all_records", "candidates")}
    rows = diagnostics.get("all_records", [])
    summary["record_counts"] = dict(total=len(rows),
        accepted=sum(bool(row.get("accepted")) for row in rows),
        eligible=sum(bool(row.get("eligible")) for row in rows))
    summary["details_asset"] = detail_path.name
    return dict(frame_id=retrieved.frame_id, time=retrieved.time,
                version=retrieved.version, coordinates=retrieved.coordinates,
                snapshot_id=retrieved.snapshot_id,
                records=[_record_for_archive(record) for record in retrieved.records],
                scores=plain(retrieved.scores), diagnostics=summary)
