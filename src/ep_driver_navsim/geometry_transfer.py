"""Lossless map geometry across SDK/core runtimes; no planner or scorer imports."""
import hashlib
import math
import re

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon


ENCODING = "wkb_hex_float64_le_2d"


def _area_geometry(geometry):
    if (geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty
            or geometry.area <= 0 or not np.isfinite(shapely.get_coordinates(geometry)).all()):
        raise ValueError("NAVSIM drivable geometry must be a finite positive polygonal area")
    return geometry


def encode_drivable(geometry):
    """Preserve every binary64 XY coordinate, including one-ULP-width holes."""
    _area_geometry(geometry)
    if not geometry.is_valid or shapely.get_coordinate_dimension(geometry) != 2:
        raise ValueError("NAVSIM drivable source must be valid two-dimensional geometry")
    raw = shapely.to_wkb(geometry, byte_order=1, output_dimension=2)
    recovered = shapely.from_wkb(raw)
    if not recovered.is_valid or not geometry.equals_exact(recovered, 0.):
        raise ValueError("NAVSIM drivable binary geometry roundtrip changed its source")
    return dict(encoding=ENCODING, data=raw.hex(), sha256=hashlib.sha256(raw).hexdigest())


def _legacy_collapsed_holes(geometry):
    """Remove only exact zero-area, <=2-point holes collapsed by old WKT.

    This does not repair self intersections, malformed shells, detached holes,
    positive-area holes or arbitrary invalid maps. Every retained coordinate is
    copied unchanged. WKB producers reject all invalid source geometries.
    """
    original_reason = shapely.is_valid_reason(geometry)
    if not original_reason.startswith("Too few points in geometry component"):
        raise ValueError("NAVSIM legacy drivable source is invalid: " + original_reason)
    parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    kept, removed = [], []
    for polygon_index, part in enumerate(parts):
        shell = list(part.exterior.coords)
        shell_geometry = Polygon(shell)
        if not shell_geometry.is_valid:
            raise ValueError("NAVSIM legacy drivable shell is invalid; no transport-only recovery")
        holes, collapsed = [], []
        for ring_index, ring in enumerate(part.interiors):
            coordinates = list(ring.coords)
            if len(set(coordinates)) <= 2 and Polygon(coordinates).area == 0.:
                points = list(dict.fromkeys(coordinates))
                support = Point(points[0]) if len(points) == 1 else LineString(points)
                if not shell_geometry.covers(support):
                    raise ValueError("NAVSIM legacy collapsed interior ring is outside its shell")
                collapsed.append(support)
                removed.append(dict(polygon_index=polygon_index, interior_index=ring_index,
                                    distinct_points=len(set(coordinates)), removed_area_m2=0.))
            else:
                holes.append(coordinates)
        for coordinates in holes:
            retained_hole = Polygon(coordinates)
            if not retained_hole.is_valid or any(support.intersects(retained_hole) for support in collapsed):
                raise ValueError("NAVSIM legacy collapsed-ring recovery conflicts with a retained hole")
        kept.append(Polygon(shell, holes))
    if not removed:
        raise ValueError("NAVSIM legacy drivable invalidity is not a collapsed zero-area interior ring")
    repaired = MultiPolygon(kept) if geometry.geom_type == "MultiPolygon" else kept[0]
    if (not repaired.is_valid or repaired.bounds != geometry.bounds
            or not math.isclose(repaired.area, geometry.area, rel_tol=0., abs_tol=0.)):
        raise ValueError("NAVSIM legacy collapsed-hole recovery changed polygon area or bounds")
    return repaired, dict(applied=True, method="remove_exact_zero_area_collapsed_interior_rings",
                         source_reason=original_reason, removed_interiors=removed,
                         removed_area_m2=0., bounds_unchanged=True, area_unchanged=True,
                         retained_coordinates_unchanged=True)


def decode_drivable(packet):
    """Read exactly one road encoding; never silently prefer conflicting maps."""
    binary, text = "drivable_geometry" in packet, "drivable_wkt" in packet
    if binary == text:
        raise ValueError("NAVSIM packet requires exactly one drivable geometry encoding")
    repair = dict(applied=False)
    if binary:
        value = packet["drivable_geometry"]
        if (not isinstance(value, dict) or set(value) != {"encoding", "data", "sha256"}
                or value.get("encoding") != ENCODING or not isinstance(value.get("data"), str)
                or len(value["data"]) % 2 or re.fullmatch(r"[0-9a-fA-F]+", value["data"]) is None):
            raise ValueError("NAVSIM drivable WKB encoding is malformed")
        raw = bytes.fromhex(value["data"])
        if hashlib.sha256(raw).hexdigest() != value.get("sha256"):
            raise ValueError("NAVSIM drivable WKB checksum differs from its payload")
        try:
            geometry = shapely.from_wkb(raw)
        except shapely.errors.GEOSException as exc:
            raise ValueError("NAVSIM drivable WKB cannot be decoded") from exc
        _area_geometry(geometry)
        if not geometry.is_valid or shapely.get_coordinate_dimension(geometry) != 2:
            raise ValueError("NAVSIM drivable WKB source is invalid; refusing implicit repair")
        if shapely.to_wkb(geometry, byte_order=1, output_dimension=2) != raw:
            raise ValueError("NAVSIM drivable WKB is not the declared complete canonical 2D encoding")
        encoding, lossless = ENCODING, True
    else:
        if not isinstance(packet["drivable_wkt"], str):
            raise ValueError("NAVSIM legacy drivable WKT must be text")
        try:
            geometry = _area_geometry(shapely.from_wkt(packet["drivable_wkt"]))
        except shapely.errors.GEOSException as exc:
            raise ValueError("NAVSIM legacy drivable WKT cannot be decoded") from exc
        if not geometry.is_valid:
            geometry, repair = _legacy_collapsed_holes(geometry)
        encoding, lossless = "legacy_wkt", False
    return geometry, dict(encoding=encoding, lossless=lossless, decoded_valid=True,
                          area_m2=float(geometry.area), legacy_wire_recovery=repair)


def drivable_roundtrip(geometry):
    """The same actual wire used by production, preflight and numerical input."""
    payload = encode_drivable(geometry)
    recovered, evidence = decode_drivable(dict(drivable_geometry=payload))
    if not geometry.equals_exact(recovered, 0.):
        raise ValueError("NAVSIM serialized drivable geometry differs from the source")
    return recovered, payload, evidence
