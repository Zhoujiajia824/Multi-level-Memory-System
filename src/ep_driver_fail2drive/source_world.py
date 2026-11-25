"""Read-only physical-source checks for official demonstration collection.

This detects gross source failures, not driving safety. It never changes world
settings, moves actors, selects driving controls or substitutes a mapped Z for
an observed value. In particular a car falling below the road cannot become a
valid demonstration merely because an official infraction event is still empty.
"""
import math


SCHEMA = 'official-source-grounding-1.0'


def source_world_geometry(snapshot, ego_id, route):
    frame = int(snapshot.frame)
    stamp = float(snapshot.timestamp.elapsed_seconds)
    frozen = snapshot.find(ego_id)
    if frozen is None:
        raise ValueError('Official ego is absent from the frozen physical source')
    pose = frozen.get_transform()
    velocity = frozen.get_velocity()
    acceleration = frozen.get_acceleration()
    xyz = [float(getattr(pose.location,k)) for k in ('x','y','z')]
    motion = [float(getattr(velocity,k)) for k in ('x','y','z')]
    accel = [float(getattr(acceleration,k)) for k in ('x','y','z')]
    if not all(math.isfinite(v) for v in [stamp,*xyz,*motion,*accel]):
        raise ValueError('Official physical source contains nonfinite XYZ measurements')
    road_z = float(route.height_at([xyz[0],-xyz[1]],xyz[2]))
    if not math.isfinite(road_z):
        raise ValueError('Public official route has no finite height reference')
    residual = xyz[2]-road_z
    # Large margins detect missing support/freefall; they are not body clearance
    # or route-following tolerances. Ordinary spawn settling, slopes and legal
    # adjacent-lane borrowing retain the actual observed source unchanged.
    below = residual < -3.
    descending_below = residual < -1. and motion[2] < -5.
    valid = not (below or descending_below)
    return dict(schema=SCHEMA,frame=frame,time=stamp,valid=valid,
        classification='physical_source_valid' if valid else 'physical_world_invalid',
        reason='within_gross_grounding_limits' if valid else 'ego_falling_below_published_road',
        cause='not_determined_by_this_check',driving_failure_automatically_excluded=False,
        official_score_unchanged=True,
        position_xyz_carla_m=xyz,velocity_xyz_carla_mps=motion,acceleration_xyz_carla_mps2=accel,
        speed_xy_mps=math.hypot(*motion[:2]),speed_xyz_mps=math.sqrt(sum(v*v for v in motion)),
        public_route_height_m=road_z,actor_origin_minus_route_height_m=residual,
        height_reference='unmodified_public_official_route;nearby_branch_selected_with_observed_z',
        limits=dict(below_route_m=3.,descending_below_route_m=1.,downward_speed_mps=5.),
        source_height_corrected=False,world_modified=False,driving_control_generated=False,
        limitation='gross physical-source validity only;does_not_certify_collision_or_driving_safety')
