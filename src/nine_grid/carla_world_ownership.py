"""Read-only CARLA 0.9.15 actor admission; Python 3.7 and standard library only.

In the official 0.9.15 source, CarlaEpisode::InitializeAtBeginPlay registers
map-native movable meshes with exactly ``mesh_path`` and ``mass`` attributes.
StaticMeshFactory's API blueprint additionally has role_name, ros_name and scale;
ActorBlueprint::MakeActorDescription serializes every attribute, including empty
role names. Thus an empty role alone is never evidence of a native map object.

Source: https://github.com/carla-simulator/carla/blob/0.9.15/
Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Game/CarlaEpisode.cpp#L342-L371
and LibCarla/source/carla/client/ActorBlueprint.cpp#L48-L56 in the same tag.
This version-specific evidence does not replace the exclusive-world contract or
authenticate a modified server/raw-RPC client. Unknown schemas fail closed.
"""
import math


CLASSIFICATION_VERSION = 'carla-0.9.15-map-mesh-admission-1.0'
NATIVE_MESH_ATTRIBUTES = frozenset(('mesh_path', 'mass'))
AUDIT_ATTRIBUTES = ('mesh_path', 'mass', 'role_name', 'ros_name', 'scale')


def _native_mesh_evidence(actor, attributes, record):
    if set(attributes) != NATIVE_MESH_ATTRIBUTES:
        return 'mesh_attribute_schema_not_native_0915'
    mesh_path, mass = attributes['mesh_path'], attributes['mass']
    if not isinstance(mesh_path, str) or not mesh_path.strip():
        return 'native_mesh_path_missing'
    try:
        valid_mass = isinstance(mass, str) and math.isfinite(float(mass)) and float(mass) >= 0
    except (ValueError, TypeError, OverflowError):
        valid_mass = False
    if not valid_mass:
        return 'native_mesh_mass_invalid'
    try:
        parent = actor.parent
        record['parent_id'] = None if parent is None else int(parent.id)
    except Exception:
        # A missing/unreadable parent is not evidence that the mesh is unattached.
        return 'native_mesh_parent_unavailable'
    if parent is not None:
        return 'native_mesh_has_parent'
    return 'map_native_mesh_0915_registration_schema'


def inspect_world_actors(actors):
    """Classify only vehicles, walkers and props, without changing the world.

    Call only after verifying the SDK and server are CARLA 0.9.15. Preserve
    native map geometry; block API-spawned/unknown props even with blank roles.
    The audit retains only relevant attribute values, never arbitrary metadata.
    """
    blocking, native = [], []
    for actor in actors:
        if not actor.type_id.startswith(('vehicle.', 'walker.', 'static.prop.')):
            continue
        record = {'id': int(actor.id), 'type_id': actor.type_id, 'role_name': ''}
        try:
            attributes = dict(actor.attributes)
        except Exception:
            record['classification_reason'] = 'actor_attributes_unavailable'
            blocking.append(record)
            continue
        record['role_name'] = str(attributes.get('role_name', ''))
        record['attribute_names'] = sorted(str(key) for key in attributes)
        record['attributes'] = {key: str(attributes[key]) for key in AUDIT_ATTRIBUTES if key in attributes}
        reason = (_native_mesh_evidence(actor, attributes, record)
                  if actor.type_id == 'static.prop.mesh' else 'existing_experiment_actor_type')
        record['classification_reason'] = reason
        if reason == 'map_native_mesh_0915_registration_schema':
            native.append(record)
        else:
            blocking.append(record)
    return {'actor_classification_version': CLASSIFICATION_VERSION,
            'map_native_mesh_count': len(native),
            'map_native_mesh_actors': sorted(native, key=lambda row: row['id']),
            'blocking_actors': sorted(blocking, key=lambda row: row['id'])}
