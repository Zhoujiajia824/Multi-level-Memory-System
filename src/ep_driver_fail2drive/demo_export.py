"""Numerical-environment conversion of real demo captures to builder input.

Run after collection/cleanup. This module does not import CARLA, tick a world,
run a planner or invent observations. Frozen raw captures remain untouched.
"""
import hashlib
import json

from nine_grid.contracts import plain
from nine_grid.paths import filesystem_path
from shapely.geometry import Polygon

from .demo_capture import atomic_json
from .protocol import SCHEMA
from .worker import decode_scene


def export_capture(directory):
    root = filesystem_path(directory)
    if (root/'capture_manifest.json').exists() or (root/'frames.jsonl').exists():
        raise ValueError('Export is immutable; do not overwrite an existing frozen capture')
    run = json.loads((root/'run.json').read_text(encoding='utf-8'))
    if run.get('complete') is not True or run.get('test_only') is not False:
        raise ValueError('Only fully closed real demonstration captures can be exported')
    count = 0
    temporary = root/'frames.jsonl.tmp'
    try:
        with (root/'raw_frames.jsonl').open(encoding='utf-8') as source, temporary.open('x',encoding='utf-8') as target:
            for line in source:
                row = json.loads(line)
                packet = dict(schema=SCHEMA, route_id=row['route_id'],frame=row['frame'],time=row['time'],
                              scene=row['scene'],images=row['images'])
                scene = decode_scene(packet)
                visibility = Polygon(row['scene']['diagnostics']['visibility_polygon'])
                if not visibility.is_valid or visibility.is_empty:
                    raise ValueError('Captured observation support is invalid')
                scene.diagnostics['visibility_wkt'] = visibility.wkt
                scene.diagnostics['driver_source'] = run['reference_driver']
                scene.diagnostics['demonstration_source'] = dict(official_test_route_used=False,
                    future_read_for_current_representation=False, raw_frame=row['frame'])
                snapshot = plain(scene)
                # The existing offline builder consumes current kinematics and
                # has no Actor.motion_model field. Preserve the observed tag
                # as metadata; do not change measured motion or object kind.
                snapshot['diagnostics']['captured_motion_models'] = {
                    actor['id']:actor.pop('motion_model',None)
                    for actor in [snapshot['ego'],*snapshot['actors']]}
                exported = dict(index=count, frame=row['frame'],time=row['time'],snapshot=snapshot,
                    native_image=row['native_image'],camera_manifest=row['camera_manifest'],audit=row['audit'])
                target.write(json.dumps(exported,ensure_ascii=False,allow_nan=False)+'\n')
                count += 1
        if count != run['raw_frame_count'] or count < 4:
            raise ValueError('Raw capture count mismatch or missing actual future evidence')
        temporary.replace(root/'frames.jsonl')
    except BaseException:
        # Leave a partial conversion as explicit evidence. A subsequent export
        # must use a fresh path or inspect it, never silently overwrite it.
        raise
    sums = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'capture_manifest.json':
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024),b''):
                    digest.update(chunk)
            sums[path.relative_to(root).as_posix()] = digest.hexdigest()
    manifest = dict(protocol='observable-driving-recording-1.0',source='carla',
        source_version=run['source_version'],split=run['split'],reference_driver=run['reference_driver'],
        frames_path='frames.jsonl',execution_path='execution.jsonl',event_evidence_path='event_evidence.jsonl',
        demo_evaluation_path='demo_evaluation.json',source_hashes=sums)
    atomic_json(root/'capture_manifest.json',manifest)
    return dict(exported_frames=count,directory=str(root),source_modified=False,memory_published=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',required=True)
    args = parser.parse_args()
    print(json.dumps(export_capture(args.capture),ensure_ascii=False))


if __name__ == '__main__':
    main()
