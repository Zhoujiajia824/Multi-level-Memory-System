"""Same-frame UE/global camera calibration shared by demonstration exports."""
from copy import deepcopy
import math

from .vehicle_geometry import _rigid_matrix,pose_rotation_agreement

VIEWS=frozenset(('front_rgb','front_left_rgb','front_right_rgb','back_rgb','back_left_rgb','back_right_rgb'))


def source_camera_calibration(camera_world_matrix_m,camera_mount_matrix_m,actor_global_matrix_m):
    """Preserve the actual UE camera pose and derive its global origin offset.

    Callers bind the image and actor snapshot to the same frame/time. Only the
    translation between UE and public-global worlds is inferred; rotations and
    images are never corrected to fit the snapshot.
    """
    import numpy as np
    camera=np.asarray(_rigid_matrix(camera_world_matrix_m),float)
    mount=np.asarray(_rigid_matrix(camera_mount_matrix_m),float)
    origin=np.asarray(_rigid_matrix(actor_global_matrix_m),float)
    actor_ue=camera@np.linalg.inv(mount)
    agreement=pose_rotation_agreement(actor_ue.tolist(),origin.tolist())
    if not agreement['rotation_within_tolerance']:
        raise ValueError('Frozen actor and actual image camera orientations disagree')
    return dict(camera_world_matrix_m=camera.tolist(),world_to_camera_ue=np.linalg.inv(camera).tolist(),
        camera_mount_matrix_m=mount.tolist(),rotation_agreement=agreement,
        api_global_minus_ue_origin_m=(origin[:3,3]-actor_ue[:3,3]).tolist(),
        coordinate_frame='CARLA_sensor_UE_world_xyz_m')


def global_camera_calibration(manifest,source_manifest):
    """Apply the official export's p_UE=p_global-offset conversion verbatim."""
    cameras=manifest.get('cameras',[])
    if not isinstance(cameras,list) or len(cameras)!=6 or {c.get('view') for c in cameras}!=VIEWS:
        raise ValueError('Source requires the exact six camera views')
    def number(value):return type(value) in (int,float) and math.isfinite(value)
    corrected=[]
    for camera in cameras:
        world_to_camera=camera.get('world_to_camera_ue');offset=camera.get('api_global_minus_ue_origin_m')
        if (not isinstance(world_to_camera,list) or len(world_to_camera)!=4
                or any(not isinstance(row,list) or len(row)!=4 or not all(map(number,row)) for row in world_to_camera)
                or not isinstance(offset,list) or len(offset)!=3 or not all(map(number,offset))):
            raise ValueError('Missing same-frame UE/global camera calibration evidence')
        frame=camera.get('carla_frame',camera.get('frame'));stamp=camera.get('timestamp')
        if type(frame) is not int or frame<0 or not number(stamp):
            raise ValueError('Missing actual calibration frame/timestamp')
        global_matrix=deepcopy(world_to_camera)
        for i in range(3):
            global_matrix[i][3]-=sum(world_to_camera[i][j]*offset[j] for j in range(3))
        corrected.append(dict(view=camera['view'],frame=frame,timestamp=stamp,
            world_to_camera_global=global_matrix,world_to_camera_ue=world_to_camera,
            api_global_minus_ue_origin_m=offset,source_manifest=source_manifest))
    return corrected
