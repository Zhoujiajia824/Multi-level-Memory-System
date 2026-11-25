"""Expert ego on unchanged official Fail2Drive scenarios, for demonstration only.

The official evaluator still owns all world ticks, actors, scenarios and scores.
This Agent never enables a paper-planner or model fallback. A failed expert run
is failed evidence, not a successful memory. Final admission is performed after
the official checkpoint has been written and every resource has closed.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from ep_driver_fail2drive.agent import (EPDriverFail2DriveAgent, CAMERA_POSES,
    PHYSICS_HZ, _control_record, CarlaDataProvider)
from ep_driver_fail2drive.archive import _png
from ep_driver_fail2drive.official_expert import OfficialExpert
from ep_driver_fail2drive.sensor_pose import SensorPoseTap
from ep_driver_fail2drive.source_world import source_world_geometry
from ep_driver_fail2drive.demo_behavior import DemonstrationBehaviorAudit, SCHEMA as BEHAVIOR_SCHEMA, POLICY as BEHAVIOR_POLICY


def get_entry_point():
    return 'OfficialDemonstrationAgent'


class OfficialDemonstrationAgent(EPDriverFail2DriveAgent):
    execution_modes = ('offline_demonstration',)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.expert = None
        self._demo_run = None
        self._demo_errors = []
        self._last_control = None
        self._closed = False
        self._native_count = 0
        self._demo_taps = {}
        self._demo_behavior = DemonstrationBehaviorAudit()

    def setup(self,path_to_conf_file):
        super().setup(path_to_conf_file)
        import carla
        try:
            spec = self.config.get('official_demo',{})
            if (spec.get('purpose') != 'demonstration_only' or spec.get('driver') != 'official_autopilot'
                    or not spec.get('official_root') or spec.get('scenario_mutations') is not False
                    or any(self.config.get(key) is not False for key in
                           ('planner_enabled','model_enabled','reflection_enabled'))
                    or self.config.get('reflection_memory_dir') is not None):
                raise ValueError('Official demonstrations require an explicit expert and isolated memory/model settings')
            for sensor in self.sensors():
                if sensor['id'] not in {name for name,_ in CAMERA_POSES} or sensor['id'] == 'front_rgb':
                    continue
                mount = carla.Transform(carla.Location(x=sensor['x'],y=sensor['y'],z=sensor['z']),
                    carla.Rotation(pitch=sensor['pitch'],yaw=sensor['yaw'],roll=sensor['roll']))
                tap = SensorPoseTap(self.sensor_interface,camera_id=sensor['id'],camera_mount_matrix_m=mount.get_matrix())
                self._demo_taps[sensor['id']] = tap
                tap.install()
        except Exception:
            for tap in self._demo_taps.values():tap.close()
            self._demo_taps.clear()
            if self._pose_tap is not None:
                self._pose_tap.close()
                self._pose_tap = None
            raise

    def sensors(self):
        return super().sensors()+[dict(type='sensor.other.imu',x=0.,y=0.,z=0.,
            roll=0.,pitch=0.,yaw=0.,sensor_tick=.05,id='imu')]

    def _create_worker(self):
        # The expert directly returns ego pedals; the numeric interpreter is
        # used later for read-only export/feature construction only.
        return None

    def _initialize(self,snapshot,input_data=None):
        super()._initialize(snapshot,input_data)
        association = self.config['archive_association']
        self._demo_run = dict(protocol='ep-driver-f2d-official-capture-1.0',
            demo_suite='fail2drive-official-demonstrations-1.0',
            id=self.archive.directory.name,source='carla',source_version=CarlaDataProvider.get_client().get_server_version(),
            scenario_family=self.config['official_demo']['scenario_family'],scenario='fail2drive',
            official_route_id=self.config['official_demo']['official_route_id'],
            split='official_scenario_demonstration',record_origin='actual_official_scenario_expert_execution',
            reference_driver='official_autopilot',driver_kind='official_autopilot',purpose='demonstration_only',
            seed=association['repetition_seed'],traffic_manager_seed=association['traffic_manager_seed'],
            official_evaluation=True,official_xml_unchanged=True,scenario_mutations=False,
            ego_vehicle_type=self.ego.type_id,world_map=self.world.get_map().name,
            official_commit=self.config['official_commit'],archive_association=association,
            normalized_source='actual_current_public_world_with_no_future',
            normalization_protocol='observable-driving-recording-1.0',
            read_future_during_current_representation=False,test_only=False,
            physics_hz=PHYSICS_HZ,record_hz=1,frames_path='frames.jsonl',
            physical_world_audit='official-source-grounding-1.0',
            demonstration_behavior_audit=BEHAVIOR_SCHEMA,
            demonstration_quality_policy=dict(BEHAVIOR_POLICY),
            human_driving_quality_review_required=True,
            model_calls=0,retrieval_enabled=False,write_memory_enabled=False,
            complete=False,completed_without_agent_error=False,errors=[])
        self.archive.json('run.json',self._demo_run)
        (self.archive.directory/'native').mkdir()
        self.expert = OfficialExpert(self.config['official_demo']['official_root'],seed=association['repetition_seed'])
        self.expert.setup(self._global_plan,self._global_plan_world_coord)
        self.archive.json('expert_world_guard.json',self.expert.evidence())

    def _display(self,input_data,frame,stamp,relative):
        image = input_data.get('rgb')
        if image is None or image[0] != frame:
            raise ValueError('Official display/state source frame mismatch')
        self.archive.capture_display(image,frame,stamp)
        if relative % 10 == 0:
            import cv2
            success,encoded = cv2.imencode('.jpg',image[1][:,:,:3])
            if not success:
                raise RuntimeError('Official demonstration visualization encode failed')
            destination = self.viz_path/('%05d.jpg'%relative)
            temporary = destination.with_suffix('.jpg.tmp')
            payload = encoded.tobytes()
            temporary.write_bytes(payload);temporary.replace(destination)
            value = dict(schema='ep-driver-f2d-display-frame-v1',jpeg=destination.name,
                jpeg_sha256=hashlib.sha256(payload).hexdigest(),carla_frame=frame,carla_timestamp=stamp,
                relative_tick=relative,route_id=self.route_id,route_run=self.viz_path.name,
                camera_id='rgb',source='official_CARLA_rendered_camera',sampling_hz_sim=2.,saved_wall_time=time.time())
            path = destination.with_suffix('.json')
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf-8');temporary.replace(path)

    def _record_source(self,input_data,scene,images,frame,stamp):
        import numpy as np
        folder = self.archive.directory
        native = 'native/%010d.png'%frame
        arrays = {name:input_data[name][1][:,:,2::-1] for name,_ in CAMERA_POSES}
        mosaic = np.concatenate((np.concatenate([arrays[n] for n in ('front_left_rgb','front_rgb','front_right_rgb')],axis=1),
            np.concatenate([arrays[n] for n in ('back_left_rgb','back_rgb','back_right_rgb')],axis=1)),axis=0)
        _png(folder/native,mosaic)
        manifest_path = 'native/%010d.cameras.json'%frame
        cameras = []
        for spec,image in zip([s for s in self.sensors() if s['id'] in arrays],images):
            if spec['id'] != image['id']:
                raise ValueError('Camera order differs from the frozen capture contract')
            tap = self._pose_tap if image['id']=='front_rgb' else self._demo_taps[image['id']]
            pose = tap.for_frame(frame,stamp,pixels=input_data[image['id']][1])
            matrix = np.asarray(pose['camera_world_matrix_m'],float)
            mount = np.asarray(pose['camera_mount_matrix_m'],float)
            origin = np.asarray(scene['diagnostics']['frozen_actor_origin_pose']['matrix_xyz_carla'],float)
            actor_ue = matrix@np.linalg.inv(mount)
            if np.max(np.abs(actor_ue[:3,:3]-origin[:3,:3])) > 1e-4:
                raise ValueError('Frozen actor and actual image camera orientations disagree')
            inverse = np.linalg.inv(matrix)
            cameras.append(dict(view=image['id'],rgb_path=Path(image['path']).relative_to(folder).as_posix(),
                frame=frame,timestamp=stamp,width=image['width'],height=image['height'],fov_deg=spec['fov'],
                world_to_camera_ue=inverse.tolist(),camera_world_matrix_m=matrix.tolist(),
                camera_mount_matrix_m=mount.tolist(),
                api_global_minus_ue_origin_m=(origin[:3,3]-actor_ue[:3,3]).tolist(),
                calibration_source=pose['capture_method'],image_selection=pose.get('image_selection'),
                coordinate_frame='CARLA_sensor_UE_world_xyz_m'))
        offsets = np.asarray([camera['api_global_minus_ue_origin_m'] for camera in cameras])
        # A LargeMap origin shift is shared by all rigid cameras. Do not absorb
        # a different camera capture pose into a per-view "origin correction".
        residual = float(np.max(np.linalg.norm(offsets-offsets[0],axis=1)))
        if residual > .02:
            raise ValueError('Same-frame rigid cameras disagree on the UE/global origin by %.6fm'%residual)
        self.archive.json(manifest_path,dict(frame_id=scene['frame_id'],time=stamp,carla_frame=frame,
            carla_timestamp=stamp,cameras=cameras,rigid_origin_residual_max_m=residual,
            rigid_origin_tolerance_m=.02))
        self.archive.append('raw_frames',dict(frame=frame,time=stamp,route_id=self.route_id,
            scene=scene,images=images,native_image=native,camera_manifest=manifest_path,
            audit=dict(source='official_checkpoint_after_run',collision=None,offroad=None,rule_violation=None)))
        self._native_count += 1

    def run_step(self,input_data,timestamp):
        snapshot = CarlaDataProvider.get_world().get_snapshot()
        frame,stamp = int(snapshot.frame),float(snapshot.timestamp.elapsed_seconds)
        started = time.monotonic()
        try:
            if self.last_frame is not None and frame != self.last_frame+1:
                raise ValueError('Official demonstration physical frames are not consecutive')
            if self.extractor is None:
                self._initialize(snapshot,input_data)
            relative = frame-self.first_frame
            source = relative % PHYSICS_HZ == 0
            health = source_world_geometry(snapshot,self.ego.id,self.extractor.route)
            self.archive.append('physical_world_audit',health)
            behavior = None
            if health['valid']:
                lanes=[row[0]['polygon'] for row in getattr(self.extractor,'_lane_bounds',())]
                try:
                    behavior=self._demo_behavior.observe(snapshot,self.ego,self.world.get_actors(),lanes)
                except Exception as error:
                    # A bad body/pose source can fail before a valid audit row
                    # exists. Still preserve this exact input frame for diagnosis;
                    # never call the teacher or return a control for that frame.
                    self.archive.capture_images(input_data,frame,stamp,[name for name,_ in CAMERA_POSES],save=True)
                    self._display(input_data,frame,stamp,relative)
                    error.evidence=dict(classification='invalid_demonstration_behavior_source',
                        frame=frame,time=stamp,reason=str(error))
                    raise
                self.archive.append('demonstration_behavior_audit',behavior)
            # Preserve the actual failure-frame images before rejecting a bad
            # physical source, including a failure between 1Hz source frames.
            images = self.archive.capture_images(input_data,frame,stamp,[name for name,_ in CAMERA_POSES],
                save=source or not health['valid'] or behavior is not None and not behavior['valid'])
            self._display(input_data,frame,stamp,relative)
            if not health['valid']:
                error = ValueError('Official physical world is invalid: '+health['reason'])
                error.evidence = health
                raise error
            if not behavior['valid']:
                error=ValueError('Official demonstration has measured bounding-box penetration')
                error.evidence=behavior
                raise error
            imu = input_data.get('imu')
            if imu is None or imu[0] != frame:
                raise ValueError('Official expert IMU/source frame mismatch')
            if source:
                scene = self.extractor.capture(frame,camera_images=images,world_snapshot=snapshot)
                observed_ego = scene['ego']
            else:
                # Full road/object representations are archived at 1Hz. Every
                # physical tick still records actual frozen ego kinematics;
                # it does not rebuild the scene or run the paper planner.
                frozen = snapshot.find(self.ego.id)
                if frozen is None:
                    raise ValueError('Official ego is absent from the frozen actor snapshot')
                observed_ego,_ = self.extractor._actor(self.ego,frozen,stamp,True)
                if observed_ego is None:
                    raise ValueError('Official ego has no finite physical body')
            actual = self.ego.get_control()
            control = self.expert.step(input_data,timestamp)
            if self.world.get_snapshot().frame != frame:
                raise RuntimeError('Expert advanced the official world')
            self.archive.append('execution',dict(frame=frame,time=stamp,official_game_time=timestamp,
                source_frame=frame,source_time=stamp,applied_control=_control_record(actual),
                observed_applied_control=_control_record(actual),returned_control=_control_record(control),
                observed_ego=observed_ego,driver='official_autopilot',control_applied_by='official_evaluator',
                expert_decision=self.expert.decision_evidence(),
                audit=dict(source='official_checkpoint_after_run',collision=None,offroad=None,rule_violation=None),
                agent_wall_s=time.monotonic()-started))
            if source:
                self._record_source(input_data,scene,images,frame,stamp)
                self.archive.append('agent_sources',dict(frame=frame,time=stamp,scene=scene))
                print('Official demonstration %s: %.2fs frame=%d speed=%.2fm/s'%(
                    self.config['official_demo']['scenario_family'],(relative/PHYSICS_HZ),frame,
                    math.hypot(*scene['ego']['velocity'])),flush=True)
            self.last_frame,self._last_control = frame,_control_record(control)
            return control
        except Exception as error:
            self._demo_errors.append(dict(frame=frame,time=stamp,type=type(error).__name__,error=str(error)))
            if self.archive is not None:
                self.archive.append('agent_errors',self._demo_errors[-1])
                self.archive.json('capture_failure.json',dict(self._demo_errors[-1],traceback=traceback.format_exc(),
                    evidence=getattr(error,'evidence',None)))
            raise

    def destroy(self):
        if self._closed:
            return
        self._closed = True
        archive = self.archive
        if archive is None:
            return super().destroy()
        errors = list(self._demo_errors)
        guard = dict(complete=False,world_mutations_allowed=False,non_ego_mutation_attempts=[],failures=[],scope='ego_control_only')
        try:
            if self.expert is not None:
                self.expert.close()
        except Exception as error:
            errors.append(dict(type=type(error).__name__,error=str(error)))
        finally:
            if self.expert is not None:
                try:
                    guard = self.expert.evidence()
                except Exception as error:
                    errors.append(dict(type=type(error).__name__,error=str(error)))
        for tap in self._demo_taps.values():
            try:
                tap.close()
            except Exception as error:
                errors.append(dict(type=type(error).__name__,error=str(error)))
        self._demo_taps.clear()
        try:
            super().destroy()
        except Exception as error:
            errors.append(dict(type=type(error).__name__,error=str(error)))
        if self._demo_run is not None:
            self._demo_run.update(complete=not errors,completed_without_agent_error=not errors,
                errors=errors,raw_frame_count=self._native_count,
                collection_closed_at=datetime.now().isoformat(),official_result_pending=True)
            # Archive.close has finalized video; write only closing metadata.
            archive.json('run.json',self._demo_run)
            archive.json('expert_world_guard.json',guard)
