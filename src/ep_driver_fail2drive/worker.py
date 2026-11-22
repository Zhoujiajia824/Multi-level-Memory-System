"""Persistent Python 3.12 planning process; no CARLA import or world ownership.

Each request is a frozen public source frame. Only this module adapts it to the
existing numerical contracts. Official route scores are never computed here.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import fields
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import OFFICIAL_COMMIT
from .protocol import SCHEMA
from nine_grid.configuration import validate_config
from nine_grid.contracts import Actor, Lane, SceneSnapshot, RetrievalResult, plain
from nine_grid.goals import build_goal_reference
from nine_grid.models import (VisionClient, CommunicationFailure, build_prompt,
                              model_image_paths, parse_answer, load_system_prompts)
from nine_grid.perception_scope import perception_scene
from nine_grid.planner_factory import create_planner
from nine_grid.planning_context import prepare_planning_context
from nine_grid.prediction import Predictor
from nine_grid.quality import evaluate, execution_audit
from nine_grid.representations import (DINOExtractor, NineGridBuilder,
                                       build_representations, render_goal_overlay)

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def _path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _redacted(value):
    if isinstance(value, dict):
        return {k: ('<redacted>' if any(word in k.lower() for word in
                    ('password', 'secret', 'api_key', 'token')) and k != 'max_tokens' else _redacted(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    return value


def decode_scene(packet):
    """Validate exact source identity before accepting any numerical geometry."""
    if packet.get('schema') != SCHEMA:
        raise ValueError('Unsupported Fail2Drive packet schema')
    raw = packet['scene']
    if raw.get('schema') != 'ep-driver-fail2drive-scene-1.0':
        raise ValueError('Unsupported public scene schema')
    frame, stamp = packet['frame'], packet['time']
    if (type(frame) is not int or type(stamp) not in (int, float) or not math.isfinite(stamp)
            or raw.get('frame_id') != f"{packet['route_id']}:{frame}"
            or raw.get('time') != stamp):
        raise ValueError('Scene frame/time differs from the frozen packet')

    def actor(row):
        allowed = {f.name for f in fields(Actor)}
        if set(row)-allowed:
            raise ValueError('Unsupported actor contract fields')
        value = dict(row)
        for key in ('position', 'velocity', 'acceleration'):
            vector = np.asarray(value[key], float)
            if vector.shape != (2,) or not np.isfinite(vector).all():
                raise ValueError('Actor vectors must be finite XY pairs')
            value[key] = vector
        for key in ('length', 'width'):
            if not math.isfinite(value[key]) or value[key] <= 0:
                raise ValueError('Actor physical dimensions must be positive')
        if not math.isfinite(value['heading']):
            raise ValueError('Actor heading is not finite')
        if value.get('motion_model') not in (None, 'free', 'stationary'):
            raise ValueError('Unsupported entity motion model')
        return Actor(**value)

    ego = actor(raw['ego'])
    actors = [actor(row) for row in raw['actors']]
    ids = [ego.id]+[a.id for a in actors]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate ego/actor identities')
    lanes = []
    for row in raw['lanes']:
        row = dict(row)
        for key in ('center', 'polygon'):
            value = np.asarray(row[key], float)
            if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
                raise ValueError('Invalid lane geometry')
            row[key] = value
        lanes.append(Lane(**row))
    if len({lane.id for lane in lanes}) != len(lanes):
        raise ValueError('Duplicate public lane identities')
    polygons = [Polygon(p) for p in raw['drivable_polygons']]
    if not polygons or any(p.is_empty or not p.is_valid for p in polygons):
        raise ValueError('Invalid public drivable geometry')
    route = np.asarray(raw['route'], float)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError('Invalid official navigation geometry')
    assets = dict(raw.get('assets', {}))
    cameras = []
    for image in packet.get('images', []):
        if image.get('frame') != frame or image.get('timestamp') != stamp:
            raise ValueError('RGB source frame/time differs from public state')
        path = Path(image['path'])
        if not path.is_file():
            raise ValueError('Frozen RGB source image is missing')
        cameras.append(dict(view=image['id'], path=str(path.resolve()),
                            frame_id=raw['frame_id'], time=stamp))
    if len({camera['view'] for camera in cameras}) != len(cameras):
        raise ValueError('Duplicate camera views')
    assets['camera_images'] = cameras
    diagnostics = dict(raw.get('diagnostics', {}))
    diagnostics['observation_policy'] = 'range_limited_ground_truth_no_occlusion-1.0'
    diagnostics['privileged_input'] = True
    return SceneSnapshot(raw['frame_id'], float(stamp), ego, actors, lanes, route,
        list(raw['route_lane_ids']), unary_union(polygons), 'fail2drive',
        rules=dict(raw.get('rules', {})), assets=assets, diagnostics=diagnostics)


class PlanningWorker:
    """Route-local planner, filters, history and original-frame model guidance."""
    def __init__(self, runtime, *, client=None, extractor=None):
        self.runtime = dict(runtime)
        if runtime.get('official_commit') != OFFICIAL_COMMIT:
            raise ValueError('The runtime must name the pinned official evaluator commit')
        parameters = _path(runtime.get('parameters_path', 'config/parameters.json'))
        config = json.loads(parameters.read_text(encoding='utf-8'))
        config['platform'] = 'carla'
        config['scenario'] = 'fail2drive'
        config['parameter_version'] += '-fail2drive-adapter-1'
        config['clock'].update(physics_hz=20, planner_hz=runtime.get('planner_hz', 10), fast_hz=1)
        # Each 0.1s control segment spans exactly two official 0.05s physics steps.
        config['mpc']['physics_substeps'] = 2
        config['clock'].pop('episode_seconds', None)
        config['memory'].update(read_enabled=False, write_enabled=False)
        config['models'].update(fast_backend='server', reflection_backend='off')
        config['perception'].update(fully_occluded_in_current_planning=True, bev_physical_obstacles=True)
        if not config.get('diagnostics', {}).get('enabled', True) or not config.get('diagnostics', {}).get('dino', True):
            raise ValueError('Fail2Drive keeps BEV/OCC/grid/DINO/reference/quality enabled')
        # Freeze this integration's cost calibration explicitly; no main-branch defaults are edited.
        config['mpc']['normalization']['speed_mps'] = float(runtime.get('speed_normalization_mps', 1.0))
        self.config = validate_config(config)
        dino = config['dino']
        dino['model'] = str(_path(runtime.get('dino_model', dino['model'])))
        if extractor is None:
            import torch
            cpu_threads = runtime.get('dino_cpu_threads', 2)
            if type(cpu_threads) is not int or not 1 <= cpu_threads <= 16:
                raise ValueError('dino_cpu_threads must be an integer from 1 to 16')
            torch.set_num_threads(cpu_threads)
        self.extractor = extractor or DINOExtractor(dino['model'], device=runtime.get('dino_device', 'cpu'), local_files_only=True)
        models_path = _path(runtime.get('models_path', 'config/fail2drive.models.example.yaml'))
        model_settings = yaml.safe_load(models_path.read_text(encoding='utf-8'))['fast']
        model_settings = dict(model_settings)
        if model_settings.get('backend') != 'server':
            raise ValueError('B_model requires backend=server; it cannot silently run base only')
        prompts_path = _path(runtime.get('prompts_path', 'config/prompts.yaml'))
        model_settings['system_prompt'] = load_system_prompts(prompts_path)['fast']
        model_settings['max_total_attempts'] = int(runtime.get('model_attempts', 2))
        model_settings['timeout_s'] = float(runtime.get('model_timeout_s', 60))
        model_settings['max_tokens'] = 4096
        attempts = model_settings['max_total_attempts']
        if not 1 <= attempts <= 2 or not 0 < model_settings['timeout_s'] <= 90:
            raise ValueError('Model attempts/timeout exceed the bounded official watchdog policy')
        budget = attempts*model_settings['timeout_s'] + sum(2**i for i in range(attempts-1))
        if budget+30 >= float(runtime.get('worker_timeout_s', 240)):
            raise ValueError('Model call budget must leave at least 30s for decoding/planning/archive')
        self.client = client or VisionClient('fast', model_settings)
        self.planner = create_planner(config)
        self.reference_planner = create_planner(config)
        self.predictor = Predictor(config)
        self.grids = NineGridBuilder(config)
        self.active_answer = None
        self.last_frame = None
        self.last_time = None
        self.route_id = None
        self.directory = None
        self.next_source_time = None
        self.first_time = None
        self.counts = dict(plans=0, model_calls=0, valid_answers=0, invalid_answers=0,
            communication_failures=0, emergency_plans=0, reference_evaluations=0)
        self.configuration_evidence = dict(parameters=self.config, official_commit=OFFICIAL_COMMIT,
            runtime=_redacted(runtime), models=_redacted(model_settings), privileged_input=True,
            memory_retrieval=False, memory_write=False, reflection=False,
            source_hashes={'parameters':_sha(parameters), 'models':_sha(models_path), 'prompts':_sha(prompts_path)},
            internal_quality_is_official_score=False, whole_route_duration_limit=None)

    def _append(self, name, value):
        with (self.directory/(name+'.jsonl')).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False)+'\n')

    def plan(self, packet):
        started = time.perf_counter()
        scene = decode_scene(packet)
        route_id = packet['route_id']
        if self.route_id is None:
            self.route_id = route_id
            self.directory = Path(packet['output_dir']).resolve()/'planning'
            self.directory.mkdir(parents=True, exist_ok=True)
            if (self.directory/'configuration.json').exists():
                raise ValueError('A route planning archive cannot overwrite an existing run')
            self.next_source_time = scene.time
            self.first_time = scene.time
            (self.directory/'configuration.json').write_text(json.dumps(plain(self.configuration_evidence),
                ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            (self.directory/'map.json').write_text(json.dumps(plain(dict(lanes=scene.lanes,
                route=scene.route, route_lane_ids=scene.route_lane_ids, drivable=scene.drivable)),
                ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if route_id != self.route_id:
            raise ValueError('A numeric worker cannot reuse filter/planner history across official routes')
        if self.last_frame is not None and (packet['frame'] <= self.last_frame or scene.time <= self.last_time):
            raise ValueError('Planning source frame/time must increase strictly')
        model_due = scene.time >= self.next_source_time-1e-7
        if bool(scene.assets['camera_images']) != model_due:
            raise ValueError('1Hz model source images are missing or present outside the agreed source cadence')
        timing = {}
        self._append('frames', dict(frame_id=scene.frame_id, time=scene.time, ego=scene.ego,
            actors=scene.actors, diagnostics=scene.diagnostics, official_score=False))
        self.grids.update(scene, build=False)
        planning_input, predictions = prepare_planning_context(scene, self.config, predictor=self.predictor)
        scope = predictions.diagnostics['perception_scope']
        self._append('predictions', predictions)
        timing['prediction_s'] = time.perf_counter()-started
        capture = None
        reps = None
        if model_due:
            self.next_source_time += max(1, int(math.floor(scene.time-self.next_source_time+1e-7))+1)
            capture = self.directory/'captures'/f"{packet['frame']:010d}"
            capture.mkdir(parents=True, exist_ok=False)
            camera_folder = self.directory/'camera_frames'/f"{packet['frame']:010d}"
            camera_folder.mkdir(parents=True, exist_ok=True)
            camera_manifest = dict(frame_id=scene.frame_id, time=scene.time,
                carla_frame=packet['frame'], carla_timestamp=scene.time,
                cameras=[dict(view=item['view'], rgb_path=item['path'], frame=packet['frame'],
                    timestamp=scene.time, calibrated=False, projection_available=False)
                    for item in scene.assets['camera_images']])
            (camera_folder/'manifest.json').write_text(json.dumps(camera_manifest,
                ensure_ascii=False, allow_nan=False), encoding='utf-8')
            scene.assets['camera_manifest_json'] = str((camera_folder/'manifest.json').resolve())
            scene.diagnostics['goal_reference'] = build_goal_reference(planning_input, self.config,
                                                    request_version=self.counts['model_calls']+1)
            planning_input.diagnostics['goal_reference'] = scene.diagnostics['goal_reference']
            stamp = time.perf_counter()
            reps = build_representations(scene, capture, self.extractor, self.grids)
            timing['representations_s'] = time.perf_counter()-stamp
            (capture/'scene.json').write_text(json.dumps(plain(scene), ensure_ascii=False,
                allow_nan=False), encoding='utf-8')
            retrieval = RetrievalResult(scene.frame_id, time=scene.time,
                        diagnostics={'reason':'B_model: retrieval disabled', 'candidates':[]})
            prompt = build_prompt(planning_input, reps, retrieval)
            image_paths = model_image_paths(planning_input, reps)
            self._append('model_inputs', dict(frame_id=scene.frame_id, time=scene.time,
                prompt=prompt, images=image_paths, object_ids=scope['selected_ids'],
                goal_reference=scene.diagnostics['goal_reference'], memory_enabled=False))
            self.active_answer = None
            self.counts['model_calls'] += 1
            stamp = time.perf_counter()
            raw, errors = None, []
            try:
                raw = self.client.call(prompt, image_paths)
                self.active_answer = parse_answer(raw, scene, reference=scene.diagnostics['goal_reference'],
                    config=self.config, max_characters=self.config['models']['answer_max_characters'])
                self.counts['valid_answers'] += 1
            except CommunicationFailure as exc:
                errors.append(str(exc))
                self.counts['communication_failures'] += 1
                self.planner.clear_semantic_guidance()
            except ValueError as exc:
                errors.append(str(exc))
                self.counts['invalid_answers'] += 1
                self.planner.clear_semantic_guidance()
            timing['model_s'] = time.perf_counter()-stamp
            self._append('model_outputs', dict(frame_id=scene.frame_id, time=scene.time,
                raw=raw, answer=self.active_answer, errors=errors, wall_seconds=timing['model_s']))
        stamp = time.perf_counter()
        if scene.rules.get('navigation_brake_required'):
            self.planner.clear_semantic_guidance()
            self.active_answer = None
            plan = self.planner.emergency_brake(planning_input, predictions,
                      reason='official route projection or current-lane traffic-light/stop-sign association invalid')
            plan.diagnostics['navigation_brake_required'] = True
        else:
            plan = self.planner.solve(planning_input, predictions, self.active_answer)
        timing['mpc_s'] = time.perf_counter()-stamp
        stamp = time.perf_counter()
        audit = execution_audit(planning_input, plan, predictions, self.config)
        if audit.get('alarm') and not plan.diagnostics.get('emergency', False):
            self._append('rejected_before_execution', dict(plan=plan, audit=audit))
            plan = self.planner.emergency_brake(planning_input, predictions,
                reason='independent current-perception execution audit alarm', answer=self.active_answer)
            audit = execution_audit(planning_input, plan, predictions, self.config)
        timing['execution_audit_s'] = time.perf_counter()-stamp
        self._append('execution_audit', dict(frame_id=scene.frame_id, time=scene.time, **audit))
        self.counts['plans'] += 1
        self.counts['emergency_plans'] += int(plan.diagnostics.get('emergency', False))
        if model_due:
            stamp = time.perf_counter()
            reference_input, reference_predictions = prepare_planning_context(scene, self.config, existing=predictions)
            if scene.rules.get('navigation_brake_required'):
                reference = self.reference_planner.emergency_brake(reference_input, reference_predictions,
                    reason='same invalid navigation/stop association as the control plan')
                reference.diagnostics['navigation_brake_required'] = True
            else:
                reference = self.reference_planner.solve(reference_input, reference_predictions, None)
            timing['reference_s'] = time.perf_counter()-stamp
            stamp = time.perf_counter()
            quality = evaluate(planning_input, plan, reference, predictions, self.config)
            if scene.rules.get('navigation_brake_required'):
                quality.valid, quality.score = False, None
                quality.reasons.append('Navigation/stop association invalid; internal quality is unavailable')
                quality.metrics['navigation_brake_required'] = True
            timing['quality_s'] = time.perf_counter()-stamp
            self.counts['reference_evaluations'] += 1
            self._append('quality', dict(frame_id=scene.frame_id, time=scene.time,
                                        quality=quality, official_score=False))
            self._append('reference_plans', reference)
            render_goal_overlay(scene, plan).save(capture/'trajectory_bev.png')
            self._append('representations', dict(frame_id=scene.frame_id, time=scene.time,
                bev_path=reps.bev_path, feature_image_path=reps.feature_image_path, grid_path=reps.grid_path,
                dino_signature=reps.dino_signature, bev_feature=reps.bev_feature,
                occ_shape=list(reps.occ.shape)))
        self._append('plans', dict(frame_id=scene.frame_id, time=scene.time, plan=plan,
            answer_source=(self.active_answer.frame_id if self.active_answer else None),
            goal_overlay_path=str(capture/'trajectory_bev.png') if capture is not None else None,
            perception_ids=scope['selected_ids'], lane_ids=scope['lane_scope_ids'], official_score=False))
        timing['total_s'] = time.perf_counter()-started
        self._append('timing', dict(frame_id=scene.frame_id, time=scene.time, **timing))
        LOG.info('Fail2Drive %s: executable=%s safety=%s MPC=%.0fms model=%s P=%d', scene.frame_id,
            plan.valid, plan.diagnostics.get('safety_feasible'), timing['mpc_s']*1000,
            self.active_answer is not None, len(scope['selected_ids']))
        self.last_frame, self.last_time = packet['frame'], scene.time
        return dict(frame_id=scene.frame_id, time=scene.time, controls=plain(plan.controls), states=plain(plan.states),
            control_dt=self.config['mpc']['step_s'], planner_hz=self.config['clock']['planner_hz'],
            executable=bool(plan.valid), diagnostics=plain(plan.diagnostics), timing=timing,
            active_answer_source=self.active_answer.frame_id if self.active_answer else None)

    def close(self):
        if self.directory is not None:
            (self.directory/'summary.json').write_text(json.dumps(dict(**self.counts,
                official_score=False, route_id=self.route_id, scenario='fail2drive',
                execution_mode='B_model', simulated_seconds=(self.last_time-self.first_time if self.last_time else 0.),
                completion='see_parent_official_results_only'), indent=2), encoding='utf-8')


def main():
    # JSONL is UTF-8 even when Windows starts Python under a legacy code page.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='strict')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    output = sys.stdout
    worker = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('op') == 'close':
                break
            try:
                if request.get('op') != 'plan':
                    raise ValueError('Unknown numeric worker operation')
                with redirect_stdout(sys.stderr):
                    if worker is None:
                        runtime = json.loads(Path(args.config).read_text(encoding='utf-8'))
                        worker = PlanningWorker(runtime)
                    plan = worker.plan(request['packet'])
                response = dict(ok=True, request_id=request.get('request_id'), plan=plan)
            except Exception as exc:
                LOG.exception('Numeric worker request failed')
                response = dict(ok=False, request_id=request.get('request_id'),
                                error={'type':type(exc).__name__, 'message':str(exc)})
            output.write(json.dumps(response, ensure_ascii=False, allow_nan=False)+'\n')
            output.flush()
    finally:
        if worker is not None:
            worker.close()


if __name__ == '__main__':
    main()
