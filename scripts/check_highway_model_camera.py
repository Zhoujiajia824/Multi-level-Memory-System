"""无需模型/权重，验证四场景的当前原生图、双图输入和冻结帧约束。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from PIL import Image
import yaml
from nine_grid.highway import HighwayAdapter
from nine_grid.models import model_image_paths
from nine_grid.representations import build_representations


def main():
    config = json.loads((ROOT / 'config/parameters.json').read_text(encoding='utf-8'))
    config['scenarios'] = yaml.safe_load((ROOT / 'config/scenarios.yaml').read_text(encoding='utf-8'))
    reports = []
    (ROOT / 'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='camera_check_', dir=ROOT / 'outputs') as directory:
        for scenario in ('highway', 'merge', 'roundabout', 'intersection'):
            current = copy.deepcopy(config)
            current['scenario'] = scenario
            adapter = HighwayAdapter(current, 1, 'camera-check-' + scenario)
            try:
                scene = adapter.snapshot()
                clock = (adapter._step, adapter.env.steps, adapter.env.time)
                position = adapter.env.vehicle.position.copy()
                viewer = adapter.env.viewer
                output = Path(directory) / scenario
                entry = adapter.capture_model_camera(scene, output)
                first_pixels = np.array(Image.open(entry['path']))
                assert (adapter._step, adapter.env.steps, adapter.env.time) == clock
                assert np.array_equal(position, adapter.env.vehicle.position)
                assert adapter.env.viewer is viewer
                assert entry['frame_id'] == scene.frame_id and entry['time'] == scene.time
                assert not entry['plan_overlay'] and first_pixels.min() != first_pixels.max()

                # 制造显著规划线；模型图必须与无规划时逐像素完全一致。
                plan = np.vstack([scene.ego.position, scene.ego.position + [15., 12.]])
                assert adapter.set_planned_trajectory(plan)
                adapter.capture_model_camera(scene, output)
                assert np.array_equal(first_pixels, np.array(Image.open(entry['path'])))
                assert np.array_equal(adapter.env._planned_trajectory, plan)
                reps = build_representations(scene, output)
                paths = model_image_paths(scene, reps)
                assert paths == [entry['path'], reps.bev_path] and len(set(paths)) == 2

                # 一次真实控制步后，旧scene不能与新物理状态重新配图。
                adapter.step(np.array([0., 0.]))
                try:
                    adapter.capture_model_camera(scene, output)
                except ValueError:
                    pass
                else:
                    raise AssertionError('过期冻结帧未拒绝')
                advanced = adapter.snapshot()
                next_entry = adapter.capture_model_camera(advanced, output)
                assert next_entry['path'] != entry['path']
                assert next_entry['frame_id'] == advanced.frame_id and next_entry['time'] == advanced.time
                assert np.array_equal(first_pixels, np.array(Image.open(entry['path'])))
                reports.append({'scenario': scenario, 'model_image_count': len(paths),
                                'native_size': entry['size'], 'times_s': [scene.time, advanced.time],
                                'capture_advanced_physics': False, 'plan_leaked_to_image': False,
                                'stale_snapshot_rejected': True})
            finally:
                adapter.close()
    print(json.dumps({'passed': True, 'scenarios': reports, 'model_calls': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
