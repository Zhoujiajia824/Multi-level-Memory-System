"""真实 highway-env：遮挡/旧半径不能提前删掉沿车道范围内的当前真值。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import shapely
import yaml
from highway_env.vehicle.kinematics import Vehicle
from nine_grid.highway import HighwayAdapter
from nine_grid.models import scene_text
from nine_grid.object_scope import scope_scene
from nine_grid.perception_scope import perception_scene
from nine_grid.planning_objects import planning_scene


def main():
    cfg = json.loads((ROOT / 'config/parameters.json').read_text(encoding='utf-8'))
    cfg['scenarios'] = yaml.safe_load((ROOT / 'config/scenarios.yaml').read_text(encoding='utf-8'))
    cfg.update(scenario='highway', vehicles=0)
    # 旧圆形范围刻意小于目标距离；对象准入应只认沿车道40/20米。
    cfg.setdefault('perception', {})['radius_m'] = 5.
    adapter = HighwayAdapter(cfg, 1, 'truth-scope-check')
    try:
        ego = adapter.env.vehicle
        start = ego.position.copy()
        index = ego.lane_index
        arc = ego.lane.local_coordinates(start)[0]
        rows = {}
        for name, ds, lane_index in (
                ('near_blocker', 10., index), ('behind_blocker', 35., index),
                ('rear_inside', -19., index), ('front_outside', 45., index),
                ('rear_outside', -21., index), ('two_lanes_away', 15., ('0', '1', 3))):
            lane = adapter.env.road.network.get_lane(lane_index)
            actor = Vehicle(adapter.env.road, lane.position(arc+ds, 0.),
                            heading=lane.heading_at(arc+ds), speed=ego.speed)
            adapter.env.road.vehicles.append(actor)
            rows[name] = adapter._identity(actor)
        assert index == ('0', '1', 1), 'fixture requires ego in lane 1'
        assert np.allclose(start, ego.position)
        scene = adapter.snapshot()
        assert set(a.id for a in scene.actors) == set(rows.values())
        assert scene.diagnostics['observation_policy'] == 'range_limited_ground_truth_no_occlusion-1.0'
        assert not scene.diagnostics['has_occlusion']
        assert shapely.from_wkt(scene.diagnostics['visibility_wkt']).equals(adapter.drivable)

        expected = {rows[k] for k in ('near_blocker', 'behind_blocker', 'rear_inside')}
        scoped, scope = scope_scene(scene, cfg)
        assert {a.id for a in scoped.actors} == expected
        scene.diagnostics['object_scope'] = scope
        prompt = json.loads(scene_text(scene, SimpleNamespace(grid={})))
        representatives = prompt['八向图']['对象']
        others = prompt['范围内其他对象']
        perceived, perception = perception_scene(scene, cfg)
        assert {r['id'] for r in representatives+others} == set(rows.values())
        assert {a.id for a in perceived.actors} == set(rows.values())
        assert rows['behind_blocker'] in {r['id'] for r in others}
        planned, diagnostic = planning_scene(scene, cfg)
        assert {a.id for a in planned.actors} == expected
        assert not set(diagnostic['selected_ids'])-expected
        print(json.dumps({'passed': True, 'raw_objects': len(scene.actors),
                          'model_objects': len(perceived.actors), 'base_planning_objects': len(planned.actors),
                          'legacy_radius_m': 5, 'lane_scope': scope['settings'],
                          'occluded_front_35m_preserved': True,
                          'bev_dynamic_outside_lane_scope_preserved': True,
                          'physics_steps': adapter.env.steps, 'model_calls': 0}, ensure_ascii=False, indent=2))
    finally:
        adapter.close()


if __name__ == '__main__':
    main()
