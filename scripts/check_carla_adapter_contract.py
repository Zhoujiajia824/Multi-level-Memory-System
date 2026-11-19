"""Exercise frozen snapshot, public rules and physical terminal transitions without a server."""
import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_adapter import CarlaAdapter


def actor(identity, x, y, lane, heading=0., velocity=(3., 0.)):
    return dict(id=identity, kind='vehicle', position=[x,y], velocity=list(velocity),
                acceleration=[0.,0.], heading=heading, length=4., width=1.8,
                lane_id=lane, visible=True, radius=None, wheelbase_m=2.7, rear_axle_to_center_m=1.35)


def lane(identity, center, polygon, successors=()):
    return dict(id=identity, center=center, polygon=polygon, width=3.5,
                speed_limit=8., successors=list(successors), boundaries=[])


def initial():
    return dict(frame=100, time=0., timestamp=20., cameras=[], collision=False,
        ego=actor('ego',10.,0.,'1|0|-1'), actors=[actor('npc',25.,10.,'2|0|-1',-math.pi/2,(0.,-5.))],
        speed_limit_mps=8., runtime=dict(physical_ticks=0,physics_hz=50,terminal_reason=None),
        current_road_evidence=dict(lane_id='1|0|-1',left=None,right=None),
        public_road_rules=dict(frame=100,physical_tick=0,rules=[dict(id='yield',kind='yield',
            source='published_road_rule',controlled_lane_ids=['1|0|-1'],priority_lane_ids=['2|0|-1'],
            stop_route_s_m=18.,conflict_route_s_m=[20.,30.])]),
        evaluation=dict(arrival=False,collision=False,route_completed=False),
        map=dict(route=[[0.,0.],[100.,0.]],route_lane_ids=['1|0|-1'],crosswalks=[],lanes=[
            lane('1|0|-1',[[0.,0.],[100.,0.]],[[-10.,-1.75],[110.,-1.75],[110.,1.75],[-10.,1.75]]),
            lane('2|0|-1',[[25.,30.],[25.,2.]],[[23.25,30.],[23.25,2.],[26.75,2.],[26.75,30.]],['3|0|-1']),
            lane('3|0|-1',[[25.,2.],[25.,-30.]],[[23.25,2.],[23.25,-30.],[26.75,-30.],[26.75,2.]])]))


class Bridge:
    def __init__(self, raw):
        self.raw=raw;self.closed=False;self.calls=[];self.response=None
    def call(self, method, **kwargs):
        self.calls.append((method,kwargs))
        if method=='init':return copy.deepcopy(self.raw)
        if method=='step':return copy.deepcopy(self.response)
        raise AssertionError(method)
    def close(self):self.closed=True


class AdapterContract(unittest.TestCase):
    def setUp(self):
        scratch=Path(__file__).resolve().parents[1]/'outputs'/'contract_checks'
        scratch.mkdir(parents=True,exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=str(scratch))
        self.bridge=Bridge(initial())
        self.adapter=CarlaAdapter(dict(clock=dict(physics_hz=50,planner_hz=10),mpc=dict(step_s=.1),
            carla=dict(runtime_mode='sync')),1,'contract',bridge_factory=lambda *a,**k:self.bridge)
        self.adapter.set_output_directory(self.directory.name)
        self.adapter.set_representation_capture_enabled(False)
    def tearDown(self):
        self.adapter.close();self.directory.cleanup()
    def plan(self):
        source=self.adapter.snapshot()
        return SimpleNamespace(valid=True,frame_id=source.frame_id,time=source.time,controls=np.zeros((30,2)))
    def advance(self, ticks, terminal=None):
        raw=copy.deepcopy(self.bridge.raw);raw.pop('map')
        raw.update(frame=100+ticks,time=ticks*.02,timestamp=20.+ticks*.02)
        raw['runtime'].update(physical_ticks=ticks,terminal_reason=terminal)
        raw['public_road_rules'].update(frame=raw['frame'],physical_tick=ticks)
        return raw
    def test_public_map_rule_retained_but_outside_actor_cannot_trigger_yield(self):
        self.bridge.raw['evaluation']['task_criteria']={'private_script_answer':'pass_and_return'}
        source=self.adapter.snapshot()
        self.assertTrue(source.rules['road_priority_available'])
        self.assertFalse(source.rules['must_wait'])
        self.assertIsNone(source.rules['stop_s'])
        self.assertEqual(source.rules['public_road_rules']['rules'][0]['stop_route_s_m'],18.)
        self.assertEqual(source.rules['public_priority_screen']['evidence']['object_scope']['selected_ids'],[])
        self.assertIs(self.adapter.snapshot(),source)
        self.assertNotIn('contract',source.frame_id)
        self.assertNotIn('independent_evaluation',source.diagnostics)
        self.assertNotIn('evaluation',source.diagnostics['cycle_result'])
    def test_priority_and_classification_use_current_scope_without_fabricating_evidence(self):
        near=actor('near',25.,0.,'1|0|-1')
        near['classification']={'kind':'vehicle','evidence':'sdk_vehicle_type','type_id':'vehicle.test'}
        far=actor('far',50.1,0.,'1|0|-1')
        self.bridge.raw['actors'] += [near,far]
        source=self.adapter.snapshot()
        state=source.rules['public_priority_screen']['evidence']
        self.assertEqual(state['object_scope']['selected_ids'],['near'])
        self.assertEqual(state['full_observed_actor_count'],1)
        self.assertEqual(source.diagnostics['object_scope']['selected_ids'],['near'])
        self.assertEqual(source.diagnostics['actor_classification'],{'near':near['classification']})
        near['classification']['evidence']='changed_after_capture'
        self.assertEqual(source.diagnostics['actor_classification']['near']['evidence'],'sdk_vehicle_type')
    def test_current_adjacency_is_applied_before_priority_scope(self):
        raw=self.bridge.raw
        raw['map']['lanes'].append(lane('adj',[[0.,3.5],[100.,3.5]],
            [[-10.,1.75],[110.,1.75],[110.,5.25],[-10.,5.25]]))
        # 原地图无邻道指针，仅本帧SDK证据提供该相邻车道。
        raw['current_road_evidence']['left']='adj'
        boundary={'points':[[0.,1.75],[100.,1.75]],'crossable':True}
        raw['map']['lanes'][0]['boundaries']=[copy.deepcopy(boundary)]
        raw['map']['lanes'][-1]['boundaries']=[copy.deepcopy(boundary)]
        raw['actors']=[actor('adjacent',20.,3.5,'adj')]
        source=self.adapter.snapshot()
        self.assertEqual(source.rules['public_priority_screen']['evidence']['object_scope']['selected_ids'],['adjacent'])
    def test_mixed_frame_public_rules_rejected(self):
        self.bridge.raw['public_road_rules']['frame']=99
        with self.assertRaisesRegex(ValueError,'冻结场景帧'):self.adapter.snapshot()
    def test_future_stop_is_not_moved_to_local_projection_search_boundary(self):
        self.bridge.raw['map']['route']=[[float(x),0.] for x in range(0,201,2)]
        self.bridge.raw['public_road_rules']['rules']=[]
        self.bridge.raw['public_road_rules']['stop_obligations']=[dict(id='sign',must_stop=True,
            lane_id='1|0|-1',position=[180.,0.])]
        scene=self.adapter.snapshot()
        self.assertAlmostEqual(scene.rules['stop_s'],180.)
        self.assertIsNone(self.adapter._project_public_stop([250.,0.]))
    def test_submitted_plan_keeps_original_source_and_accepts_early_physical_arrival(self):
        plan=self.plan();self.bridge.response=self.advance(2,'arrival')
        self.bridge.response['evaluation'].update(arrival=True,route_completed=True)
        result=self.adapter.step_plan(plan)
        self.assertTrue(result['arrival']);self.assertTrue(result['terminated'])
        self.assertEqual(self.bridge.calls[-1][1]['source_frame'],100)
        self.assertEqual(self.bridge.calls[-1][1]['source_time'],0.)
        self.assertEqual(self.adapter.snapshot().frame_id,self.adapter._observation_namespace+':000102')
    def test_unexplained_short_physics_step_rejected(self):
        plan=self.plan();self.bridge.response=self.advance(2)
        with self.assertRaisesRegex(RuntimeError,'执行前缀'):self.adapter.step_plan(plan)
    def test_endpoint_position_does_not_override_independent_task_evaluation(self):
        self.bridge.raw['ego']['position']=[99.,0.]
        result=self.adapter.snapshot().diagnostics['cycle_result']
        self.assertFalse(result['arrival'])


if __name__=='__main__':unittest.main(verbosity=2)
