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
from nine_grid.geometry import checked_route_projection
from nine_grid.local_reference import LocalReferenceManager


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
        self.bridge.raw['map']['route']=[[0.,0.],[12.,0.]]
        self.bridge.raw['public_road_rules']['rules']=[]
        plan=self.plan();self.bridge.response=self.advance(2,'arrival')
        self.bridge.response['ego']['position']=[10.1,0.]
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

    def test_long_segment_window_and_endpoint_longitudinal_residual(self):
        p=checked_route_projection([90.,0.],[[0.,0.],[100.,0.]],0.,0.,window=8.)
        self.assertFalse(p['valid']);self.assertAlmostEqual(p['s_m'],8.)
        self.assertAlmostEqual(p['distance_m'],82.)
        p=checked_route_projection([120.,0.],[[0.,0.],[100.,0.]],100.,0.)
        self.assertFalse(p['valid']);self.assertEqual(p['lateral_m'],0.)
        self.assertAlmostEqual(p['distance_m'],20.)

    def test_loop_projection_cannot_relocate_to_another_lap(self):
        route=[[0.,0.],[200.,0.],[200.,100.],[0.,100.],[0.,0.]]
        p=checked_route_projection([100.,0.],route,500.,0.,window=10.)
        self.assertFalse(p['valid']);self.assertGreater(p['distance_m'],100.)

    def test_current_adjacent_lane_red_is_mapped_and_unrelated_lights_excluded(self):
        raw=self.bridge.raw
        raw['ego']=actor('ego',10.,3.5,'1|0|-2')
        raw['map']['lanes'].append(lane('1|0|-2',[[0.,3.5],[100.,3.5]],
            [[-10.,1.75],[110.,1.75],[110.,5.25],[-10.,5.25]]))
        raw['traffic_lights']=[dict(id='current',state='red',stops=[dict(lane_id='1|0|-2',position=[30.,3.5])]),
            dict(id='other',state='red',stops=[dict(lane_id='not-on-route',position=[5.,20.])])]
        scene=self.adapter.snapshot()
        self.assertAlmostEqual(scene.rules['stop_s'],30.)
        self.assertEqual([item['id'] for item in scene.rules['traffic_lights']],['current'])

    def test_navigation_is_not_moved_by_confirmed_lane_change(self):
        raw=self.bridge.raw
        raw['map']['lanes'][0]['left']='1|0|-2'
        raw['map']['lanes'].append(lane('1|0|-2',[[0.,3.5],[100.,3.5]],
            [[-10.,1.75],[110.,1.75],[110.,5.25],[-10.,5.25]]))
        self.adapter.snapshot()
        original=self.adapter.route.copy();goal=self.adapter._goal_position.copy()
        moved=self.adapter._convert_actor(actor('ego',12.,3.5,'1|0|-2'))
        self.adapter._confirm_lane(moved,.1);self.adapter._confirm_lane(moved,.5)
        np.testing.assert_array_equal(self.adapter.route,original)
        np.testing.assert_array_equal(self.adapter._goal_position,goal)
        self.assertEqual(self.adapter._goal_lane_id,'1|0|-1')

    def test_wrong_goal_lane_stops_with_explicit_reason_and_does_not_arrive(self):
        raw=self.bridge.raw;raw['map']['route']=[[0.,0.],[12.,0.]]
        raw['public_road_rules']['rules']=[]
        raw['ego']=actor('ego',10.,3.5,'1|0|-2',velocity=(0.,0.))
        raw['map']['lanes'].append(lane('1|0|-2',[[0.,3.5],[100.,3.5]],
            [[-10.,1.75],[110.,1.75],[110.,5.25],[-10.,5.25]]))
        scene=self.adapter.snapshot();result=scene.diagnostics['cycle_result']
        self.assertTrue(scene.rules['navigation_brake_required'])
        self.assertTrue(result['terminated']);self.assertFalse(result['arrival'])
        self.assertEqual(result['reason'],'destination_missed')
        self.assertEqual(scene.rules['goal_position'],[12.,0.])

    def test_destination_lane_intent_and_future_lane_membership(self):
        raw=self.bridge.raw;raw['map']['lanes'][0]['left']='1|0|-2'
        raw['map']['lanes'].append(lane('1|0|-2',[[0.,3.5],[100.,3.5]],
            [[-10.,1.75],[110.,1.75],[110.,5.25],[-10.,5.25]]))
        scene=self.adapter.snapshot();lanes={item.id:item for item in scene.lanes}
        current,target=lanes['1|0|-1'],lanes['1|0|-2']
        manager=LocalReferenceManager({})
        scene.route_lane_ids=['1|0|-1','1|0|-2'];scene.rules['goal_remaining_m']=120.
        # 邻道仅属于未来终点，不得复用当前位置的原车道中心。
        path=manager._continuation(scene,target,lanes)
        self.assertAlmostEqual(path[0,1],3.5)
        scene.rules['goal_remaining_m']=50.
        self.assertTrue(manager._navigation_change_needed(scene,current,target,lanes))
        self.assertIsNotNone(manager._legal(scene,target,current,lanes))

    def test_public_three_lane_cross_section_does_not_accept_an_unrelated_parallel_road(self):
        raw=self.bridge.raw
        raw['ego']=actor('ego',10.,7.,'1|0|-3')
        raw['map']['lanes'][0]['left']='1|0|-2'
        for identity,y,left,right in (('1|0|-2',3.5,'1|0|-3','1|0|-1'),('1|0|-3',7.,None,'1|0|-2')):
            item=lane(identity,[[0.,y],[100.,y]],
                [[-10.,y-1.75],[110.,y-1.75],[110.,y+1.75],[-10.,y+1.75]])
            item.update(left=left,right=right);raw['map']['lanes'].append(item)
        scene=self.adapter.snapshot()
        self.assertTrue(scene.diagnostics['route_projection']['valid'])
        self.assertAlmostEqual(scene.diagnostics['route_projection']['lateral_m'],7.)
        self.assertGreater(scene.diagnostics['route_projection']['maximum_distance_m'],7.)
        self.assertEqual(self.adapter._cross_section_limit('2|0|-1',[10.,0.]),6.)

    def test_destination_stop_enters_mpc_and_navigation_loss_forces_certified_or_emergency_braking(self):
        import json
        from nine_grid.planner import Planner
        from nine_grid.contracts import PredictionBundle
        raw=self.bridge.raw;raw['public_road_rules']['rules']=[]
        raw['actors']=[];raw['map']['route']=[[0.,0.],[20.,0.]]
        scene=self.adapter.snapshot()
        cfg=json.loads((Path(__file__).resolve().parents[1]/'config/parameters.json').read_text(encoding='utf-8'))
        predictions=PredictionBundle(scene.frame_id,scene.time,np.arange(31)*.1,{}, {})
        planner=Planner(cfg);plan=planner.solve(scene,predictions)
        self.assertEqual(plan.diagnostics['stop_s'],scene.rules['destination_stop_s'])
        self.assertLess(plan.states[-1,3],np.linalg.norm(scene.ego.velocity))
        self.assertLessEqual(plan.states[:,0].max(),20.)
        scene.rules['navigation_brake_required']=True
        scene.rules['navigation_status']='route_projection_invalid'
        plan=planner.solve(scene,predictions)
        self.assertTrue(plan.diagnostics['navigation_brake_required'])
        self.assertTrue(np.all(plan.controls[:,0]<=1e-8))
        self.assertIn(plan.diagnostics['execution_mode'],('safe_brake','emergency_unverified'))
        if not plan.diagnostics['safety_feasible']:
            self.assertEqual(plan.diagnostics['execution_mode'],'emergency_unverified')


if __name__=='__main__':unittest.main(verbosity=2)
