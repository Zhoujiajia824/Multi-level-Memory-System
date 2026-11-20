"""Regression checks for frozen instances, event causality and signal exclusion.

No CARLA world connection, physical tick or model call is performed.
"""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_experiment_spec import resolve_spec, settings_from_manifest, COORDINATE_CONTRACT
from nine_grid.carla_experiment_routes import RouteGeometry
from nine_grid.carla_experiments import EventState, SignalProgram, Experiment


def specimen():
    return {'schema_version':'carla-experiment-2.0','name':'case','family':'cut_in',
        'description':'test','map':'Town01','desired_speed_mps':8.,'duration_s':60.,
        'experiment':{'ego_blueprint':'vehicle.tesla.model3','initial_speed_mps':8.,
            'route':{'opendrive_sha256':'a'*64,'coordinate_contract':copy.deepcopy(COORDINATE_CONTRACT),
                'points':[{'road_id':1,'lane_id':1,'s_m':x} for x in (0.,2.)]},
            'actors':[{'role':'other','type':'vehicle','blueprint':'vehicle.audi.a2',
                'placement':{'s_m':30.,'lane':'left'},'behavior':'cut_in','speed_mps':6.,
                'duration_ticks':200,'trigger':{'ego_s_m':[10.,40.],'gap_m':[5.,25.],'earliest_tick':50}}],
            'variations':[{'path':'actors.0.speed_mps','min':5.,'max':7.}]}}


class InstanceTests(unittest.TestCase):
    def test_same_seed_is_exact_and_different_seed_changes_parameter(self):
        data=specimen();original=copy.deepcopy(data)
        a,ma=resolve_spec(data,1);b,mb=resolve_spec(data,1);c,mc=resolve_spec(data,2)
        self.assertEqual(ma,mb);self.assertEqual(data,original)
        self.assertNotEqual(a['actors'][0]['speed_mps'],c['actors'][0]['speed_mps'])
        self.assertEqual(a['route'],c['route'])
        self.assertNotEqual(ma['instance_hash'],mc['instance_hash'])

    def test_manifest_roundtrip_and_tamper(self):
        settings,manifest=resolve_spec(specimen(),3)
        self.assertEqual(settings,settings_from_manifest(json.loads(json.dumps(manifest))))
        manifest['settings']['actors'][0]['speed_mps']+=1.
        with self.assertRaises(ValueError):settings_from_manifest(manifest)

    def test_variation_cannot_change_route_or_unknown_key(self):
        data=specimen();data['experiment']['variations'][0]['path']='route.points.0.s_m'
        with self.assertRaises(ValueError):resolve_spec(data,1)
        data=specimen();data['experiment']['actors'][0]['trigger']['time_s']=1.
        with self.assertRaises(ValueError):resolve_spec(data,1)

    def test_mutating_runtime_does_not_mutate_frozen_manifest(self):
        settings,manifest=resolve_spec(specimen(),1)
        settings['actors'][0]['speed_mps']=0.
        self.assertGreater(manifest['settings']['actors'][0]['speed_mps'],0.)

    def test_coordinate_contract_is_required_and_frozen(self):
        data=specimen();del data['experiment']['route']['coordinate_contract']
        with self.assertRaises(ValueError):resolve_spec(data,1)
        _,manifest=resolve_spec(specimen(),1)
        self.assertEqual(manifest['coordinate_contract'],COORDINATE_CONTRACT)


class EventTests(unittest.TestCase):
    def test_trigger_has_exact_condition_and_effective_ticks(self):
        event=EventState(specimen()['experiment']['actors'][0])
        event.consider(49,11.,10.);self.assertEqual(event.state,'armed')
        event.consider(50,11.,30.);self.assertEqual(event.state,'armed')
        event.consider(51,11.,10.);self.assertEqual(event.state,'active')
        self.assertEqual(event.trigger_tick,51);self.assertEqual(event.action_start_tick,52)
        event.consider(52,11.,10.);self.assertEqual(len(event.transitions),2)

    def test_missed_event_is_not_silently_retriggered(self):
        event=EventState(specimen()['experiment']['actors'][0])
        event.consider(100,41.,10.);self.assertEqual(event.state,'missed')
        event.consider(101,20.,10.);self.assertEqual(event.state,'missed')

    def test_early_wait_is_not_a_scene_error(self):
        event=EventState(specimen()['experiment']['actors'][0])
        for tick in range(200):event.consider(tick,5.,20.)
        self.assertEqual(event.state,'waiting');self.assertIsNone(event.trigger_tick)


class Value:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)


class CoordinateTests(unittest.TestCase):
    @staticmethod
    def route(points):
        return RouteGeometry([Value(transform=Value(location=Value(x=x,y=y,z=z))) for x,y,z in points])

    def test_slope_arc_and_fraction_use_xy_height_only_selects_level(self):
        geometry=self.route([(0.,0.,0.),(10.,0.,10.)])
        self.assertAlmostEqual(geometry.length,10.)
        # A box may sit above the road surface. Its z must not move its s forward.
        s,d,index,error=geometry.project(Value(x=4.,y=2.,z=8.))
        self.assertAlmostEqual(s,4.)
        self.assertAlmostEqual(d,-2.)
        self.assertAlmostEqual(error,2.)
        layered=self.route([(0.,0.,0.),(10.,0.,0.),(20.,0.,10.),(10.,0.,10.),(0.,0.,10.)])
        self.assertEqual(layered.project(Value(x=4.,y=0.,z=10.))[2],3)

    def test_box_height_at_grade_vertex_cannot_change_xy_projection(self):
        geometry=self.route([(0.,0.,0.),(10.,0.,5.),(20.,0.,0.)])
        for x in (9.9,10.1):
            for z in (0.,6.,10.):
                s,d,_,error=geometry.project(Value(x=x,y=0.,z=z))
                self.assertAlmostEqual(s,x)
                self.assertAlmostEqual(d,0.)
                self.assertAlmostEqual(error,0.)

    def test_event_progress_and_gap_use_rotated_nonzero_box_center(self):
        def actor(identity,x,y,yaw,box_offset_x):
            def transform(point):
                # Emulate CARLA's in-place transform; source bounding box must survive.
                c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
                px,py=point.x,point.y
                point.x=x+c*px-s*py;point.y=y+s*px+c*py
                return point
            pose=Value(rotation=Value(yaw=yaw),transform=transform)
            return Value(id=identity,bounding_box=Value(location=Value(x=box_offset_x,y=0.,z=0.),
                rotation=Value(yaw=0.),extent=Value(x=2.,y=1.)),get_transform=lambda:pose,
                get_velocity=lambda:Value(x=0.,y=0.,z=0.))
        engine=Experiment.__new__(Experiment)
        engine.geometry=self.route([(10.,0.,0.),(10.,40.,4.)])
        engine.carla=Value(Location=lambda **values:Value(**values))
        engine.ego=actor(1,10.,20.,90.,3.)
        engine._tick=-1;engine._frame=None;engine._ego_hint=None
        engine._ego_stops={};engine._rows=[];engine._errors=[]
        engine.settings={'instance_hash':'fake'};engine.physics_hz=50
        engine.after_tick(100,0)
        self.assertAlmostEqual(engine._ego_s,23.)
        self.assertAlmostEqual(engine.ego.bounding_box.location.x,3.)
        other=actor(2,10.,31.,90.,1.)
        self.assertAlmostEqual(engine._gap({'actor':other}),5.)
        self.assertEqual(engine.evaluation_state()['coordinate_contract'],COORDINATE_CONTRACT)

    def test_director_freezes_invalid_route_progress_and_recovers_locally(self):
        position=Value(x=2.,y=0.,z=0.);pose=Value(rotation=Value(yaw=0.),transform=lambda point:position)
        actor=Value(id=1,bounding_box=Value(location=Value(x=0.,y=0.,z=0.)),
            get_transform=lambda:pose,get_velocity=lambda:Value(x=0.,y=0.,z=0.))
        engine=Experiment.__new__(Experiment)
        engine.geometry=self.route([(0.,0.,0.),(10.,0.,0.),(20.,0.,0.)]);engine.ego=actor
        engine.carla=Value(Location=lambda **values:Value(**values))
        engine._tick=-1;engine._frame=None;engine._ego_hint=None;engine._ego_s=0.
        engine._ego_stops={};engine._rows=[];engine._errors=[];engine.events=[]
        engine.settings={'instance_hash':'fake'};engine.physics_hz=50
        engine.after_tick(100,0);self.assertEqual(engine._ego_s,2.)
        position.x=120.;engine.after_tick(101,1)
        self.assertEqual(engine._ego_s,2.)
        self.assertFalse(engine.evaluation_state()['ego_route_projection']['valid'])
        self.assertEqual(engine.evaluation_state()['ego_route_projection']['euclidean_error_m'],100.)
        self.assertTrue(engine.evaluation_state()['scenario_valid'])
        position.x=2.1;engine.after_tick(102,2)
        self.assertAlmostEqual(engine._ego_s,2.1)
        self.assertTrue(engine.evaluation_state()['ego_route_projection']['valid'])
        pose.rotation.yaw=180.;engine.after_tick(103,3)
        self.assertFalse(engine.evaluation_state()['ego_route_projection']['valid'])
        self.assertEqual([row['event'] for row in engine.events],[
            'scenario_route_projection_invalid','scenario_route_projection_recovered','scenario_route_projection_invalid'])


class Light:
    def __init__(self,identity,x,stops=None):
        self.id=identity;self.x=x;self.state='Yellow';self.frozen=False;self.is_alive=True;self.group=[]
        self.stops=[] if stops is None else stops
    def get_transform(self):return Value(location=Value(x=self.x,y=0.,z=0.),rotation=Value(yaw=0.))
    def get_group_traffic_lights(self):return self.group
    def get_stop_waypoints(self):return self.stops
    def get_state(self):return self.state
    def set_state(self,state):self.state=state
    def is_frozen(self):return self.frozen
    def freeze(self,value):self.frozen=value


class SignalTests(unittest.TestCase):
    @staticmethod
    def stop(road,lane,yaw,section=0):
        return Value(road_id=road,section_id=section,lane_id=lane,transform=Value(rotation=Value(yaw=yaw)))

    def test_same_incoming_approach_heads_share_phase_other_directions_are_exclusive(self):
        lights=[Light(1,0.,[self.stop(10,-1,0.)]),Light(2,1.,[self.stop(10,-2,2.)]),
            Light(3,2.,[self.stop(10,1,180.)]),Light(4,3.,[self.stop(20,-1,90.)])]
        for light in lights:light.group=list(lights)
        world=Value(get_actors=lambda:Value(filter=lambda pattern:list(lights)))
        sdk=Value(TrafficLightState=Value(Red='Red',Green='Green',Yellow='Yellow'))
        program=SignalProgram(world,sdk,{'green_ticks':5,'yellow_ticks':2,'all_red_ticks':1})
        self.assertEqual(len(program.groups[0]),3)
        self.assertEqual([light.state for light in lights],['Green','Green','Red','Red'])
        for tick in range(72):
            program.apply(tick)
            self.assertEqual(lights[0].state,lights[1].state)
            active=[phase for phase in program.groups[0] if any(light.state in ('Green','Yellow') for light in phase)]
            self.assertLessEqual(len(active),1)
        self.assertEqual(program.phase_evidence[0]['phases'][0]['incoming_sections'],[[10,0,-1]])
        program.close()

    def test_shared_pole_yaw_does_not_merge_incompatible_stop_waypoint_directions(self):
        lights=[Light(1,0.,[self.stop(10,-1,0.)]),Light(2,1.,[self.stop(10,-2,90.)]),Light(3,2.)]
        for light in lights:light.group=list(lights)
        world=Value(get_actors=lambda:Value(filter=lambda pattern:list(lights)))
        sdk=Value(TrafficLightState=Value(Red='Red',Green='Green',Yellow='Yellow'))
        program=SignalProgram(world,sdk,{})
        self.assertEqual(len(program.groups[0]),3)
        self.assertEqual(program.phase_evidence[0]['phases'][2]['source'],'unknown_single_light')
        program.close()

    def test_no_conflicting_greens_and_restoration(self):
        lights=[Light(22,3.),Light(3,1.),Light(90,2.)]
        for light in lights:light.group=list(lights)
        world=Value(get_actors=lambda:Value(filter=lambda pattern:list(lights)))
        sdk=Value(TrafficLightState=Value(Red='Red',Green='Green',Yellow='Yellow'))
        program=SignalProgram(world,sdk,{'green_ticks':5,'yellow_ticks':2,'all_red_ticks':1})
        self.assertEqual([x.state for x in sorted(lights,key=lambda x:x.x)],['Green','Red','Red'])
        for tick in range(100):
            program.apply(tick)
            self.assertLessEqual(sum(x.state in ('Green','Yellow') for x in lights),1)
            self.assertTrue(all(x.frozen for x in lights))
        self.assertEqual(program.close(),[])
        self.assertTrue(all(x.state=='Yellow' and not x.frozen for x in lights))


class ShortQueueTests(unittest.TestCase):
    def test_short_route_keeps_original_goal_and_continuing_queue_actors(self):
        import yaml
        root=Path(__file__).resolve().parents[1]
        path=root/'config/carla_scenarios/town10hd_queue.yaml'
        spec=yaml.safe_load(path.read_text(encoding='utf-8'))
        settings=spec['experiment']
        for route in [settings['route']]+[actor['placement']['route'] for actor in settings['actors']]:
            route.update(json.loads((path.parent/route['file']).read_text(encoding='utf-8')))
        settings,_=resolve_spec(spec,1)
        points=settings['route']['points'];end=points[-1]
        self.assertEqual(spec['duration_s'],60)
        self.assertAlmostEqual(settings['route']['length_m'],139.23249601660032)
        self.assertEqual((end['road_id'],end['section_id'],end['lane_id']),(2,0,1))
        self.assertAlmostEqual(end['x_m'],102.5548324584961)
        self.assertAlmostEqual(end['y_m'],45.63352584838867)
        self.assertEqual(settings['route']['task']['entry'],points[0])
        self.assertEqual(settings['route']['task']['exit'],end)
        self.assertEqual(settings['route']['server_validation']['status'],'pending')
        for actor,expected_offset in zip(settings['actors'],(17.69595477440491,38.71684293441891)):
            route=actor['placement']['route'];start=route['points'][0]
            self.assertIn(start,points)
            self.assertEqual(start['lane_id'],points[0]['lane_id'])
            index=points.index(start)
            offset=sum(math.hypot(a['x_m']-b['x_m'],a['y_m']-b['y_m']) for a,b in zip(points[:index],points[1:index+1]))
            self.assertAlmostEqual(offset,expected_offset)
            # NPC routes go beyond the ego goal; otherwise endpoint braking
            # creates an artificial stationary blockade at the destination.
            self.assertIn(end,route['points'])
            goal_index=route['points'].index(end)
            remaining=sum(math.hypot(a['x_m']-b['x_m'],a['y_m']-b['y_m']) for a,b in zip(route['points'][goal_index:],route['points'][goal_index+1:]))
            self.assertGreater(remaining,50.)
            trigger=actor['trigger'];self.assertLess(trigger['ego_s_m'][1],40.)
            self.assertLess(trigger['earliest_tick']+actor['hold_ticks'],750)
            self.assertGreater(settings['route']['length_m']-trigger['ego_s_m'][1],90.)
        self.assertEqual(settings['signals'],{'green_ticks':250,'yellow_ticks':100,'all_red_ticks':50,'offset_ticks':0})


class ProductionWiringTests(unittest.TestCase):
    def test_platform_expands_and_binds_effective_execution_calibration_without_sdk(self):
        import runpy
        from nine_grid.carla_execution import SignedAccelerationController
        from nine_grid.carla_experiment_validation import infrastructure_contract
        root=Path(__file__).resolve().parents[1];entry=runpy.run_path(str(root/'scripts/run_carla.py'))
        platform=entry['load_yaml'](root/'config/carla.yaml')
        args=Value(seed=1,map=None,runtime_mode=None,world_plan=None,carla_python=None,
            host=None,port=None,exclusive_world=False,allow_world_reset=False)
        path=root/'config/carla_scenarios/town10hd_queue.yaml'
        cfg=entry['platform_configuration'](platform,args,path)
        defaults=SignedAccelerationController(.6).settings
        self.assertEqual(cfg['longitudinal_control'],defaults)
        without=copy.deepcopy(platform);without['carla'].pop('longitudinal_control')
        self.assertEqual(entry['platform_configuration'](without,args,path)['longitudinal_control'],defaults)
        changed=copy.deepcopy(cfg);changed['longitudinal_control']['speed_kp']+=.1
        self.assertNotEqual(infrastructure_contract(cfg),infrastructure_contract(changed))
        self.assertEqual(infrastructure_contract(cfg)['longitudinal_controller_version'],SignedAccelerationController.VERSION)
        invalid=copy.deepcopy(platform);invalid['carla']['longitudinal_control']['unknown_gain']=1.
        with self.assertRaises(ValueError):entry['platform_configuration'](invalid,args,path)
        self.assertNotIn('carla',sys.modules)

    @staticmethod
    def raw_waypoint(identity,x,y=0.,heading=0.):
        return {'lane_id':identity,'position':[x,y],'z':0.,'heading_rad':heading,
            'width_m':3.5,'lane_type':'Driving'}

    def replay_fixture(self):
        from nine_grid.carla_task_criteria import TaskCriteria
        from nine_grid.carla_experiment_validation import _ProjectionMapReplay
        points=[{'road_id':1,'section_id':0,'lane_id':1,'x_m':x,'y_m':0.,'z_m':0.,
            'yaw_deg':0.,'s_m':x,'junction':False} for x in (0.,10.,20.,30.,40.)]
        settings={'route':{'points':points},'family':'route'};task=TaskCriteria(settings)
        raw={'version':'carla-route-map-evidence-1.0','source':'native_public_waypoint_topology',
            'physical_tick':0,'boundary_gates':[],'corridors':{},'route_widths_m':[3.5]*len(points)}
        for gate in task.checkpoints:
            index=gate['index'];wp=self.raw_waypoint('1|0|1',points[index]['x_m'])
            raw['boundary_gates'].append({'index':index,'route_waypoint':wp,'distance_m':3.,
                'previous':[self.raw_waypoint('676|0|1',-2.)] if index==0 else [],'next':[]})
        raw['corridors']['0']={'route_waypoint':self.raw_waypoint('1|0|1',0.),
            'neighbour_links':[{'side':'left','from_lane_id':'1|0|1',
                'to_waypoint':self.raw_waypoint('1|0|2',0.,3.5)}]}
        replay=_ProjectionMapReplay(settings,task);replay.update(raw,0);task.projection_metadata=replay.corridor
        return settings,task,replay,raw

    def test_native_boundary_queries_replay_start_gate_without_trusting_passed_ticks(self):
        _,task,_,_=self.replay_fixture()
        ego={'id':'ego','lane_id':'676|0|1','position':[.18,0.],'heading':0.,'velocity':[1.,0.],
            'length':4.,'width':2.,'is_junction':False}
        task.observe(0,ego,[],{})
        self.assertEqual(task.report()['route_checkpoints_completed'],1)
        self.assertTrue(task.report()['checkpoints'][0]['passed_tick']==0)
        self.assertTrue(task.report()['events'][0]['boundary_topology_match'])

    def test_native_cross_section_geometry_is_recomputed_and_spoofed_map_links_rejected(self):
        settings,task,replay,raw=self.replay_fixture()
        ego={'lane_id':'1|0|2','position':[.2,3.5],'heading':0.}
        self.assertTrue(replay.corridor(0,ego,.2)['contains_observed_center'])
        ego['position'][1]=40.
        self.assertFalse(replay.corridor(0,ego,.2)['contains_observed_center'])
        from nine_grid.carla_task_criteria import TaskCriteria
        from nine_grid.carla_experiment_validation import _ProjectionMapReplay
        raw['corridors']['0']['neighbour_links'][0]['to_waypoint']['position'][1]=40.
        with self.assertRaises(ValueError):_ProjectionMapReplay(settings,TaskCriteria(settings)).update(raw,0)

    def test_physical_validator_replays_boundary_gate_and_rejects_stale_projection(self):
        import io
        from unittest.mock import patch
        from nine_grid.carla_experiment_validation import _physics,REQUIRED_CHECKS
        settings,task,_,raw=self.replay_fixture();settings['instance_hash']='test-instance'
        ego={'id':'ego','lane_id':'676|0|1','position':[.18,0.],'heading':0.,'velocity':[1.,0.],
            'length':4.,'width':2.,'is_junction':False,'z':0.}
        initial={'source_frame':100,'actors_at_source':[{'role':'ego','carla_actor_id':'ego',**copy.deepcopy(ego)}]}
        rows=[{'physical_tick':tick,'frame':100+tick,'timestamp':1000.+tick/50.,'sim_time_s':tick/50.,
            'instance_hash':settings['instance_hash'],'ego':copy.deepcopy(ego),'actors':[],
            'route_map_evidence':raw if tick==0 else None,'route_s_m':.18} for tick in range(6)]
        rows[-1]['ego']['position']=[120.,0.]
        rows[-1]['route_map_evidence']={'version':'carla-route-map-evidence-1.0',
            'source':'native_public_waypoint_topology','physical_tick':5,'corridors':{
                '1':{'route_waypoint':self.raw_waypoint('1|0|1',10.),'neighbour_links':[]}}}
        evaluation={'physical_ticks':5,'first_frame':100,'task_criteria':{
            'route_checkpoints_completed':1,'route_checkpoints_total':len(task.checkpoints),'wrong_exit':False,
            'completed':False,'projection_invalid_ticks':1,'route_projection':{'valid':False}}}
        errors={key:[] for key in REQUIRED_CHECKS}
        with patch.object(Path,'open',return_value=io.StringIO('\n'.join(json.dumps(row) for row in rows))):
            clock,recomputed=_physics(Path('unused'),settings,initial,evaluation,errors)
        self.assertFalse(any(errors.values()),errors)
        self.assertEqual(recomputed['route_checkpoints_completed'],1)
        self.assertEqual(recomputed['projection_invalid_ticks'],1)
        self.assertIsNone(clock[5]['director_route_s_m'])
        self.assertGreater(clock[5]['director_projection_error_m'],50.)


if __name__=='__main__':unittest.main(verbosity=2)
