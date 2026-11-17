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


class Light:
    def __init__(self,identity,x):
        self.id=identity;self.x=x;self.state='Yellow';self.frozen=False;self.is_alive=True;self.group=[]
    def get_transform(self):return Value(location=Value(x=self.x,y=0.,z=0.),rotation=Value(yaw=0.))
    def get_group_traffic_lights(self):return self.group
    def get_state(self):return self.state
    def set_state(self,state):self.state=state
    def is_frozen(self):return self.frozen
    def freeze(self,value):self.frozen=value


class SignalTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main(verbosity=2)
