"""No simulator connection: independently test actual task success and failures."""
from pathlib import Path
import json
import math
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from nine_grid.carla_task_criteria import TaskCriteria


def settings(kind='route',junction=False):
    points=[]
    for x in range(0,152,2):
        road=1 if not junction or x<20 else 100 if x<=30 else 2
        points.append({'road_id':road,'section_id':0,'lane_id':1,'s_m':x,
            'x_m':x,'y_m':0.,'z_m':0.,'yaw_deg':0.,'junction':road==100})
    return {'route':{'points':points,'task':{'kind':kind}}}


def actor(identity,x,y=0.,road=1,lane=1,junction=False):
    return {'id':str(identity),'position':[float(x),float(y)],'z':0.,'heading':0.,
        'length':4.,'width':2.,'lane_id':'{}|0|{}'.format(road,lane),
        'velocity':[10.,0.],'is_junction':junction}


class TaskTests(unittest.TestCase):
    def test_catalogue_prescribed_path_completes_navigation(self):
        root=Path(__file__).resolve().parents[1]
        checked=0
        for config in (root/'config/carla_scenarios').glob('*.yaml'):
            path=root/'config/carla_routes'/(config.stem+'.json')
            if not path.is_file():continue
            route=json.loads(path.read_text(encoding='utf-8'))
            check=TaskCriteria({'route':route})
            for tick,p in enumerate(route['points']):
                ego=actor(1,p['x_m'],-p['y_m'],road=p['road_id'],lane=p['lane_id'],junction=p.get('junction',False))
                ego.update(heading=-math.radians(p['yaw_deg']),z=p['z_m'],
                    lane_id='{}|{}|{}'.format(p['road_id'],p.get('section_id',0),p['lane_id']))
                result=check.observe(tick,ego,[])
            with self.subTest(scenario=config.stem):
                self.assertEqual(result['route_checkpoints_completed'],result['route_checkpoints_total'])
                self.assertFalse(result['wrong_exit'])
                if check.kind=='signal_junction':self.assertTrue(result['turn']['completed'])
                if check.kind=='roundabout':self.assertTrue(result['maneuver_completed'])
            checked+=1
        self.assertGreaterEqual(checked,20)

    def test_end_position_does_not_replace_ordered_traversal(self):
        check=TaskCriteria(settings())
        check.observe(0,actor(1,0),[])
        result=check.observe(1,actor(1,150),[])
        self.assertFalse(result['completed'])
        self.assertLess(result['route_checkpoints_completed'],result['route_checkpoints_total'])

    def test_start_boundary_requires_native_topology_and_near_aligned_gate(self):
        for linked, y, heading, expected in ((False, 0., 0., False), (True, 0., 0., True),
                                            (True, 4., 0., False), (True, 0., math.pi, False)):
            check=TaskCriteria(settings())
            if linked:check.set_boundary_topology({0:['676|0|2']})
            ego=actor(1,.18,y,road=676,lane=2);ego['heading']=heading
            result=check.observe(0,ego,[])
            with self.subTest(linked=linked,y=y,heading=heading):
                self.assertEqual(result['route_checkpoints_completed']>0,expected)
        check=TaskCriteria(settings());check.set_boundary_topology({0:['676|0|2']})
        check.observe(0,actor(1,.18,road=676,lane=2),[])
        for tick in range(1,151):result=check.observe(tick,actor(1,tick),[])
        self.assertTrue(result['completed'])

    def test_projection_invalid_cannot_consume_checkpoints_or_be_task_complete(self):
        check=TaskCriteria(settings());check.observe(0,actor(1,0),[])
        before=check._next;progress=check.progress
        result=check.observe(1,actor(1,100,50),[])
        self.assertFalse(result['route_projection']['valid'])
        self.assertEqual(check._next,before)
        self.assertEqual(check.progress,progress)
        self.assertIn('route_projection_invalid',result['reasons'])
        self.assertFalse(result['completed'])

    def test_only_verified_cross_section_metadata_can_extend_projection_distance(self):
        for verified in (False,True):
            check=TaskCriteria(settings())
            check.projection_metadata=lambda index,ego,s:{'contains_observed_center':verified,
                'maximum_distance_m':8.75,'source':'native_test_fixture'}
            result=check.observe(0,actor(1,.1,7.,lane=3),[])
            self.assertEqual(result['route_projection']['valid'],verified)
            if not verified:self.assertEqual(result['route_projection']['max_distance_m'],6.)

    def test_declared_route_is_completed_with_actual_samples(self):
        check=TaskCriteria(settings())
        for tick in range(151):result=check.observe(tick,actor(1,tick),[])
        self.assertTrue(result['completed'])

    def test_actual_wrong_exit_is_preserved(self):
        check=TaskCriteria(settings(junction=True))
        for tick in range(151):
            road=1 if tick<20 else 100 if tick<=30 else 3
            result=check.observe(tick,actor(1,tick,road=road,junction=road==100),[])
        self.assertFalse(result['completed']);self.assertTrue(result['wrong_exit'])

    def test_legitimate_junction_exit(self):
        check=TaskCriteria(settings(junction=True))
        for tick in range(151):
            road=1 if tick<20 else 100 if tick<=30 else 2
            result=check.observe(tick,actor(1,tick,road=road,junction=road==100),[])
        self.assertTrue(result['completed']);self.assertFalse(result['wrong_exit'])

    def test_overtake_requires_left_pass_and_stable_return(self):
        check=TaskCriteria(settings('overtake_return'))
        for tick in range(301):
            y=0. if tick<=20 else min(3.5,(tick-20)*.35) if tick<=100 else max(0.,3.5-(tick-100)*.35)
            ego=actor(1,tick*.5,y,lane=2 if y>1.7 else 1)
            result=check.observe(tick,ego,[actor(2,20+tick*.2)],{'slow_lead':'2'})
        self.assertTrue(result['completed']);self.assertEqual(result['overtake']['phase'],'completed')

    def test_following_to_endpoint_is_not_overtaking(self):
        check=TaskCriteria(settings('overtake_return'))
        for tick in range(151):
            result=check.observe(tick,actor(1,tick),[actor(2,tick+20)],{'slow_lead':'2'})
        self.assertFalse(result['completed']);self.assertFalse(result['maneuver_completed'])

    def test_right_pass_does_not_count_as_left_overtake(self):
        check=TaskCriteria(settings('overtake_return'))
        for tick in range(301):
            y=-3.5 if 20<tick<110 else 0.
            result=check.observe(tick,actor(1,tick*.5,y,lane=2 if y else 1),[actor(2,20+tick*.2)],{'slow_lead':'2'})
        self.assertFalse(result['completed']);self.assertTrue(result['overtake']['wrong_side_pass'])

    def test_missing_physical_tick_is_rejected(self):
        check=TaskCriteria(settings());check.observe(0,actor(1,0),[])
        with self.assertRaises(ValueError):check.observe(2,actor(1,1),[])


if __name__=='__main__':unittest.main(verbosity=2)
