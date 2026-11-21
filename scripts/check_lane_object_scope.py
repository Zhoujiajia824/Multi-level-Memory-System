"""无CARLA/模型调用：沿车道对象准入与八向规划的回归检查。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from types import SimpleNamespace
from dataclasses import replace
import unittest
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from nine_grid.contracts import Actor, Lane, SceneSnapshot
from nine_grid.object_scope import scope_scene, lane_role_geometry
from nine_grid.planning_objects import planning_scene, road_rule_observations


def actor(key, x, y=0., kind='vehicle', heading=0., lane='c', speed=0.):
    return Actor(key,kind,np.array([x,y],float),np.array([speed*np.cos(heading),speed*np.sin(heading)]),
                 np.zeros(2),heading,4. if kind!='pedestrian' else .6,1.8 if kind!='pedestrian' else .6,lane_id=lane,
                 radius=.3 if kind=='pedestrian' else None)


def lane(key, points, width=3.5, **kwargs):
    points=np.asarray(points,float)
    polygon=np.array(LineString(points).buffer(width/2,cap_style=2).exterior.coords)
    return Lane(key,points,polygon,width,15.,**kwargs)


def straight(crossable=True):
    ls=[lane('c',[[-60,0],[100,0]],left='l',right='r'),
        lane('l',[[-60,3.5],[100,3.5]],right='c'),lane('r',[[-60,-3.5],[100,-3.5]],left='c'),
        lane('far',[[-60,7],[100,7]])]
    for item in ls:
        y=item.center[0,1]
        item.boundaries=[dict(points=[[-60,y-1.75],[100,y-1.75]],crossable=crossable),
                         dict(points=[[-60,y+1.75],[100,y+1.75]],crossable=crossable)]
    return SceneSnapshot('test:1',0.,actor('ego',0.,speed=10.),[],ls,ls[0].center,['c'],
                         unary_union([Polygon(i.polygon) for i in ls]),'scope_test',
                         diagnostics={'lane_map_complete':True})


def ids(scene, config=None):
    value, diagnostic=scope_scene(scene, config)
    return {a.id for a in value.actors},diagnostic


class ScopeTests(unittest.TestCase):
    def test_types_ranges_and_adjacent_only(self):
        s=straight();s.actors=[actor('front',35),actor('beyond',41),actor('rear',-19),actor('rear_far',-21),
            actor('left',15,3.5,lane='l'),actor('two_lanes',15,7,lane='far'),
            actor('pole',15,kind='pole'),actor('obstacle',20,kind='static_obstacle'),
            actor('offroad_obstacle',20,13,kind='static_obstacle'),actor('walker',8,kind='pedestrian'),
            actor('bicycle',12,kind='bicycle')]
        self.assertEqual(ids(s)[0],{'front','rear','left','obstacle','walker','bicycle'})

    def test_exact_default_boundaries_and_configured_shorter_range(self):
        s=straight();s.actors=[actor('front_edge',40),actor('front_out',40.01),
            actor('rear_edge',-20),actor('rear_out',-20.01)]
        chosen,d=ids(s)
        self.assertEqual(d['settings'],{'front_m':40.,'rear_m':20.})
        self.assertEqual(chosen,{'front_edge','rear_edge'})
        self.assertEqual(ids(s,{'planning_objects':{'front_m':30.,'rear_m':10.}})[0],set())

    def test_solid_lane_disallows_neighbor(self):
        s=straight(False);s.actors=[actor('own',10),actor('adjacent',10,3.5,lane='l')]
        self.assertEqual(ids(s)[0],{'own'})
        self.assertEqual(ids(s)[1]['roads']['left']['reason'],'noncrossable_boundary')

    def test_map_unknown_does_not_invent_adjacency(self):
        s=straight();s.diagnostics['lane_map_complete']=False;s.actors=[actor('adjacent',10,3.5,lane='l')]
        self.assertEqual(ids(s)[0],set())

    def test_curved_lane_uses_arc_not_radius(self):
        theta=np.linspace(-1,2.8,150)
        line=np.c_[15*np.cos(theta),15*np.sin(theta)]
        ls=lane('c',line,width=3.5)
        s=straight();s.ego=actor('ego',15,0,heading=np.pi/2)
        s.lanes=[ls];s.route=line;s.drivable=Polygon(ls.polygon)
        s.actors=[actor('inside',15*np.cos(1.5),15*np.sin(1.5),heading=1.5+np.pi/2),
                  actor('near_but_long_arc',15*np.cos(2.7),15*np.sin(2.7),heading=2.7+np.pi/2)]
        self.assertLess(np.linalg.norm(s.actors[1].position-s.ego.position),40.)
        self.assertEqual(ids(s)[0],{'inside'})

    def test_overlapping_junction_owner_disambiguation(self):
        s=straight();cross=lane('cross',[[10,-30],[10,30]],width=16)
        s.lanes.append(cross)
        s.actors=[actor('cross_traffic',10,0,heading=np.pi/2,lane='cross',speed=3),
                  actor('own_traffic',12,0,lane='c',speed=3)]
        chosen, diagnostic = ids(s)
        self.assertEqual(chosen, {'own_traffic', 'cross_traffic'})
        self.assertEqual(diagnostic['objects']['cross_traffic']['owner_lane_id'], 'cross')
        self.assertIn('front', diagnostic['objects']['cross_traffic']['roles'])

    def test_center_and_true_lane_straddle(self):
        s=straight();s.actors=[actor('straddle',8,1.7,lane='c')]
        chosen,d=ids(s)
        self.assertEqual(chosen,{'straddle'})
        self.assertEqual(set(d['objects']['straddle']['roles']),{'front','left_front'})

    def test_body_intrudes_lane_without_center_owner(self):
        s=straight();s.lanes=s.lanes[:1];s.lanes[0].left=s.lanes[0].right=None
        s.actors=[actor('intrudes',10,2.2,kind='static_obstacle',lane=None),
                  actor('offroad',10,4.0,kind='static_obstacle',lane=None)]
        self.assertEqual(ids(s)[0],{'intrudes'})

    def test_same_lane_wrong_way_preserved(self):
        s=straight();s.actors=[actor('wrongway',12,heading=np.pi,speed=3)]
        self.assertEqual(ids(s)[0],{'wrongway'})

    def test_side_and_longitudinal_net_gap(self):
        s=straight();s.actors=[actor('beside',1,3.5,lane='l'),actor('ahead',10,3.5,lane='l')]
        _,d=ids(s)
        self.assertEqual(d['objects']['beside']['roles'],['left_side'])
        self.assertEqual(d['objects']['ahead']['roles'],['left_front'])
        self.assertAlmostEqual(d['objects']['ahead']['memberships']['left_front']['longitudinal_gap_m'],6.)

    def test_navigation_chain_follows_turn_and_not_branch(self):
        s=straight();s.lanes=[lane('c',[[-20,0],[10,0]],successors=['straight','turn']),
            lane('straight',[[10,0],[70,0]]),lane('turn',[[10,0],[13,0],[17,2],[20,6],[20,30]])]
        s.route=np.vstack((s.lanes[0].center,s.lanes[2].center[1:]));s.route_lane_ids=['c','turn']
        s.actors=[actor('turning',20,12,heading=np.pi/2,lane='turn'),actor('offroute',28,0,lane='straight')]
        chosen,d=ids(s)
        self.assertEqual(chosen,{'turning'})
        self.assertEqual(d['chains']['center']['lane_ids'],['c','turn'])

    def test_rear_predecessor_chain(self):
        s=straight();s.lanes=[lane('prev',[[-40,0],[0,0]],successors=['c']),lane('c',[[0,0],[70,0]])]
        s.ego=actor('ego',2);s.route=s.lanes[1].center;s.route_lane_ids=['prev','c']
        s.actors=[actor('rear',-16,lane='prev'),actor('rear_far',-25,lane='prev')]
        self.assertEqual(ids(s)[0],{'rear'})

    def test_attention_and_conflict_cannot_expand_range(self):
        s=straight();s.ego.velocity=np.array([30.,0.]);s.actors=[actor('close',20),actor('far',45),actor('fast_rear',-25,speed=50)]
        answer=SimpleNamespace(attention=[dict(rank=1,id='far'),dict(rank=2,id='fast_rear')])
        out,d=planning_scene(s,{'planning_objects':{'adaptive_front':True}},answer)
        self.assertEqual({a.id for a in out.actors},{'close'})
        self.assertEqual(len(d['attention_rejected']),2)
        self.assertFalse(d['adaptive_expansion'])

    def test_rule_cache_cannot_restore_outside_actors(self):
        s=straight();s.rules['road_priority_available']=True;s.actors=[actor('near',12),actor('far',90)]
        s.diagnostics['road_rule_observations']=road_rule_observations(s)
        out,d=planning_scene(s)
        self.assertEqual([a['id'] for a in road_rule_observations(out)['actors']],['near'])
        self.assertEqual([a.id for a in s.actors],['near','far'])

    def test_unknown_ego_lane_does_not_use_stable_history(self):
        s=straight();s.ego.lane_id=None;s.diagnostics['stable_navigation_lane_id']='c'
        s.actors=[actor('a',10)]
        chosen,d=ids(s)
        self.assertEqual(chosen,set())
        self.assertIsNone(d['chains']['center'])
        self.assertFalse(d['roads']['left']['known'])

    def test_same_frame_scope_rechecks_mutable_states(self):
        s=straight();s.actors=[actor('a',10)]
        scoped,_=scope_scene(s);scoped.actors[0].position[0]=80
        self.assertEqual(ids(scoped)[0],set())

    def test_same_frame_selected_planning_mutation_is_rejected(self):
        s=straight();s.actors=[actor('a',10)]
        selected,_=planning_scene(s)
        for position in (12.,80.):
            selected.actors[0].position[0]=position
            with self.assertRaisesRegex(ValueError,'发生变化'):
                planning_scene(selected)

    def test_selected_planning_reentry_preserves_attention(self):
        s=straight();s.ego.velocity[:]=0;s.actors=[actor('near',10),actor('far',20)]
        answer=SimpleNamespace(attention=[dict(rank=1,id='far')])
        selected,d=planning_scene(s,answer=answer)
        again,other=planning_scene(selected)
        self.assertEqual({a.id for a in again.actors},{'near','far'})
        self.assertEqual(other['selected_input_signature'],d['selected_input_signature'])

    def test_config_none_inherits_bound_scope_range(self):
        s=straight();s.actors=[actor('a',60)]
        scoped,_=scope_scene(s,{'planning_objects':{'front_m':70.,'rear_m':20.}})
        selected,d=planning_scene(scoped)
        self.assertEqual({a.id for a in selected.actors},{'a'})
        self.assertEqual(d['settings']['front_m'],70.)
        self.assertEqual(d['settings']['rear_m'],20.)
        self.assertEqual(d['effective_front_m'],70.)

    def test_shared_diagnostic_on_raw_scene_does_not_admit_all(self):
        s=straight();s.actors=[actor('in',10),actor('out',60)]
        _,d=scope_scene(s);s.diagnostics['object_scope']=d
        self.assertEqual(ids(s)[0],{'in'})
        self.assertEqual(ids(s,{'planning_objects':{'front_m':70.}})[0],{'in','out'})


if __name__=='__main__':
    unittest.main(verbosity=2)
