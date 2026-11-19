"""CARLA road participant category regressions; standard library, no live server.

The blueprint allowlist is based on the CARLA 0.9.15 Road props catalogue:
https://carla.readthedocs.io/en/0.9.15/catalogue_props/
These checks cover category evidence, not downstream lane/range eligibility.
"""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nine_grid.carla_bridge import (CarlaSession, actor_category, environment_category,
                                   environment_catalogue, OBJECT_CATEGORY_VERSION)


def vector(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def rotation(yaw=0., pitch=0., roll=0.):
    return NS(yaw=yaw, pitch=pitch, roll=roll)


class Box:
    def __init__(self, location=None, extent=None):
        self.location = location or vector(10., 0., 1.)
        self.extent = extent or vector(2., 1., 1.)
        self.rotation = rotation()

    def get_world_vertices(self, transform):
        return [vector(self.location.x + x*self.extent.x,
                       self.location.y + y*self.extent.y,
                       self.location.z + z*self.extent.z)
                for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]


LABELS = ('Any', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Vehicles',
          'Bicycle', 'Pedestrians', 'Static', 'Other', 'Fences', 'GuardRail',
          'Walls', 'Poles', 'TrafficSigns', 'TrafficLight', 'Buildings', 'Roads')
CARLA = NS(CityObjectLabel=NS(**{name: name for name in LABELS}),
           BoundingBox=Box, Location=vector, Vector3D=vector, Rotation=rotation,
           Transform=lambda: NS(), LaneType=NS(Driving='Driving'))


def item(identity, label, name=None, box=None):
    return NS(id=identity, type=label, name=name or 'component_' + str(identity),
              bounding_box=box or Box(vector(identity*10., 0., 1.)))


class Categories(unittest.TestCase):
    def catalogue(self, values):
        # There is deliberately no world mutation API in this fixture.
        return environment_catalogue(NS(get_environment_objects=lambda _: values), CARLA)

    def test_map_infrastructure_is_excluded_even_with_broken_boxes(self):
        labels = ('Fences', 'GuardRail', 'Walls', 'Poles', 'TrafficSigns', 'TrafficLight',
                  'Buildings', 'Roads')
        values = [item(i, label, box=NS()) for i, label in enumerate(labels)]
        objects, audit = self.catalogue(values)
        self.assertEqual(objects, [])
        self.assertEqual(audit['excluded_label_counts'], {label: 1 for label in labels})
        self.assertTrue(all(not row['included'] for row in audit['category_decisions']))

    def test_static_other_are_not_obstacles_by_default(self):
        values = [item(1, 'Static', 'SM_PlasticTable'),
                  item(2, 'Other', 'SM_Wall_TrafficCone_Advert'),
                  item(3, 'Other', 'my_constructioncone'),
                  item(4, 'Other', 'static.prop.trafficcone01')]
        objects, audit = self.catalogue(values)
        self.assertEqual([entry['id'] for entry in objects], ['environment_4'])
        self.assertEqual(objects[0]['kind'], 'static_obstacle')
        self.assertEqual(sum(audit['excluded_label_counts'].values()), 3)

    def test_native_table_samples_do_not_enter_object_table(self):
        objects, audit = self.catalogue([
            item(i, 'Static', 'SM_PlasticTable_SM_' + str(i)) for i in range(115)])
        self.assertEqual(objects, [])
        self.assertEqual(audit['excluded_label_counts']['Static'], 115)
        self.assertEqual(len(audit['category_decisions']), 115)

    def test_parked_vehicle_bicycle_pedestrian_keep_categories(self):
        objects, audit = self.catalogue([item(1, 'Car'), item(2, 'Bicycle'), item(3, 'Pedestrians')])
        self.assertEqual([entry['kind'] for entry in objects], ['vehicle', 'bicycle', 'pedestrian'])
        session = CarlaSession()
        session.carla, session.map = CARLA, NS(get_waypoint=lambda *a, **kw: None)
        for entry in objects:
            state = session._environment_actor(entry)
            self.assertEqual(state['kind'], entry['kind'])
            self.assertEqual(state['classification']['version'], OBJECT_CATEGORY_VERSION)
            self.assertEqual(state['velocity'], [0., 0.])
            if state['kind'] == 'pedestrian':
                self.assertGreater(state['radius'], 0.)
        self.assertEqual(audit['category_counts']['static_obstacle'], 0)

    def test_all_vehicle_semantics_are_vehicles_not_generic_obstacles(self):
        for label in ('Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Vehicles'):
            self.assertEqual(environment_category(label, 'parked_mesh')[0], 'vehicle')

    def test_mirrored_retained_geometry_is_normalized_not_discarded(self):
        box = Box(vector(10., 0., 1.), vector(-2., 1., 1.))
        objects, audit = self.catalogue([item(1, 'Car', box=box)])
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['box'].extent.x, 2.)
        self.assertEqual(box.extent.x, -2.)
        self.assertEqual(audit['signed_extent_normalized_count'], 1)

    def test_invalid_retained_geometry_is_still_an_error(self):
        for extent in (vector(0., 1., 1.), vector(math.nan, 1., 1.)):
            with self.assertRaises(ValueError):
                self.catalogue([item(1, 'Car', box=Box(extent=extent))])

    def test_prop_allowlist_is_exact_not_any_static_prefix_or_role(self):
        for type_id in ('static.prop.trafficcone01', 'static.prop.constructioncone',
                        'static.prop.streetbarrier', 'static.prop.ironplank',
                        'static.prop.brokentile01', 'static.prop.dirtdebris01'):
            self.assertEqual(actor_category(NS(type_id=type_id))[0], 'static_obstacle')
        for type_id in ('static.prop.mesh', 'static.prop.chainbarrier', 'static.prop.bench01',
                        'static.prop.trafficwarning', 'static.trigger.friction',
                        'static.prop.trafficcone01_extra'):
            self.assertIsNone(actor_category(NS(type_id=type_id, attributes={'role_name': 'obstacle'}))[0])

    def test_runtime_actor_table_filters_experiment_furniture(self):
        session = CarlaSession()
        session.frame, session.cfg, session.carla = 5, {}, CARLA
        session.environment_audit = {}
        session.ego = NS(id=1)
        cone = NS(id=2, type_id='static.prop.trafficcone01', bounding_box=Box(),
                  get_transform=lambda: NS(transform=lambda point: point))
        # No geometry: exclusion must happen before geometry/state acquisition.
        bench = NS(id=3, type_id='static.prop.bench01')
        session.experiment = NS(actors=[cone, bench])
        session.world = NS(get_actors=lambda: [cone, bench])
        session._actor = lambda actor, is_ego=False: {'id': str(actor.id), 'position': [0., 0.],
                                                     'z': 1., 'height_m': 2.}
        session._range_environment = lambda ego, radius: []
        _, actors, _ = session._entities()
        self.assertEqual([actor['id'] for actor in actors], ['2'])
        decisions = session.environment_audit['runtime_actor_category_decisions']
        self.assertEqual([row['included'] for row in decisions], [True, False])


if __name__ == '__main__':
    unittest.main()
