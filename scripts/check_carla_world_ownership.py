"""CARLA 0.9.15 admission/cleanup regressions with SDK-shaped fakes; no server.

Attribute fixtures reflect the tagged server registration and client blueprint
serialization code. They do not prove the attributes of any live Town01 actor.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nine_grid.carla_bridge import CarlaSession
from nine_grid.carla_world_ownership import inspect_world_actors
from check_carla_camera_clock import blockers, run_probe


SCRATCH = Path(__file__).resolve().parents[1] / 'outputs' / 'ownership_contract_checks'
NATIVE_ATTRIBUTES = {'mesh_path': '/Game/Carla/Static/Test/TestMesh.TestMesh', 'mass': '12.5'}


class Actor:
    def __init__(self, identity, type_id='static.prop.mesh', attributes=None, parent=None):
        self.id, self.type_id, self.parent = identity, type_id, parent
        self.attributes = dict(NATIVE_ATTRIBUTES if attributes is None else attributes)
        self.world = None

    def destroy(self):
        self.world.mutations.append(('destroy', self.id))
        self.world.actors.remove(self)
        return True

    def stop(self):
        self.world.mutations.append(('stop', self.id))


class Snapshot(list):
    def __init__(self, actors, frame):
        super().__init__(actors)
        self.frame = frame


class World:
    def __init__(self, actors=(), synchronous=False):
        self.actors, self.mutations, self.frame = list(actors), [], 50
        self.settings = SimpleNamespace(synchronous_mode=synchronous,
            fixed_delta_seconds=.02 if synchronous else None,
            no_rendering_mode=False, substepping=True,
            max_substep_delta_time=.01, max_substeps=10)
        for actor in self.actors:
            actor.world = self

    def get_actors(self):
        return list(self.actors)

    def get_map(self):
        return SimpleNamespace(name='Carla/Maps/Town01')

    def get_snapshot(self):
        return Snapshot(self.actors, self.frame)

    def get_settings(self):
        return copy.copy(self.settings)

    def apply_settings(self, settings):
        self.mutations.append(('apply_settings', settings.synchronous_mode))
        self.settings = copy.copy(settings)

    def wait_for_tick(self, timeout):
        if self.settings.synchronous_mode:
            raise AssertionError('Must never wait on a synchronous world during cleanup')
        self.frame += 1
        return self.get_snapshot()

    def tick(self, *args):
        raise AssertionError('Admission and cleanup must not tick the world')


class Director:
    def __init__(self, actors):
        self.actors, self.events = list(actors), []

    def close(self):
        for actor in self.actors:
            actor.destroy()


class WorldOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)

    def session(self, world, directory):
        session = CarlaSession()
        session.world, session.root = world, Path(directory)
        return session

    def test_native_115_survive_owned_cleanup_and_next_episode_admission(self):
        native = [Actor(identity) for identity in range(82, 197)]
        world = World(native)
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            first = self.session(world, directory)
            first._check_world_ownership(world.get_settings())
            self.assertEqual(world.mutations, [])
            self.assertTrue(first.world_ownership['accepted'])
            self.assertEqual(first.world_ownership['map_native_mesh_count'], 115)
            first.saved_settings = world.get_settings()
            first._settings_change_attempted = True
            world.settings.synchronous_mode = True
            world.settings.fixed_delta_seconds = .02
            vehicles = [Actor(identity, 'vehicle.test', {'role_name': 'experiment'})
                        for identity in (197, 198)]
            sensors = [Actor(identity, 'sensor.camera.rgb', {'role_name': 'front'})
                       for identity in range(199, 207)]
            for actor in vehicles + sensors:
                actor.world = world
            world.actors.extend(vehicles + sensors)
            first.experiment, first.sensors = Director(vehicles), sensors
            first.close()
            self.assertTrue(first.cleanup_report['verified'])
            self.assertEqual(first.cleanup_report['remaining_owned_ids'], [])
            self.assertFalse(world.settings.synchronous_mode)
            self.assertEqual({identity for operation, identity in world.mutations
                              if operation == 'destroy'}, set(range(197, 207)))
            self.assertEqual(world.actors, native)
            before = list(world.mutations)
            second = self.session(world, directory)
            second._check_world_ownership(world.get_settings())
            self.assertEqual(world.mutations, before)
            report = json.loads((Path(directory) / 'carla_world_ownership.json').read_text(encoding='utf-8'))
            self.assertTrue(report['accepted'])
            self.assertEqual(report['map_native_mesh_count'], 115)
            self.assertEqual(report['blocking_actors'], [])
            self.assertEqual(report['map_native_mesh_actors'][0]['attribute_names'], ['mass', 'mesh_path'])

    def test_sdk_spawned_mesh_with_blank_role_still_blocks_without_mutation(self):
        attributes = dict(NATIVE_ATTRIBUTES, role_name='', ros_name='static.prop.mesh', scale='1.0')
        world = World([Actor(82), Actor(197, attributes=attributes)])
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = self.session(world, directory)
            with self.assertRaisesRegex(RuntimeError, 'existing_experiment_actor_types'):
                session._check_world_ownership(world.get_settings())
            session.close()
            self.assertEqual(world.mutations, [])
            self.assertEqual(session.cleanup_report['status'], 'not_acquired')
            self.assertEqual(session.world_ownership['map_native_mesh_count'], 1)
            self.assertEqual([row['id'] for row in session.world_ownership['blocking_actors']], [197])

    def test_empty_role_or_missing_or_additional_attributes_never_prove_native(self):
        variants = ({}, {'role_name': ''}, {'mesh_path': NATIVE_ATTRIBUTES['mesh_path']},
                    {'mass': '12.5'}, dict(NATIVE_ATTRIBUTES, role_name=''),
                    dict(NATIVE_ATTRIBUTES, scale='1.0'),
                    dict(NATIVE_ATTRIBUTES, unknown='anything'))
        for attributes in variants:
            with self.subTest(attributes=attributes):
                report = inspect_world_actors([Actor(1, attributes=attributes)])
                self.assertEqual(report['map_native_mesh_count'], 0)
                self.assertEqual(len(report['blocking_actors']), 1)

    def test_invalid_mass_or_path_is_blocked(self):
        for key, value in [('mass', ''), ('mass', 'nan'), ('mass', 'inf'), ('mass', '-1'),
                           ('mass', 'bad'), ('mass', True), ('mesh_path', ''), ('mesh_path', '   ')]:
            with self.subTest(key=key, value=value):
                attributes = dict(NATIVE_ATTRIBUTES)
                attributes[key] = value
                self.assertEqual(inspect_world_actors([Actor(1, attributes=attributes)])['map_native_mesh_count'], 0)
        self.assertEqual(inspect_world_actors([Actor(1, attributes=dict(NATIVE_ATTRIBUTES, mass='0'))])[
            'map_native_mesh_count'], 1)

    def test_attached_or_missing_parent_evidence_is_blocked(self):
        attached = Actor(1, parent=SimpleNamespace(id=5))
        missing = SimpleNamespace(id=2, type_id='static.prop.mesh', attributes=NATIVE_ATTRIBUTES)
        report = inspect_world_actors([attached, missing])
        self.assertEqual(report['map_native_mesh_count'], 0)
        self.assertEqual([row['classification_reason'] for row in report['blocking_actors']],
                         ['native_mesh_has_parent', 'native_mesh_parent_unavailable'])

    def test_vehicles_walkers_and_nonmesh_props_always_block(self):
        actors = [Actor(identity, kind) for identity, kind in enumerate((
            'vehicle.test', 'walker.pedestrian.test', 'static.prop.cone', 'static.prop.mesh_extra'))]
        report = inspect_world_actors(actors)
        self.assertEqual(report['map_native_mesh_count'], 0)
        self.assertEqual(len(report['blocking_actors']), len(actors))

    def test_synchronous_native_only_world_is_still_rejected(self):
        world = World([Actor(1)], synchronous=True)
        with tempfile.TemporaryDirectory(dir=str(SCRATCH)) as directory:
            session = self.session(world, directory)
            with self.assertRaisesRegex(RuntimeError, 'world_already_synchronous'):
                session._check_world_ownership(world.get_settings())
            self.assertEqual(world.mutations, [])
            self.assertEqual(session.world_ownership['rejection_reasons'], ['world_already_synchronous'])

    def test_audit_does_not_expose_arbitrary_attribute_values(self):
        attributes = dict(NATIVE_ATTRIBUTES, private_token='do-not-serialize-this-value')
        report = inspect_world_actors([Actor(1, attributes=attributes)])
        self.assertNotIn(attributes['private_token'], json.dumps(report))
        self.assertEqual(len(report['blocking_actors']), 1)

    def test_camera_probe_uses_same_conservative_admission(self):
        native = Actor(1)
        foreign = Actor(2, attributes=dict(NATIVE_ATTRIBUTES, role_name=''))
        self.assertEqual(blockers(World([native]))['count'], 0)
        report = blockers(World([native, foreign]))
        self.assertEqual(report['count'], 1)
        self.assertEqual(report['map_native_mesh_count'], 1)
        self.assertEqual(report['first_25'][0]['id'], 2)

    def test_camera_probe_requires_supported_version_before_world_access(self):
        class Client:
            def set_timeout(self, value):
                pass

            def get_client_version(self):
                return '0.9.15'

            def get_server_version(self):
                return '0.9.16'

            def get_world(self):
                raise AssertionError('Version check must precede world access')

        report = {}
        args = SimpleNamespace(exclusive_world=True, host='127.0.0.1', port=2000, timeout=1)
        run_probe(args, SimpleNamespace(Client=lambda *args: Client()), report)
        self.assertEqual(report['status'], 'refused')
        self.assertFalse(report['cleanup']['settings_restore_attempted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
