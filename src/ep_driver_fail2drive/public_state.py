"""Frozen current public CARLA state, without world or actor lifecycle ownership.

Only standard-library imports are allowed here: this runs in the official SDK
environment. The numerical worker constructs NumPy/Shapely contracts. This is a
privileged current-state adapter, not an image-only perception implementation.
"""
import math

from .route import RouteAdapter, lane_key, point_in_polygon, polygons_intersect, project, xy
from .traffic_rules import PublicTrafficRules
from .vehicle_geometry import SCHEMA as GEOMETRY_SCHEMA


SCHEMA = 'ep-driver-fail2drive-scene-1.0'
CATEGORY_VERSION = 'public-physical-entity-registry-1.0'
BICYCLES = frozenset(('vehicle.bh.crossbike','vehicle.diamondback.century','vehicle.gazelle.omafiets'))
ROAD_OBSTACLE_TYPES = frozenset((
    'static.prop.streetbarrier', 'static.prop.trafficcone01', 'static.prop.trafficcone02',
    'static.prop.constructioncone', 'static.prop.warningconstruction',
    'static.prop.barrel', 'static.prop.box01', 'static.prop.box02', 'static.prop.box03',
    'static.prop.creasedbox01', 'static.prop.creasedbox02', 'static.prop.creasedbox03',
    'static.prop.plasticbag', 'static.prop.trashbag', 'static.prop.trashcan',
    'static.prop.dirtdebris01', 'static.prop.dirtdebris02', 'static.prop.dirtdebris03',
    'static.prop.garbage01', 'static.prop.garbage02', 'static.prop.garbage03',
    'static.prop.container', 'static.prop.vendingmachine',
))
# Asset *identity* is public metadata, never a scenario name or an event timer.
# Unknown props are not silently admitted/rejected by their broad static prefix:
# their frozen physical body must actually intersect a same-level driving lane.
INFRASTRUCTURE_PREFIXES = ('sensor.', 'traffic.', 'controller.', 'spectator', 'spectator.')


def _point(matrix, point):
    values = [point.x,point.y,point.z,1.]
    return [sum(matrix[i][j]*values[j] for j in range(4)) for i in range(3)]


def _rectangle(center, heading, length, width):
    c,s = math.cos(heading),math.sin(heading)
    return [[center[0]+u*c-v*s,center[1]+u*s+v*c]
            for u,v in ((length/2,width/2),(length/2,-width/2),(-length/2,-width/2),(-length/2,width/2))]


def _location(template, position):
    return type(template)(x=position[0],y=position[1],z=position[2])


def _state_name(state):
    return str(state).split('.')[-1].lower()


class PublicStateExtractor:
    """Read current-frame actors and a frozen official route; never tick/spawn."""
    def __init__(self, world, ego, global_plan_world_coord, config=None, route_id='route', ego_geometry=None):
        self.world,self.ego,self.config = world,ego,dict(config or {})
        self.route_id = str(route_id)
        self.map = world.get_map()
        self.route = RouteAdapter(self.map,global_plan_world_coord,self.config)
        self.traffic_rules=PublicTrafficRules(self.map,self.route)
        self.angular_history,self.history,self.kinematic_history = {},{},{}
        self.next_history = None
        self.last_frame = None
        self.ego_geometry = ego_geometry
        self._lane_bounds=[(lane,min(p[0] for p in lane['polygon']),max(p[0] for p in lane['polygon']),
                           min(p[1] for p in lane['polygon']),max(p[1] for p in lane['polygon']))
                          for lane in self.route.lanes]
        self._light_stops,self._light_stop_errors={},{}
        self.crosswalks = self._crosswalks()

    def _crosswalks(self):
        result,part = [],[]
        for location in self.map.get_crosswalks():
            point = xy(location)
            if abs(location.z-self.route.height_at(point,location.z)) > float(self.config.get('same_level_tolerance_m',1.5)):
                part=[]
                continue
            if part and math.dist(point,part[0]) < .05:
                if len(part) >= 3:
                    result.append(part)
                part=[]
            else:
                part.append(point)
        return result

    def _actor(self, actor, frozen, timestamp, is_ego=False):
        transform = frozen.get_transform()
        box = actor.bounding_box
        center = _point(transform.get_matrix(),box.location)
        heading = -math.radians(transform.rotation.yaw+box.rotation.yaw)
        vertices = box.get_world_vertices(transform)
        flat = [[v.x,-v.y] for v in vertices]
        c,s = math.cos(heading),math.sin(heading)
        length = 2.*max(abs((p[0]-center[0])*c+(p[1]+center[1])*s) for p in flat)
        width = 2.*max(abs(-(p[0]-center[0])*s+(p[1]+center[1])*c) for p in flat)
        if not all(math.isfinite(v) and v > .001 for v in (length,width)):
            return None,None
        velocity,acceleration,omega = frozen.get_velocity(),frozen.get_acceleration(),frozen.get_angular_velocity()
        angular = math.radians(omega.z)
        previous = self.angular_history.get(actor.id)
        alpha = (angular-previous[1])/(timestamp-previous[0]) if previous and timestamp > previous[0]+1e-8 else 0.
        self.angular_history[actor.id] = (timestamp,angular)
        ox,oy = center[0]-transform.location.x,center[1]-transform.location.y
        position = [center[0],-center[1]]
        motion = [velocity.x-angular*oy,-(velocity.y+angular*ox)]
        accel = [acceleration.x-alpha*oy-angular*angular*ox,
                 -(acceleration.y+alpha*ox-angular*angular*oy)]
        location = _location(transform.location,center)
        wp = self.map.get_waypoint(location,project_to_road=False)
        lane_id = lane_key(wp) if wp is not None and str(wp.lane_type).split('.')[-1].lower() == 'driving' else None
        height = max(v.z for v in vertices)-min(v.z for v in vertices)
        row = dict(id='ego' if is_ego else 'carla_{}'.format(actor.id),kind='vehicle',position=position,
            velocity=motion,acceleration=accel,heading=heading,length=length,width=width,lane_id=lane_id,
            visible=True,history=[],radius=None,wheelbase_m=None,rear_axle_to_center_m=None,motion_model=None)
        evidence = dict(type_id=actor.type_id,version=CATEGORY_VERSION,height_m=height,z_m=center[2],
            z_min_m=min(v.z for v in vertices),z_max_m=max(v.z for v in vertices),
            position_source='snapshot_transform_times_actor_local_bbox_center',
            velocity_source='actor_snapshot_velocity_plus_angular_center_offset',
            acceleration_source='actor_snapshot_acceleration_plus_causal_angular_offset',
            angular_acceleration_source='past_difference' if previous else 'initial_zero_assumption')
        old_motion=self.kinematic_history.get(actor.id)
        observed_rate=(math.dist(position,old_motion['position'])/(timestamp-old_motion['time'])
                       if old_motion and timestamp>old_motion['time']+1e-8 else
                       old_motion['observed_rate'] if old_motion else 0.)
        self.kinematic_history[actor.id]=dict(time=timestamp,position=list(position),observed_rate=observed_rate)
        evidence['observed_center_motion_mps']=observed_rate
        attributes=getattr(actor,'attributes',{})
        if attributes.get('mesh_path'):
            evidence['mesh_path']=attributes['mesh_path']
        if is_ego:
            geometry = self.ego_geometry
            if not geometry or geometry.get('schema') != GEOMETRY_SCHEMA or geometry.get('status') != 'valid':
                raise ValueError('Verified same-frame wheel geometry must be supplied before ego extraction')
            row['wheelbase_m'] = geometry['wheelbase_m']
            row['rear_axle_to_center_m'] = geometry['rear_axle_to_center_m']
            evidence['geometry'] = dict(schema=geometry['schema'], measured_frame=geometry['frame'],
                source=geometry['source'], wheelbase_m=row['wheelbase_m'],
                rear_axle_to_center_m=row['rear_axle_to_center_m'])
        return row,evidence

    def _category(self, actor, row, evidence):
        identity = actor.type_id
        if identity == 'static.prop.mesh' and evidence.get('mesh_path','').startswith('/Game/Carla/Static/Car/'):
            return 'vehicle','public_parked_vehicle_mesh_asset'
        if identity in BICYCLES:
            return 'bicycle','public_bicycle_blueprint'
        if identity.startswith('vehicle.'):
            return 'vehicle','public_vehicle_blueprint'
        if identity.startswith('walker.pedestrian.'):
            return 'pedestrian','public_pedestrian_blueprint'
        if identity.startswith(('walker.animal.','animal.')):
            return 'static_obstacle','public_animal_entity'
        if identity in ROAD_OBSTACLE_TYPES:
            return 'static_obstacle','exact_road_obstacle_blueprint'
        if not identity.startswith(('static.prop.','walker.')):
            return None,'unregistered_nonphysical_or_background_type'
        # Unknown meshes (including abnormal assets) need physical evidence on
        # the road; ordinary roadside furniture is not a global obstacle list.
        body = _rectangle(row['position'],row['heading'],row['length'],row['width'])
        if evidence['height_m'] < .03:
            return None,'no_obstacle_height_evidence'
        if (math.hypot(*row['velocity']) > .05 or math.hypot(*row['acceleration']) > .1
                or evidence['observed_center_motion_mps']>.05):
            return 'static_obstacle','unregistered_moving_physical_prop'
        xmin,xmax=min(p[0] for p in body),max(p[0] for p in body)
        ymin,ymax=min(p[1] for p in body),max(p[1] for p in body)
        intersects = any(polygons_intersect(body,lane['polygon'])
            for lane,lx0,lx1,ly0,ly1 in self._lane_bounds
            if xmax >= lx0 and xmin <= lx1 and ymax >= ly0 and ymin <= ly1)
        if not intersects:
            return None,'unregistered_prop_outside_driving_surface'
        return 'static_obstacle','unregistered_physical_body_intersects_driving_surface'

    def _lights(self, actors, ego, frame):
        records, stop_s, unmapped = [],None,False
        current_light = self.ego.get_traffic_light() if self.ego.is_at_traffic_light() else None
        for actor in actors:
            if not actor.type_id.startswith('traffic.traffic_light'):
                continue
            if actor.id not in self._light_stops:
                try:
                    self._light_stops[actor.id]=[
                        dict(lane_id=lane_key(wp),position=xy(wp.transform.location),
                             elevation_m=float(wp.transform.location.z))
                        for wp in actor.get_stop_waypoints()]
                except (AttributeError,RuntimeError) as exc:
                    self._light_stops[actor.id]=[]
                    self._light_stop_errors[actor.id]=dict(id=actor.id,
                        reason='public_controlled_stop_waypoints_unavailable',error_type=type(exc).__name__)
            stops=self._light_stops[actor.id]
            # Geometry is a static public map property; light state is always
            # read anew for this frozen frame, never replayed from the cache.
            if not any(stop['lane_id'] in self.route.lane_ids or stop['lane_id']==ego['lane_id'] for stop in stops) and not (current_light is not None and current_light.id==actor.id):
                continue
            state = _state_name(actor.get_state())
            relevant,considered = [],[]
            for stop in stops:
                identity = stop['lane_id']
                if identity not in self.route.lane_ids and identity != ego['lane_id']:
                    continue
                position = stop['position']
                considered.append(dict(lane_id=identity,position=position,elevation_m=stop['elevation_m']))
                s = self.route.stop_projection(position,identity,ego['lane_id'],stop['elevation_m'])
                if s is not None:
                    relevant.append(s)
                elif identity == ego['lane_id'] and state in ('red','yellow'):
                    lane = next((l for l in self.route.lanes if l['id'] == identity),None)
                    if lane:
                        along = project(position,lane['center'])['s_m']-project(ego['position'],lane['center'])['s_m']
                        unmapped |= 0. <= along <= 40.
            if current_light is not None and current_light.id == actor.id and state in ('red','yellow') and not relevant:
                unmapped = True
            if relevant or (current_light is not None and current_light.id == actor.id):
                selected = min(relevant) if relevant else None
                trigger = getattr(actor,'trigger_volume',None)
                records.append(dict(id=actor.id,state=state,stop_s=selected,stops=considered,
                    association='public_controlled_stop_waypoints' if relevant else 'current_lane_red_stop_unmapped',
                    source_frame=frame,trigger_volume_local=({
                        'center':[trigger.location.x,trigger.location.y,trigger.location.z],
                        'extent':[trigger.extent.x,trigger.extent.y,trigger.extent.z]} if trigger is not None else None)))
                if selected is not None and state in ('red','yellow'):
                    stop_s = selected if stop_s is None else min(stop_s,selected)
        return records,stop_s,unmapped

    def capture(self, frame, timestamp=None, camera_images=None, world_snapshot=None):
        snapshot = world_snapshot if world_snapshot is not None else self.world.get_snapshot()
        if snapshot.frame != frame:
            raise ValueError('Frozen public state does not match the official input frame')
        observed_time = float(snapshot.timestamp.elapsed_seconds)
        if timestamp is not None and abs(float(timestamp)-observed_time) > 1e-5:
            raise ValueError('Official image/state timestamp mismatch')
        if self.last_frame is not None and frame < self.last_frame:
            raise ValueError('Public-state frame moved backwards')
        frozen_ego = snapshot.find(self.ego.id)
        if frozen_ego is None:
            raise ValueError('Official ego is absent from the frozen actor snapshot')
        ego,ego_evidence = self._actor(self.ego,frozen_ego,observed_time,True)
        if ego is None:
            raise ValueError('Official ego has no finite physical body')
        route_projection = self.route.update(ego['position'],ego['heading'],frame,observed_time,
                                             math.hypot(*ego['velocity']),ego['lane_id'])
        current_actors = list(self.world.get_actors())
        actors,classifications,excluded = [],{'ego':ego_evidence},[]
        radius = float(self.config.get('public_capture_radius_m',120.))
        for actor in current_actors:
            if actor.id == self.ego.id or actor.type_id.startswith(INFRASTRUCTURE_PREFIXES):
                continue
            frozen = snapshot.find(actor.id)
            if frozen is None:
                continue
            location=frozen.get_transform().location
            if math.dist(xy(location),ego['position']) > radius+max(float(actor.bounding_box.extent.x),float(actor.bounding_box.extent.y)):
                continue
            row,evidence = self._actor(actor,frozen,observed_time)
            if row is None:
                excluded.append(dict(id=actor.id,type_id=actor.type_id,reason='invalid_physical_body'))
                continue
            if evidence['z_max_m'] < ego_evidence['z_min_m']-.2 or evidence['z_min_m'] > ego_evidence['z_max_m']+.2:
                excluded.append(dict(id=actor.id,type_id=actor.type_id,reason='different_vertical_occupancy'))
                continue
            kind,reason = self._category(actor,row,evidence)
            evidence.update(kind=kind,category_evidence=reason)
            if kind is None:
                excluded.append(dict(id=actor.id,type_id=actor.type_id,reason=reason))
                continue
            row['kind']=kind
            if kind == 'pedestrian':
                row['radius']=max(row['length'],row['width'])/2.
            if kind == 'static_obstacle':
                speed=math.hypot(*row['velocity'])
                acceleration=math.hypot(*row['acceleration'])
                moving = speed > .05 or acceleration > .1 or evidence['observed_center_motion_mps']>.05
                # Animals remain free-motion entities even at a momentary stop.
                row['motion_model']='free' if moving or reason == 'public_animal_entity' else 'stationary'
                evidence['motion_evidence']=dict(speed_mps=speed,acceleration_mps2=acceleration,
                    model=row['motion_model'],source='frozen_current_physical_kinematics',
                    observed_center_motion_mps=evidence['observed_center_motion_mps'],
                    future_intent_used=False)
            actors.append(row)
            classifications[row['id']]=evidence
        history_due = self.next_history is None or observed_time+1e-8 >= self.next_history
        if history_due:
            for row in [ego]+actors:
                self.history.setdefault(row['id'],[]).append(dict(time=observed_time,
                    position=list(row['position']),velocity=list(row['velocity']),
                    acceleration=list(row['acceleration']),heading=row['heading'],observed=True))
            self.next_history=observed_time+.5
        for identity in list(self.history):
            self.history[identity]=[value for value in self.history[identity] if observed_time-value['time'] <= 3.+1e-8]
            if not self.history[identity]:
                del self.history[identity]
        for row in [ego]+actors:
            row['history']=[dict(value) for value in self.history.get(row['id'],[])]
        lights,light_stop_s,unmapped = self._lights(current_actors,ego,frame)
        stop_s=light_stop_s
        public_rules=self.traffic_rules.update(current_actors,snapshot,ego,frozen_ego,observed_time)
        if public_rules['stop_s'] is not None:
            stop_s=public_rules['stop_s'] if stop_s is None else min(stop_s,public_rules['stop_s'])
        speed_limit=float(self.ego.get_speed_limit())/3.6
        if not math.isfinite(speed_limit) or speed_limit <= 0:
            raise ValueError('Current public CARLA ego speed limit is unavailable')
        desired=min(speed_limit,float(self.config.get('desired_speed_mps',10.)))
        valid=route_projection['valid']
        stop_sign_active=(public_rules['stop_s'] is not None and
                          (light_stop_s is None or public_rules['stop_s']<=light_stop_s))
        rules=dict(desired_speed=desired,speed_limit=speed_limit,stop_s=stop_s,must_wait=stop_s is not None,
            wait_reason=('当前公共停牌要求实测停车' if stop_sign_active
                         else '当前受控车道红灯或黄灯要求停车') if stop_s is not None else None,
            traffic_lights=lights,crosswalks=self.crosswalks,
            map_available=dict(crosswalk=True,traffic_light=True,
                stop_line=not self._light_stop_errors and not unmapped and public_rules['stop_sign_geometry_available']),
            road_priority_available=False,public_road_rules=dict(public_rules,rules=[]),
            navigation_status='following' if valid else 'route_projection_invalid',
            navigation_brake_required=not valid or unmapped or bool(public_rules['unmapped_current_stops']),
            unmapped_current_red=unmapped,stop_sign_obligations=public_rules['stop_obligations'])
        # An exact-input frame must remain frozen throughout all public reads.
        if self.world.get_snapshot().frame != frame:
            raise ValueError('World advanced while converting the official frozen input')
        frame_id='{}:{}'.format(self.route_id,frame)
        ids={lane['id'] for lane in self.route.lanes}
        observation_radius = float(self.config.get('public_capture_radius_m',120.))
        visibility = [[ego['position'][0]+observation_radius*math.cos(i*2*math.pi/128),
                       ego['position'][1]+observation_radius*math.sin(i*2*math.pi/128)] for i in range(128)]
        self.last_frame=frame
        return dict(schema=SCHEMA,frame_id=frame_id,time=observed_time,ego=ego,actors=sorted(actors,key=lambda a:a['id']),
            lanes=self.route.lanes,route=self.route.points,route_lane_ids=self.route.lane_ids,
            drivable_polygons=self.route.drivable_polygons,scenario='fail2drive',rules=rules,
            assets=dict(camera_images=list(camera_images or []),depth_images=[]),
            diagnostics=dict(platform='fail2drive',carla_frame=frame,carla_timestamp=observed_time,
                route_s=route_projection['trusted_progress_m'] or 0.,reference_anchor_s=route_projection['trusted_progress_m'] or 0.,
                route_projection=route_projection,navigation_status=rules['navigation_status'],
                public_map_sample_audit=self.route.map_sample_audit,
                public_light_geometry_unavailable=list(self._light_stop_errors.values()),
                stable_navigation_lane_id=ego['lane_id'],lane_map_complete=ego['lane_id'] in ids,
                actor_classification=classifications,excluded_public_entities=excluded,
                visibility_polygon=visibility,visibility_evidence=dict(valid=True,
                    method='range_limited_privileged_current_state_no_occlusion',radius_m=observation_radius,
                    occlusion_tested=False,source_frame=frame),
                observation_policy='privileged_current_physical_state_and_public_map_no_future',
                official_completion_is_external=True,official_world_ownership=True))
