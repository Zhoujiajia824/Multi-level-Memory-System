"""Frozen current public CARLA state, without world or actor lifecycle ownership.

Only standard-library imports are allowed here: this runs in the official SDK
environment. The numerical worker constructs NumPy/Shapely contracts. This is a
privileged current-state adapter, not an image-only perception implementation.
"""
import math

from .route import RouteAdapter, lane_key, point_in_polygon, polygons_intersect, project, xy, wrap
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


def _actor_origin_pose(transform, frame, timestamp):
    """Archive the frozen CARLA Actor origin, independently of the box centre."""
    matrix=[[float(v) for v in row] for row in transform.get_matrix()]
    position=[float(getattr(transform.location,key)) for key in ('x','y','z')]
    if (len(matrix)!=4 or any(len(row)!=4 for row in matrix)
            or not all(math.isfinite(v) for row in matrix for v in row)
            or not all(math.isfinite(v) for v in position)
            or any(abs(matrix[i][3]-position[i])>1e-5 for i in range(3))):
        raise ValueError('Invalid frozen CARLA Actor origin transform')
    forward=[matrix[i][0] for i in range(3)]
    if abs(sum(v*v for v in forward)-1.)>1e-5:
        raise ValueError('Invalid frozen CARLA Actor forward vector')
    orientation={key:(float(getattr(transform.rotation,key)) if hasattr(transform.rotation,key) else None)
                 for key in ('yaw','pitch','roll')}
    if any(v is not None and not math.isfinite(v) for v in orientation.values()):
        raise ValueError('Invalid frozen CARLA Actor orientation')
    return dict(valid=True,source='frozen_ego_actor_snapshot_transform',reference_point='carla_actor_origin',
        position_xyz_carla=position,forward_xyz_carla=forward,orientation_deg=orientation,
        matrix_xyz_carla=matrix,frame=int(frame),timestamp=float(timestamp))


def _scope_lane_projection(position, center):
    """Reject ambiguous lane stations rather than projecting to another loop."""
    rows=[];arc=0.
    for a,b in zip(center,center[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        if length<=1e-8:continue
        fraction=max(0.,min(1.,((position[0]-a[0])*dx+(position[1]-a[1])*dy)/(length*length)))
        foot=[a[0]+fraction*dx,a[1]+fraction*dy]
        rows.append(dict(s_m=arc+fraction*length,position=foot,heading_rad=math.atan2(dy,dx),
                         distance_m=math.dist(position,foot)))
        arc+=length
    if not rows:raise ValueError('lane_has_no_projectable_segment')
    best=min(rows,key=lambda row:row['distance_m'])
    close=[row for row in rows if row['distance_m']<=best['distance_m']+.05]
    best['near_station_span_m']=max(row['s_m'] for row in close)-min(row['s_m'] for row in close)
    if best['distance_m']>.1:raise ValueError('stop_or_finish_not_on_lane_center')
    if best['near_station_span_m']>1.:raise ValueError('ambiguous_lane_station')
    return best


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

    def _continuation_stop_association(self, position, lane_id, elevation_m, stop_heading_rad, ego):
        """Map a real stop to the same public lane beyond the finite task route.

        This only supplies a traffic-rule coordinate. The paper planner decides
        the trajectory; no red light is waived and no braking control is chosen.
        """
        evidence=dict(scope='unresolved',resolved=False,reason='public_stop_coordinate_unavailable')
        try:
            if (not self.route.lane_ids or lane_id!=ego['lane_id']
                    or lane_id!=self.route.lane_ids[-1]):
                raise ValueError('stop_not_on_current_and_terminal_lane')
            end=list(self.route.points[-1]);end_z=float(self.route.elevations[-1])
            if (not all(math.isfinite(v) for v in [*position,*end,end_z,elevation_m,stop_heading_rad])):
                raise ValueError('nonfinite_stop_or_route_geometry')
            lane=next(row for row in self.route.lanes if row['id']==lane_id)
            stop_projection=_scope_lane_projection(position,lane['center'])
            finish_projection=_scope_lane_projection(end,lane['center'])
            gap=stop_projection['s_m']-finish_projection['s_m']
            evidence.update(lane_id=lane_id,stop_lane_s_m=stop_projection['s_m'],
                finish_lane_s_m=finish_projection['s_m'],gap_after_finish_m=gap,
                stop_center_residual_m=stop_projection['distance_m'],
                finish_center_residual_m=finish_projection['distance_m'])
            if gap<=0.:raise ValueError('stop_not_after_official_finish')
            height_limit=float(self.config.get('same_level_tolerance_m',1.5))
            stop_height=self.route.lane_height_at(lane_id,position,elevation_m)
            finish_height=self.route.lane_height_at(lane_id,end,end_z)
            if not all(math.isfinite(v) for v in (height_limit,stop_height,finish_height)):
                raise ValueError('lane_elevation_unavailable')
            evidence.update(stop_elevation_error_m=abs(stop_height-elevation_m),
                            finish_elevation_error_m=abs(finish_height-end_z))
            if (height_limit<=0. or abs(stop_height-elevation_m)>height_limit
                    or abs(finish_height-end_z)>height_limit):
                raise ValueError('stop_or_finish_elevation_ambiguous')
            before=self.route.points[-2]
            heading=math.atan2(end[1]-before[1],end[0]-before[0])
            if (abs(wrap(finish_projection['heading_rad']-heading))>.35
                    or abs(wrap(stop_projection['heading_rad']-stop_heading_rad))>.1):
                raise ValueError('stop_or_finish_direction_not_confirmed')
            station=float(self.route.length)+gap
            if not math.isfinite(station):raise ValueError('nonfinite_continuation_stop_station')
            evidence.update(scope='public_lane_continuation',resolved=True,stop_s_m=station,
                reason='same_lane_public_stop_coordinate_after_task_end',
                traffic_constraint_retained=True,trajectory_decided_by='paper_planner')
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError,StopIteration) as exc:
            evidence['reason']=str(exc) if isinstance(exc,ValueError) else 'scope_geometry_unavailable:'+type(exc).__name__
        return evidence

    def _lights(self, actors, ego, frame):
        records, stop_s, unmapped = [],None,False
        self._last_light_association_audit=[]
        current_light = self.ego.get_traffic_light() if self.ego.is_at_traffic_light() else None
        for actor in actors:
            if not actor.type_id.startswith('traffic.traffic_light'):
                continue
            if actor.id not in self._light_stops:
                try:
                    self._light_stops[actor.id]=[
                        dict(lane_id=lane_key(wp),position=xy(wp.transform.location),
                             elevation_m=float(wp.transform.location.z),
                             heading_rad=-math.radians(wp.transform.rotation.yaw))
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
            light_unmapped=False
            current_reason=None
            for stop in stops:
                identity = stop['lane_id']
                if identity not in self.route.lane_ids and identity != ego['lane_id']:
                    continue
                position = stop['position']
                detail=dict(lane_id=identity,position=position,elevation_m=stop['elevation_m'],
                            position_xyz_carla=[position[0],-position[1],stop['elevation_m']],
                            heading_rad=stop.get('heading_rad'))
                considered.append(detail)
                s = self.route.stop_projection(position,identity,ego['lane_id'],stop['elevation_m'])
                detail['task_stop_s']=s
                if s is not None:
                    detail['association_reason']='upcoming_stop_on_official_route'
                    relevant.append(s)
                elif identity == ego['lane_id'] and state in ('red','yellow'):
                    association=self._continuation_stop_association(
                        position,identity,stop['elevation_m'],stop.get('heading_rad'),ego)
                    detail['continuation_association']=association
                    if association['resolved']:
                        s=association['stop_s_m']
                        detail.update(task_stop_s=s,association_reason='public_lane_continuation_stop')
                        relevant.append(s)
                    lane = next((l for l in self.route.lanes if l['id'] == identity),None)
                    if lane:
                        along = project(position,lane['center'])['s_m']-project(ego['position'],lane['center'])['s_m']
                        detail['ahead_on_current_lane_m']=along
                        detail['unmapped_affected']=bool(0.<=along<=40. and not association['resolved'])
                        light_unmapped |= detail['unmapped_affected']
            if current_light is not None and current_light.id == actor.id and state in ('red','yellow') and not relevant:
                current_stops=[row for row in considered if row['lane_id']==ego['lane_id']]
                # Unknown rule coordinates are an input error, not permission
                # to replace the paper planner with an external brake command.
                if not current_stops or not all(row.get('continuation_association',{}).get('resolved',False) for row in current_stops):
                    light_unmapped=True
                    current_reason=('current_affected_light_without_stop_waypoints' if not stops else
                                    'current_affected_light_without_proven_stop_scope')
            unmapped |= light_unmapped
            current=bool(current_light is not None and current_light.id==actor.id)
            if (considered and any('continuation_association' in row for row in considered)) or current or light_unmapped:
                self._last_light_association_audit.append(dict(id=actor.id,state=state,source_frame=frame,
                    current_affected_light=current,unmapped_affected=light_unmapped,stops=considered,
                    reason=current_reason or ('public_stop_association_unresolved' if light_unmapped else 'public_stop_scope_classified'),
                    geometry_error=self._light_stop_errors.get(actor.id)))
            if relevant or current or light_unmapped:
                selected = min(relevant) if relevant else None
                trigger = getattr(actor,'trigger_volume',None)
                records.append(dict(id=actor.id,state=state,stop_s=selected,stops=considered,
                    association=('public_controlled_stop_waypoints' if relevant else
                                 'current_lane_red_stop_unmapped' if light_unmapped else
                                 'current_light_without_upcoming_task_stop'),
                    unmapped_affected=light_unmapped,
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
        actor_origin_pose=_actor_origin_pose(frozen_ego.get_transform(),frame,observed_time)
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
        speed_limit=float(self.ego.get_speed_limit())/3.6
        if not math.isfinite(speed_limit) or speed_limit <= 0:
            raise ValueError('Current public CARLA ego speed limit is unavailable')
        lights,light_stop_s,unmapped = self._lights(current_actors,ego,frame)
        stop_s=light_stop_s
        public_rules=self.traffic_rules.update(current_actors,snapshot,ego,frozen_ego,observed_time)
        if public_rules['stop_s'] is not None:
            stop_s=public_rules['stop_s'] if stop_s is None else min(stop_s,public_rules['stop_s'])
        desired=min(speed_limit,float(self.config.get('desired_speed_mps',10.)))
        valid=route_projection['valid']
        input_errors=[]
        if not valid:input_errors.append('route_projection_invalid')
        if unmapped:input_errors.append('unresolved_current_red_stop')
        if public_rules['unmapped_current_stops']:input_errors.append('unresolved_current_stop_sign')
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
            navigation_input_valid=not input_errors,navigation_input_errors=input_errors,
            finish_gate=dict(getattr(self.route,'finish_gate',{})),
            actor_origin_pose={key:actor_origin_pose[key] for key in
                               ('valid','source','reference_point','position_xyz_carla','forward_xyz_carla','frame','timestamp')},
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
                public_light_association_audit=list(self._last_light_association_audit),
                frozen_actor_origin_pose=actor_origin_pose,
                stable_navigation_lane_id=ego['lane_id'],lane_map_complete=ego['lane_id'] in ids,
                actor_classification=classifications,excluded_public_entities=excluded,
                visibility_polygon=visibility,visibility_evidence=dict(valid=True,
                    method='range_limited_privileged_current_state_no_occlusion',radius_m=observation_radius,
                    occlusion_tested=False,source_frame=frame),
                observation_policy='privileged_current_physical_state_and_public_map_no_future',
                official_completion_is_external=True,official_world_ownership=True))
