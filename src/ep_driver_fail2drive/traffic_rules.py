"""Causal stop-sign obligations from current public CARLA trigger geometry.

No ScenarioRunner state, event schedule, hidden actor route or official scoring
object is read. The public geometry/threshold convention is checked against
RunningStopTest in the pinned official source; these are planning obligations,
not replacement official scores. Yield priority remains explicitly unavailable.
"""
import math

from .route import lane_key, project, xy, wrap


VERSION='public-stop-sign-observed-hold-1.0'


def _world_point(transform, point):
    matrix=transform.get_matrix()
    p=[point.x,point.y,point.z,1.]
    return [sum(matrix[i][j]*p[j] for j in range(4)) for i in range(3)]


class PublicTrafficRules:
    """Require a genuinely observed stop before releasing each route encounter."""
    # The official criterion uses <0.1m/s, 4m proximity, public waypoints every
    # 0.5m and 1.2*max(0.5,trigger extent) in world XY. Keep those semantics
    # visible; a separate 0.1s observed hold avoids a one-sample touch-and-go.
    def __init__(self, road_map, route, hold_s=.1):
        self.map,self.route=road_map,route
        self.hold_s=float(hold_s)
        if not math.isfinite(self.hold_s) or self.hold_s<.1:
            raise ValueError('Public stop requires at least 0.1s of observed stationary state')
        self.encounters={}
        self.geometry_cache={}
        self.last_frame=None
        self.last_time=None

    def _geometry(self, actor, frozen):
        transform=frozen.get_transform()
        box=actor.trigger_volume
        center=_world_point(transform,box.location)
        key=(actor.id,tuple(center),float(box.extent.x),float(box.extent.y),float(box.extent.z))
        if key in self.geometry_cache:
            return self.geometry_cache[key]
        extent=[max(.5,abs(float(box.extent.x)))*1.2,max(.5,abs(float(box.extent.y)))*1.2]
        position=[center[0],-center[1]]
        controlled=[]
        for lane in self.route.lanes:
            projection=project(position,lane['center'],elevations=self.route.lane_elevations[lane['id']],
                               elevation=center[2])
            p=projection['position']
            if abs(p[0]-position[0])<extent[0] and abs(p[1]-position[1])<extent[1]:
                elevation=self.route.lane_height_at(lane['id'],p,center[2])
                if abs(elevation-center[2])<=abs(float(box.extent.z))+1.5:
                    controlled.append(dict(lane_id=lane['id'],position=list(p),elevation_m=elevation,
                                           heading_rad=projection['heading_rad']))
        value=dict(id=actor.id,type_id=actor.type_id,position=position,z_m=center[2],
            extent_xy_m=extent,controlled=controlled,
            source='public_stop_trigger_volume_intersection_with_public_lane_center',
            geometry_semantics='official_stop_criterion_axis_aligned_xy_1.2_times_extent_min_0.5m')
        self.geometry_cache[key]=value
        return value

    def _check_waypoints(self, ego_snapshot):
        transform=ego_snapshot.get_transform()
        wp=self.map.get_waypoint(transform.location)
        if wp is None:
            return [],transform
        points=[wp]
        for _ in range(8):
            successors=points[-1].next(.5)
            if not successors:
                break
            # This is the official criterion's public-map local lookahead,
            # not an NPC's future route or a scenario's hidden path.
            points.append(successors[0])
        return points,transform

    @staticmethod
    def _affected(geometry, waypoints, ego_transform, ego_snapshot):
        if not waypoints:
            return False
        current=waypoints[0].transform
        location=current.location
        position=geometry['position']
        if math.hypot(location.x-position[0],-location.y-position[1])>4.:
            return False
        actor_heading=-math.radians(ego_transform.rotation.yaw)
        lane_heading=-math.radians(current.rotation.yaw)
        velocity=ego_snapshot.get_velocity()
        forward_speed=velocity.x*math.cos(-actor_heading)+velocity.y*math.sin(-actor_heading)
        if forward_speed<-.17 or math.cos(wrap(actor_heading-lane_heading))<-.17:
            return False
        ex,ey=geometry['extent_xy_m']
        return any(abs(wp.transform.location.x-position[0])<ex
                   and abs(-wp.transform.location.y-position[1])<ey for wp in waypoints)

    def update(self, actors, snapshot, ego, ego_snapshot, timestamp):
        frame=int(snapshot.frame)
        if self.last_frame is not None and (frame<self.last_frame or timestamp<self.last_time):
            raise ValueError('Public stop-sign observation clock moved backwards')
        self.last_frame,self.last_time=frame,float(timestamp)
        waypoints,ego_transform=self._check_waypoints(ego_snapshot)
        velocity=ego_snapshot.get_velocity()
        speed=math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        obligations,unsupported,unmapped,unavailable=[],[],[],[]
        trusted=self.route.previous_s
        for actor in actors:
            if actor.type_id.startswith('traffic.yield'):
                unsupported.append(dict(id=actor.id,type_id=actor.type_id,reason='public_yield_priority_not_implemented'))
                continue
            if not actor.type_id.startswith('traffic.stop'):
                continue
            frozen=snapshot.find(actor.id)
            if frozen is None:
                unavailable.append(dict(id=actor.id,reason='stop_sign_absent_from_frozen_actor_snapshot'))
                continue
            geometry=self._geometry(actor,frozen)
            possible=[]
            for controlled in geometry['controlled']:
                if controlled['lane_id'] not in self.route.lane_ids and controlled['lane_id']!=ego['lane_id']:
                    continue
                stop=self.route.stop_projection(controlled['position'],controlled['lane_id'],ego['lane_id'],
                                                controlled['elevation_m'])
                if stop is not None:
                    possible.append((stop,controlled))
            affected=self._affected(geometry,waypoints,ego_transform,ego_snapshot)
            # A just-passed sign must keep its current encounter until it leaves
            # the criterion's affected area; it is never reset merely because
            # its forward stop projection has become unavailable.
            current=[entry for entry in self.encounters.values() if entry['id']==actor.id
                     and entry['status'] in ('approaching','observing_hold','fulfilled')
                     and not entry.get('cleared')]
            if not possible and not current:
                if affected:
                    unmapped.append(dict(id=actor.id,reason='current_affected_stop_missing_forward_route_projection'))
                continue
            if current:
                encounter=current[-1]
            else:
                stop,controlled=min(possible,key=lambda pair:pair[0])
                if trusted is None or stop-trusted>70.:
                    continue
                key=(actor.id,round(stop,2))
                encounter=self.encounters.setdefault(key,dict(id=actor.id,status='approaching',
                    stop_s=stop-.2,trigger_s=stop,lane_id=controlled['lane_id'],
                    position=list(controlled['position']),first_observed_frame=frame,hold_started_at=None,
                    completed_frame=None,cleared=False,was_affected=False,source=geometry['source']))
            if affected and speed<.1 and encounter['status']!='fulfilled':
                if encounter['hold_started_at'] is None:
                    encounter['hold_started_at']=float(timestamp)
                encounter['status']='observing_hold'
                if timestamp-encounter['hold_started_at']+1e-8>=self.hold_s:
                    encounter.update(status='fulfilled',completed_frame=frame,completed_at=float(timestamp))
            elif encounter['status']!='fulfilled':
                encounter['hold_started_at']=None
                encounter['status']='approaching'
            if not affected and (encounter['was_affected'] or
                                 (trusted is not None and trusted>encounter['trigger_s']+4.)):
                if encounter['status']!='fulfilled':
                    encounter['status']='passed_without_observed_stop'
                encounter['cleared']=True
            encounter['was_affected'] |= affected
            row=dict(encounter,affected_now=affected,observed_speed_mps=speed,
                     must_stop=encounter['status'] in ('approaching','observing_hold'),
                     rule_version=VERSION,source_frame=frame,
                     stop_fulfilment_source='actual_frozen_ego_velocity_below_0.1mps_in_public_trigger_approach',
                     internal_rule_state_is_official_score=False)
            obligations.append(row)
        active=[row['stop_s'] for row in obligations if row['must_stop']]
        return dict(version=VERSION,frame=frame,time=float(timestamp),stop_obligations=obligations,
            stop_s=min(active) if active else None,unmapped_current_stops=unmapped,
            unavailable_stop_signs=unavailable,
            stop_sign_geometry_available=not unmapped,unsupported_signs=unsupported,
            road_priority_available=False,limitation='Only public stop-sign obligations are implemented; yield priority is unavailable')
