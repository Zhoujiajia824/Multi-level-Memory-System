"""Current-frame demonstration quality evidence; never a driving controller.

The thresholds below are explicit demonstration-review standards, not traffic
law. Bounding-box penetration is not a mesh/physics collision measurement, and
20Hz samples are not a continuous-time safety proof. No world or actor writes.
"""
from copy import deepcopy
import math

from .public_state import ROAD_OBSTACLE_TYPES, BICYCLES
from .route import polygons_intersect

SCHEMA = 'official-demonstration-behavior-1.0'
POLICY = dict(standard='demonstration_quality_review_not_traffic_law',
    clearance_radius_m=20., interaction_clearance_m=10.,
    penetration_xy_m=.05, penetration_z_m=.08,
    pedestrian_clearance_m=1., object_clearance_m=.5,
    pedestrian_bypass_clearance_m=3., pedestrian_bypass_speed_mps=3.,
    object_bypass_clearance_m=2., object_bypass_speed_mps=4.,
    lateral_acceleration_review_mps2=2.5, lateral_jerk_review_mps3=5.,
    sdk_vertex_center_tolerance_m=.01)


def _finite(values):
    return all(type(v) in (int,float) and math.isfinite(v) for v in values)


def _vector(value):
    return [float(getattr(value,k)) for k in ('x','y','z')]


def _hull(points):
    points=sorted(set(map(tuple,points)))
    if len(points)<3:raise ValueError('Demonstration body has a degenerate footprint')
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[];upper=[]
    for p in points:
        while len(lower)>1 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper)>1 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    result=[list(p) for p in lower[:-1]+upper[:-1]]
    if len(result)<3:raise ValueError('Demonstration body has a degenerate footprint')
    return result


def _point_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def separation(a,b):
    """Euclidean polygon clearance and sampled SAT penetration, in metres."""
    overlap=math.inf
    for poly in (a,b):
        for p,q in zip(poly,poly[1:]+poly[:1]):
            length=math.dist(p,q)
            if length<=1e-12:continue
            axis=[-(q[1]-p[1])/length,(q[0]-p[0])/length]
            aa=[sum(v[k]*axis[k] for k in (0,1)) for v in a]
            bb=[sum(v[k]*axis[k] for k in (0,1)) for v in b]
            overlap=min(overlap,min(max(aa)-min(bb),max(bb)-min(aa)))
    if overlap>=0:return 0.,overlap
    distance=min(_point_distance(p,x,y) for poly,other in ((a,b),(b,a))
        for p in poly for x,y in zip(other,other[1:]+other[:1]))
    return distance,0.


def _geometry(body):
    """Reconstruct the audit footprint from the actual archived SDK vertices."""
    vertices=body.get('world_vertices_xyz_carla_m')
    if (not isinstance(vertices,list) or len(vertices)!=8
            or any(not isinstance(p,list) or len(p)!=3 or not _finite(p) for p in vertices)):
        raise ValueError('Demonstration body lacks eight finite same-frame SDK vertices')
    matrix=body.get('actor_matrix_xyz_carla_m');box=body.get('local_bbox',{})
    center=box.get('center_xyz_m');extent=box.get('extent_xyz_m');rotation=box.get('rotation_deg',{})
    if (not isinstance(matrix,list) or len(matrix)!=4 or any(not isinstance(r,list) or len(r)!=4
            or not _finite(r) for r in matrix) or not isinstance(center,list) or len(center)!=3
            or not _finite(center) or not isinstance(extent,list) or len(extent)!=3
            or not _finite(extent) or any(v<=0 for v in extent)
            or set(rotation)!={'pitch','yaw','roll'} or not _finite(rotation.values())):
        raise ValueError('Demonstration body lacks a finite frozen pose/local bounding box')
    for key in ('velocity_xyz_carla_mps','acceleration_xyz_carla_mps2'):
        motion=body.get(key)
        if not isinstance(motion,list) or len(motion)!=3 or not _finite(motion):
            raise ValueError('Demonstration body lacks finite frozen motion')
    world=[sum(matrix[i][j]*(*center,1.)[j] for j in range(4)) for i in range(3)]
    # CARLA SDK vertices are float32 global-world coordinates. At Town13's
    # kilometre-scale coordinates their averaged centre need not be bit-exact
    # with Python matrix multiplication; allow 1cm without altering any vertex.
    if any(abs(sum(p[k] for p in vertices)/8.-world[k])>POLICY['sdk_vertex_center_tolerance_m'] for k in range(3)):
        raise ValueError('SDK body vertices disagree with frozen pose and local bbox center')
    polygon=_hull([[p[0],-p[1]] for p in vertices])
    return polygon,[min(p[2] for p in vertices),max(p[2] for p in vertices)],world


def _body(actor,frozen):
    pose=frozen.get_transform();box=actor.bounding_box
    rotation=box.rotation
    result=dict(id=int(actor.id),type_id=str(actor.type_id),
        actor_matrix_xyz_carla_m=[[float(v) for v in r] for r in pose.get_matrix()],
        local_bbox=dict(center_xyz_m=_vector(box.location),extent_xyz_m=_vector(box.extent),
            rotation_deg={k:float(getattr(rotation,k,0.)) for k in ('pitch','yaw','roll')}),
        world_vertices_xyz_carla_m=[_vector(p) for p in box.get_world_vertices(pose)],
        velocity_xyz_carla_mps=_vector(frozen.get_velocity()),
        acceleration_xyz_carla_mps2=_vector(frozen.get_acceleration()))
    _geometry(result)
    if not _finite(result['velocity_xyz_carla_mps']+result['acceleration_xyz_carla_mps2']):
        raise ValueError('Demonstration body has nonfinite frozen motion')
    return result


def _category(actor,polygon,lanes):
    name=actor.type_id
    if name in BICYCLES:return 'bicycle'
    if name.startswith('vehicle.'):return 'vehicle'
    if name.startswith('walker.pedestrian.'):return 'pedestrian'
    if name.startswith(('walker.animal.','animal.')):return 'animal'
    if name in ROAD_OBSTACLE_TYPES:return 'obstacle'
    if name=='static.prop.mesh' and getattr(actor,'attributes',{}).get('mesh_path','').startswith('/Game/Carla/Static/Car/'):
        return 'vehicle'
    # Unknown scene props need current physical road occupancy, not a family ID
    # or access to the scenario's future actor/trigger lists.
    if name.startswith(('static.prop.','walker.')) and any(polygons_intersect(polygon,p) for p in lanes):
        return 'obstacle'
    return None


def _measures(ego,objects,previous,excluded=()):
    polygon,zrange,_=_geometry(ego)
    matrix=ego['actor_matrix_xyz_carla_m'];v=ego['velocity_xyz_carla_mps'];a=ego['acceleration_xyz_carla_mps2']
    forward=[matrix[0][0],matrix[1][0]];length=math.hypot(*forward)
    if length<.5:raise ValueError('Demonstration ego has an invalid planar forward direction')
    forward=[x/length for x in forward]
    lateral=-forward[1]*a[0]+forward[0]*a[1]
    jerk=(lateral-previous['lateral_acceleration_mps2'])/.05 if previous is not None else None
    speed=math.hypot(v[0],v[1]);flags=[];hard=[];near=[];values=[]
    for body in objects:
        other,oz,_=_geometry(body)
        gap,penetration=separation(polygon,other)
        vertical=min(zrange[1],oz[1])-max(zrange[0],oz[0])
        contact=penetration>POLICY['penetration_xy_m'] and vertical>POLICY['penetration_z_m']
        identity=body['id'];kind=body['kind'];risks=[]
        if vertical>0:
            if gap<=POLICY['interaction_clearance_m']:near.append(identity)
            if contact:hard.append(identity)
            vulnerable=kind in ('pedestrian','animal','bicycle')
            if speed>.5 and gap<(POLICY['pedestrian_clearance_m'] if vulnerable else POLICY['object_clearance_m']):
                risks.append('close_vulnerable_object' if vulnerable else 'close_object')
            if speed>(POLICY['pedestrian_bypass_speed_mps'] if vulnerable else POLICY['object_bypass_speed_mps']) and gap<(
                    POLICY['pedestrian_bypass_clearance_m'] if vulnerable else POLICY['object_bypass_clearance_m']):
                risks.append('fast_bypass_near_vulnerable_object' if vulnerable else 'fast_bypass_near_object')
        flags.extend(risks)
        values.append(dict(id=identity,clearance_xy_m=gap,sat_penetration_xy_m=penetration,
            vertical_overlap_m=vertical,hard_geometry_penetration=contact,review_flags=risks))
    if abs(lateral)>POLICY['lateral_acceleration_review_mps2']:flags.append('high_lateral_acceleration')
    if jerk is not None and abs(jerk)>POLICY['lateral_jerk_review_mps3']:flags.append('high_lateral_jerk')
    if excluded:flags.append('nonvolumetric_prop_excluded_from_bbox_audit')
    return dict(speed_xy_mps=speed,lateral_acceleration_mps2=lateral,lateral_jerk_mps3=jerk,
        objects=values,near_actor_ids=sorted(near),hard_geometry_penetration_ids=sorted(hard),
        review_flags=sorted(set(flags)),valid=not hard)


class DemonstrationBehaviorAudit:
    def __init__(self):self.previous=None

    def observe(self,snapshot,ego,actors,lane_polygons=()):
        frame=int(snapshot.frame);stamp=float(snapshot.timestamp.elapsed_seconds)
        if frame<0 or not math.isfinite(stamp):raise ValueError('Invalid demonstration behavior source identity')
        if self.previous is not None and (frame!=self.previous['frame']+1 or abs(stamp-self.previous['time']-.05)>1e-5):
            raise ValueError('Demonstration behavior source is not consecutive official 20Hz')
        frozen=snapshot.find(ego.id)
        if frozen is None:raise ValueError('Demonstration ego absent from frozen behavior source')
        current=_body(ego,frozen);polygon,zrange,center=_geometry(current);objects=[];excluded=[]
        ego_radius=max(math.hypot(p[0]-center[0],p[1]+center[1]) for p in polygon)
        for actor in actors:
            if actor.id==ego.id or not actor.type_id.startswith(('vehicle.','walker.','animal.','static.prop.')):continue
            state=snapshot.find(actor.id)
            if state is None:continue
            extents=_vector(actor.bounding_box.extent)
            if not _finite(extents):raise ValueError('Nonfinite physical demonstration object extent')
            # Cheap source-frame broad phase before storing exact SDK vertices.
            origin=state.get_transform().location;extent=actor.bounding_box.extent;offset=actor.bounding_box.location
            bound=POLICY['clearance_radius_m']+ego_radius+math.sqrt(extent.x**2+extent.y**2+extent.z**2)
            bound+=math.sqrt(offset.x**2+offset.y**2+offset.z**2)+2.*POLICY['sdk_vertex_center_tolerance_m']
            if math.hypot(origin.x-center[0],origin.y-center[1])>bound:continue
            if any(v<=0 for v in extents):
                if actor.type_id.startswith(('vehicle.','walker.','animal.')) or actor.type_id in ROAD_OBSTACLE_TYPES:
                    raise ValueError('A traffic entity lacks a volumetric physical bounding box')
                excluded.append(dict(id=int(actor.id),type_id=actor.type_id,extent_xyz_m=extents,
                    reason='nonvolumetric_render_prop_not_an_audited_physical_body'))
                continue
            body=_body(actor,state);other,oz,_=_geometry(body)
            if min(zrange[1],oz[1])-max(zrange[0],oz[0])<=0:continue
            if separation(polygon,other)[0]>POLICY['clearance_radius_m']:continue
            kind=_category(actor,other,lane_polygons)
            if kind is not None:body['kind']=kind;objects.append(body)
        objects.sort(key=lambda x:x['id'])
        measures=_measures(current,objects,self.previous,excluded)
        row=dict(schema=SCHEMA,frame=frame,time=stamp,ego=current,nearby_bodies=objects,
            excluded_nonvolumetric_props=excluded,**measures,
            human_review_required=True,model_input=False,driving_control_generated=False,
            scope='frozen_current_20Hz_nearby_bbox_quality_evidence',
            limitation='bbox_penetration_not_mesh_collision;sampled_not_continuous;no_human_quality_certification')
        self.previous=row
        return row


def validate_rows(rows,execution,physical_rows=()):
    """Recompute quality metrics and bind them to recorded actual execution."""
    if len(rows)!=len(execution):raise ValueError('Behavior audit does not cover every actual execution frame')
    physical={r['frame']:r for r in physical_rows};previous=None;flags=set();hard=[]
    for row,step in zip(rows,execution):
        if (row.get('schema')!=SCHEMA or row.get('frame')!=step['frame'] or abs(row.get('time',math.inf)-step['time'])>1e-6
                or row.get('human_review_required') is not True or row.get('model_input') is not False
                or row.get('driving_control_generated') is not False):
            raise ValueError('Behavior audit is not bound to the actual execution source')
        bodies=row.get('nearby_bodies')
        if (not isinstance(bodies,list) or any(not isinstance(b,dict) or type(b.get('id')) is not int
                or b.get('kind') not in ('vehicle','bicycle','pedestrian','animal','obstacle') for b in bodies)
                or len({b['id'] for b in bodies})!=len(bodies)):
            raise ValueError('Invalid behavior-audit neighborhood identities')
        excluded=row.get('excluded_nonvolumetric_props')
        if (not isinstance(excluded,list) or any(not isinstance(r,dict) or type(r.get('id')) is not int
                or not str(r.get('type_id','')).startswith('static.prop.') or r['type_id'] in ROAD_OBSTACLE_TYPES
                or not isinstance(r.get('extent_xyz_m'),list) or len(r['extent_xyz_m'])!=3
                or not _finite(r['extent_xyz_m']) or not any(v<=0 for v in r['extent_xyz_m'])
                or r.get('reason')!='nonvolumetric_render_prop_not_an_audited_physical_body' for r in excluded)):
            raise ValueError('Invalid nonvolumetric-prop exclusion evidence')
        expected=_measures(row['ego'],bodies,previous,excluded)
        if any(row.get(k)!=v for k,v in expected.items()):
            raise ValueError('Behavior audit metrics contradict their archived current-frame geometry')
        _,_,center=_geometry(row['ego']);observed=step['observed_ego']['position']
        if math.dist([center[0],-center[1]],observed)>1e-4:
            raise ValueError('Behavior and executed ego body centers disagree')
        source=physical.get(row['frame'])
        if source is not None:
            for key,field in (('position_xyz_carla_m','actor_matrix_xyz_carla_m'),
                              ('velocity_xyz_carla_mps','velocity_xyz_carla_mps'),
                              ('acceleration_xyz_carla_mps2','acceleration_xyz_carla_mps2')):
                value=[row['ego'][field][i][3] for i in range(3)] if key=='position_xyz_carla_m' else row['ego'][field]
                if len(value)!=3 or any(abs(a-b)>1e-6 for a,b in zip(value,source[key])):
                    raise ValueError('Behavior ego and physical frozen source differ')
        flags.update(row['review_flags']);hard.extend(row['hard_geometry_penetration_ids']);previous=row
    if hard:raise ValueError('Demonstration has measured bounding-box penetration; not a physics collision certificate')
    return dict(declared=True,verified=True,schema=SCHEMA,frames=len(rows),
        human_review_required=True,automatic_human_quality_approval=False,
        review_flags=sorted(flags),quality_policy=deepcopy(POLICY),
        limitation='sampled nearby bounding boxes;not_mesh_contact_or_continuous_safety_proof')


def interaction_windows(rows):
    """Use observed neighboring entities, never family/route IDs or scripts."""
    windows=[];current=None
    for row in rows:
        ids=row['near_actor_ids']
        if not ids:continue
        if current is None or row['time']-current['end_time']>2.:
            current=dict(start_frame=row['frame'],end_frame=row['frame'],start_time=row['time'],end_time=row['time'],
                actor_ids=[],review_flags=[],minimum_clearance_m=math.inf)
            windows.append(current)
        current.update(end_frame=row['frame'],end_time=row['time'])
        current['actor_ids']=sorted(set(current['actor_ids'])|set(ids))
        current['review_flags']=sorted(set(current['review_flags'])|set(row['review_flags']))
        gaps=[x['clearance_xy_m'] for x in row['objects'] if x['id'] in ids]
        current['minimum_clearance_m']=min(current['minimum_clearance_m'],min(gaps))
    return windows
