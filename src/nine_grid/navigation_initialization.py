"""Explicit NAVSIM current-state recovery onto the supplied navigation.

This is a planning corridor, not a replacement native lane observation. It is
shared by references, object scope and the potential domain, and is only
registered from current state and public topology. Reference feasibility is
reported separately: a map baseline is not an already executable trajectory.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

from .geometry import rectangle_corners, clean_polyline, polyline_lengths

VERSION = 'navsim-initialization-recovery-1.0'
KEY = 'navigation_initialization'


def _project(line, point):
    s = float(line.project(Point(point)))
    a, b = max(0., s-.1), min(line.length, s+.1)
    p, q = np.array(line.interpolate(a).coords[0]), np.array(line.interpolate(b).coords[0])
    return s, np.array(line.interpolate(s).coords[0]), math.atan2(*(q-p)[::-1])


def _quintic(start, end, distance, wheelbase, rear, step=.25,reference_only=False):
    """Spatial Hermite curve with pose/curvature boundary conditions."""
    p0, h0, k0 = start; p1, h1, k1 = end
    length = float(distance)
    c = np.zeros((6, 2))
    c[0] = p0
    c[1] = length*np.array([math.cos(h0), math.sin(h0)])
    c[2] = .5*length**2*k0*np.array([-math.sin(h0), math.cos(h0)])
    rhs = np.vstack((p1-c[:3].sum(axis=0),
        length*np.array([math.cos(h1), math.sin(h1)])-c[1]-2*c[2],
        length**2*k1*np.array([-math.sin(h1), math.cos(h1)])-2*c[2]))
    c[3:] = np.linalg.solve([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]], rhs)
    u = np.linspace(0., 1., max(5, int(math.ceil(length/step))+1))
    p = np.polynomial.polynomial.polyval(u, c).T
    first = np.polynomial.polynomial.polyval(u, c[1:]*np.arange(1,6)[:,None]).T
    second = np.polynomial.polynomial.polyval(u, c[2:]*(np.arange(2,6)*np.arange(1,5))[:,None]).T
    norm = np.linalg.norm(first, axis=1)
    if np.any(norm < 1e-6):
        raise ValueError('initialization curve is not forward regular')
    curvature = (first[:,0]*second[:,1]-first[:,1]*second[:,0])/norm**3
    if np.any(np.abs(rear*curvature) >= .95) and not reference_only:
        raise ValueError('initialization curvature is not bicycle-reachable')
    beta = np.arcsin(np.clip(rear*curvature,-.999999,.999999))
    headings = np.unwrap(np.arctan2(first[:,1], first[:,0]))-beta
    steering = np.arctan(wheelbase*curvature/np.sqrt(np.maximum(1e-8,1-(rear*curvature)**2)))
    return p, headings, curvature, steering


def _body_check(points, headings, road, length, width):
    bodies = shapely.polygons(np.stack([rectangle_corners(p, h, length, width)
                                       for p,h in zip(points, headings)]))
    return bool(np.all(shapely.covers(road.buffer(1e-6), bodies)))


def _reference_quality(points,headings,road,ego,steering=None):
    bodies=shapely.polygons(np.stack([rectangle_corners(p,h,ego.length,ego.width)
                                   for p,h in zip(points,headings)]))
    outside=shapely.area(shapely.difference(bodies,road))
    return dict(full_body_on_road=bool(np.max(outside)<=1e-8),
        maximum_body_outside_area_m2=float(np.max(outside)),
        center_on_road=bool(road.buffer(1e-6).covers(LineString(points))),
        maximum_steering_rad=None if steering is None else float(np.max(np.abs(steering))),
        role='reference_feasibility_diagnostic_only;MPC_optimizes_actual_vehicle_constraints')


def _native_connections(scene, lanes, steering_limit, route_ids, *, points=None, front_m=None,rear_m=None):
    """Validate only short gaps of consecutive, directed public lane pairs."""
    result = []
    route = LineString(scene.route if points is None else points)
    ego_s = route.project(Point(scene.ego.position))
    for first, second in zip(route_ids,route_ids[1:]):
        a,b = lanes[first],lanes[second]
        if second not in a.successors or np.linalg.norm(a.center[-1]-b.center[0])>2.:
            raise ValueError('NAVSIM public continuation lacks a short directed native connection')
        left,right = LineString(a.center),LineString(b.center)
        # The map can contain a much longer public tail than this domain uses.
        # Do not require unused distant junctions to satisfy the local contract.
        seam_s = route.project(Point(a.center[-1]))
        if ((front_m is not None and seam_s>ego_s+float(front_m)+20.)
                or (rear_m is not None and seam_s<ego_s-float(rear_m)-20.)):
            continue
        # nuPlan connector baselines can extend several metres beyond their
        # own polygons. Attach inside both actual lane areas, with the same
        # bounded 20m end trim used by confirmed navigation connections.
        trim = max(3.,scene.ego.length/2)
        def covered_span(line, polygon):
            arc = np.linspace(0.,line.length,max(3,int(math.ceil(line.length/.25))+1))
            inside = [float(s) for s in arc if polygon.covers(line.interpolate(float(s)))]
            if not inside:
                raise ValueError('NAVSIM native baseline has no covered attachment')
            return inside[0],inside[-1]
        first_left,last = covered_span(left,Polygon(a.polygon))
        first_covered,last_right = covered_span(right,Polygon(b.polygon))
        # A short lane's covered span can itself be shorter than the nominal
        # trim. Keep both attachments inside that actual span, not past its
        # opposite boundary or on an extrapolated baseline endpoint.
        start_arc = max(first_left+min(.25,(last-first_left)/4),last-trim)
        end_arc = min(last_right-min(.25,(last_right-first_covered)/4),first_covered+trim)
        if left.length-start_arc>20. or end_arc>20.:
            raise ValueError('NAVSIM native attachment needs unsupported end trimming')
        start = np.array(left.interpolate(start_arc).coords[0])
        end = np.array(right.interpolate(end_arc).coords[0])
        s0,s1 = route.project(Point(start)),route.project(Point(end))
        if (s1<=s0 or not Polygon(a.polygon).covers(route.interpolate(s0))
                or not Polygon(b.polygon).covers(route.interpolate(s1))):
            raise ValueError('NAVSIM native gap is not on the frozen public route')
        arc = np.linspace(s0,s1,max(5,int(math.ceil((s1-s0)/.25))+1))
        points = np.array([route.interpolate(float(s)).coords[0] for s in arc])
        heading = np.unwrap(np.arctan2(np.gradient(points[:,1],arc),np.gradient(points[:,0],arc)))
        curvature = np.gradient(heading,arc)
        quality=_reference_quality(points,heading,scene.drivable,scene.ego,
                                  np.arctan(scene.ego.wheelbase_m*curvature))
        if not quality['center_on_road']:
            error = ValueError('NAVSIM native center connection leaves actual drivable: '+str(first)+'->'+str(second))
            error.navigation_interval_m = (float(s0-ego_s),float(s1-ego_s))
            raise error
        shape = LineString(points).buffer(min(a.width,b.width)/2,cap_style=2).intersection(scene.drivable)
        result.append(dict(from_lane_id=first,to_lane_id=second,shape=shape,
            kind='validated_short_directed_native_navigation_gap',reference_quality=quality))
    return result


def native_navigation_connections(scene, trace, *, front_m=120.,rear_m=20.,require_forward=True):
    """Same independently checked native seam band for every NAVSIM lane trace."""
    if scene.scenario!='navsim' or trace is None:
        return []
    lanes = {lane.id:lane for lane in scene.lanes}
    route = LineString(trace['points'])
    ego_s = route.project(Point(scene.ego.position))
    required = max(40.,float(scene.diagnostics.get('route_association',{}).get('required_forward_reference_m',40.)))
    links,rejected = [],[]
    for first,second in zip(trace['lane_ids'],trace['lane_ids'][1:]):
        try:
            links.extend(_native_connections(scene,lanes,.6,[first,second],points=trace['points'],
                                              front_m=front_m,rear_m=rear_m))
        except ValueError as error:
            seam = route.project(Point(lanes[first].center[-1]))-ego_s
            interval = list(getattr(error,'navigation_interval_m',(seam-20.,seam+20.)))
            needed = interval[1]>=0. and interval[0]<=min(float(front_m),required)
            rejection = dict(from_lane_id=first,to_lane_id=second,reason=str(error),
                longitudinal_interval_m=interval,required_forward=needed,
                policy='no_unvalidated_band_authorized')
            rejected.append(rejection)
    trace['native_connection_rejections'] = rejected
    trace['required_forward_connection_m'] = required
    trace['native_forward_valid'] = not any(row['required_forward'] for row in rejected)
    trace['verified_forward_until_m'] = min([float(route.length-ego_s)]+[
        max(0.,row['longitudinal_interval_m'][0]) for row in rejected
        if row['longitudinal_interval_m'][1]>=0.])
    # Backward/distant failures remain explicit unconnected context; they must
    # not authorize a strip or reject unrelated current forward navigation.
    # Verify the actually needed forward native path, not the entire map tail.
    extent = min(float(front_m),required)
    if route.length-ego_s+1e-6 < extent:
        trace['native_forward_valid'] = False
    native = [Polygon(lanes[key].polygon).intersection(
        LineString(lanes[key].center).buffer(lanes[key].width/2,cap_style=2)) for key in trace['lane_ids']]
    domain = shapely.union_all([*native,*[link['shape'] for link in links]]).intersection(scene.drivable)
    needed_path = substring(route,max(ego_s,float(trace.get('domain_start_s',ego_s))),max(ego_s+extent,float(trace.get('domain_start_s',ego_s))))
    if not domain.buffer(1e-6).covers(needed_path):
        trace['native_forward_valid'] = False
        missing = needed_path.difference(domain.buffer(1e-6))
        endpoints = [np.array(part.coords[0]) for part in getattr(missing,'geoms',[missing])
                     if part.geom_type=='LineString' and not part.is_empty]
        if endpoints:
            trace['verified_forward_until_m'] = min(trace['verified_forward_until_m'],
                min(max(0.,route.project(Point(p))-ego_s) for p in endpoints))
    return links


def build_initialization(ego, lanes, route, route_ids, road, association,
                         yaw_rate=0., steering_limit=.6):
    """Produce only a current/map recovery; never consult a future ego pose."""
    base = dict(version=VERSION, status='disabled', reason='native_current_lane_on_navigation',
                raw_lane_id=ego['lane_id'], navigation_lane_id=association['current_lane_id'],
                geometry_source='current_pose_motion_directed_native_map_and_public_navigation_only')
    groups = association['lane_roadblock_ids']
    if groups.get(ego['lane_id']) in {groups.get(key) for key in route_ids}:
        return np.asarray(route, float), base
    table = {lane['id']: lane for lane in lanes}
    if ego['lane_id'] not in table:
        return np.asarray(route, float), dict(base, status='failed', reason='native_current_lane_missing')
    try:
        position = np.array(ego['position'], float)
        velocity = np.array(ego['velocity'], float)
        speed = float(np.linalg.norm(velocity))
        wheelbase, rear = float(ego['wheelbase_m']), float(ego['rear_axle_to_center_m'])
        # Match the planning bicycle and its causal observed steering estimate.
        # Raw lateral velocity may contain rear-axle slip/noise; forcing it as
        # both body heading and yaw curvature gives incompatible boundaries.
        observed_steer = float(np.clip(math.atan(wheelbase*float(yaw_rate)/max(abs(velocity[0]),.3)),
                                       -steering_limit,steering_limit)) if speed>.3 else 0.
        beta = math.atan(rear/wheelbase*math.tan(observed_steer))
        course = float(ego['heading'])+beta
        curvature = math.sin(beta)/rear
        length, width = float(ego['length']), float(ego['width'])
        native = table[ego['lane_id']]
        nav = table[association['current_lane_id']]
        # Trace only the actual directed predecessors; never import the raw
        # connector's future (possibly wrong-turn) branch into the corridor.
        back_points = np.asarray(native['center'], float)
        back_ids, current = [native['id']], native
        for _ in range(32):
            line = LineString(back_points)
            if line.project(Point(position)) >= 22.:
                break
            options = [lane for lane in lanes if current['id'] in lane['successors'] and lane['id'] not in back_ids
                and np.linalg.norm(np.asarray(lane['center'])[-1]-back_points[0]) <= 2.]
            if not options:
                break
            # Incoming heading and distance disambiguate only the observed past.
            target_h = math.atan2(*(back_points[1]-back_points[0])[::-1])
            def rank(lane):
                center = np.array(lane['center'])
                h = math.atan2(*(center[-1]-center[-2])[::-1])
                return abs((h-target_h+math.pi)%(2*math.pi)-math.pi), lane['id']
            current = min(options, key=rank)
            back_points = clean_polyline(np.vstack((current['center'], back_points)))
            back_ids.insert(0, current['id'])
        line = LineString(back_points)
        along = float(line.project(Point(position)))
        rear_curve = None
        rear_fallback = []
        # A quintic's parameter length is not its geometric arc length. Extend
        # only along the same directed native predecessor until the measured
        # current-state curve actually supplies the agreed 20m rear context.
        distances = list(np.arange(20., min(along,30.)+.01, .5))
        if not distances and along>.1:
            distances = [float(min(along,20.))]
        for distance in distances:
            _, p0, h0 = _project(line, line.interpolate(along-distance).coords[0])
            try:
                candidate = _quintic((p0,h0,0.), (position,course,curvature),
                                      distance, wheelbase, rear)
            except ValueError:
                try:
                    candidate=_quintic((p0,h0,0.),(position,course,curvature),distance,wheelbase,rear,reference_only=True)
                except ValueError:
                    continue
            p, h, _, st = candidate
            rear_fallback.append((float(np.max(np.abs(st))),candidate))
            if (polyline_lengths(p)[-1] >= 20. and np.max(np.abs(st)) <= steering_limit
                    and _body_check(p,h,road,length,width)):
                rear_curve = candidate
                break
        reasons=[]
        if rear_curve is None:
            # Rear geometry only defines the 20m object chart. It is not a
            # trajectory that the vehicle must now drive backwards through.
            reasons.append('rear_reference_not_vehicle_feasible')
            if rear_fallback:
                rear_curve=min(rear_fallback,key=lambda row:row[0])[1]
            else:
                rear_curve=(position[None,:],np.array([ego['heading']]),np.array([curvature]),np.array([observed_steer]))
        back, back_h, _, back_steer = rear_curve
        if polyline_lengths(back)[-1]<20.-1e-6:
            reasons.append('native_rear_coverage_short')
        public = LineString(route)
        s0, _, _ = _project(public, position)
        available = public.length-s0
        chosen = None
        fallback=[]
        candidates = list(np.arange(max(15., min(30., speed*2.)), min(60., available-length)+.01, 5.))
        if not candidates and available>.1:
            candidates=[float(min(15.,available))]
        for distance in candidates:
            end_s = s0+float(distance)
            _, target, target_h = _project(public, public.interpolate(end_s).coords[0])
            _, _, ha = _project(public, public.interpolate(max(s0,end_s-.5)).coords[0])
            _, _, hb = _project(public, public.interpolate(min(public.length,end_s+.5)).coords[0])
            target_k = ((hb-ha+math.pi)%(2*math.pi)-math.pi)
            try:
                points, headings, ks, steer = _quintic((position,course,curvature),
                    (target,target_h,target_k), distance, wheelbase, rear)
            except ValueError:
                try:
                    points,headings,ks,steer=_quintic((position,course,curvature),
                        (target,target_h,target_k),distance,wheelbase,rear,reference_only=True)
                except ValueError:
                    continue
            # Keep a causal current-pose/public-route geometric reference even
            # when none of these seeds is already a feasible vehicle motion.
            # It is explicitly unverified, and clipped road bands never expand
            # the real map. The MPC still owns its body/dynamic constraints.
            center_outside=float(LineString(points).difference(road).length)
            fallback.append(((center_outside,float(np.max(np.abs(steer)))),
                             (points,headings,ks,steer,end_s)))
            if (np.max(np.abs(steer)) > steering_limit
                    or np.max(np.abs(ks))*speed**2 > max(3.5,abs(curvature)*speed**2)+1e-6
                    or not _body_check(points,headings,road,length,width)):
                continue
            chosen = points, headings, ks, steer, end_s
            break
        if chosen is None:
            if not fallback:
                raise ValueError('no finite forward-regular current/public reference connection')
            chosen=min(fallback,key=lambda row:row[0])[1]
            reasons.append('forward_reference_not_vehicle_feasible')
        points, headings, ks, steer, end_s = chosen
        connection = np.vstack((back[:-1], points))
        connection_h = np.r_[back_h[:-1], headings]
        arc = polyline_lengths(route)
        combined = clean_polyline(np.vstack((connection, np.asarray(route)[arc>end_s+1e-6])))
        end_lane = min((table[key] for key in route_ids),
                      key=lambda lane: LineString(lane['center']).distance(Point(points[-1])))['id']
        rear_anchor=min(back_ids,key=lambda key:LineString(table[key]['center']).distance(Point(back[0]))) if len(back)>1 else None
        # The sampled rear start can lie on an interior predecessor, rather
        # than the first lane fetched to obtain enough past context.
        rear_anchor_kind='current_pose_without_native_rear'
        if rear_anchor is not None:
            if LineString(table[rear_anchor]['center']).distance(Point(back[0]))<=.05:
                rear_anchor_kind='native_lane_center'
            else:
                rear_anchor=None
                rear_anchor_kind='directed_native_center_gap'
        result = dict(base, status='reference_only_unverified' if reasons else 'active',
            reason='off_navigation_overlapping_connector_initialization',
            points_world=connection.tolist(), headings_world=connection_h.tolist(), ego_index=len(back)-1,
            width_m=min(float(native['width']),float(nav['width'])),
            rear_native_lane_ids=back_ids,rear_anchor_lane_id=rear_anchor,
            rear_anchor_kind=rear_anchor_kind,end_lane_id=end_lane,
            ego_length_m=length, ego_width_m=width, wheelbase_m=wheelbase, rear_axle_to_center_m=rear,
            steering_limit_rad=float(steering_limit), maximum_steering_rad=float(max(np.max(np.abs(back_steer)),np.max(np.abs(steer)))),
            maximum_forward_lateral_acceleration_mps2=float(np.max(np.abs(ks))*speed**2),
            rear_context_m=float(polyline_lengths(back)[-1]), forward_recovery_m=float(polyline_lengths(points)[-1]),
            public_route_join_s_m=float(end_s),reference_quality_reasons=reasons,
            validation='current_pose_heading;native_directed_rear;public_forward_route;vehicle_feasibility_diagnostic_only')
        return combined, result
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return np.asarray(route, float), dict(base, status='failed', reason=str(error))


def initialization_geometry(scene):
    """Verify coordinate/topology identity; diagnose reference vehicle feasibility."""
    if scene.scenario != 'navsim':
        return None
    row = scene.diagnostics.get(KEY)
    if not row or row.get('status') == 'disabled':
        return None
    if row.get('version') != VERSION or row.get('status') not in ('active','reference_only_unverified'):
        raise ValueError('NAVSIM initialization recovery failed: '+str(row.get('reason')))
    digest = hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest()
    digest += hashlib.sha256(scene.route.tobytes()+scene.drivable.wkb).hexdigest()
    state = (scene.frame_id,scene.time,scene.ego.lane_id,scene.ego.heading,
             scene.ego.length,scene.ego.width,scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m)
    geometry_digest = hashlib.sha256(repr(state).encode()+np.asarray(scene.ego.position,float).tobytes())
    required_ids = set(scene.route_lane_ids)|set(row['rear_native_lane_ids'])
    for lane in scene.lanes:
        if lane.id in required_ids:
            geometry_digest.update(repr((lane.id,lane.width,tuple(lane.successors))).encode())
            geometry_digest.update(np.asarray(lane.center,float).tobytes()+np.asarray(lane.polygon,float).tobytes())
    digest += geometry_digest.hexdigest()
    cached = scene.assets.get(KEY)
    if cached is not None and cached['fingerprint'] == digest:
        return cached['geometry']
    points = np.asarray(row['points_world'],float)
    headings = np.asarray(row['headings_world'],float)
    index = row['ego_index']
    lanes = {lane.id:lane for lane in scene.lanes}
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<5 or headings.shape!=(len(points),)
            or not np.isfinite(points).all() or not np.isfinite(headings).all()
            or type(index) is not int or not 0<=index<len(points)-1
            or row['raw_lane_id']!=scene.ego.lane_id
            or row['navigation_lane_id'] not in scene.route_lane_ids
            or row['end_lane_id'] not in scene.route_lane_ids
            or np.linalg.norm(points[index]-scene.ego.position)>1e-6
            or abs((headings[index]-scene.ego.heading+math.pi)%(2*math.pi)-math.pi)>1e-3):
        raise ValueError('NAVSIM initialization contract differs from current state/navigation')
    expected_width = min(lanes[row['raw_lane_id']].width,lanes[row['navigation_lane_id']].width)
    if (not math.isfinite(float(row['width_m'])) or abs(float(row['width_m'])-expected_width)>1e-6
            or not math.isfinite(float(row['steering_limit_rad'])) or not 0.<float(row['steering_limit_rad'])<=.6):
        raise ValueError('NAVSIM initialization width/steering contract expands native limits')
    chords = np.linalg.norm(np.diff(points,axis=0),axis=1)
    if np.any(chords<1e-8) or chords.max()>1.:
        raise ValueError('NAVSIM initialization reference is not densely covered')
    if not np.allclose([row['ego_length_m'],row['ego_width_m'],row['wheelbase_m'],row['rear_axle_to_center_m']],
        [scene.ego.length,scene.ego.width,scene.ego.wheelbase_m,scene.ego.rear_axle_to_center_m],rtol=0.,atol=1e-6):
        raise ValueError('NAVSIM initialization vehicle geometry mismatch')
    route = LineString(scene.route)
    along = np.array([route.project(Point(p)) for p in points])
    if max(route.distance(Point(p)) for p in points)>.01 or np.any(np.diff(along)<-1e-6):
        raise ValueError('NAVSIM initialization is not on the frozen directed reference')
    # Recompute curvature from sampled centre geometry, independently of the
    # producer's maximum. Dense .25m geometry has a small endpoint difference.
    arc = np.r_[0.,np.cumsum(chords)]
    tangent = np.unwrap(np.arctan2(np.gradient(points[:,1],arc),np.gradient(points[:,0],arc)))
    curvature = np.gradient(tangent,arc)
    rear = float(scene.ego.rear_axle_to_center_m)
    steer = np.arctan(scene.ego.wheelbase_m*curvature/np.sqrt(np.maximum(1e-8,1-(rear*curvature)**2)))
    quality=_reference_quality(points,headings,scene.drivable,scene.ego,steer)
    quality['rear_coverage_m']=float(chords[:index].sum())
    quality['reasons']=list(row.get('reference_quality_reasons',[]))
    if not quality['full_body_on_road'] and 'reference_body_outside_actual_road' not in quality['reasons']:
        quality['reasons'].append('reference_body_outside_actual_road')
    if np.max(np.abs(steer))>float(row['steering_limit_rad'])+.01:
        quality['reasons'].append('reference_steering_exceeds_limit')
    back = row['rear_native_lane_ids']
    if (not back or back[-1]!=row['raw_lane_id'] or any(key not in lanes for key in back)
            or any(b not in lanes[a].successors for a,b in zip(back,back[1:]))
            or (index>0 and LineString(clean_polyline(np.vstack([lanes[key].center for key in back]))).distance(Point(points[0]))>.05)
            or LineString(lanes[row['end_lane_id']].center).distance(Point(points[-1]))>.05):
        raise ValueError('NAVSIM initialization is detached from current/public map endpoints')
    anchor=row.get('rear_anchor_lane_id')
    if anchor is not None and (anchor not in back or LineString(lanes[anchor].center).distance(Point(points[0]))>.05):
        raise ValueError('NAVSIM rear anchor lane differs from its actual directed segment')
    band = LineString(points).buffer(float(row['width_m'])/2,cap_style=2).intersection(scene.drivable)
    # Recovery replaces the public prefix up to its proven native endpoint.
    # Validating or restoring that old prefix would reintroduce a competing
    # route and reject valid recoveries which cross the first lane boundary.
    route_ids = list(scene.route_lane_ids[scene.route_lane_ids.index(row['end_lane_id']):])
    trace=dict(points=scene.route,lane_ids=route_ids,domain_start_s=float(route.project(Point(points[-1]))))
    connections=native_navigation_connections(scene,trace,front_m=120.,rear_m=20.,require_forward=False)
    native_parts, native_geometry = [], {}
    for offset,key in enumerate(route_ids):
        center = LineString(lanes[key].center)
        if offset == 0:
            center = substring(center,center.project(Point(points[-1])),center.length)
        if center.geom_type != 'LineString' or center.length <= 1e-6:
            continue
        polygon = Polygon(lanes[key].polygon).intersection(center.buffer(lanes[key].width/2,cap_style=2))
        native_parts.append(polygon)
        native_geometry[key] = polygon,center
    corridor = shapely.union_all([band,*native_parts,*[link['shape'] for link in connections]]).intersection(scene.drivable)
    # GEOS intersections can retain detached zero-area boundary fragments.
    # Discard only lines/points and retain every actual-road area component.
    # Multipart remote context is not a vehicle-feasibility input veto.
    from .potential_domain import polygonal_area
    corridor = polygonal_area(corridor)
    if corridor.is_empty or not corridor.is_valid or corridor.geom_type not in ('Polygon','MultiPolygon'):
        raise ValueError('NAVSIM initialization reference has no valid narrow road area')
    forward_curve=LineString(points[index:])
    missing=forward_curve.difference(scene.drivable.buffer(1e-6))
    forward_valid=missing.is_empty
    verified_until=float(polyline_lengths(points[index:])[-1])
    if not forward_valid:
        components=getattr(missing,'geoms',[missing])
        entries=[forward_curve.project(Point(g.coords[0])) for g in components if g.geom_type=='LineString' and not g.is_empty]
        verified_until=min(entries,default=0.)
    native_until=max(0.,float(trace['verified_forward_until_m']))
    if trace['native_forward_valid']:
        verified_until=max(verified_until,native_until) if forward_valid else verified_until
    else:
        verified_until=min(verified_until,native_until)
    forward_valid=forward_valid and trace['native_forward_valid']
    quality.update(native_forward_valid=forward_valid,verified_forward_until_m=verified_until,
                   native_connection_rejections=trace['native_connection_rejections'])
    result = dict(points=points, headings=headings, shape=band, corridor=corridor,
        native_connections=connections,
        lane_ids=route_ids, native_geometry=native_geometry, navigation_lane_id=row['navigation_lane_id'],
        width_m=float(row['width_m']), evidence={key:row[key] for key in
            ('version','reason','raw_lane_id','navigation_lane_id','end_lane_id','geometry_source','rear_context_m','forward_recovery_m','validation')},
        reference_quality=quality,native_forward_valid=forward_valid,verified_forward_until_m=verified_until)
    scene.assets[KEY] = dict(fingerprint=digest, geometry=result)
    return result


def initialization_lane(scene, lanes):
    geometry = initialization_geometry(scene)
    if geometry is None:
        return None
    lane = lanes[geometry['navigation_lane_id']]
    # This temporary planning view is never written to scene.lanes or ego's raw
    # association. All consumers use the same registered current-state reference corridor.
    # A planning-lane view does not grant its polygon any new drivable area.
    # Scope/APF consume the exact (possibly multipart) road-clipped corridor.
    return replace(lane, center=scene.route.copy(), polygon=lane.polygon.copy(),
                   width=geometry['width_m'], left=None, right=None)
