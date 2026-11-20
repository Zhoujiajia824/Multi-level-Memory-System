"""Lane-domain, centre-sampled Frenet artificial potentials.

The reference is one frozen, publicly confirmed navigation branch.  Its C2
interpolant is parametrised by *geometric* arc length, not by x or the chord
parameter of the spline.  Neither ego dimensions nor source dimensions enter
the potential.  Complete-body collision/rule checks remain the planner's job.

Outside ``domain`` the public field is undefined (NaN plus an explicit mask).
Callers must impose their separate lane/body constraints, never treat a masked
field as a safe zero-cost route.  Source centres already admitted upstream may
lie just outside the lane when their body intrudes; only their projection
coverage, rather than their centre's domain membership, limits the source.
"""
from __future__ import annotations

import numpy as np
import shapely
from scipy.interpolate import CubicSpline

from .geometry import clean_polyline, ego_to_world


VERSION = 'lane-frenet-centre-potential-1.0'

CLASS_DEFAULTS = {
    'vehicle': {'longitudinal_m': 20., 'lateral_m': 6., 'peak': 8.},
    'bicycle': {'longitudinal_m': 10., 'lateral_m': 3., 'peak': 12.},
    'pedestrian': {'longitudinal_m': 10., 'lateral_m': 10., 'peak': 16.},
    'static_obstacle': {'longitudinal_m': 12., 'lateral_m': 4., 'peak': 8.},
}


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name+' must be a finite numeric value')
    number = float(value)
    if not np.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(name+' must be finite and '+('nonnegative' if zero else 'positive'))
    return number


def compact_shape(rho):
    """Monotone C2 potential kernel; support is exactly rho < 1.

    Stable factorisation of 1-10*rho**3+15*rho**4-6*rho**5 avoids
    cancellation near the support boundary.  A smooth hill has zero gradient
    at its centre; potential height is not the same quantity as force.
    """
    rho = np.asarray(rho, float)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError('elliptical radius must be finite and nonnegative')
    t = np.maximum(1.-rho, 0.)
    return t**3*(1.+3.*rho+6.*rho**2)


class LanePotentialGeometry:
    """A frozen smooth lane chart plus its explicit admissible field domain.

    ``reference`` and query points use one Cartesian frame. ``domain`` uses
    that frame too unless ``domain_pose=(world_origin, world_heading)`` maps
    queries from an ego frame into the domain's unchanged world frame. Only
    points and normal vectors are transformed, never the polygon topology.
    ``anchor_s`` / ``anchors`` are coordinates in THIS chart's integrated arc
    length.  They must not be old polyline arc lengths from another reference.
    Per-point anchors and a bounded search window prevent selecting a remote
    return branch solely because it is geometrically close.

    Projection is piecewise smooth at branch/window choices; regular-region
    Jacobians include the curvature factor t/(1-kappa*d).  A singular normal
    chart, uncovered endpoint or unconverged projection is invalid explicitly.
    """

    def __init__(self, reference, domain, *, anchor_s=None, domain_pose=None,
                 projection_window_m=120., minimum_metric=.05):
        points = np.asarray(reference, float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('potential reference must be a finite (M,2) array')
        self.reference = clean_polyline(points).copy()
        if domain is None or domain.is_empty or not domain.is_valid:
            raise ValueError('potential lane domain must be a nonempty valid geometry')
        if domain.geom_type not in ('Polygon', 'MultiPolygon'):
            raise ValueError('potential lane domain must be polygonal')
        self.domain = domain
        self._domain_boundary = domain.boundary
        self._domain_pose = None
        self._domain_rotation = None
        if domain_pose is not None:
            origin, heading = domain_pose
            origin = np.asarray(origin, float)
            heading = float(heading)
            if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(heading):
                raise ValueError('potential domain pose must be a finite 2D origin and heading')
            self._domain_pose = (origin.copy(), heading)
            self._domain_rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                             [np.sin(heading), np.cos(heading)]])
        shapely.prepare(self.domain)
        self.projection_window_m = _positive(projection_window_m, 'projection_window_m')
        self.minimum_metric = _positive(minimum_metric, 'minimum_metric')
        if self.minimum_metric >= 1.:
            raise ValueError('minimum_metric must be less than one')
        if anchor_s is not None and not np.isfinite(float(anchor_s)):
            raise ValueError('anchor_s must be finite')
        self.anchor_s = None if anchor_s is None else float(anchor_s)
        self._segments = np.diff(self.reference, axis=0)
        self._chords = np.linalg.norm(self._segments, axis=1)
        self._u = np.r_[0., np.cumsum(self._chords)]
        self._curve = CubicSpline(self._u, self.reference, axis=0, bc_type='natural')
        self._gauss_nodes, self._gauss_weights = np.polynomial.legendre.leggauss(12)
        self._arc = np.r_[0., np.cumsum(self._integral(self._u[:-1], self._u[1:]))]
        self.length = float(self._arc[-1])
        self.arc_nodes = self._arc.copy()
        self.diagnostics = {
            'version': VERSION, 'reference': 'frozen_C2_cubic_interpolant',
            'arc_length': '12_point_Gauss_Legendre_geometric_arc_per_cubic_piece',
            'coordinates': 's_forward_m;d_left_positive_m',
            'potential_samples': 'ego_centre;source_centre;no_body_dimensions',
            'outside_domain': 'undefined_NaN_and_valid_false;separate_constraints_required',
            'projection_jacobian': 'implicit_closest_point;curvature_metric_included',
            'projection_window_m': self.projection_window_m,
            'reference_length_m': self.length,
            'domain_coordinates': 'query_frame' if domain_pose is None else 'unchanged_world_frame',
            'domain_query': 'point_transform_only;polygon_topology_unchanged',
            'limitation': 'local_normal_chart;branch/window switches are not globally smooth',
        }

    def _domain_points(self, points):
        points = np.atleast_2d(np.asarray(points, float))
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('potential domain queries must be finite (N,2) points')
        return points if self._domain_pose is None else ego_to_world(points, *self._domain_pose)

    def domain_contains(self, points):
        """Inclusive membership in the original domain, including its boundary."""
        query = self._domain_points(points)
        return np.asarray(shapely.intersects_xy(self.domain, query[:, 0], query[:, 1]), bool)

    def domain_clearance(self, points):
        """Signed distance and inward normal, expressed in the query frame.

        Positive inside, negative outside (including holes); at non-smooth
        nearest-boundary choices this retains the existing local derivative.
        The normal is a vector, so changing frames applies rotation only.
        """
        query = self._domain_points(points)
        inside = np.asarray(shapely.intersects_xy(self.domain, query[:, 0], query[:, 1]), bool)
        nearest = shapely.get_coordinates(shapely.get_point(
            shapely.shortest_line(self._domain_boundary, shapely.points(query)), 0))
        offset = query-nearest
        distance = np.linalg.norm(offset, axis=1)
        sign = np.where(inside, 1., -1.)
        normal = sign[:, None]*offset/np.maximum(distance[:, None], 1e-10)
        if self._domain_rotation is not None:
            normal = normal @ self._domain_rotation
        return sign*distance, normal

    def _integral(self, lo, hi):
        lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
        half = (hi-lo)/2.
        u = (hi+lo)[..., None]/2.+half[..., None]*self._gauss_nodes
        speed = np.linalg.norm(self._curve(u, 1), axis=-1)
        return half*np.sum(speed*self._gauss_weights, axis=-1)

    def _arc_at(self, u):
        u = np.asarray(u, float)
        j = np.clip(np.searchsorted(self._u, u, side='right')-1, 0, len(self._u)-2)
        return self._arc[j]+self._integral(self._u[j], u)

    def chart_arc_from_polyline_s(self, old_s):
        """Convert THIS reference's chord arc to its spline's geometric arc.

        The spline parameter is the original polyline chord arc.  Integrating
        the containing fractional cubic piece gives the exact same chart as
        ``project``; simply interpolating ``arc_nodes`` would not do so.
        This operation changes coordinates, not a frozen goal/event identity.
        """
        old_s = np.asarray(old_s, float)
        if (np.any(~np.isfinite(old_s)) or np.any(old_s < -1e-7)
                or np.any(old_s > self._u[-1]+1e-7)):
            raise ValueError('polyline arc is outside this frozen reference')
        return self._arc_at(np.clip(old_s, 0., self._u[-1]))

    def _parameter_at(self, s):
        s = np.asarray(s, float)
        if np.any(~np.isfinite(s)) or np.any(s < -1e-7) or np.any(s > self.length+1e-7):
            raise ValueError('sample arc length is outside the frozen reference')
        s = np.clip(s, 0., self.length)
        j = np.clip(np.searchsorted(self._arc, s, side='right')-1, 0, len(self._u)-2)
        fraction = (s-self._arc[j])/(self._arc[j+1]-self._arc[j])
        u = self._u[j]+fraction*(self._u[j+1]-self._u[j])
        for _ in range(6):
            error = self._arc_at(u)-s
            if np.max(np.abs(error), initial=0.) < 1e-11:
                break
            speed = np.maximum(np.linalg.norm(self._curve(u, 1), axis=-1), 1e-12)
            u = np.clip(u-error/speed, self._u[j], self._u[j+1])
        return u

    def frame(self, s):
        """Position, unit tangent/normal and signed curvature at geometric s."""
        u = self._parameter_at(s)
        first, second = self._curve(u, 1), self._curve(u, 2)
        speed = np.linalg.norm(first, axis=-1)
        tangent = first/speed[..., None]
        normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        curvature = (first[..., 0]*second[..., 1]-first[..., 1]*second[..., 0])/speed**3
        return {'position': self._curve(u), 'tangent': tangent,
                'normal': normal, 'curvature': curvature}

    def sample(self, s, d=0.):
        """Map THIS chart's arc length and signed normal offset to Cartesian."""
        s, d = np.broadcast_arrays(np.asarray(s, float), np.asarray(d, float))
        if not np.isfinite(d).all():
            raise ValueError('normal offsets must be finite')
        frame = self.frame(s)
        return frame['position']+d[..., None]*frame['normal']

    def project(self, points, *, anchors=None, require_domain=True, window_m=None):
        """Return batched s/d and matching Cartesian Jacobian, including masks.

        The leading dimension is retained even for a single input point.
        ``coordinate_valid`` includes reference coverage and local metric;
        ``domain_valid`` tests the supplied lane domain independently.
        ``valid`` is their conjunction unless ``require_domain=False``.
        ``window_m`` is a per-call scalar or per-point override; it never
        mutates this frozen geometry's default projection window.
        Values remain finite where possible for infeasible SQP iterates; field
        methods below publish NaN for invalid field samples.
        """
        points = np.atleast_2d(np.asarray(points, float))
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('potential query must be a finite (N,2) array')
        if len(points) == 0:
            return {'s': np.empty(0), 'd': np.empty(0), 'heading': np.empty(0),
                    'jacobian': np.empty((0, 2, 2)), 'valid': np.empty(0, bool),
                    'coordinate_valid': np.empty(0, bool), 'domain_valid': np.empty(0, bool),
                    'coverage': np.empty(0, bool), 'metric': np.empty(0),
                    'feet': np.empty((0, 2))}
        requested = self.anchor_s if anchors is None else anchors
        anchor = None if requested is None else np.broadcast_to(np.asarray(requested, float), (len(points),))
        if anchor is not None and not np.isfinite(anchor).all():
            raise ValueError('projection anchors must be finite coordinates in this chart')
        raw_window = self.projection_window_m if window_m is None else window_m
        window = np.broadcast_to(np.asarray(raw_window, float), (len(points),))
        if (isinstance(raw_window, (bool, np.bool_)) or not np.isfinite(window).all()
                or np.any(window <= 0.)):
            raise ValueError('projection windows must be finite positive metres')
        output = []
        for start in range(0, len(points), 1024):
            subset = points[start:start+1024]
            relative = subset[:, None]-self.reference[:-1]
            fraction = np.clip(np.einsum('nsi,si->ns', relative, self._segments)/self._chords**2, 0., 1.)
            feet = self.reference[:-1]+fraction[..., None]*self._segments
            cost = np.sum((subset[:, None]-feet)**2, axis=-1)
            permitted = np.ones_like(cost, bool)
            if anchor is not None:
                local_anchor = anchor[start:start+len(subset)]
                local_window = window[start:start+len(subset)]
                permitted = ((self._arc[:-1][None] <= local_anchor[:, None]+local_window[:, None])
                             & (self._arc[1:][None] >= local_anchor[:, None]-local_window[:, None]))
                cost = np.where(permitted, cost, np.inf)
            chosen = np.argmin(cost, axis=1)
            row = np.arange(len(subset))
            has_branch = permitted.any(axis=1)
            u = self._u[chosen]+fraction[row, chosen]*self._chords[chosen]
            lower = self._u[np.maximum(chosen-1, 0)]
            upper = self._u[np.minimum(chosen+2, len(self._u)-1)]
            if anchor is not None:
                # Restrict Newton as well as the initial nearest-segment seed.
                lower = np.maximum(lower, self._parameter_at(
                    np.clip(local_anchor-local_window, 0., self.length)))
                upper = np.minimum(upper, self._parameter_at(
                    np.clip(local_anchor+local_window, 0., self.length)))
                # Out-of-coverage anchors keep invalid diagnostics finite.
                upper = np.maximum(upper, lower)
                u = np.clip(u, lower, upper)
            for _ in range(16):
                curve, first, second = self._curve(u), self._curve(u, 1), self._curve(u, 2)
                residual = curve-subset
                numerator = np.sum(residual*first, axis=1)
                denominator = np.sum(first**2+residual*second, axis=1)
                safe = np.where(np.abs(denominator) > 1e-12, denominator, 1.)
                step = np.clip(numerator/safe, -5., 5.)
                updated = np.clip(u-step, lower, upper)
                if np.max(np.abs(updated-u), initial=0.) < 1e-11:
                    u = updated
                    break
                u = updated
            curve, first, second = self._curve(u), self._curve(u, 1), self._curve(u, 2)
            speed = np.linalg.norm(first, axis=1)
            tangent = first/np.maximum(speed[:, None], 1e-12)
            normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
            residual = subset-curve
            d = np.sum(residual*normal, axis=1)
            along = np.sum(residual*tangent, axis=1)
            curvature = (first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0])/np.maximum(speed, 1e-12)**3
            metric = 1.-curvature*d
            coverage = has_branch & (np.abs(along) <= 1e-6)
            coordinate_valid = coverage & (speed > 1e-8) & (metric >= self.minimum_metric)
            # Keep the actual signed denominator for diagnosis; invalid charts
            # are not silently repaired into valid inverse geometry.
            safe_metric = np.where(np.abs(metric) > 1e-10, metric, 1.)
            jacobian = np.stack((tangent/safe_metric[:, None], normal), axis=1)
            output.append({'s': self._arc_at(u), 'd': d, 'jacobian': jacobian,
                           'heading': np.arctan2(tangent[:, 1], tangent[:, 0]),
                           'coverage': coverage, 'coordinate_valid': coordinate_valid,
                           'metric': metric, 'feet': curve})
        result = {key: np.concatenate([part[key] for part in output], axis=0) for key in output[0]}
        result['domain_valid'] = self.domain_contains(points)
        result['valid'] = result['coordinate_valid'] & (result['domain_valid'] if require_domain else True)
        return result

    def project_sequence(self, points, initial_s, *, require_domain=True,
                         max_step_m=None, initial_window_m=10.):
        """Project a time sequence without jumping to a nearby return branch.

        ``initial_s`` belongs to this chart.  Every subsequent normal-foot
        search is anchored to the preceding foot, with a movement-derived
        local window capped at ``max_step_m`` (20 m by default).  The first
        search uses ``initial_window_m`` (10 m).  Exceeding the local window
        is diagnosed as uncovered, never repaired by a global projection.
        """
        points = np.atleast_2d(np.asarray(points, float))
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('potential sequence must be a finite (N,2) array')
        previous_s = float(initial_s)
        if not np.isfinite(previous_s):
            raise ValueError('sequence initial_s must be a finite chart coordinate')
        cap = 20. if max_step_m is None else _positive(max_step_m, 'max_step_m')
        first_window = min(cap, _positive(initial_window_m, 'initial_window_m'))
        if len(points) == 0:
            return self.project(points, require_domain=require_domain)
        rows = []
        for index, point in enumerate(points):
            movement = 0. if index == 0 else float(np.linalg.norm(point-points[index-1]))
            local_window = first_window if index == 0 else min(cap, max(3., 2.*movement+1.))
            projected = self.project(point, anchors=previous_s,
                require_domain=require_domain, window_m=local_window)
            rows.append(projected)
            previous_s = float(projected['s'][0])
        return {key: np.concatenate([row[key] for row in rows], axis=0) for key in rows[0]}

    def repulsion(self, points, source_positions, settings, attention_gain=1., *,
                  anchors=None, source_anchors=None):
        """Frenet ellipse energy, Cartesian gradient and sqrt-energy residual.

        Source coordinates are fixed when taking the candidate gradient.  A
        single source centre broadcasts to the batch.  No heading or physical
        rectangle/circle dimensions are accepted by this interface.
        """
        points = np.atleast_2d(np.asarray(points, float))
        source = np.broadcast_to(np.asarray(source_positions, float), points.shape)
        query = self.project(points, anchors=anchors)
        other = self.project(source, anchors=source_anchors, require_domain=False)
        return self.repulsion_from_projection(query, other, settings, attention_gain)

    def repulsion_from_projection(self, query, other, settings, attention_gain=1.):
        """Evaluate the same field from cached projections of THIS chart.

        A singleton source projection broadcasts to all query rows.  This
        helper avoids projecting the same MPC candidate for every actor; its
        masks, values and analytic derivatives match ``repulsion`` exactly.
        The caller owns the projections and must not mix frozen references.
        """
        a = _positive(settings.get('longitudinal_m', 20.), 'longitudinal_m')
        b = _positive(settings.get('lateral_m', 6.), 'lateral_m')
        peak = _positive(settings.get('peak', 8.), 'peak', zero=True)
        gain = _positive(attention_gain, 'attention_gain', zero=True)
        peak *= gain
        delta = np.column_stack((query['s']-other['s'], query['d']-other['d']))
        inverse = np.array([1./a**2, 1./b**2])
        rho = np.sqrt(np.sum(delta**2*inverse, axis=1))
        shape = compact_shape(rho)
        energy = peak*shape
        residual = np.sqrt(peak)*np.sqrt(shape)
        inside = rho < 1.
        t = np.maximum(1.-rho, 0.)
        basis = 1.+3.*rho+6.*rho**2
        # Analytic centre/cutoff limits avoid dividing by rho or sqrt(U).
        energy_sd = (-30.*peak*rho*t**2*inside)[:, None]*delta*inverse
        residual_sd = (-15.*np.sqrt(peak)*rho*np.sqrt(t)/np.sqrt(basis)*inside)[:, None]*delta*inverse
        gradient = np.einsum('ni,nij->nj', energy_sd, query['jacobian'])
        residual_jacobian = np.einsum('ni,nij->nj', residual_sd, query['jacobian'])
        valid = query['valid'] & other['coordinate_valid']
        for array in (energy, residual, gradient, residual_jacobian):
            array[~valid] = np.nan
        return {'energy': energy, 'gradient': gradient, 'residual': residual,
                'residual_jacobian': residual_jacobian, 'valid': valid, 'rho': rho,
                'ego_projection': query, 'source_projection': other,
                'peak': peak, 'longitudinal_m': a, 'lateral_m': b}

    def attraction(self, points, goal_position, settings=None, *, anchors=None,
                   goal_anchor=None, require_domain=True):
        """Frenet quadratic valley and its equivalent nonnegative residual.

        ``energy=sum(residual**2)-shift``.  The negative energy itself MUST NOT
        be passed into an interface that squares its argument.  ``shift`` is a
        constant energy-zero choice; it does not change trajectory gradients.
        """
        settings = settings or {}
        weight = _positive(settings.get('weight', settings.get('attraction_weight', 3.)), 'attraction weight', zero=True)
        longitudinal = _positive(settings.get('longitudinal_scale_m', 30.), 'longitudinal_scale_m')
        lateral = _positive(settings.get('lateral_scale_m', 15.), 'lateral_scale_m')
        shift = _positive(settings.get('shift', 6.), 'attraction shift', zero=True)
        query = self.project(points, anchors=anchors, require_domain=require_domain)
        goal = self.project(np.asarray(goal_position, float).reshape(1, 2),
                            anchors=goal_anchor, require_domain=False)
        scale = np.sqrt(weight/2.)/np.array([longitudinal, lateral])
        delta = np.column_stack((query['s']-goal['s'][0], query['d']-goal['d'][0]))
        residual = delta*scale
        jacobian = query['jacobian']*scale[None, :, None]
        energy = np.sum(residual**2, axis=1)-shift
        gradient = 2.*np.einsum('ni,nij->nj', residual, jacobian)
        valid = query['valid'] & goal['coordinate_valid'][0]
        for array in (energy, residual, jacobian, gradient):
            array[~valid] = np.nan
        return {'energy': energy, 'residual': residual, 'residual_jacobian': jacobian,
                'gradient': gradient, 'valid': valid, 'shift': shift,
                'ego_projection': query, 'goal_projection': goal}
