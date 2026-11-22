"""空间五次连接：位置、切向和曲率边界；不指定到达时间或速度。"""
from __future__ import annotations

import numpy as np

from .geometry import clean_polyline, polyline_lengths, wrap_angle


def spatial_samples(start, end, step=.5, anchors=()):
    """保留精确边界点，排除arange浮点尾部重合，避免曲率差分除零。"""
    start,end,step=float(start),float(end),float(step)
    if not np.isfinite([start,end,step]).all() or step<=0 or end-start<=1e-6:
        raise ValueError('空间参考采样区间必须有限且有正长度')
    # 1e-7米只用于数值节点合并，远小于车体几何与目标到达容差。
    fixed=[start]
    for x in sorted(set(float(x) for x in anchors)):
        if fixed[-1]+1e-7<x<end-1e-7:fixed.append(x)
    fixed.append(end)
    regular=np.arange(start,end,step)
    for x in fixed:regular=regular[np.abs(regular-x)>1e-7]
    return np.sort(np.r_[regular,fixed])


def quintic_offsets(distance, length, start, end):
    """给定两端(d,d',d'')；导数均对参考线弧长，量纲m、1、1/m。"""
    length = float(length)
    values = np.asarray([*start, *end, length], float)
    if not np.isfinite(values).all() or length <= 1e-6:
        raise ValueError('五次连接需要有限边界与正的连接距离')
    c = np.zeros(6)
    c[:3] = [start[0], length*start[1], .5*length**2*start[2]]
    rhs = [end[0]-sum(c[:3]), length*end[1]-c[1]-2*c[2],
           length**2*end[2]-2*c[2]]
    c[3:] = np.linalg.solve([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]],rhs)
    u = np.clip(np.asarray(distance, float)/length, 0., 1.)
    return np.polynomial.polynomial.polyval(u,c)


def frame_geometry(reference, s):
    """保持原折线坐标定义；曲率仅用于连接边界，不改写导航或冻结目标。"""
    reference = clean_polyline(reference)
    arc = polyline_lengths(reference)
    tangent = np.unwrap(np.arctan2(np.gradient(reference[:,1],arc),
                                  np.gradient(reference[:,0],arc)))
    curvature = np.gradient(tangent,arc)
    derivative = np.gradient(curvature,arc)
    return float(np.interp(s,arc,curvature)),float(np.interp(s,arc,derivative))


def initial_lateral_state(scene, reference, s, d, heading):
    """从实际车体状态连接。低速不计算yaw_rate/v，避免零速奇异。"""
    k, kp = frame_geometry(reference,s)
    velocity = np.asarray(scene.ego.velocity,float)
    speed = float(np.linalg.norm(velocity))
    course = float(np.arctan2(velocity[1],velocity[0])) if speed>.3 else float(scene.ego.heading)
    denominator = 1-k*d
    if denominator <= .05 or abs(float(wrap_angle(course-heading))) >= 1.2:
        raise ValueError('当前状态不满足单值前向Frenet连接条件')
    slope = denominator*np.tan(float(wrap_angle(course-heading)))
    # 已执行转角只提供局部曲率边界，不能用未来自车轨迹或隐藏状态。
    control = np.asarray(scene.diagnostics.get('previous_control',[]),float)
    if control.shape==(2,) and np.isfinite(control).all():
        steer = float(control[1])
        wheelbase = scene.ego.wheelbase_m or scene.ego.length
        rear = scene.ego.rear_axle_to_center_m or wheelbase/2
        curvature = np.sin(np.arctan(rear/wheelbase*np.tan(steer)))/rear
    elif speed>.3 and np.isfinite(scene.ego.acceleration).all():
        # 首帧没有执行转角缓存时，用当前运动学曲率，不读未来，不设曲率为零。
        ax,ay=np.asarray(scene.ego.acceleration,float)
        curvature=(velocity[0]*ay-velocity[1]*ax)/speed**3
    else:
        curvature=k/max(denominator,.05)
    second = (curvature*(denominator**2+slope**2)**1.5
              -k*denominator**2-kp*d*slope-2*k*slope**2)/denominator
    return float(d),float(slope),float(second)


class ContinuousReference:
    """Interpolating geometry for a sampled reference, with consistent normals.

    The parameter remains the frozen polyline's metre-labelled arc coordinate;
    source nodes are interpolated exactly. It is not a replacement task route
    or a change to the model's original Frenet target definition.
    """
    def __init__(self, reference):
        from scipy.interpolate import make_interp_spline
        self.points=clean_polyline(reference)
        self.arc=polyline_lengths(self.points)
        self.total=float(self.arc[-1])
        degree=5 if len(self.points)>=6 else 3 if len(self.points)>=4 else len(self.points)-1
        self.spline=make_interp_spline(self.arc,self.points,k=degree,axis=0)
        self.degree=degree

    def derivative(self, parameter, order=0):
        parameter=np.asarray(parameter,float)
        if order>self.degree:
            return np.zeros(parameter.shape+(2,))
        return np.asarray(self.spline(parameter,nu=order),float)

    def geometry(self, parameter):
        first=self.derivative(parameter,1);second=self.derivative(parameter,2);third=self.derivative(parameter,3)
        speed=np.linalg.norm(first,axis=-1)
        if np.any(speed<1e-6):
            raise ValueError('连续参考出现退化切向，不能生成五次目标连接')
        tangent=first/speed[...,None]
        normal=np.stack((-tangent[...,1],tangent[...,0]),axis=-1)
        cross=first[...,0]*second[...,1]-first[...,1]*second[...,0]
        curvature=cross/speed**3
        derivative=(first[...,0]*third[...,1]-first[...,1]*third[...,0])/speed**3
        derivative-=3.*cross*np.sum(first*second,axis=-1)/speed**5
        return dict(position=self.derivative(parameter),tangent=tangent,normal=normal,
            speed=speed,speed_derivative=np.sum(first*second,axis=-1)/speed,
            curvature=curvature,curvature_derivative=derivative)

    def project(self, positions, initial):
        """Local closest-point refinement; never jump a loop or parallel road."""
        positions=np.asarray(positions,float)
        q=np.asarray(initial,float).copy()
        radius=max(1.,float(np.max(np.diff(self.arc)))*1.5)
        lower=np.maximum(0.,q-radius);upper=np.minimum(self.total,q+radius)
        for _ in range(10):
            delta=self.derivative(q)-positions
            first=self.derivative(q,1);second=self.derivative(q,2)
            denominator=np.sum(first*first+delta*second,axis=-1)
            if np.any(denominator<=1e-8):
                raise ValueError('连续参考局部投影失效，不允许跨段重定位')
            change=np.sum(delta*first,axis=-1)/denominator
            q=np.clip(q-change,lower,upper)
            if np.max(np.abs(change))<1e-10:break
        return q


def continuous_initial_lateral_state(scene, curve, parameter, lateral):
    """Actual heading/curvature boundary in the continuous reference parameter."""
    frame=curve.geometry(float(parameter))
    h=float(frame['speed']);hp=float(frame['speed_derivative'])
    k=float(frame['curvature']);kp=float(frame['curvature_derivative'])
    heading=float(np.arctan2(frame['tangent'][1],frame['tangent'][0]))
    velocity=np.asarray(scene.ego.velocity,float);speed=float(np.linalg.norm(velocity))
    course=float(np.arctan2(velocity[1],velocity[0])) if speed>.3 else float(scene.ego.heading)
    denominator=1.-k*lateral
    if denominator<=.05 or abs(float(wrap_angle(course-heading)))>=1.2:
        raise ValueError('当前状态不满足连续参考单值前向连接条件')
    a=h*denominator;slope=a*np.tan(float(wrap_angle(course-heading)))
    control=np.asarray(scene.diagnostics.get('previous_control',[]),float)
    if control.shape==(2,) and np.isfinite(control).all():
        wheelbase=scene.ego.wheelbase_m or scene.ego.length
        rear=scene.ego.rear_axle_to_center_m or wheelbase/2.
        curvature=np.sin(np.arctan(rear/wheelbase*np.tan(float(control[1]))))/rear
    elif speed>.3 and np.isfinite(scene.ego.acceleration).all():
        ax,ay=np.asarray(scene.ego.acceleration,float)
        curvature=(velocity[0]*ay-velocity[1]*ax)/speed**3
    else:
        curvature=k/denominator
    tangential_second=hp*denominator-h*kp*lateral-2.*h*k*slope
    second=(curvature*(a*a+slope*slope)**1.5+slope*tangential_second)/a-h*h*k*denominator
    return float(lateral),float(slope),float(second)
