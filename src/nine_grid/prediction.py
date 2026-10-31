"""仅使用当前可见观测的运动滤波与公开道路约束预测。"""
from __future__ import annotations

from functools import lru_cache
import math
import time
import numpy as np
import casadi as ca
from scipy.linalg import expm
from scipy.interpolate import CubicSpline
import shapely
from shapely.geometry import Polygon
from .contracts import Actor, PredictionBundle, SceneSnapshot

_CTRA_PREPARATION_SECONDS=0.


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def ctra_derivative(z):
    if isinstance(z,(ca.SX,ca.MX)):
        return ca.vertcat(z[2]*ca.cos(z[4]),z[2]*ca.sin(z[4]),z[3],0,z[5],0)
    _, _, v, a, chi, omega = z
    return np.array([v * np.cos(chi), v * np.sin(chi), a, 0, omega, 0])


def ctra_jacobian(z):
    symbolic=isinstance(z,(ca.SX,ca.MX))
    f=type(z).zeros(6,6) if symbolic else np.zeros((6,6))
    trig=ca if symbolic else np
    f[0, 2], f[0, 4] = trig.cos(z[4]), -z[2] * trig.sin(z[4])
    f[1, 2], f[1, 4] = trig.sin(z[4]), z[2] * trig.cos(z[4])
    f[2, 3], f[4, 5] = 1, 1
    return f


@lru_cache(maxsize=32)
def _ctra_kernel(substeps):
    """缓存同一联合RK4计算图；dt、积分步长和两个噪声均为显式输入。"""
    global _CTRA_PREPARATION_SECONDS
    started=time.perf_counter()
    packed=ca.SX.sym('state_and_covariance',42)
    settings=ca.SX.sym('dt_step_qj_qw',4)
    z=packed[:6];covariance=ca.reshape(packed[6:],6,6)
    remaining=settings[0]
    q=ca.diag(ca.vertcat(0,0,0,settings[2],0,settings[3]))
    def rhs(x,p):
        f=ctra_jacobian(x)
        return ctra_derivative(x),f@p+p@f.T+q
    for _ in range(substeps):
        active=remaining>1e-12
        braking_stop=ca.logic_and(active,ca.logic_and(z[2]<=1e-10,z[3]<0))
        z[2]=ca.if_else(braking_stop,0,z[2]);z[3]=ca.if_else(braking_stop,0,z[3])
        stationary=ca.logic_and(active,ca.logic_and(z[2]<=1e-10,z[3]<=0))
        z[5]=ca.if_else(stationary,0,z[5])
        h=ca.fmin(settings[1],remaining)
        braking=ca.logic_and(z[3]<0,z[2]>0)
        stop_time=z[2]/ca.fmax(-z[3],1e-300)
        h=ca.if_else(active,ca.if_else(braking,ca.fmin(h,stop_time),h),0)
        a,ap=rhs(z,covariance)
        b,bp=rhs(z+h*a/2,covariance+h*ap/2)
        c,cp=rhs(z+h*b/2,covariance+h*bp/2)
        d,dp=rhs(z+h*c,covariance+h*cp)
        z=z+h*(a+2*b+2*c+d)/6
        covariance=covariance+h*(ap+2*bp+2*cp+dp)/6
        remaining=remaining-h
        negative=ca.logic_and(active,z[2]<0)
        z[2]=ca.if_else(negative,0,z[2]);z[3]=ca.if_else(negative,0,z[3]);z[5]=ca.if_else(negative,0,z[5])
    modulo=ca.fmod(z[4]+np.pi,2*np.pi)
    z[4]=ca.if_else(modulo<0,modulo+2*np.pi,modulo)-np.pi
    covariance=(covariance+covariance.T)/2
    function=ca.Function('ctra_joint_rk4',[packed,settings],[ca.vertcat(z,ca.vec(covariance))])
    _CTRA_PREPARATION_SECONDS+=time.perf_counter()-started
    return function


def _ctra_substeps(dt,integration_step):
    # 一次减速停车可在固定积分步中额外插入一个子步；停车后加速度归零。
    return max(0,int(math.ceil(max(0.,dt)/integration_step)))+1


def ctra_propagate(z, covariance, dt, qj=1., qw=.04, integration_step=.02):
    """同一同步RK4与停车切分，数值计算图替代逐级Python调度。"""
    function=_ctra_kernel(_ctra_substeps(float(dt),integration_step))
    result=np.asarray(function(np.r_[z,np.asarray(covariance).reshape(-1,order='F')],
                               [dt,integration_step,qj,qw])).reshape(-1)
    return result[:6],result[6:].reshape(6,6,order='F')


@lru_cache(maxsize=32)
def _ctra_accumulator(dt,count,qj,qw,integration_step):
    """完整参数标识缓存批量传播；每个逻辑步仍单独wrap和协方差对称化。"""
    global _CTRA_PREPARATION_SECONDS
    kernel=_ctra_kernel(_ctra_substeps(dt,integration_step))
    started=time.perf_counter()
    function=kernel.mapaccum(count)
    settings=np.tile(np.array([dt,integration_step,qj,qw])[:,None],(1,count))
    _CTRA_PREPARATION_SECONDS+=time.perf_counter()-started
    return function,settings


def _ctra_sequence(z,covariance,dt,count,qj=1.,qw=.04,integration_step=.02):
    if not count:
        return np.empty((0,42))
    function,settings=_ctra_accumulator(float(dt),int(count),float(qj),float(qw),float(integration_step))
    return np.asarray(function(np.r_[z,np.asarray(covariance).reshape(-1,order='F')],settings)).T


def joseph_update(z, p, measurement, indices, variances, angle_index=None):
    h = np.eye(len(z))[indices]
    # 同一速度、加速度推导出的运动观测存在相关噪声，不能强行独立。
    r = np.asarray(variances,float)
    if r.ndim == 1:
        r = np.diag(r)
    innovation = measurement - h @ z
    if angle_index in indices:
        k = indices.index(angle_index)
        innovation[k] = wrap(innovation[k])
    k = np.linalg.solve(h @ p @ h.T + r, h @ p).T
    z = z + k @ innovation
    i = np.eye(len(z)) - k @ h
    p = i @ p @ i.T + k @ r @ k.T
    return z, (p+p.T)/2


def motion_measurement(actor, cfg):
    """当前可见速度与加速度的运动观测及一阶噪声传播，不差分侧偏跳变。"""
    noise=cfg.get('measurement_std',{})
    rp=noise.get('position_m',.05)**2
    rv=noise.get('speed_mps',.1)**2
    ra=noise.get('acceleration_mps2',.2)**2
    rh=noise.get('motion_heading_rad',.02)**2
    rw=noise.get('yaw_rate_radps',.03)**2
    velocity=np.asarray(actor.velocity,float);acceleration=np.asarray(actor.acceleration,float)
    speed=float(np.linalg.norm(velocity))
    moving=speed>=cfg.get('low_speed_mps',.3)
    chi=float(np.arctan2(velocity[1],velocity[0])) if moving else float(actor.heading)
    unit=np.array([np.cos(chi),np.sin(chi)]);normal=np.array([-unit[1],unit[0]])
    longitudinal=float(unit@acceleration)
    omega=float(normal@acceleration/speed) if moving else 0.
    measurement=np.array([*actor.position,speed,longitudinal,chi,omega])
    covariance=np.diag([rp,rp,rv,ra,rh,cfg.get('initial_turn_rate_variance',.1)])
    if moving:
        # 原始速度噪声由速度大小与运动方向噪声变换到世界平面。
        velocity_covariance=rv*np.outer(unit,unit)+speed**2*rh*np.outer(normal,normal)
        raw_covariance=np.zeros((4,4));raw_covariance[:2,:2]=velocity_covariance
        raw_covariance[2:,2:]=ra*np.eye(2)
        jacobian=np.zeros((4,4))
        jacobian[0,:2]=unit
        jacobian[1,:2]=omega*normal;jacobian[1,2:]=unit
        jacobian[2,:2]=normal/speed
        jacobian[3,:2]=-omega*unit/speed-longitudinal*normal/speed**2
        jacobian[3,2:]=normal/speed
        covariance[2:,2:]=jacobian@raw_covariance@jacobian.T
        covariance[5,5]+=rw
    return measurement,covariance,moving


@lru_cache(maxsize=32)
def singer_matrices(dt, tau=1., sigma=1.5):
    """Van Loan 块指数精确离散化每轴 Singer 模型。"""
    a = np.array([[0., 1, 0], [0, 0, 1], [0, 0, -1/tau]])
    q = np.diag([0., 0, 2*sigma*sigma/tau])
    m = np.block([[a, q], [np.zeros((3, 3)), -a.T]]) * dt
    e = expm(m)
    f = e[:3, :3]
    return f, e[:3, 3:] @ f.T


class _Curve:
    """同一自然三次样条系数及几何量，避免标量求值反复进入通用插值器。"""
    def __init__(self,line):
        self.line=line
        self.segments=np.diff(line,axis=0)
        self.length=np.linalg.norm(self.segments,axis=1)
        self.arc=np.r_[0.,np.cumsum(self.length)]
        self.coefficients=CubicSpline(self.arc,line,axis=0,bc_type='natural',extrapolate=False).c

    def evaluate(self,ss):
        ss=np.clip(ss,0,self.arc[-1])
        index=np.clip(np.searchsorted(self.arc,ss,side='right')-1,0,len(self.arc)-2)
        r=np.asarray(ss-self.arc[index])[...,None]
        a,b,c,d=self.coefficients[:,index]
        xy=((a*r+b)*r+c)*r+d
        first=(3*a*r+2*b)*r+c
        second=6*a*r+2*b
        return xy,first,second

    def metric(self,ss):
        """逐步纵向积分只需要标量曲率/尺度，不构造未使用的位置和航向。"""
        position=min(max(float(ss),0.),self.arc[-1])
        index=min(max(int(np.searchsorted(self.arc,position,side='right'))-1,0),len(self.arc)-2)
        r=position-self.arc[index]
        a,b,c,_=self.coefficients[:,index]
        dx=(3*a[0]*r+2*b[0])*r+c[0];dy=(3*a[1]*r+2*b[1])*r+c[1]
        ddx=6*a[0]*r+2*b[0];ddy=6*a[1]*r+2*b[1]
        scale=math.sqrt(dx*dx+dy*dy)
        return (dx*ddy-dy*ddx)/max(scale,1e-8)**3,scale


@lru_cache(maxsize=64)
def _cached_curve(shape, content):
    """仅缓存最近的公开参考，不为每帧复制整张地图或保存交通真值。"""
    line=np.frombuffer(content,dtype=np.float64).reshape(shape)
    line=line[np.r_[True,np.linalg.norm(np.diff(line,axis=0),axis=1)>1e-8]]
    if len(line)<2:
        raise ValueError('公开参考至少需要两个不同位置')
    return _Curve(line)


def _curve(line):
    if isinstance(line,_Curve):
        return line
    line=np.ascontiguousarray(line,dtype=np.float64)
    return _cached_curve(line.shape,line.tobytes())


def _curve_state(line,ss):
    """位置、切向、曲率共享同一 C2 曲线，保留非严格弧长参数的尺度 g。"""
    xy,first,second=_curve(line).evaluate(ss)
    scale=np.linalg.norm(first,axis=-1)
    heading=np.arctan2(first[...,1],first[...,0])
    curvature=(first[...,0]*second[...,1]-first[...,1]*second[...,0])/np.maximum(scale,1e-8)**3
    return xy,heading,curvature,scale


def _projections(points,line):
    curve=_curve(line)
    line,arc,segments,length=curve.line,curve.arc,curve.segments,curve.length
    points=np.asarray(points)
    t = np.clip(np.sum((points[:,None]-line[:-1])*segments, axis=2)/np.maximum(length**2, 1e-12), 0, 1)
    closest = line[:-1]+t[...,None]*segments
    i = np.argmin(np.linalg.norm(closest-points[:,None], axis=2),axis=1)
    # 先找最近折线段，再在相邻局部曲线上优化正交投影，不跨环跳到远处分支。
    lower,upper=arc[np.maximum(0,i-1)],arc[np.minimum(len(arc)-1,i+2)]
    ss=arc[i]+t[np.arange(len(points)),i]*length[i]
    active=np.ones(len(points),dtype=bool)
    for _ in range(6):
        xy,first,second=curve.evaluate(ss);offset=xy-points
        denominator=np.sum(first*first+offset*second,axis=1)
        valid=active&(denominator>1e-10)
        updated=np.clip(ss-np.sum(offset*first,axis=1)/np.where(valid,denominator,1),lower,upper)
        converged=abs(updated-ss)<1e-9
        ss=np.where(valid,updated,ss)
        active=valid&~converged
        if not active.any():break
    foot,heading,_,_=_curve_state(curve,ss)
    d=np.sum((points-foot)*np.column_stack([-np.sin(heading),np.cos(heading)]),axis=1)
    return ss,d,heading


def _projection(p,line):
    return tuple(value[0] for value in _projections(np.asarray(p)[None],line))


def _at(line, ss):
    xy,heading,_,_=_curve_state(line,ss)
    return xy,heading


def _polynomial_peak(coeff,duration,order):
    derivative=np.polynomial.polynomial.polyder(coeff,order)
    next_derivative=np.polynomial.polynomial.polyder(derivative)
    roots=np.polynomial.polynomial.polyroots(next_derivative) if len(next_derivative)>1 else []
    times=[0.,duration]+[float(root.real) for root in roots if abs(root.imag)<1e-8 and 0<root.real<duration]
    return float(np.max(abs(np.polynomial.polynomial.polyval(times,derivative))))


def lateral_convergence(d,vd,ad,target,times,cfg):
    """初态超限时先完成不可避免的 jerk 恢复，再接原五次收敛；不伪造初态。"""
    d,vd,ad,target=map(float,(d,vd,ad,target))
    nominal=float(cfg.get('lateral_speed_limit_mps',3.))
    acceleration_limit=max(float(cfg.get('lateral_acceleration_limit_mps2',4.)),abs(ad))
    jerk_limit=float(cfg.get('lateral_jerk_limit_mps3',10.))
    recovery=abs(vd)>=nominal and vd*ad>0
    sign=float(np.sign(vd)) if recovery else 0.
    tau=abs(ad)/jerk_limit if recovery else 0.
    recovery_coeff=np.array([d,vd,.5*ad,-sign*jerk_limit/6])
    end_d=float(np.polynomial.polynomial.polyval(tau,recovery_coeff))
    end_v=float(np.polynomial.polynomial.polyval(tau,np.polynomial.polynomial.polyder(recovery_coeff)))
    end_a=0. if recovery else ad
    speed_limit=max(nominal,abs(end_v))
    diagnostic={'initial_lateral_position_m':float(d),'initial_lateral_speed_mps':float(vd),
                'initial_lateral_acceleration_mps2':float(ad),'target_lateral_position_m':float(target),
                'recovery_applied':recovery,'recovery_duration_s':float(tau),
                'unavoidable_peak_lateral_speed_mps':float(abs(end_v)),
                'lateral_speed_bound_mps':float(speed_limit),'lateral_acceleration_bound_mps2':float(acceleration_limit),
                'lateral_jerk_bound_mps3':jerk_limit}
    best_violation=None
    for duration in np.arange(cfg.get('lane_change_duration_min_s',1.),cfg.get('lane_change_duration_max_s',5.)+.001,cfg.get('lane_change_duration_step_s',.2)):
        remaining=duration-tau
        if remaining<=1e-8:continue
        mat=np.array([[remaining**3,remaining**4,remaining**5],[3*remaining**2,4*remaining**3,5*remaining**4],
                      [6*remaining,12*remaining**2,20*remaining**3]])
        coeff=np.r_[end_d,end_v,.5*end_a,np.linalg.solve(mat,[target-end_d-end_v*remaining-.5*end_a*remaining**2,-end_v-end_a*remaining,-end_a])]
        peaks=[_polynomial_peak(coeff,remaining,order) for order in (1,2,3)]
        ratio=max(peaks[0]/speed_limit,peaks[1]/acceleration_limit,peaks[2]/jerk_limit)
        best_violation=ratio if best_violation is None else min(best_violation,ratio)
        if peaks[0]>speed_limit+1e-6 or peaks[1]>acceleration_limit+1e-6 or peaks[2]>jerk_limit+1e-6:
            continue
        tt=np.asarray(times,dtype=float)
        local=np.clip(tt-tau,0,remaining)
        values=[np.polynomial.polynomial.polyval(local,np.polynomial.polynomial.polyder(coeff,order)) for order in (0,1,2)]
        if recovery:
            prefix=tt<tau
            for order in (0,1,2):
                values[order][prefix]=np.polynomial.polynomial.polyval(tt[prefix],np.polynomial.polynomial.polyder(recovery_coeff,order))
        values[1][tt>=duration]=0.;values[2][tt>=duration]=0.
        diagnostic.update(convergence_duration_s=float(duration),recovery_end_position_m=end_d,recovery_end_speed_mps=end_v,
                          convergence_coefficients=coeff.tolist(),recovery_coefficients=recovery_coeff.tolist())
        return values,diagnostic
    diagnostic['minimum_constraint_ratio']=best_violation
    return None,diagnostic


class Predictor:
    """每个 track 保存滤波状态；不保存遮挡对象的可执行预测。"""
    def __init__(self, config=None):
        self.cfg = (config or {}).get('prediction', {})
        self.tracks = {}
        self.preparation_seconds=0.

    def _observe(self, actor, time):
        noise=self.cfg.get('measurement_std',{})
        rp=noise.get('position_m',.05)**2
        rv=noise.get('speed_mps',.1)**2
        ra=noise.get('acceleration_mps2',.2)**2
        old = self.tracks.get(actor.id)
        if actor.kind == 'pedestrian':
            measurement = np.array([actor.position[0], actor.velocity[0], actor.acceleration[0], actor.position[1], actor.velocity[1], actor.acceleration[1]])
            z, p = measurement.copy(), np.diag(self.cfg.get('pedestrian_initial_variances',[.1]*6))
            if old is not None:
                dt = time-old['time']
                if dt > 0:
                    f, q = singer_matrices(round(dt, 8), self.cfg.get('singer_tau_s', 1.), self.cfg.get('singer_accel_std_mps2', 1.5))
                    ff = np.kron(np.eye(2), f)
                    z, p = ff@old['z'], ff@old['p']@ff.T + np.kron(np.eye(2), q)
                    z, p = joseph_update(z, p, measurement, list(range(6)), [rp,rv,ra]*2)
                else:
                    z, p = old['z'], old['p']
        else:
            measurement,measurement_covariance,moving=motion_measurement(actor,self.cfg)
            z,p=measurement.copy(),measurement_covariance.copy()
            if old is not None:
                dt = time-old['time']
                if dt > 0:
                    z, p = ctra_propagate(old['z'], old['p'], dt, self.cfg.get('ctra_jerk_psd', 1.), self.cfg.get('ctra_yaw_accel_psd', .04),self.cfg.get('integration_step_s',.02))
                    inds=[0,1,2,3,4,5] if moving else [0,1,2,3]
                    z,p=joseph_update(z,p,measurement[inds],inds,measurement_covariance[np.ix_(inds,inds)],4)
                    z[4] = wrap(z[4])
                else:
                    z, p = old['z'], old['p']
        track = dict(old or {})
        track['observation_dt'] = max(0.,time-old['time']) if old else 0.
        track.update(z=z, p=p, time=time)
        self.tracks[actor.id] = track
        return track

    def _mapped(self, actor, scene, track, free, times):
        # vT 仅用于首轮准备参考；内侧偏移的实际参考弧长可能更长。若公开
        # 后继仍在，补齐后用同一观测重算，不重复累计变道计时或篡改初态。
        original=dict(track)
        required_distance=0.
        for retries in range(len(scene.lanes)+1):
            working=dict(original)
            result,diagnostic=self._mapped_once(actor,scene,working,free,times,required_distance)
            if not diagnostic.get('reference_extension_available',False):
                track.update(working)
                diagnostic['reference_extension_recomputations']=retries
                return result,diagnostic
            required_distance=max(required_distance,diagnostic['required_reference_parameter_m'],
                                  diagnostic['available_reference_parameter_m'])+5.
        raise RuntimeError('公开参考补充未收敛；车道图或参考长度异常')

    def _mapped_once(self, actor, scene, track, free, times, required_distance):
        lanes = {l.id: l for l in scene.lanes}
        if actor.lane_id not in lanes or actor.kind == 'pedestrian':
            return free, {'map_applied': False, 'valid':True,'reason':'行人或无机动车道关联，使用观测物理运动'}
        lane = lanes[actor.lane_id]
        last_lane=lanes.get(track.get('observed_lane'))
        if last_lane is not None and last_lane.id!=lane.id and lane.id not in (last_lane.left,last_lane.right):
            migrated={}
            for key in ('source_lane','target_lane'):
                identity=track.get(key)
                if identity is None:continue
                old=lanes.get(identity)
                preferred=lane.id if identity==last_lane.id else lane.left if identity==last_lane.left else lane.right if identity==last_lane.right else None
                if old is not None and preferred in old.successors:
                    migrated[key]=preferred
                elif old is not None and lane.id in old.successors:
                    migrated[key]=lane.id
                else:
                    migrated[key]=None
            track.update(migrated)
            if migrated and migrated.get('target_lane') is None:
                track.update(source_lane=None,target_lane=None,lateral_duration=0.,settled=0.,observation_dt=0.)
        track['observed_lane']=lane.id
        if np.max(free[:,3])<1e-8:
            # 停车不意味着向车道中心横移；静止车体朝向仍是可观测状态。
            return np.tile([*actor.position,actor.heading,0.],(len(times),1)), {'map_applied':True,'valid':True,'successors':[],'stopped':True}
        line = lane.center.copy()
        predecessors=[candidate for candidate in lanes.values() if lane.id in candidate.successors and candidate.id!=lane.id]
        def predecessor_cost(candidate):
            h0=math.atan2(*(candidate.center[-1]-candidate.center[-2])[::-1])
            h1=math.atan2(*(lane.center[1]-lane.center[0])[::-1])
            angle=abs(float(wrap(h1-h0)))
            return float(np.linalg.norm(candidate.center[-1]-lane.center[0]))+5*angle,candidate.id
        predecessors=[candidate for candidate in predecessors if abs(float(wrap(
            math.atan2(*(candidate.center[-1]-candidate.center[-2])[::-1])-
            math.atan2(*(lane.center[1]-lane.center[0])[::-1]))))<np.pi/2]
        predecessor=min(predecessors,key=predecessor_cost) if predecessors else None
        if predecessor is not None:
            # 接缝导数必须带有真实公开前驱上下文；禁止在当前段首强置曲率为零。
            prefix=predecessor.center[-min(12,len(predecessor.center)):]
            line=np.vstack([prefix,line[1:] if np.linalg.norm(prefix[-1]-line[0])<1e-6 else line])
        s, d, heading = _projection(actor.position, line)
        # 当前公开车道的后继以短时运动几何选取，绝不读取真实路线。
        visited = {lane.id} | ({predecessor.id} if predecessor is not None else set())
        successors = []
        while np.sum(np.linalg.norm(np.diff(line, axis=0), axis=1))-s < max(20., float(free[:, 3].max())*times[-1]+5,required_distance):
            options = [lanes[k] for k in lane.successors if k in lanes and k not in visited]
            if not options:
                break
            prefix=min(int(round(self.cfg.get('map_successor_prefix_s',.8)/(times[1]-times[0]))),len(free)-1)
            indexes=np.unique(np.linspace(1,max(1,prefix),8,dtype=int))
            def geometry_cost(candidate):
                endpoint_gap=float(np.linalg.norm(lane.center[-1]-candidate.center[0]))
                outgoing=lane.center[-1]-lane.center[-2]
                incoming=candidate.center[1]-candidate.center[0]
                angle=abs(float(wrap(np.arctan2(incoming[1],incoming[0])-np.arctan2(outgoing[1],outgoing[0]))))
                return round(endpoint_gap,6),round(angle,6),candidate.id
            def motion_cost(candidate):
                trial=np.vstack([line,candidate.center[1:] if np.linalg.norm(line[-1]-candidate.center[0])<1e-6 else candidate.center])
                curve=_curve(trial)
                probes=free[indexes,:2]
                ps,_,ph=_projections(probes,curve)
                distance=np.linalg.norm(_at(curve,ps)[0]-probes,axis=1)
                errors=(distance/max(candidate.width/2,.1))**2+(
                    wrap(ph-free[indexes,2])/self.cfg.get('map_successor_heading_scale_rad',np.pi/6))**2
                return float(np.mean(errors)),*geometry_cost(candidate)
            if not successors:
                # 同一后继道路的平行车道只留一个几何延续；已锁定的横向
                # 收敛负责换道，不能在参考选择中再叠加一次横移。
                remaining={candidate.id for candidate in options}
                representatives=[]
                while remaining:
                    group={min(remaining)};pending=list(group)
                    while pending:
                        identity=pending.pop()
                        neighbors={lanes[identity].left,lanes[identity].right}|{
                            other for other in remaining if identity in (lanes[other].left,lanes[other].right)}
                        found=(neighbors&remaining)-group
                        group.update(found);pending.extend(found)
                    remaining-=group
                    representatives.append(min((lanes[identity] for identity in group),key=geometry_cost))
                # 仅第一层真实分叉使用当前运动证据；远端不能重复使用原
                # 时刻的短前缀，也不能让合法出口间隙绝对压倒驶出证据。
                lane=min(representatives,key=motion_cost)
            else:
                lane=min(options,key=geometry_cost)
            visited.add(lane.id)
            successors.append(lane.id)
            line = np.vstack([line, lane.center[1:] if np.linalg.norm(line[-1]-lane.center[0])<1e-6 else lane.center])
        lane = lanes[actor.lane_id]
        # 后继拼接会影响接缝附近的样条导数，初态须按最终同一参考重新投影。
        reference=_curve(line)
        s,d,heading=_projection(actor.position,reference)
        vd = float(np.dot(actor.velocity, [-np.sin(heading), np.cos(heading)]))
        _,_,curvature,metric=_curve_state(reference,s)
        curvature,metric=float(curvature),float(metric)
        q=1-curvature*d
        if q<.1 or metric<.1:
            return free, {'map_applied':False,'valid':False,'reason':'Frenet 投影奇异'}
        tangential_speed=float(np.dot(actor.velocity,[np.cos(heading),np.sin(heading)]))
        sd=tangential_speed/(q*metric)
        # d 的二阶导数由同一连续曲线得到；不可将向心加速度当作横向换道加速度。
        ad = float(np.dot(actor.acceleration, [-np.sin(heading), np.cos(heading)]))-curvature*tangential_speed**2/q
        target_id = track.get('target_lane')
        elapsed=track.get('observation_dt',0.)
        if target_id is not None and target_id in lanes:
            _, offset, _ = _projection(actor.position, lanes[target_id].center)
            if abs(offset) < self.cfg.get('lane_change_finish_offset_m',.35) and abs(vd) < self.cfg.get('lane_change_finish_lateral_speed_mps',.2):
                track['settled'] = track.get('settled', 0.) + elapsed
                if track['settled'] >= self.cfg.get('lane_change_finish_hold_s',.5):
                    track['target_lane'] = None
            else:
                track['settled'] = 0.
        elif abs(vd) >= self.cfg.get('lane_change_min_lateral_speed_mps',.3):
            track['lateral_duration'] = track.get('lateral_duration', 0.)+elapsed
            candidate = lane.left if vd > 0 else lane.right
            if track['lateral_duration'] >= self.cfg.get('lane_change_detect_hold_s',.3) and candidate in lanes:
                track['target_lane'] = candidate
                track['source_lane'] = lane.id
                target_id = candidate
        else:
            track['lateral_duration'] = 0.
        cancelled_target=None
        target_side='left' if target_id==lane.left and target_id else 'right' if target_id==lane.right and target_id else None
        if target_side and target_id in lanes:
            target_segment=target_id
            for identity in successors:
                next_target=getattr(lanes[identity],target_side)
                if next_target not in lanes or next_target not in lanes[target_segment].successors:
                    # 分支不再有对应平行目标，撤销过期锁定。仅改变终端d，
                    # 仍从真实d、vd、ad连续收敛，禁止将当前车体瞬移到中心。
                    cancelled_target=target_id
                    track.update(target_lane=None,source_lane=None,lateral_duration=0.,settled=0.)
                    target_id=None
                    break
                target_segment=next_target
        target_d = 0.
        if target_id in lanes:
            target_point = _at(lanes[target_id].center, _projection(actor.position, lanes[target_id].center)[0])[0]
            target_d = _projection(target_point, reference)[1]
        lateral,lateral_diagnostic=lateral_convergence(d,vd,ad,target_d,times,self.cfg)
        geometry_diagnostic={'reference_curvature_per_m':curvature,'reference_parameter_scale':metric,
                             'reference_heading_rad':float(heading),'predecessor':predecessor.id if predecessor else None,
                             'successors':successors,'lateral':lateral_diagnostic,
                             'cancelled_target_lane':cancelled_target}
        if lateral is None:
            return free, {'map_applied': False, 'valid':False,'reason': '横向收敛无可行参数',**geometry_diagnostic}
        ds,lateral_speeds,_=lateral
        stopped=np.flatnonzero(free[:,3]<=1e-8)
        if len(stopped) and stopped[0]>0:
            # 预测制动至停车后，横向收敛同样停止，不让矩形原地漂移。
            stop=int(stopped[0])
            ds[stop:]=ds[stop]
            lateral_speeds[stop:]=0.
        # 对非严格弧长样条参数逐步积分；曲率与参数尺度随位置更新。
        speeds = free[:, 3]
        if np.any(abs(lateral_speeds)>speeds+1e-6):
            return free,{'map_applied':False,'valid':False,'reason':'横向恢复速度超过滤波总速度',**geometry_diagnostic}
        longitudinal=np.empty(len(times));longitudinal[0]=s
        def parameter_rate(position,index):
            kap,g=reference.metric(position)
            denominator=(1-kap*ds[index])*g
            if denominator<.1:return None
            return float(np.sqrt(max(0.,speeds[index]**2-lateral_speeds[index]**2))/denominator)
        for index,dt in enumerate(np.diff(times)):
            first=max(0.,sd) if index==0 else parameter_rate(longitudinal[index],index)
            if first is None:
                return free,{'map_applied':False,'valid':False,'reason':'预测路径进入 Frenet 奇异区域',**geometry_diagnostic}
            last=parameter_rate(longitudinal[index]+dt*first,index+1)
            if last is None:
                return free,{'map_applied':False,'valid':False,'reason':'预测路径进入 Frenet 奇异区域',**geometry_diagnostic}
            longitudinal[index+1]=longitudinal[index]+dt*(first+last)/2
        length = reference.arc[-1]
        if longitudinal[-1] > length+.1:
            terminal=lanes[successors[-1]] if successors else lane
            extension_available=any(identity in lanes and identity not in visited for identity in terminal.successors)
            return free, {'map_applied': False, 'valid':False,'reason': '公开道路不足以覆盖时域',
                          'reference_extension_available':extension_available,
                          'available_reference_parameter_m':float(length-s),
                          'required_reference_parameter_m':float(longitudinal[-1]-s),**geometry_diagnostic}
        xy, tangents = _at(reference, longitudinal)
        xy = xy+np.stack([-np.sin(tangents),np.cos(tangents)],axis=1)*ds[:,None]
        direction = np.gradient(xy, times, axis=0)
        chi = np.arctan2(direction[:,1],direction[:,0])
        body_offset = wrap(actor.heading-track['z'][4])
        headings=wrap(chi+body_offset)
        for i in range(len(headings)):
            if np.linalg.norm(direction[i])<self.cfg.get('low_speed_mps',.3):
                headings[i]=actor.heading if i==0 else headings[i-1]
        result = np.column_stack([xy, headings, speeds])
        result[0] = [*actor.position,actor.heading,np.linalg.norm(actor.velocity)]
        allowed_ids=visited|{identity for identity in (track.get('source_lane'),track.get('target_lane')) if identity in lanes}
        # 横向目标在后续道路段也须拥有同侧公开车道；参考保持源车道，不
        # 再额外切换到目标车道后叠加同一横向位移。
        side='left' if track.get('target_lane')==lane.left and lane.left else 'right' if track.get('target_lane')==lane.right and lane.right else None
        if side:
            allowed_ids.update(getattr(lanes[identity],side) for identity in visited if getattr(lanes[identity],side) in lanes)
        corridor=shapely.union_all([Polygon(lanes[identity].polygon) for identity in allowed_ids])
        legal=shapely.covers(corridor,shapely.points(xy))
        if not np.all(legal):
            return free,{'map_applied':False,'valid':False,'reason':'连续地图预测越出所选公开车道区域',
                         'first_outside_time_s':float(times[np.flatnonzero(~legal)[0]]),**geometry_diagnostic}
        return result, {'map_applied': True, 'valid':True,'target_lane': track.get('target_lane'),**geometry_diagnostic}

    def predict(self, scene: SceneSnapshot, horizon=None, step=None):
        preparation_before=_CTRA_PREPARATION_SECONDS
        horizon=self.cfg.get('diagnostic_horizon_s',3.9) if horizon is None else horizon
        step=self.cfg.get('step_s',.1) if step is None else step
        times = np.arange(round(horizon/step)+1)*step
        trajectories, actors, diag = {}, {}, {}
        for actor in scene.actors:
            if not actor.visible:
                continue
            actors[actor.id] = actor
            if actor.kind == 'static_obstacle':
                trajectories[actor.id] = np.tile([*actor.position,actor.heading,0.],(len(times),1))
                continue
            track = self._observe(actor,scene.time)
            z, p = track['z'].copy(), track['p'].copy()
            output = [[*actor.position,actor.heading,float(np.linalg.norm(actor.velocity))]]
            if actor.kind == 'pedestrian':
                f,q = singer_matrices(step,self.cfg.get('singer_tau_s',1.),self.cfg.get('singer_accel_std_mps2',1.5))
                ff=np.kron(np.eye(2),f);qq=np.kron(np.eye(2),q)
                for _ in times[1:]:
                    z,p=ff@z,ff@p@ff.T+qq
                    output.append([z[0],z[3],np.arctan2(z[4],z[1]),np.hypot(z[1],z[4])])
            else:
                sequence=_ctra_sequence(z,p,step,len(times)-1,self.cfg.get('ctra_jerk_psd',1.),
                                        self.cfg.get('ctra_yaw_accel_psd',.04),self.cfg.get('integration_step_s',.02))
                headings=wrap(sequence[:,4]+wrap(actor.heading-track['z'][4]))
                output.extend(np.column_stack([sequence[:,:2],headings,sequence[:,2]]))
            output,info=self._mapped(actor,scene,track,np.array(output),times)
            # 地图先验失败不等于基础物理预测不存在；保留对象和失败原因，明确实际来源。
            info['map_valid']=bool(info.get('map_applied') and info.get('valid'))
            info['trajectory_source']='map_conditioned_ekf' if info['map_valid'] else ('singer' if actor.kind=='pedestrian' else 'ctra_ekf')
            info['kinematic_valid']=bool(np.isfinite(output).all() and np.all(output[:,3]>=-1e-8))
            if not info['kinematic_valid']:
                raise ValueError(f'对象 {actor.id} 的物理预测包含非法数值')
            trajectories[actor.id],diag[actor.id]=output,info
        # 留历史只用于再次观测滤波；从不为被遮挡对象输出当前预测。
        self.tracks={k:v for k,v in self.tracks.items() if scene.time-v['time'] <= 3.}
        self.preparation_seconds=_CTRA_PREPARATION_SECONDS-preparation_before
        return PredictionBundle(scene.frame_id,scene.time,times,trajectories,actors,diag)
