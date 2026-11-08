"""固定结构动力学导数与实时 SQP 子问题，不提供备用控制器。"""
from __future__ import annotations

import time
import numpy as np
import casadi as ca


def bicycle_geometry(length, wheelbase_m=None, rear_axle_to_center_m=None):
    """返回轴距和后轴至几何中心距离；缺省严格保留highway等轴臂模型。"""
    if wheelbase_m is None and rear_axle_to_center_m is None:
        wheelbase_m, rear_axle_to_center_m = length, length/2
    if wheelbase_m is None or rear_axle_to_center_m is None:
        raise ValueError('车辆轴距和后轴至几何中心距离必须同时提供')
    wheelbase, rear = float(wheelbase_m), float(rear_axle_to_center_m)
    if not np.isfinite([wheelbase,rear]).all() or not 0. < rear < wheelbase:
        raise ValueError('车辆几何要求有限正轴距且几何中心位于前后轴之间')
    return wheelbase, rear


class Dynamics:
    """一次构建符号链；每次同时返回原动力学轨迹和精确控制敏感度。"""
    def __init__(self,n,dt,substeps,length,wheelbase_m=None,rear_axle_to_center_m=None):
        begin=time.perf_counter()
        wheelbase,rear=bicycle_geometry(length,wheelbase_m,rear_axle_to_center_m)
        ratio=rear/wheelbase
        u=ca.SX.sym('u',2*n);initial=ca.SX.sym('initial',4)
        state=initial;columns=[state]
        for k in range(n):
            acceleration,steering=u[2*k],u[2*k+1]
            beta=ca.atan(ratio*ca.tan(steering));step=dt/substeps
            for _ in range(substeps):
                x,y,heading,speed=ca.vertsplit(state)
                state=ca.vertcat(x+speed*ca.cos(heading+beta)*step,
                    y+speed*ca.sin(heading+beta)*step,
                    heading+speed/rear*ca.sin(beta)*step,speed+acceleration*step)
            columns.append(state)
        states=ca.horzcat(*columns)
        sensitivity=ca.jacobian(ca.vec(states),u)
        self.function=ca.Function('vehicle_rollout',[u,initial],[states,sensitivity])
        self.state_function=ca.Function('vehicle_states',[u,initial],[states])
        self.sensitivity_function=ca.Function('vehicle_sensitivity',[u,initial],[sensitivity])
        self.n=n;self.build_seconds=time.perf_counter()-begin

    def __call__(self,controls,initial):
        states,sensitivity=self.function(np.asarray(controls).ravel(),initial)
        return states.full().T,sensitivity.full().reshape(self.n+1,4,2*self.n)

    def rollout(self,controls,initial):
        return self.state_function(np.asarray(controls).ravel(),initial).full().T

    def sensitivity(self,controls,initial):
        return self.sensitivity_function(np.asarray(controls).ravel(),initial).full().reshape(self.n+1,4,2*self.n)


class QuadraticSolver:
    """OSQP 只负责当前线性化子问题；最终执行资格由真实几何复核决定。"""
    def __init__(self,n,maximum_iterations=600,tolerance=1e-6):
        self.n=n;self.maximum_iterations=maximum_iterations;self.tolerance=tolerance
        self.solvers={}
        self.cache_builds=0;self.cache_hits=0;self.setup_seconds=0.
        self.last_diagnostics={}

    def solve(self,hessian,gradient,matrix,lower,upper,step_lower,step_upper):
        begin=time.perf_counter()
        # 空的物理控制箱与信赖域交集没有可解QP，不能交给CasADi抛异常，
        # 更不能交换上下界或裁剪边界来掩盖不可行。调用方仍只可执行已验收候选。
        self.last_diagnostics={'valid':False,'scaled_stationarity_inf':None,'scaled_primal_violation':None}
        invalid_box=np.flatnonzero(np.asarray(step_lower)>np.asarray(step_upper))
        if len(invalid_box):
            self.last_diagnostics['invalid_step_bound_indices']=invalid_box.tolist()
            return np.zeros(self.n),False,'invalid_step_bounds',time.perf_counter()-begin
        scale=np.maximum(np.maximum(np.abs(step_lower),np.abs(step_upper)),1e-4)
        scaled=matrix*scale[None,:]
        low_box=step_lower/scale;high_box=step_upper/scale
        least=np.sum(np.where(scaled>=0,scaled*low_box,scaled*high_box),axis=1)
        greatest=np.sum(np.where(scaled>=0,scaled*high_box,scaled*low_box),axis=1)
        active=(least<lower-1e-10)|(greatest>upper+1e-10)
        scaled=scaled[active];lower=lower[active];upper=upper[active]
        row_scale=np.maximum(np.linalg.norm(scaled,axis=1),1e-8)
        scaled=scaled/row_scale[:,None];lower=lower/row_scale;upper=upper/row_scale
        rows=len(lower);mask=np.abs(scaled)>1e-14
        # 动力学敏感度在零转角等位置出现数值零；按因果前缀固定稀疏结构，
        # 避免同一时域结构因这些零反复重建求解器。两点差分行保持原稀疏度。
        prefix=np.maximum.accumulate(np.where(mask,np.arange(self.n)[None,:]+1,0),axis=1)[:,-1]
        dynamic=np.sum(mask,axis=1)>2
        mask[dynamic]=np.arange(self.n)[None,:]<prefix[dynamic,None]
        key=(rows,mask.tobytes())
        if key not in self.solvers:
            if len(self.solvers)>=8:self.solvers.pop(next(iter(self.solvers)))
            rr,cc=np.nonzero(mask);sparsity=ca.Sparsity.triplet(rows,self.n,rr.tolist(),cc.tolist())
            # 仅消去在信赖域内必然不活跃的线性约束；真实几何仍完整复核。
            setup_started=time.perf_counter()
            self.solvers[key]=(ca.conic('real_time_qp','osqp',
                {'h':ca.Sparsity.dense(self.n,self.n),'a':sparsity},
                {'error_on_fail':False,'osqp':{'verbose':False,'max_iter':self.maximum_iterations,
                 'eps_abs':self.tolerance,'eps_rel':self.tolerance,'polish':True}}),sparsity)
            self.setup_seconds+=time.perf_counter()-setup_started;self.cache_builds+=1
        else:
            self.cache_hits+=1
            # 有限LRU只淘汰一个最久未使用结构，不能每次满后清空全部缓存。
            self.solvers[key]=self.solvers.pop(key)
        solver,sparsity=self.solvers[key]
        # 加速度与转角量纲相差明显；只作变量变换，不改变目标或约束。
        rr,cc=sparsity.get_triplet()
        scaled_hessian=hessian*scale[:,None]*scale[None,:];scaled_gradient=gradient*scale
        result=solver(h=scaled_hessian,g=scaled_gradient,
            a=ca.DM(sparsity,scaled[rr,cc]),lba=lower,uba=upper,lbx=low_box,ubx=high_box)
        z=result['x'].full().ravel();step=z*scale
        dual_rows=result['lam_a'].full().ravel();dual_box=result['lam_x'].full().ravel()
        # 这些是当前缩放QP的残差，不是非线性MPC已达全局最优的证明。
        self.last_diagnostics={'valid':False,'scaled_stationarity_inf':None,'scaled_primal_violation':None}
        if np.all(np.isfinite(z)) and np.all(np.isfinite(dual_rows)) and np.all(np.isfinite(dual_box)):
            stationarity=scaled_hessian@z+scaled_gradient+scaled.T@dual_rows+dual_box
            values=scaled@z
            stationarity_norm=float(np.max(np.abs(stationarity)))
            primal_violation=float(max(0.,np.max(lower-values,initial=0.),
                np.max(values-upper,initial=0.),np.max(low_box-z),np.max(z-high_box)))
            if np.isfinite(stationarity_norm) and np.isfinite(primal_violation):
                self.last_diagnostics.update(valid=True,scaled_stationarity_inf=stationarity_norm,
                    scaled_primal_violation=primal_violation)
        stats=solver.stats()
        status=str(stats.get('return_status','未知'))
        # 固定迭代RTI允许检查已优化但未达到QP终止精度的迭代点；
        # 只有后续原动力学、实体、道路与全部控制边界验收合格才能执行。
        eligible=bool(stats.get('success',False)) or status in ('maximum iterations reached','solved inaccurate')
        return step,eligible,status,time.perf_counter()-begin


def rectangle_gap(states,other,length,width,other_length,other_width):
    """分离轴最大间隙及其活动分支导数；车体保持真实矩形尺寸。"""
    center=other[:,:2]-states[:,:2]
    angles=np.stack((states[:,2],states[:,2]+np.pi/2,other[:,2],other[:,2]+np.pi/2),axis=1)
    axes=np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    projection=np.einsum('ni,nai->na',center,axes)
    own=states[:,2,None]-angles;remote=other[:,2,None]-angles
    own_support=length/2*np.abs(np.cos(own))+width/2*np.abs(np.sin(own))
    remote_support=other_length/2*np.abs(np.cos(remote))+other_width/2*np.abs(np.sin(remote))
    distances=np.abs(projection)-own_support-remote_support
    selected=np.argmax(distances,axis=1);row=np.arange(len(states))
    normal=axes[row,selected];signed_projection=projection[row,selected]
    # 平行同车道完全对齐时 |q| 在零点不可微，零梯度会锁死避让。
    # 选定可核对的一侧次梯度，后续道路/边线硬约束仍决定其可行性。
    sign=np.where(np.abs(signed_projection)<1e-9,-1.,np.sign(signed_projection))
    result=distances[row,selected];gradient=np.zeros((len(states),4))
    gradient[:,:2]=-sign[:,None]*normal
    rotates=(selected<2).astype(float)
    phi=own[row,selected];remote_phi=remote[row,selected]
    own_derivative=-length/2*np.sin(phi)*np.sign(np.cos(phi))+width/2*np.cos(phi)*np.sign(np.sin(phi))
    remote_derivative=-other_length/2*np.sin(remote_phi)*np.sign(np.cos(remote_phi))+other_width/2*np.cos(remote_phi)*np.sign(np.sin(remote_phi))
    perpendicular=np.stack((-normal[:,1],normal[:,0]),axis=1)
    gradient[:,2]=sign*np.sum(center*perpendicular,axis=1)*rotates-own_derivative*(1-rotates)+remote_derivative*rotates
    return result,gradient


def circle_gap(states,centers,radius,length,width):
    """圆心至有向矩形的解析有符号距离及导数。"""
    relative=centers-states[:,:2];c=np.cos(states[:,2]);s=np.sin(states[:,2])
    local=np.stack((c*relative[:,0]+s*relative[:,1],-s*relative[:,0]+c*relative[:,1]),axis=1)
    q=np.abs(local)-np.array([length/2,width/2]);outside=np.maximum(q,0.)
    norm=np.linalg.norm(outside,axis=1)
    distance=norm+np.minimum(np.max(q,axis=1),0.)-radius
    local_gradient=np.zeros_like(local);external=norm>1e-10
    local_gradient[external]=outside[external]/norm[external,None]*np.sign(local[external])
    active=np.argmax(q,axis=1);row=np.arange(len(states))
    local_gradient[row[~external],active[~external]]=np.sign(local[row[~external],active[~external]])
    gradient=np.zeros((len(states),4))
    gradient[:,0]=-c*local_gradient[:,0]+s*local_gradient[:,1]
    gradient[:,1]=-s*local_gradient[:,0]-c*local_gradient[:,1]
    gradient[:,2]=local_gradient[:,0]*local[:,1]-local_gradient[:,1]*local[:,0]
    return distance,gradient


def rectangle_distance(states,vertices,other_vertices,separation):
    """全部顶点—线段见证点给出精确矩形欧氏净距和活动特征导数。"""
    count=len(states);best=np.full(count,np.inf);own_point=np.zeros((count,2));remote_point=np.zeros((count,2))
    for swap in (False,True):
        points=other_vertices if swap else vertices
        starts=vertices if swap else other_vertices
        segments=np.roll(starts,-1,axis=1)-starts
        delta=points[:,:,None,:]-starts[:,None,:,:]
        fraction=np.clip(np.sum(delta*segments[:,None,:,:],axis=-1)/np.maximum(np.sum(segments**2,axis=-1)[:,None,:],1e-12),0.,1.)
        closest=starts[:,None,:,:]+fraction[:,:,:,None]*segments[:,None,:,:]
        distances=np.sum((points[:,:,None,:]-closest)**2,axis=-1)
        selected=np.argmin(distances.reshape(count,-1),axis=1);row=np.arange(count)
        pi,si=selected//4,selected%4;candidate=np.sqrt(distances[row,pi,si]);replace=candidate<best
        point=points[row,pi];nearest=closest[row,pi,si]
        own_point[replace]=(nearest if swap else point)[replace]
        remote_point[replace]=(point if swap else nearest)[replace]
        best=np.minimum(best,candidate)
    best=np.where(separation>0,best,0.)
    normal=(own_point-remote_point)/np.maximum(best[:,None],1e-10)
    normal[separation<=0]=0.
    gradient=np.zeros((count,4));gradient[:,:2]=normal
    arm=own_point-states[:,:2]
    gradient[:,2]=-normal[:,0]*arm[:,1]+normal[:,1]*arm[:,0]
    return best,gradient
