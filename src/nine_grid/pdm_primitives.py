"""从EP-Driver当前MPC抽取的共享物理/接口/道路规则；无优化器调用。
用于PDM分支的公平对照；保持同一自行车积分、预测权限和路权语义。
来源 src/nine_grid/planner.py SHA256: 6da129b3a4e97d9c915899b09cd126a64df175feb597605ced379f216dc79752
"""
import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from nine_grid.geometry import clean_polyline, polyline_lengths, project_polyline, sample_polyline, rectangle_corners, wrap_angle
from nine_grid.realtime_solver import bicycle_geometry

def propagate(state, control, length=5., dt=.1, substeps=5,
              wheelbase_m=None,rear_axle_to_center_m=None):
    """几何中心自行车模型显式Euler；未提供轴距时保持highway原公式。"""
    state=np.asarray(state,float).copy();a,delta=np.asarray(control,float)
    wheelbase,rear=bicycle_geometry(length,wheelbase_m,rear_axle_to_center_m)
    beta=np.arctan(rear/wheelbase*np.tan(delta));step=dt/substeps
    for _ in range(substeps):
        x,y,psi,v=state
        state=np.array([x+v*np.cos(psi+beta)*step,y+v*np.sin(psi+beta)*step,
                        psi+v/rear*np.sin(beta)*step,v+a*step])
    return state

def _validate_prediction_input(scene, predictions, horizon):
    """拒绝错帧、漏对象和不足时域；不得把接口错误当作空交通或静止尾段。"""
    if (predictions.frame_id != scene.frame_id or predictions.version != scene.version
            or predictions.coordinates != 'world_xy_m' or scene.coordinates != 'world_xy_m'
            or not np.isclose(predictions.time, scene.time, rtol=0., atol=1e-8)):
        raise ValueError('规划预测输入的原帧、时间、版本或世界坐标不一致')
    visible = [actor for actor in scene.actors if actor.visible]
    expected = {actor.id for actor in visible}
    if len(expected) != len(visible) or set(predictions.actors) != expected or set(predictions.trajectories) != expected:
        raise ValueError('规划预测对象集合必须完整对应当前可见对象，不能遗漏或引入不可见对象')
    times = np.asarray(predictions.times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or abs(times[0]) > 1e-8 or np.any(np.diff(times) <= 0)):
        raise ValueError('规划预测时刻必须从零开始、严格递增且数值有限')
    if expected and times[-1] < horizon-1e-8:
        raise ValueError('对象预测未覆盖MPC完整时域，不允许静默重复末端位置')
    for actor in visible:
        predicted = predictions.actors[actor.id]
        if (predicted.id != actor.id or not predicted.visible or predicted.kind != actor.kind
                or predicted.length != actor.length or predicted.width != actor.width
                or predicted.radius != actor.radius
                or not np.allclose(predicted.position, actor.position, rtol=0., atol=1e-8)
                or not np.allclose(predicted.velocity, actor.velocity, rtol=0., atol=1e-8)
                or abs(float(wrap_angle(predicted.heading-actor.heading))) > 1e-8):
            raise ValueError(f'对象 {actor.id} 的预测实体描述与当前观测不一致')
        trajectory = np.asarray(predictions.trajectories[actor.id], float)
        if (trajectory.shape != (len(times), 4) or not np.isfinite(trajectory).all()
                or np.any(trajectory[:, 3] < -1e-8)):
            raise ValueError(f'对象 {actor.id} 的预测维度或运动数值非法')

def public_rules(self,scene,predictions):
        """公共道路优先级和可见预测确定让行，不能用自身停车解释应该等待。"""
        stop_s=scene.rules.get("stop_s")
        ego_s=project_polyline(scene.ego.position,scene.route,scene.diagnostics.get("route_s"),scene.ego.heading)[0]
        lanes={lane.id:lane for lane in scene.lanes}
        ego_lane=lanes.get(scene.ego.lane_id)
        reason=scene.rules.get("wait_reason")
        rules=self.config.get("rules",{})
        stop_margin=float(rules.get("stop_margin_m",1.))
        self.rule_evidence=[]
        if ego_lane and scene.rules.get("road_priority_available"):
            route=scene.route;route_shape=LineString(route)
            for identity,actor in predictions.actors.items():
                other_lane=lanes.get(actor.lane_id)
                audit={"actor_id":identity,"source":"visible_public_lane_geometry","active":False}
                self.rule_evidence.append(audit)
                if not actor.visible or not other_lane:
                    audit["reason"]="对象不可见或缺少公开车道";continue
                # 已在自车同一公开路段的对象属于跟驰；不能仅凭当前朝向
                # 相似删除汇入冲突，也不能把对向车当作同向跟驰。
                if actor.lane_id in scene.route_lane_ids:
                    audit["reason"]="已在自车同一公开路段，按跟驰处理";continue
                line=other_lane.center.copy();lane=other_lane;visited={lane.id};path_ids=[lane.id]
                inferred=list(predictions.diagnostics.get(identity,{}).get("successors",[]))
                actor_s=project_polyline(actor.position,line,heading=actor.heading)[0]
                while polyline_lengths(line)[-1]-actor_s<55.:
                    options=[lanes[key] for key in lane.successors if key in lanes and key not in visited]
                    if not options:break
                    next_id=inferred.pop(0) if inferred else None
                    # 优先复用本帧预测器已经推断的公开合法后继。停车时
                    # 预测可能没有后继，按已观察运动朝向选最一致者；这只
                    # 定义公开路权通路，不生成额外恒速轨迹或读取隐藏路线。
                    candidates={candidate.id:candidate for candidate in options}
                    if next_id in candidates:
                        lane=candidates[next_id]
                    else:
                        low_speed=float(self.config.get("prediction",{}).get("low_speed_mps",.3))
                        motion=actor.heading if np.linalg.norm(actor.velocity)<low_speed else np.arctan2(actor.velocity[1],actor.velocity[0])
                        def alignment(candidate):
                            direction=candidate.center[-1]-candidate.center[0]
                            return abs(float(wrap_angle(np.arctan2(direction[1],direction[0])-motion))),candidate.id
                        lane=min(options,key=alignment)
                    visited.add(lane.id)
                    path_ids.append(lane.id)
                    line=clean_polyline(np.vstack((line,lane.center)))
                # 首次入口由完整公共通路的空间几何决定，不受短时预测
                # 刹停或更远处共同出口的最近点影响。
                width=other_lane.width
                corridor=LineString(line).buffer(width/2,cap_style="flat")
                intersection=route_shape.intersection(corridor)
                coordinates=shapely.get_coordinates(intersection)
                audit["public_lane_ids"]=path_ids
                if not len(coordinates):
                    audit["reason"]="公开通路与自车导航没有交汇";continue
                entry_s=float(np.min(project_polyline(coordinates,route)[0]))
                audit["entry_s"]=entry_s
                if entry_s>ego_s+55:
                    audit["reason"]="首次入口在自车局部规则范围之外";continue
                if entry_s<ego_s+scene.ego.length/2-.001:
                    audit["reason"]="自车已通过首次入口，不重新选择远端重合处";continue
                entry=sample_polyline(route,entry_s)
                # 只保留对象车尾至其前方55m通路。完整通路只用于固定
                # 首次入口，已经驶过的道路不会继续触发让行。
                arc=polyline_lengths(line)
                actor_s=project_polyline(actor.position,line,heading=actor.heading)[0]
                rear_s=float(np.min(project_polyline(rectangle_corners(actor.position,actor.heading,actor.length,actor.width),line)[0]))
                end_s=min(arc[-1],actor_s+55.)
                if end_s<=rear_s+1e-6:
                    audit["reason"]="对象前方公共通路不足";continue
                forward=clean_polyline(np.vstack((sample_polyline(line,rear_s),line[(arc>rear_s)&(arc<end_s)],sample_polyline(line,end_s))))
                if not LineString(forward).buffer(width/2,cap_style="flat").buffer(1e-8).covers(shapely.Point(entry)):
                    audit["reason"]="首次入口不在对象车尾及前方55m的当前通路中";continue
                priorities=[candidate.priority for candidate in scene.lanes if candidate.id in scene.route_lane_ids
                            and Polygon(candidate.polygon).buffer(1e-8).covers(shapely.Point(entry))]
                ego_priority=min(priorities,default=ego_lane.priority)
                audit.update(ego_priority=ego_priority,other_priority=other_lane.priority)
                if other_lane.priority<=ego_priority:
                    audit["reason"]="对象不具有更高的公开路权";continue
                proposed=entry_s-stop_margin
                audit.update(active=True,stop_s=proposed,reason="可见高优先级对象尚未清空首次冲突入口")
                if stop_s is None or proposed<stop_s:
                    stop_s=proposed;reason=f"向可见高优先级交通对象 {identity} 让行（公开通路首次入口）"
        return stop_s,reason
