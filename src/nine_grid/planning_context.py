"""模型候选全部做EKF；优化集合在planning_view中独立筛选。"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np

from .perception_scope import perception_scene
from .prediction import Predictor


def prepare_planning_context(scene, config, answer=None, predictor=None, existing=None):
    scoped, selection = perception_scene(scene, config)
    if existing is None:
        predictions = (predictor or Predictor(config)).predict(scoped)
    else:
        if existing.frame_id != scene.frame_id or not np.isclose(existing.time, scene.time, rtol=0, atol=1e-8):
            raise ValueError('补算预测必须绑定同一原帧，禁止用当前帧替代历史状态')
        frozen = existing.diagnostics.get('perception_scope', {})
        if (frozen.get('version') == selection['version']
                and frozen.get('selected_input_signature') != selection['selected_input_signature']):
            raise ValueError('复用预测的原帧感知状态或地图签名发生变化')
        ids = {actor.id for actor in scoped.actors}
        for actor in scoped.actors:
            if actor.id not in existing.trajectories:
                continue
            old = existing.actors.get(actor.id)
            trajectory = np.asarray(existing.trajectories[actor.id], float)
            if (old is None or old.kind != actor.kind or old.motion_model != actor.motion_model or old.length != actor.length
                    or old.width != actor.width or old.radius != actor.radius
                    or not np.allclose(old.position, actor.position, rtol=0, atol=1e-8)
                    or not np.allclose(old.velocity, actor.velocity, rtol=0, atol=1e-8)
                    or not np.allclose(old.acceleration, actor.acceleration, rtol=0, atol=1e-8)
                    or trajectory.shape != (len(existing.times), 4)
                    or not np.isfinite(trajectory).all()
                    or not np.allclose(trajectory[0,:2], actor.position, rtol=0, atol=1e-8)
                    or abs((trajectory[0,2]-actor.heading+np.pi)%(2*np.pi)-np.pi) > 1e-8
                    or not np.isclose(trajectory[0,3], np.linalg.norm(actor.velocity), rtol=0, atol=1e-8)):
                raise ValueError('复用预测的对象状态或预测首点与原帧不一致：'+actor.id)
        missing = [actor for actor in scoped.actors if actor.id not in existing.trajectories]
        trajectories = {key: value.copy() for key, value in existing.trajectories.items() if key in ids}
        diagnostics = {key: copy.deepcopy(value) for key, value in existing.diagnostics.items() if key in ids}
        if missing:
            # 兼容缺少候选预测的历史原帧证据；不能读取未来或后台当前世界。
            extension = Predictor(config).predict(replace(scoped, actors=missing))
            if not np.allclose(extension.times, existing.times):
                raise ValueError('原帧补算预测时域与既有证据不一致')
            trajectories.update(extension.trajectories)
            diagnostics.update(extension.diagnostics)
            diagnostics['source_frame_extension'] = dict(
                actor_ids=sorted(actor.id for actor in missing),
                source='原帧已观测当前状态重新初始化，不冒充当时EKF后验')
        predictions = replace(existing, trajectories=trajectories,
            actors={actor.id: actor for actor in scoped.actors}, diagnostics=diagnostics)
    predictions.diagnostics.pop('planning_scope', None)
    predictions.diagnostics['perception_scope'] = copy.deepcopy(selection)
    predictions.diagnostics['prediction_actor_ids'] = sorted(predictions.trajectories)
    return scoped, predictions
