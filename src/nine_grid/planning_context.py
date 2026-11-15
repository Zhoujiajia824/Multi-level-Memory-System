"""预测前筛选规划对象；后台原帧验收只补算该帧已提供的状态。"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np

from .planning_objects import planning_scene
from .prediction import Predictor


def prepare_planning_context(scene, config, answer=None, predictor=None, existing=None):
    scoped, selection = planning_scene(scene, config, answer)
    if existing is None:
        predictions = (predictor or Predictor(config)).predict(scoped)
    else:
        if existing.frame_id != scene.frame_id or not np.isclose(existing.time, scene.time, rtol=0, atol=1e-8):
            raise ValueError('补算预测必须绑定同一原帧，禁止用当前帧替代历史状态')
        ids = {actor.id for actor in scoped.actors}
        missing = [actor for actor in scoped.actors if actor.id not in existing.trajectories]
        trajectories = {key: value.copy() for key, value in existing.trajectories.items() if key in ids}
        diagnostics = {key: copy.deepcopy(value) for key, value in existing.diagnostics.items() if key in ids}
        if missing:
            # 原快答案未选中而反思关注的对象，没有当时滤波后验；明确使用
            # 已冻结原帧当前状态初始化，不能读取未来或后台当前世界。
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
    predictions.diagnostics['planning_scope'] = copy.deepcopy(selection)
    return scoped, predictions
