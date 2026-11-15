"""主规划、无语义参照和原帧反思使用同一后端契约。"""
def create_planner(config):
    backend = config.get('planner', {}).get('backend', 'mpc')
    if backend == 'mpc':
        from .planner import Planner
        return Planner(config)
    if backend == 'pdm':
        try:
            from .pdm_planner import Planner as PDMPlanner
        except ImportError as exc:
            raise ValueError('本分支未安装PDM规划后端，请使用EP-Driver-CARLA-PDM分支') from exc
        return PDMPlanner(config)
    raise ValueError('未知规划后端：' + str(backend))
