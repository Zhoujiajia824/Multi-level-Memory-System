"""启动只读结果查看器；读取已有归档，不连接或推进CARLA。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'ep_agent_viewer'))


def main(argv=None):
    from epviz.server import main as viewer_main
    args = list(sys.argv[1:] if argv is None else argv)
    for flag, value in (('--root', ROOT / 'output'), ('--port', 39081), ('--engine-root', ROOT)):
        if not any(item == flag or item.startswith(flag + '=') for item in args):
            args += [flag, str(value)]
    return viewer_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
