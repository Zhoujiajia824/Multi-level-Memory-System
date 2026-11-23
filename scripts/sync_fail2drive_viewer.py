"""将独立查看器运行文件同步到本分支；不复制测试、缓存或实验产物。"""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def selected(source):
    files = [source / name for name in ('run_viewer.py', 'README.md', 'requirements.txt')]
    files += list((source / 'epviz').glob('*.py'))
    files += [p for p in (source / 'web').rglob('*') if p.is_file()]
    return sorted(p for p in files if p.is_file() and '__pycache__' not in p.parts
                  and not p.name.endswith(('.log', '.pid', '.tmp')))


def content(path):
    if path.name != 'README.md':
        return path.read_bytes()
    sections, validation = [], False
    for paragraph in path.read_text(encoding='utf-8').split('\n\n'):
        if paragraph.startswith('## 验证'):
            validation = True
        elif validation and paragraph.startswith('## '):
            validation = False
        if not validation and not paragraph.startswith(('NAVSIM 回归测试：', '可选浏览器回归：')):
            sections.append(paragraph)
    return ('\n\n'.join(sections).rstrip() + '\n').encode('utf-8')


def sync(source, check=False):
    source = Path(source).resolve()
    target = (ROOT / 'tools' / 'ep_agent_viewer').resolve()
    if source == target or not target.is_relative_to(ROOT.resolve()):
        raise ValueError('查看器部署目录必须位于当前分支内，且与来源不同')
    if not (source / 'run_viewer.py').is_file():
        raise ValueError('缺少独立查看器入口')
    files, changed = selected(source), []
    for original in files:
        output = target / original.relative_to(source)
        data = content(original)
        if not output.is_file() or hashlib.sha256(output.read_bytes()).digest() != hashlib.sha256(data).digest():
            changed.append(original.relative_to(source).as_posix())
            if not check:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(data)
    print('查看器文件：', len(files), '差异：', len(changed), '只检查：', check)
    if check and changed:
        raise RuntimeError('查看器部署副本需要更新：' + ', '.join(changed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT.parent.parent / 'ep_agent_viewer'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    sync(args.source, args.check)
