"""查看经验与明确停用已确认错误的记录；不删除历史资产。"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nine_grid.memory import MemoryStore


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='经验查看及确认错误后的停用')
    parser.add_argument('--memory-dir',required=True)
    sub=parser.add_subparsers(dest='operation',required=True)
    sub.add_parser('list')
    show=sub.add_parser('show');show.add_argument('id')
    deactivate=sub.add_parser('invalidate');deactivate.add_argument('id');deactivate.add_argument('--reason',required=True)
    args=parser.parse_args()
    root=Path(args.memory_dir)
    root=root if root.is_absolute() else ROOT/root
    store=MemoryStore(root,root if args.operation=='invalidate' else None)
    records=store.snapshot()
    if args.operation=='list':
        result={'可检索记录':[{'id':r['id'],'revision':r['revision'],'frame_id':r['frame_id'],'time':r['time'],'assets_directory':r['_asset_root']} for r in records],
                '排除记录':store.audit,'snapshot_id':store.snapshot_id}
    else:
        matches=[r for r in records if r['id']==args.id]
        if not matches:
            parser.error('未找到当前可用的该 ID；先检查 list 的排除记录')
        if args.operation=='show':
            result=[{k:v for k,v in r.items() if k!='_representations'} for r in matches]
        else:
            store.invalidate(args.id,args.reason)
            result={'已停用':args.id,'原因':args.reason,'生效范围':'后续建立的检索快照，保留审计资产'}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
