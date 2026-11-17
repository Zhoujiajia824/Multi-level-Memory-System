"""Validate at least two same-instance server runs without connecting to CARLA."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nine_grid.carla_experiment_validation import validate_runs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episodes',nargs='+',help='episode output directories, not batch directories')
    p.add_argument('--coverage-witness',action='append',default=[],help='additional independent development episode proving full route and required-event coverage; repeatable')
    p.add_argument('--output',required=True)
    args=p.parse_args();output=Path(args.output).resolve()
    try:result=validate_runs(args.episodes,args.coverage_witness)
    except (OSError,ValueError,KeyError,TypeError) as exc:p.error(str(exc))
    # Portable together with the experiment evidence directory tree.
    for row in result['runs']:row['directory']=Path(os.path.relpath(row['directory'],output.parent)).as_posix()
    result['coverage_witnesses']=[Path(os.path.relpath(path,output.parent)).as_posix() for path in result['coverage_witnesses']]
    for row in result['evidence_files']:row['path']=Path(os.path.relpath(row['path'],output.parent)).as_posix()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'报告':str(output),'正式准入':result['passed'],'技术烟雾':result['technical_smoke_passed'],
        '实际场景覆盖':result['scenario_coverage_passed'],'实例':result['instance_hash'],'问题':result['reasons']},ensure_ascii=False))
    return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
