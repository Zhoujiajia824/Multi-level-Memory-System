"""Compile an explicit development route from a real OpenDRIVE file, without a server."""
import argparse,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nine_grid.carla_experiment_spec import COORDINATE_CONTRACT

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--xodr',type=Path,required=True)
    p.add_argument('--map',required=True)
    p.add_argument('--start',required=True,help='road_id,lane_id,s_m')
    p.add_argument('--length-m',type=float,required=True)
    p.add_argument('--branches',default='',help='ordered road:lane choices at ambiguous successor branches; comma separated')
    p.add_argument('--task',required=True,help='public task description, including intended turn/exit')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        import carla
        if not math.isfinite(args.length_m) or not 20<=args.length_m<=3000:raise ValueError('length must be 20..3000m')
        road,lane,s=args.start.split(',');road=int(road);lane=int(lane);s=float(s)
        choices=[tuple(int(v) for v in item.split(':')) for item in args.branches.split(',') if item]
        if any(len(item)!=2 for item in choices):raise ValueError('branches use road:lane pairs')
        m=carla.Map(args.map,args.xodr.read_text());start=m.get_waypoint_xodr(road,lane,s)
        if start is None or start.is_junction or str(start.lane_type)!='Driving':raise ValueError('start must be a non-junction driving lane')
        route=[start];length=0;index=0
        while length<args.length_m:
            options=list(route[-1].next(2.))
            if not options:raise ValueError('route ends before requested distance')
            if len(options)>1:
                if index>=len(choices):
                    raise ValueError('explicit next branch required at %.1fm; choices=%s'%(length,[(w.road_id,w.lane_id) for w in options]))
                wanted=choices[index];matches=[w for w in options if (w.road_id,w.lane_id)==wanted]
                if len(matches)!=1:raise ValueError('branch does not identify one public successor: '+str(wanted))
                nxt=matches[0];index+=1
            else:nxt=options[0]
            gap=route[-1].transform.location.distance(nxt.transform.location)
            if not .01<gap<=4.5:raise ValueError('noncontiguous step')
            if any(nxt.transform.location.distance(x.transform.location)<.5 for x in route[:-5:5]):raise ValueError('route repeats itself; specify a task exit')
            a,b=route[-1].transform.location,nxt.transform.location
            route.append(nxt);length+=math.hypot(b.x-a.x,b.y-a.y)
        if index!=len(choices):raise ValueError('unused branch choices; route/task lengths disagree')
        points=[]
        for w in route:
            t=w.transform;l=t.location
            points.append(dict(road_id=w.road_id,section_id=w.section_id,lane_id=w.lane_id,s_m=w.s,
                x_m=l.x,y_m=l.y,z_m=l.z,yaw_deg=t.rotation.yaw,junction=w.is_junction))
        result=dict(map=args.map,opendrive_sha256=hashlib.sha256(m.to_opendrive().encode()).hexdigest(),
            sampling_step_m=2,points=points,length_m=length,coordinate_contract=COORDINATE_CONTRACT,
            task={'description':args.task,'entry':points[0],'exit':points[-1],'branch_choices':choices},
            server_validation={'status':'pending','scope':'offline_sdk_topology_only'})
        if args.output.exists():raise ValueError('output already exists; use a new filename')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf8')
        print(json.dumps({'route':str(args.output),'length_m':length,'points':len(points),'server_validated':False},ensure_ascii=False))
    except (ValueError,OSError,ImportError) as exc:p.error(str(exc))
if __name__=='__main__':main()
