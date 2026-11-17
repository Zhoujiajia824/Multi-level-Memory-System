"""Catalogue regression checks; optional real 0.9.15 OpenDRIVE validation (no server)."""
import argparse,copy,importlib.util,json,sys,tempfile
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nine_grid.carla_experiment_spec import resolve_spec,settings_from_manifest,COORDINATE_CONTRACT
loader=importlib.util.spec_from_file_location('carla_cli',ROOT/'scripts/run_carla.py')
cli=importlib.util.module_from_spec(loader);loader.loader.exec_module(cli)

def check_admission_contract(manifest):
    from nine_grid.carla_experiment_validation import validate_runs,verify_report,MEASUREMENTS,_event_coverage
    from nine_grid.carla_task_criteria import TaskCriteria
    # Deliberately synthetic evidence tests admission semantics, never CARLA physics.
    settings=copy.deepcopy(manifest['settings'])
    settings['route']={'opendrive_sha256':'f'*64,'length_m':20.,'task':{'kind':'following'},'coordinate_contract':COORDINATE_CONTRACT,
        'points':[{'road_id':1,'section_id':0,'lane_id':-1,'s_m':2.*i,
                   'x_m':2.*i,'y_m':0.,'z_m':0.,'yaw_deg':0.,'junction':False} for i in range(11)]}
    settings['actors']=[];settings['public_rules']=[];settings['evaluation']={}
    _,manifest=resolve_spec({'schema_version':'carla-experiment-2.0','name':'synthetic_admission_contract',
        'family':'following','description':'Synthetic contract fixture; not CARLA validation',
        'map':'Town01','desired_speed_mps':5.,'duration_s':10.,'experiment':settings},1)
    settings=settings_from_manifest(manifest)
    (ROOT/'outputs').mkdir(exist_ok=True)
    with nullcontext(tempfile.mkdtemp(prefix='catalogue_contract_',dir=str(ROOT/'outputs'))) as temporary:
        root=Path(temporary);dirs=[]
        for index in range(2):
            directory=root/str(index);directory.mkdir();dirs.append(directory)
            initial={'client_version':'0.9.15-local','server_version':'0.9.15-local','source_frame':100,'physics_hz':50,'camera_hz':10,
                'instance_hash':manifest['instance_hash'],'map_identity':{'name':manifest['map'],'opendrive_sha256':'f'*64},
                'rgb_enabled':True,'rgb_initialization':{'status':'aligned'},
                'actors_at_source':[{'id':'ego','carla_actor_id':'1','role':'ego','position':[0.,0.],'velocity':[5.,0.],'heading':0.}]}
            physics=[];video_rows=[];task=TaskCriteria(settings)
            views=('front_rgb','front_left_rgb','front_right_rgb','back_rgb','back_left_rgb','back_right_rgb')
            for tick in range(201):
                ego={'id':'1','position':[tick*.1,0.],'heading':0.,'velocity':[5.,0.],
                    'lane_id':'1|0|-1','is_junction':False,'length':4.,'width':2.,'actor_origin_carla_xyz':[tick*.1,0.,0.]}
                task.observe(tick,ego,[],{'ego':'1'})
                physics.append({'instance_hash':manifest['instance_hash'],'physical_tick':tick,'frame':100+tick,
                    'timestamp':10.+tick/50.,'sim_time_s':tick/50.,'ego':ego,'actors':[],'route_s_m':tick*.1})
                if tick%5==0:
                    video_rows.append({'video_index':len(video_rows),'carla_frame':100+tick,'physical_tick':tick,'sim_time':tick/50.,
                        'source_cameras':[{'view':view,'frame':100+tick,'timestamp':10.+tick/50.} for view in views]})
            values={'resolved_instance.json':manifest,'configuration.json':{'parameters':{'carla':{'runtime_mode':'sync',
                'experiment_phase':'development','implementation_sha256':'a'*64,
                'camera_width':640,'camera_height':360,'camera_fov':100,'camera_height_m':2.2,'camera_hz':10,
                'camera_save_hz':1,'rgb_postprocess_effects':True,'map_resolution_m':2,'map_speed_cap_mps':30,
                'same_level_tolerance_m':1.5,'arrival_margin_m':3}}},
                'summary.json':{'termination_reason':'collision'},'carla_initialization.json':initial,
                'evaluation.json':{'version':'carla-physical-evaluation-2.0','physical_hz':50,
                    'physical_ticks':200,'first_frame':100,'scenario_valid':True,'evaluation_valid':True,
                    'task_criteria':task.report(),'scenario_events':[],
                    'coverage':{key:True for key in MEASUREMENTS},'strict_success':False,'driving_outcome':'collision'},
                'carla_cleanup.json':{'status':'complete','verified':True},
                'video_recording.json':{'status':'complete','playback_verified':True,'frames':41,'decoded_frames':41,
                    'first_carla_frame':100,'last_carla_frame':300,'source':'CARLA_same_frame_six_rgb_10hz','path':'video.mp4'}}
            for name,value in values.items():(directory/name).write_text(json.dumps(value),encoding='utf8')
            (directory/'physical_evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in physics),encoding='utf8')
            (directory/'video_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in video_rows),encoding='utf8')
            (directory/'video.mp4').write_bytes(b'Synthetic contract bytes, not a playable CARLA movie')
        report=validate_runs(dirs)
        assert report['passed'],report['reasons']
        out=root/'admission.json';out.write_text(json.dumps(report),encoding='utf8')
        verify_report(out,manifest['instance_hash'])
        # An active event is observed one tick after the condition frame; its
        # completion must match an actual physics frame and the saved event log.
        event_dir=root/'event_clock';event_dir.mkdir()
        active={'id':'ped','state':'active','required':True,'behavior':'cross',
            'trigger_tick':0,'action_start_tick':1,'completed_tick':None}
        complete=dict(active,state='completed',completed_tick=20)
        log=[{'instance_hash':manifest['instance_hash'],'physical_tick':tick,'frame':100+tick,
              'timestamp':10.+tick/50.,'scenario_event':event} for tick,event in ((1,active),(20,complete))]
        (event_dir/'scenario_events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in log),encoding='utf8')
        event_settings={'instance_hash':manifest['instance_hash'],'actors':[{'role':'ped','required':True,
            'behavior':'cross','trigger':{'ego_s_m':[0.,5.],'earliest_tick':0}}]}
        errors={'events':[]};clock={row['physical_tick']:dict(row,director_route_s_m=row['route_s_m']) for row in physics}
        covered=_event_coverage(event_dir,event_settings,{'scenario_events':[complete]},clock,errors,[])
        assert covered['complete'] and not errors['events']
        fake=dict(complete,completed_tick=300)
        assert not _event_coverage(event_dir,event_settings,{'scenario_events':[fake]},clock,{'events':[]},[])['complete']
        # Two intact 0.1s smoke runs cannot claim full-route admission.
        for directory in dirs:
            short=TaskCriteria(settings)
            for row in physics[:6]:short.observe(row['physical_tick'],row['ego'],[],{'ego':'1'})
            ev=copy.deepcopy(values['evaluation.json']);ev.update(physical_ticks=5,task_criteria=short.report())
            video=copy.deepcopy(values['video_recording.json']);video.update(frames=2,decoded_frames=2,last_carla_frame=105)
            (directory/'evaluation.json').write_text(json.dumps(ev),encoding='utf8')
            (directory/'video_recording.json').write_text(json.dumps(video),encoding='utf8')
            (directory/'physical_evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in physics[:6]),encoding='utf8')
            (directory/'video_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in video_rows[:2]),encoding='utf8')
        smoke=validate_runs(dirs)
        assert smoke['technical_smoke_passed'] and not smoke['passed'],smoke['reasons']
        # A wrong camera source frame invalidates even otherwise aligned video.
        bad=copy.deepcopy(video_rows[:2]);bad[1]['source_cameras'][0]['frame']-=1
        (dirs[1]/'video_frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in bad),encoding='utf8')
        assert not validate_runs(dirs)['checks']['video']
        # Stable scenario identifiers alone do not certify the actual starting layout.
        changed=json.loads((dirs[1]/'carla_initialization.json').read_text())
        changed['actors_at_source'][0]['position'][0]+=2
        (dirs[1]/'carla_initialization.json').write_text(json.dumps(changed),encoding='utf8')
        assert not validate_runs(dirs)['checks']['repeatability'],'different initial positions must fail paired admission'
        try:verify_report(out,manifest['instance_hash'])
        except ValueError:pass
        else:raise AssertionError('a stale report accepted changed evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-dir',type=Path,help='real CARLA Maps directory; requires matching SDK')
    args=parser.parse_args();maps={};results=[]
    for path in sorted((ROOT/'config/carla_scenarios').glob('*.yaml')):
        data=cli.expand_routes(cli.load_yaml(path),path.parent)
        one,manifest=resolve_spec(data,1);two,m2=resolve_spec(data,2)
        assert resolve_spec(data,1)[1]==manifest,'same seed is unstable'
        assert settings_from_manifest(manifest)['instance_hash']==one['instance_hash']
        if data['experiment']['variations']:
            assert manifest['sampled']!=m2['sampled'],'seed does not change intended physical parameters'
        corrupted=copy.deepcopy(manifest);corrupted['settings']['initial_speed_mps']+=1
        try:settings_from_manifest(corrupted)
        except ValueError:pass
        else:raise AssertionError('tampered frozen manifest accepted')
        if args.maps_dir:
            import carla
            from nine_grid.carla_experiment_routes import inspect_instance
            town=data['map']
            if town not in maps:
                xodr=next(args.maps_dir.rglob(town+'.xodr'))
                maps[town]=carla.Map(town,xodr.read_text())
            from nine_grid.carla_bridge import RouteElevationSupport
            for seed in (1,2,3):
                route,_,_=inspect_instance(maps[town],resolve_spec(data,seed)[0],carla)
                if seed==1:RouteElevationSupport(route,1.5,radius_m=120.)
            for endpoint in ('min','max'):
                extreme=copy.deepcopy(data)
                for row in extreme['experiment']['variations']:row['min']=row[endpoint];row['max']=row[endpoint]
                inspect_instance(maps[town],resolve_spec(extreme,1)[0],carla)
        results.append({'name':data['name'],'map':data['map'],'actors':len(one['actors']),
                        'route_m':one['route']['length_m'],'status':data['status']})
    check_admission_contract(manifest)
    assert len(set(x['map'] for x in results))==12,'installed town coverage regressed'
    assert not any(x['status']=='validated' for x in results),'catalogue cannot assert server acceptance'
    print(json.dumps({'passed':True,'scenarios':len(results),'maps':len(set(x['map'] for x in results)),
                      'real_opendrive_checked':bool(args.maps_dir),'server_checked':False,'catalogue':results},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
