"""Official-world demonstration launch plans, separate from paper A/B/C methods.

The official evaluator remains the sole owner of the world, actors, route,
weather, event triggers and scoring. Only its ego Agent entry point changes.
Planning this command never constructs a CARLA client or starts a simulator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

from . import OFFICIAL_COMMIT
from .demo_catalog import FAMILIES

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'fail2drive-official-demonstrations-1.0'
CATALOG = ROOT/'config'/'fail2drive_official_demonstrations.json'
AGENT = ROOT/'src'/'ep_driver_fail2drive'/'official_demo_agent.py'


def _launcher():
    spec = importlib.util.spec_from_file_location('_official_demo_original_launcher', ROOT/'scripts'/'run_fail2drive.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                     allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_catalog(path=None):
    value = _read(Path(path or CATALOG))
    if (value.get('schema') != SCHEMA or value.get('official_commit') != OFFICIAL_COMMIT
            or value.get('purpose') != 'demonstration_only'
            or value.get('scenario_mutations') is not False):
        raise ValueError('Official demonstration catalog identity or world-preservation policy is invalid')
    rows = value.get('demonstrations')
    if not isinstance(rows,list) or len(rows) != 17:
        raise ValueError('Expected one canonical original Generalization route for each of 17 families')
    families,identities = set(),set()
    for row in rows:
        if (not isinstance(row,dict) or row.get('family') not in FAMILIES or row['family'] in families
                or type(row.get('route_id')) is not int or row['route_id'] in identities):
            raise ValueError('Unknown, duplicate or invalid official demonstration family/route')
        filename = 'Generalization_%s_%04d.xml'%(row['family'],row['route_id'])
        if row.get('routes') != 'fail2drive_split/'+filename:
            raise ValueError('Canonical demonstration must name its original official Generalization XML')
        if not isinstance(row.get('scenario_types'),list) or not row['scenario_types']:
            raise ValueError('Canonical route needs its expected official scenario types')
        families.add(row['family']);identities.add(row['route_id'])
    if families != set(FAMILIES):
        raise ValueError('Canonical route catalog does not cover all 17 official families')
    return deepcopy(value)


def _route_details(path, source, launcher):
    identity = launcher.route_identity(path)
    match = re.fullmatch(r'(Base|Generalization)_(.+)_\d{4}\.xml',path.name)
    family = match.group(2)
    if family not in FAMILIES:
        raise ValueError('Official XML has an unknown Fail2Drive family: '+family)
    if not path.resolve().is_relative_to(source.resolve()):
        raise ValueError('Demonstration routes must remain inside the verified official checkout')
    root = ET.parse(launcher.io_path(path)).getroot()
    route = root.find('route')
    if route is None or not route.attrib.get('town'):
        raise ValueError('Original official route lacks its map identity')
    return dict(official_route_id=identity,scenario_family=family,variant=match.group(1),
        map=route.attrib['town'],official_routes_path=str(path),
        routes_sha256=hashlib.sha256(launcher.io_path(path).read_bytes()).hexdigest(),
        scenario_types=[row.attrib['type'] for row in route.findall('.//scenario')],
        declared_actor_models=[dict(scenario_type=scenario.attrib['type'],parameter=child.tag,attributes=dict(child.attrib))
            for scenario in route.findall('.//scenario') for child in scenario
            if child.tag in ('vehicle','walker','pedestrian','prop','image','objects','blocker_model')],
        ego_model_policy='unmodified_official_RouteScenario_spawn;actual_actor_identity_is_captured_in_run',
        original_xml_unchanged=True,source_relative_path=path.relative_to(source).as_posix())


def _expert_preflight(source):
    # This module uses only the standard library at import time. Validate the
    # exact expert sources and real speed-limit data before any evaluator or
    # CARLA world can be constructed (Git LFS pointer files must fail here).
    from .official_expert import source_identity
    return source_identity(source)


def _expert_hashes(source, launcher):
    evidence = _expert_preflight(source)
    files = sorted(path for path in launcher.io_path(source/'team_code').rglob('*.py')
                   if '__pycache__' not in path.parts)
    if not any(path.name=='autopilot.py' for path in files):
        raise ValueError('Pinned official checkout lacks its data-collection AutoPilot')
    result = {path.relative_to(launcher.io_path(source)).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    result.update(evidence['data_sha256'])
    return result


def _code_hashes(launcher):
    files = [ROOT/'scripts'/'capture_fail2drive_official.py',Path(__file__),AGENT,
             ROOT/'src'/'ep_driver_fail2drive'/'official_expert.py']
    result = {}
    for path in files:
        if not path.is_file():
            raise ValueError('Official demonstration implementation is incomplete: '+str(path))
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['paper_adapter_source_sha256'] = launcher.adapter_source_hash()
    return result


def _dependency_preflight(evaluator_python, env, source):
    """Import only expert prerequisites; do not instantiate an Agent or Client."""
    script = r'''
import hashlib, importlib, json, sys
from pathlib import Path
requirements = (
    ('carla', ('VehicleControl',)), ('numpy', ('ndarray',)), ('cv2', ('imencode',)),
    ('scipy.interpolate', ('interp1d',)), ('scipy.spatial', ('cKDTree',)),
    ('scipy.integrate', ('RK45',)),
    ('agents.navigation.local_planner', ('RoadOption',)),
    ('agents.navigation.global_route_planner', ('GlobalRoutePlanner',)),
    ('srunner.scenariomanager.carla_data_provider', ('CarlaDataProvider',)),
    ('leaderboard.autoagents.autonomous_agent', ('AutonomousAgent',)),
    ('leaderboard.autoagents.autonomous_agent_local', ('AutonomousAgent',)),
)
modules, failures = {}, []
for name, attributes in requirements:
    try:
        module = importlib.import_module(name)
        for attribute in attributes:
            if not hasattr(module, attribute):
                raise ImportError(name+'.'+attribute+' is absent')
        path = Path(module.__file__).resolve()
        modules[name] = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'version':getattr(module, '__version__', None)}
    except Exception as error:
        failures.append({'module':name,'error':type(error).__name__+': '+str(error)})
print(json.dumps({'python':sys.executable,'modules':modules,'failures':failures,
                  'connected_to_carla':False,'agent_constructed':False,'expert_setup_called':False}))
sys.exit(2 if failures else 0)
'''
    result = subprocess.run([str(evaluator_python),'-c',script],env=dict(env),cwd=ROOT,
                            capture_output=True,text=True,timeout=60)
    try:
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError,IndexError) as error:
        raise ValueError('Expert dependency import preflight produced no valid result: '+result.stderr.strip()) from error
    if result.returncode or evidence.get('failures'):
        raise ValueError('Expert dependency import preflight failed; no evaluator/world started: '+
                         json.dumps(evidence.get('failures') or result.stderr.strip(),ensure_ascii=False))
    from .official_expert import LOCAL_BASE_SHA256
    local_base = evidence.get('modules',{}).get('leaderboard.autoagents.autonomous_agent_local',{})
    if local_base.get('sha256') != LOCAL_BASE_SHA256:
        raise ValueError('Expert dependency import resolved a different official local agent base')
    expected = (source/'leaderboard'/'leaderboard'/'autoagents'/'autonomous_agent_local.py').resolve()
    if Path(local_base.get('path','')).resolve() != expected:
        raise ValueError('Expert dependency import resolved outside the verified official checkout')
    return evidence


def select_routes(args, catalog, source, launcher):
    route_input = launcher.absolute(args.routes,source)
    if route_input.is_dir():
        candidates = sorted(route_input.glob('*.xml'))
    elif route_input.is_file():
        candidates = [route_input]
    else:
        raise ValueError('Official route input is missing')
    if not candidates:
        raise ValueError('Official route input contains no split XML')
    indexed = {}
    for path in candidates:
        details = _route_details(path,source,launcher)
        if details['official_route_id'] in indexed:
            raise ValueError('Official route IDs repeat in the selected input')
        indexed[details['official_route_id']] = (path,details)
    if args.route_ids is not None:
        wanted = list(args.route_ids)
        if not wanted or len(set(wanted))!=len(wanted) or any(type(x) is not int for x in wanted):
            raise ValueError('Choose distinct explicit official route IDs')
    else:
        selected = catalog['demonstrations'] if args.all_families else [
            row for row in catalog['demonstrations'] if row['family']==args.family]
        wanted = [row['route_id'] for row in selected]
        if not wanted:
            raise ValueError('Select one family, explicit route IDs or all 17 families')
    if not set(wanted).issubset(indexed):
        raise ValueError('Selected original official routes are absent: '+str(sorted(set(wanted)-set(indexed))))
    rows = [indexed[identity] for identity in wanted]
    if args.route_ids is None:
        canonical = {row['route_id']:row for row in catalog['demonstrations']}
        for path,details in rows:
            expected = canonical[details['official_route_id']]
            if (details['source_relative_path'] != expected['routes']
                    or details['scenario_types'] != expected['scenario_types']):
                raise ValueError('Canonical original route changed its path or official scenario composition')
    return rows


def plan(args, *, preflight=None, launcher=None):
    """Build exact commands/config identities without writing or connecting CARLA."""
    launcher = launcher or _launcher()
    catalog = load_catalog(args.catalog)
    if args.port != 39000 or args.traffic_manager_port != 39003:
        raise ValueError('Use the isolated official Fail2Drive ports 39000/39003')
    seeds = args.seeds or [args.seed if args.seed is not None else 1]
    if any(type(seed) is not int or seed<1 for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Official repetition seeds must be distinct positive integers')
    source = launcher.absolute(args.fail2drive_root)
    if preflight is None:
        official = launcher.verify_official(source)
        simulator = launcher.verify_simulator(launcher.absolute(args.simulator_root))
        sdk = launcher.verify_sdk(args.evaluator_python or sys.executable,simulator)
        preflight = official,simulator,sdk
    routes = select_routes(args,catalog,source,launcher)
    implementation = _code_hashes(launcher)
    expert = _expert_hashes(source,launcher)
    expert_identity = _digest(expert)
    base = launcher.absolute(args.output_dir or ROOT/'output'/'official_demonstrations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    allowed = (ROOT/'output'/'official_demonstrations').resolve()
    if not base.is_relative_to(allowed) or base==allowed:
        raise ValueError('Choose a new run directory below this branch output/official_demonstrations/')
    jobs=[]
    dependencies=None
    for seed in seeds:
        for route,details in routes:
            original_args = argparse.Namespace(fail2drive_root=str(source),simulator_root=args.simulator_root,
                config=args.config,execution_mode='A_base',reflection='off',reflection_memory_dir=None,
                parameters_path=args.parameters_path,models_path=None,prompts_path=None,
                worker_python=args.worker_python,evaluator_python=args.evaluator_python,
                planner_hz=10,video=True,output_dir=str(base),routes=str(route),seed=seed,
                routes_subset='',host=args.host,port=args.port,traffic_manager_port=args.traffic_manager_port,
                debug=args.debug,record=True,resume=False,_adapter_source_sha256=implementation['paper_adapter_source_sha256'])
            config,runtime,command,env,official,_,_ = launcher.prepare(original_args,preflight=preflight)
            config.update(execution_mode='offline_demonstration',official_root=str(source),
                planner_enabled=False,model_enabled=False,reflection_enabled=False,reflection_memory_dir=None)
            config['official_demo'] = dict(driver='official_autopilot',purpose='demonstration_only',
                official_root=str(source),scenario_family=details['scenario_family'],
                official_route_id=details['official_route_id'],scenario_mutations=False,
                official_expert_source_sha256=expert_identity,expert_privileged_only_offline=True,
                data_use='targeted_seen_scenario_demonstration')
            command=list(command)
            command[command.index('--agent')+1] = str(AGENT)
            # Preserve the official import search path. The expert adapter
            # privately loads its pinned helpers to avoid bare-module clashes;
            # paper and SDK imports retain precedence.
            env=dict(env)
            env['PYTHONPATH'] += os.pathsep+str(source/'team_code')
            env['DATAGEN']='0';env['DEBUG_CHALLENGE']='0';env['HISTOGRAM']='0';env['TP_STATS']='0'
            env.pop('SAVE_PATH',None)
            if dependencies is None:
                dependencies = _dependency_preflight(command[0],env,source)
            config['official_demo']['dependency_preflight'] = dependencies
            identity,inputs = launcher.fingerprint(config,original_args,route)
            inputs.update(demonstration_schema=SCHEMA,demonstration_command=command,
                demonstration_implementation_sha256=implementation,
                official_expert_files_sha256=expert,official_expert_source_sha256=expert_identity,
                expert_dependency_preflight=dependencies,
                batch_scheduling_policy=dict(continue_on_known_route_rejection=bool(getattr(args,'continue_on_route_failure',False))),
                controlled_environment={key:env.get(key) for key in
                    ('PYTHONPATH','DATAGEN','DEBUG_CHALLENGE','HISTOGRAM','TP_STATS','SAVE_PATH')})
            identity=_digest(inputs)
            checkpoint=Path(config['archive_association']['official_checkpoint'])
            launch=runtime.with_name('%04d_launch.json'%details['official_route_id'])
            metadata=dict(details,schema=SCHEMA,official_commit=OFFICIAL_COMMIT,purpose='demonstration_only',
                driver_kind='official_autopilot',scenario_mutations=False,repetition_seed=seed,
                traffic_manager_seed=config['archive_association']['traffic_manager_seed'],
                command=command,config=config,official=official,fingerprint=identity,fingerprint_inputs=inputs,
                official_expert_source_sha256=expert_identity,official_expert_files_sha256=expert,
                implementation_sha256=implementation,official_routes_unmodified=True,custom_episode_limit=None,
                acceptance='official_result_and_finalized_physical_evidence_required',
                data_use='targeted_seen_scenario_demonstration',simulator_started_by_launcher=False,
                ego_policy_is_the_only_substitution=True,
                batch_scheduling_policy=inputs['batch_scheduling_policy'],
                trajectory_identity_note='NPC positions and activation times may react to the changed ego trajectory; scenario code and seed are unchanged')
            jobs.append(dict(config=config,runtime_path=runtime,launch_path=launch,checkpoint=checkpoint,
                agent_output=Path(config['output_dir']),command=command,env=env,metadata=metadata))
    return dict(output_dir=base,jobs=jobs,implementation_sha256=implementation,expert_sha256=expert,
                catalog=catalog,official_root=source,launcher=launcher,
                continue_on_route_failure=bool(getattr(args,'continue_on_route_failure',False)))


def preview(planned):
    return dict(schema=SCHEMA,purpose='demonstration_only',dry_run=True,
        output_dir=str(planned['output_dir']),jobs=[dict(
            official_route_id=job['metadata']['official_route_id'],scenario_family=job['metadata']['scenario_family'],
            map=job['metadata']['map'],repetition_seed=job['metadata']['repetition_seed'],
            traffic_manager_seed=job['metadata']['traffic_manager_seed'],command=job['command'],
            runtime_path=str(job['runtime_path']),checkpoint=str(job['checkpoint']),
            agent_output=str(job['agent_output']),fingerprint=job['metadata']['fingerprint']) for job in planned['jobs']],
        official_routes_unmodified=True,scenario_mutations=False,custom_episode_limit=None,
        simulator_started_by_launcher=False,accepted_for_memory=False,memory_published=False,
        continue_on_known_route_rejection=planned.get('continue_on_route_failure',False))


def _write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def _assert_prepared_or_empty(job):
    runtime,launch,checkpoint,output=job['runtime_path'],job['launch_path'],job['checkpoint'],job['agent_output']
    exists=runtime.exists() or launch.exists() or checkpoint.exists() or (output.exists() and any(output.iterdir()))
    prepared=(runtime.is_file() and launch.is_file() and not checkpoint.exists()
        and (not output.exists() or not any(output.iterdir()))
        and _read(runtime)==job['config'] and _read(launch)==job['metadata'])
    if exists and not prepared:
        raise ValueError('Official demonstration output is not empty; use a new run directory: '+str(output))


def write_plan(planned):
    for job in planned['jobs']:
        _assert_prepared_or_empty(job)
    for job in planned['jobs']:
        seed_root=job['runtime_path'].parent.parent
        for folder in ('run','res','out','err','agent_runs'):
            (seed_root/folder).mkdir(parents=True,exist_ok=True)
        job['agent_output'].mkdir(parents=True,exist_ok=True)
        Path(job['env']['VIZ_PATH']).mkdir(parents=True,exist_ok=True)
        (seed_root/'recorder'/('%04d'%job['metadata']['official_route_id'])).mkdir(parents=True,exist_ok=True)
        _write(job['runtime_path'],job['config']);_write(job['launch_path'],job['metadata'])


def _failure_summary(root,capture):
    """Print bounded diagnostics while retaining the complete raw evidence."""
    failure=None
    try:
        path=root/'capture_failure.json'
        if path.is_file():failure=_read(path)
    except (OSError,ValueError,TypeError):pass
    if not isinstance(failure,dict):
        errors=capture.get('errors',[]) if isinstance(capture,dict) else []
        failure=errors[0] if isinstance(errors,list) and errors and isinstance(errors[0],dict) else {}
    evidence=failure.get('evidence')
    evidence=evidence if isinstance(evidence,dict) else {}
    result=dict(type=str(failure.get('type','unknown_capture_failure'))[:100],
        message=str(failure.get('error','No complete failure record was archived'))[:1200],
        classification=str(evidence.get('classification',evidence.get('schema','unclassified')))[:160])
    for key in ('frame','time'):
        value=failure.get(key)
        if type(value) in (int,float) and math.isfinite(value):result[key]=value
    return result,failure


def _known_route_rejection(root,capture,failure,verdict):
    """Whitelist route-level quality stops; never skip unknown runtime faults."""
    if (not isinstance(capture,dict) or capture.get('complete') is not False
            or not isinstance(capture.get('errors'),list) or len(capture['errors'])!=1
            or verdict.get('official_result_valid') is not True
            or verdict.get('entry_status')!='Finished'
            or verdict.get('official_status')!='Failed - Agent crashed'
            or verdict.get('route_completed') is not False):return None
    recorded=capture['errors'][0]
    if (not isinstance(recorded,dict) or failure.get('type')!='ValueError'
            or any(recorded.get(k)!=failure.get(k) for k in ('frame','time','type','error'))):return None
    try:video=_read(root/'planning'/'video_recording.json')
    except (OSError,ValueError,TypeError):return None
    if (not isinstance(video,dict) or video.get('status')!='complete'
            or video.get('playback_verified') is not True):return None
    evidence=failure.get('evidence')
    if not isinstance(evidence,dict) or evidence.get('valid') is not False:return None
    if (any(type(failure.get(k)) not in (int,float) or not math.isfinite(failure[k]) for k in ('frame','time'))
            or any(evidence.get(k)!=failure.get(k) for k in ('frame','time'))):return None
    from .source_world import SCHEMA as PHYSICAL_SCHEMA
    from .demo_behavior import SCHEMA as BEHAVIOR_SCHEMA
    if (evidence.get('schema')==PHYSICAL_SCHEMA
            and evidence.get('classification')=='physical_world_invalid'
            and evidence.get('reason')=='ego_falling_below_published_road'
            and evidence.get('world_modified') is False
            and failure.get('error')=='Official physical world is invalid: ego_falling_below_published_road'):
        return 'route_physical_grounding_rejection'
    contacts=evidence.get('hard_geometry_penetration_ids')
    if (evidence.get('schema')==BEHAVIOR_SCHEMA and isinstance(contacts,list) and contacts
            and all(type(v) is int for v in contacts)
            and evidence.get('driving_control_generated') is False
            and failure.get('error')=='Official demonstration has measured bounding-box penetration'):
        return 'route_demonstration_geometry_rejection'
    return None


def run_plan(planned):
    write_plan(planned)
    outcomes=[]
    stop_reason=None
    for job in planned['jobs']:
        launcher=planned['launcher']
        if (_code_hashes(launcher)!=planned['implementation_sha256']
                or _expert_hashes(planned['official_root'],launcher)!=planned['expert_sha256']):
            raise ValueError('Demonstration code changed after planning; use a new consistent run')
        metadata=job['metadata']
        source=Path(metadata['official_routes_path'])
        if hashlib.sha256(launcher.io_path(source).read_bytes()).hexdigest()!=metadata['routes_sha256']:
            raise ValueError('Original official route changed after planning')
        route=metadata['official_route_id'];seed_root=job['runtime_path'].parent.parent
        print('Running official-world demonstration %04d (%s), seed %d'%(route,metadata['scenario_family'],metadata['repetition_seed']),flush=True)
        with (seed_root/'out'/('%04d_out.log'%route)).open('x',encoding='utf-8') as out, \
                (seed_root/'err'/('%04d_err.log'%route)).open('x',encoding='utf-8') as err:
            code=subprocess.call(job['command'],cwd=ROOT,env=job['env'],stdout=out,stderr=err)
        verdict=launcher.official_result(job['checkpoint'],route)
        archives=sorted(job['agent_output'].glob('agent/*/run.json'))
        result=dict(official_route_id=route,scenario_family=metadata['scenario_family'],
            repetition_seed=metadata['repetition_seed'],evaluator_returncode=code,official_result=verdict,
            checkpoint=str(job['checkpoint']),launch_path=str(job['launch_path']),runtime_path=str(job['runtime_path']),
            agent_output=str(job['agent_output']),run_dir=str(archives[0].parent) if len(archives)==1 else None,
            archive_count=len(archives),finalization_required=True,accepted_for_memory=False,memory_published=False)
        fatal=None;capture=None;failure={}
        if code:
            fatal='evaluator_process_failed'
        elif len(archives)!=1:
            fatal='missing_or_ambiguous_agent_capture'
        else:
            try:
                capture=_read(archives[0])
                if not isinstance(capture,dict):raise ValueError('Agent closing metadata must be an object')
                if capture.get('complete') is not True or capture.get('errors'):
                    fatal='agent_or_physical_source_failed'
            except (OSError,ValueError,TypeError):
                fatal='invalid_agent_closing_metadata'
        if fatal is None and 'Agent crashed' in str(verdict.get('official_status','')):
            fatal='official_agent_crashed'
        if fatal is None and (verdict.get('official_result_valid') is False
                or verdict.get('official_status')=='result_missing_or_invalid'):
            fatal='invalid_official_result'
        if len(archives)==1 and fatal:
            result['failure_summary'],failure=_failure_summary(archives[0].parent,capture)
        elif fatal:
            result['failure_summary']=dict(type=fatal,classification='capture_unavailable',
                message='No unique complete Agent capture; inspect the preserved evaluator logs',
                stdout_path=str(seed_root/'out'/('%04d_out.log'%route)),
                stderr_path=str(seed_root/'err'/('%04d_err.log'%route)))
        rejection=(_known_route_rejection(archives[0].parent,capture,failure,verdict)
                   if fatal and code==0 and len(archives)==1 else None)
        continue_route=bool(planned.get('continue_on_route_failure',False) and rejection)
        result.update(route_failure_reason=fatal,known_route_rejection=rejection,
            continued_after_route_rejection=continue_route,
            batch_stop_reason=None if continue_route else fatal)
        if len(archives)==1:
            command=[job['config']['worker_python'],str(ROOT/'scripts'/'build_fail2drive_official_memory.py'),
                'finalize','--capture',str(archives[0].parent),'--checkpoint',str(job['checkpoint']),
                '--launch',str(job['launch_path'])]
            result['finalize_command']=command
            result['finalize_command_posix']=shlex.join(command)
        _write(job['launch_path'].with_name('%04d_collection.json'%route),result)
        outcomes.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        if fatal is not None and not continue_route:
            stop_reason=fatal
            break
    finished=len(outcomes)==len(planned['jobs'])
    # Finishing every scheduled process does not certify collection success:
    # the last process may return zero/Perfect before Agent teardown reports a
    # physical-source or archive failure. Keep attempt counts separate while
    # making every batch-success flag fail closed on that outcome.
    healthy=not any(row.get('route_failure_reason') for row in outcomes)
    processes_ok=finished and healthy and all(row['evaluator_returncode']==0 for row in outcomes)
    routes_ok=finished and healthy and all(row['official_result'].get('route_completed') is True for row in outcomes)
    return dict(schema=SCHEMA,collection_finished=finished,results=outcomes,
        planned_jobs=len(planned['jobs']),finished_jobs=len(outcomes),stopped_early=not finished,batch_stop_reason=stop_reason,
        all_routes_attempted=finished,
        completed_with_route_failures=finished and any(row.get('route_failure_reason') or
            row['official_result'].get('route_completed') is not True for row in outcomes),
        continue_on_known_route_rejection=planned.get('continue_on_route_failure',False),
        collection_succeeded=processes_ok and routes_ok,
        evaluator_runs_completed=processes_ok,
        official_routes_completed=routes_ok,
        finalization_required=True,accepted_for_memory=False,memory_published=False)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--list',action='store_true',help='Print canonical official routes without simulator/SDK preflight')
    result.add_argument('--catalog',default=str(CATALOG))
    selection=result.add_mutually_exclusive_group()
    selection.add_argument('--family',choices=FAMILIES)
    selection.add_argument('--route-ids',nargs='+',type=int)
    selection.add_argument('--all-families',action='store_true')
    result.add_argument('--fail2drive-root')
    result.add_argument('--simulator-root')
    result.add_argument('--routes',default='fail2drive_split')
    result.add_argument('--config',default='config/fail2drive.example.json')
    result.add_argument('--parameters-path')
    result.add_argument('--worker-python')
    result.add_argument('--evaluator-python')
    result.add_argument('--output-dir')
    seeds=result.add_mutually_exclusive_group()
    seeds.add_argument('--seed',type=int)
    seeds.add_argument('--seeds',nargs='+',type=int)
    mode=result.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='Default: validate and print commands; no write or evaluator')
    mode.add_argument('--prepare-only',action='store_true',help='Write immutable runtime/launch provenance only')
    mode.add_argument('--run',action='store_true',help='Explicitly run the unmodified official evaluator with the demonstration Agent')
    result.add_argument('--host',default='127.0.0.1')
    result.add_argument('--port',type=int,default=39000)
    result.add_argument('--traffic-manager-port',type=int,default=39003)
    result.add_argument('--debug',type=int,choices=(0,1,2,3),default=0)
    result.add_argument('--continue-on-route-failure',action='store_true',
        help='Continue after a structured route grounding/geometry rejection only; unknown Agent, SDK, encoder and source failures still stop; any failure retains exit 2')
    return result


def main(argv=None):
    command=parser();args=command.parse_args(argv)
    try:
        if args.list:
            print(json.dumps(load_catalog(args.catalog),ensure_ascii=False,indent=2))
            return 0
        if not args.fail2drive_root or not args.simulator_root:
            command.error('--fail2drive-root and --simulator-root are required for validated plans')
        if not (args.family or args.route_ids is not None or args.all_families):
            command.error('Select --family, --route-ids or --all-families explicitly')
        planned=plan(args)
        if args.run:
            result=run_plan(planned)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result['collection_succeeded'] else 2
        if args.prepare_only:
            write_plan(planned)
        result=preview(planned)
        result.update(dry_run=not args.prepare_only,prepared_only=args.prepare_only)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:
        print(json.dumps(dict(error=type(error).__name__+': '+str(error),accepted_for_memory=False,
                              memory_published=False),ensure_ascii=False),file=sys.stderr)
        return 2
