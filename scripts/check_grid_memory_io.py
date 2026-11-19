"""隔离临时库验证八向 NPZ 读写、旧库兼容及冻结审核；不修改正式记忆。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from shapely.geometry import box

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from nine_grid import offline_memory as offline
from nine_grid.configuration import (GRID_METRIC_VERSION, OCC_METRIC_VERSION,
    RETRIEVAL_METRIC_VERSION, grid_metric_signature, occupancy_distance_config,
    occupancy_metric_signature)
from nine_grid.contracts import Actor, Lane, SceneSnapshot, Representations, VERSION, ANSWER_PROTOCOL_VERSION
from nine_grid.grid_storage import GRID_STORAGE_VERSION, save_grid, load_grid
from nine_grid.memory import MemoryStore
from nine_grid.paths import filesystem_path
from nine_grid.representations import build_grid, FEATURE_VERSION, GRID_REPRESENTATION_VERSION, OCC_REPRESENTATION_VERSION
from nine_grid.retrieval import grid_comparison


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False),encoding='utf-8')


def comparison(a,b):
    return {k:v for k,v in grid_comparison(a,b).items() if k!='timing_s'}


def representations(root):
    lane=Lane('current',np.array([[-100.,0.],[200.,0.]]),
              np.array([[-100.,-2.],[200.,-2.],[200.,2.],[-100.,2.]]),4.,30.)
    ego=Actor('ego','vehicle',np.zeros(2),np.array([10.,0.]),np.zeros(2),0.,4.5,1.8,'current')
    front=Actor('front_car','vehicle',np.array([20.,0.]),np.array([8.,0.]),np.zeros(2),0.,4.5,1.8,'current')
    scene=SceneSnapshot('fixture:1',0.,ego,[front],[lane],lane.center.copy(),['current'],
        box(-100,-2,200,2),'fixture',diagnostics={
            'visibility_polygon':[[-100.,-25.],[200.,-25.],[200.,25.],[-100.,25.]],
            'has_occlusion':False,'lane_map_complete':True})
    image=root/'fixture.png'
    Image.new('RGB',(4,4),(0,80,0)).save(image)
    return Representations(scene.frame_id,scene.time,str(image),str(image),
        np.zeros((6,50,25),np.float32),np.ones((6,50,25),np.float32),build_grid(scene),
        np.ones(768,np.float32)/np.sqrt(768),dino_signature='test:grid-memory-io',
        occupancy_config=occupancy_distance_config())


def online_record(reps):
    stamp=dict(frame_id=reps.frame_id,time=reps.time)
    return dict(id='online-fixture',version=VERSION,revision=1,status='active',**stamp,
        answer_protocol_version=ANSWER_PROTOCOL_VERSION,assets={},
        goal_reference=dict(**stamp,reference_id='fixture-reference',reference_version=1,
            request_version=1,s_ego_m=0.,d_ego_m=0.,points=[[0.,0.],[100.,0.]],
            source_pose=[0.,0.,0.],source_lane_id='current',route_lane_ids=['current'],
            definition='fixture Frenet',limits=dict(minimum_delta_s_m=1.,maximum_delta_s_m=40.,maximum_lateral_slope=1.)),
        situation=dict(**stamp,scenario='fixture',坐标='world_xy_m',情境='保持车距',
            原点世界姿态=[0.,0.,0.],自车速度=10.,可见对象=[]),
        answer=dict(attention=[],local_goal=dict(delta_s_m=20.,d_m=0.),
            explanation=dict(情境='保持车距',依据='合法车道',参考答案='直行',适用条件='道路可通行')),
        quality=dict(**stamp,version=VERSION,valid=True,score=1.,collision=False,precollision=False,reasons=[],
            metrics=dict(gates=dict(collision_free=True,on_road=True,correct_direction=True,rules=True),
                progress=1.,ttc=1.,comfort=1.,progress_m=20.,reference_progress_m=20.)))


def offline_library(root,reps,extension='.npz',storage_version=GRID_STORAGE_VERSION):
    """构造完整、显式标记为测试的四帧审核库，无假装通过真实人工审核。"""
    identity='a'*32
    folder=root/'records'/identity
    folder.mkdir(parents=True)
    annotation_id='b'*64
    experience=dict(status='accept',reject_reason=None,sections={name:'测试夹具' for name in offline.SECTIONS},uncertainty='')
    text=offline._experience_text(experience,offline.ANNOTATION_PROTOCOL,10000)
    (folder/'experience.txt').write_text(text,encoding='utf-8')
    annotation=dict(status='completed',test_only=True,protocol_version=offline.ANNOTATION_PROTOCOL,
        request_fingerprint=annotation_id,identity=dict(protocol_version=offline.ANNOTATION_PROTOCOL,mode='api',max_characters=10000))
    write_json(folder/f'audit/annotations/{annotation_id}.json',annotation)
    draft=dict(task_id=identity,protocol=offline.LIBRARY_PROTOCOL,annotation=annotation_id,experience=experience)
    write_json(folder/'audit/experience.json',draft)
    supervision=dict(audit=[dict(collision=False,offroad=False)])
    write_json(folder/'audit/supervision.json',supervision)
    timeline=[dict(offset_s=float(i),time_s=float(i),frame_id=reps.frame_id if i==0 else f'future:{i}',native_image='native.png') for i in range(4)]
    task=dict(id=identity,protocol=offline.LIBRARY_PROTOCOL,anchor_frame_id=reps.frame_id,
        anchor_time_s=0.,sampling=dict(time_tolerance_s=.001),timeline=timeline)
    write_json(folder/'audit/task.json',task)
    hashes={'task':offline._digest(task),'supervision':digest_file(folder/'audit/supervision.json')}
    for i,row in enumerate(timeline):
        target=folder/'evidence'/str(i)
        target.mkdir(parents=True)
        (target/'native.png').write_bytes(Path(reps.bev_path).read_bytes())
        write_json(target/'scene.json',dict(frame_id=row['frame_id'],time=row['time_s']))
        if i==0:
            paths=dict(bev='bev.png',feature_image='feature.png',occupancy='occupancy.npz',
                       grid='grid'+extension,grid_preview='grid_preview.png',bev_feature='bev_feature.npy')
            for key in ('bev','feature_image','grid_preview'):
                (target/paths[key]).write_bytes(Path(reps.bev_path).read_bytes())
            np.savez_compressed(target/paths['occupancy'],occ=reps.occ,known=reps.occ_known)
            np.save(target/paths['bev_feature'],reps.bev_feature,allow_pickle=False)
            if extension=='.npz':
                save_grid(target/paths['grid'],reps.grid)
            else:
                write_json(target/paths['grid'],reps.grid)
            meta=dict(asset_protocol=offline.ASSET_PROTOCOL,version=VERSION,representation_version=FEATURE_VERSION,
                grid_representation_version=GRID_REPRESENTATION_VERSION,occupancy_representation_version=OCC_REPRESENTATION_VERSION,
                grid_metric_version=GRID_METRIC_VERSION,occupancy_metric_version=OCC_METRIC_VERSION,
                retrieval_metric_version=RETRIEVAL_METRIC_VERSION,status='ready',current_only=True,
                frame_id=reps.frame_id,time=0.,coordinates=reps.coordinates,dino_signature=reps.dino_signature,
                occupancy_config=reps.occupancy_config,occupancy_metric_signature=occupancy_metric_signature(reps.occupancy_config),
                grid_metric_signature=grid_metric_signature(reps.grid['distance_config']),paths=paths,
                sha256={key:digest_file(target/name) for key,name in paths.items()})
            if storage_version is not None:
                meta['grid_storage_version']=storage_version
        else:
            meta=dict(frame_id=row['frame_id'],time=row['time_s'])
        write_json(target/'features.json',meta)
        hashes[str(row['offset_s'])]={name:digest_file(target/name) for name in ('features.json','scene.json','native.png')}
    evidence_digest=offline._digest(hashes)
    write_json(folder/'audit/review.json',dict(decision='approve',reviewer='ISOLATED TEST FIXTURE',
        draft_sha256=digest_file(folder/'audit/experience.json'),evidence_digest=evidence_digest))
    write_json(folder/'record.json',dict(protocol=offline.LIBRARY_PROTOCOL,record_origin='offline_retrospective',
        id=identity,status='active',revision=1,test_only=True,frame_id=reps.frame_id,time=0.,
        feature_dir='evidence/0',text_path='experience.txt',annotation_is_ground_truth=False,mpc_tested=False,pdm_tested=False,
        evidence=[dict(offset_s=row['offset_s'],time_s=row['time_s'],path=f'evidence/{i}') for i,row in enumerate(timeline)],
        evidence_digest_at_review=evidence_digest,experience=experience))
    write_json(folder/'checksums.json',{p.relative_to(folder).as_posix():digest_file(p) for p in folder.rglob('*') if p.is_file()})
    write_json(root/'manifest.json',dict(protocol=offline.LIBRARY_PROTOCOL,frozen=True,contains_test_records=True,
        records=[dict(id=identity,path=f'records/{identity}/record.json',checksums_sha256=digest_file(folder/'checksums.json'))]))
    write_json(root/'freeze.json',dict(manifest_sha256=digest_file(root/'manifest.json')))
    return folder


class MemoryIO(unittest.TestCase):
    def setUp(self):
        scratch=filesystem_path(ROOT/'outputs'/'validation')
        scratch.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='grid-memory-io-',dir=scratch)
        self.root=filesystem_path(self.temporary.name)
        self.assertTrue(self.root.is_relative_to(scratch))
        self.addCleanup(self.temporary.cleanup)
        self.reps=representations(self.root)
        self.store=MemoryStore(self.root/'online',self.root/'online',allow_test_features=True)

    def write_online(self):
        payload=self.store.write(online_record(self.reps),self.reps)
        path=next((self.root/'online'/'records').glob('*/record.json'))
        return path,payload

    def test_online_npz_write_read(self):
        path,payload=self.write_online()
        self.assertEqual(payload['features']['grid'],'grid.npz')
        self.assertEqual(payload['grid_storage_version'],GRID_STORAGE_VERSION)
        self.assertNotIn('grid.json',payload['checksums'])
        self.assertFalse((path.parent/'grid.json').exists())
        records=self.store.snapshot()
        self.assertEqual(len(records),1,self.store.audit)
        self.assertEqual(comparison(self.reps.grid,self.reps.grid),
                         comparison(self.reps.grid,records[0]['_representations'].grid))
        self.assertEqual(records[0]['_representations'].grid_path,str(path.parent/'grid.npz'))

    def test_online_old_json_read_only(self):
        path,payload=self.write_online()
        (path.parent/'grid.npz').unlink()
        write_json(path.parent/'grid.json',self.reps.grid)
        payload['features']['grid']='grid.json'
        del payload['grid_storage_version'],payload['checksums']['grid.npz']
        payload['checksums']['grid.json']=digest_file(path.parent/'grid.json')
        write_json(path,payload)
        old=(path.parent/'grid.json').read_bytes()
        records=self.store.snapshot()
        self.assertEqual(len(records),1,self.store.audit)
        self.assertEqual(records[0]['_representations'].grid,self.reps.grid)
        self.assertEqual((path.parent/'grid.json').read_bytes(),old)
        self.assertFalse((path.parent/'grid.npz').exists())

    def test_online_npz_corruption_does_not_fallback(self):
        path,payload=self.write_online()
        (path.parent/'grid.npz').write_bytes(b'corrupt npz')
        payload['checksums']['grid.npz']=digest_file(path.parent/'grid.npz')
        write_json(path,payload)
        self.assertEqual(self.store.snapshot(),[])
        self.assertTrue(self.store.audit)
        write_json(path.parent/'grid.json',self.reps.grid)
        self.assertEqual(self.store.snapshot(),[])
        self.assertIn('同时存在',self.store.audit[0]['reason'])

    def test_online_declared_path_containment(self):
        path,payload=self.write_online()
        payload['features']['grid']='../grid.npz'
        write_json(path,payload)
        self.assertEqual(self.store.snapshot(),[])
        self.assertIn('越出',self.store.audit[0]['reason'])

    def test_online_checksum_required(self):
        path,payload=self.write_online()
        del payload['checksums']['grid.npz']
        write_json(path,payload)
        self.assertEqual(self.store.snapshot(),[])
        self.assertIn('校验项',self.store.audit[0]['reason'])

    def test_offline_json_and_npz_frozen_snapshot(self):
        for suffix in ('.json','.npz'):
            with self.subTest(suffix=suffix):
                root=self.root/suffix[1:]
                offline_library(root,self.reps,suffix,GRID_STORAGE_VERSION if suffix=='.npz' else None)
                store=MemoryStore(root,allow_test_features=True)
                records=store.snapshot()
                self.assertEqual(len(records),1,store.audit)
                self.assertEqual(comparison(self.reps.grid,self.reps.grid),
                                 comparison(self.reps.grid,records[0]['_representations'].grid))
                self.assertTrue(records[0]['_representations'].grid_path.endswith(suffix))
                self.assertEqual(MemoryStore(root).snapshot(),[])

    def test_offline_checksum_and_audit_retained(self):
        root=self.root/'frozen'
        folder=offline_library(root,self.reps)
        (folder/'evidence/0/grid.npz').write_bytes(b'changed after review')
        store=MemoryStore(root,allow_test_features=True)
        self.assertEqual(store.snapshot(),[])
        self.assertIn('校验不符',store.audit[0]['reason'])

    def test_offline_unknown_storage_version_rejected(self):
        root=self.root/'unknown-version'
        offline_library(root,self.reps,storage_version='unsupported-v999')
        store=MemoryStore(root,allow_test_features=True)
        self.assertEqual(store.snapshot(),[])
        self.assertIn('存储版本',store.audit[0]['reason'])

    def test_derivation_evidence_preserves_motion(self):
        path=self.root/'grid.npz'
        save_grid(path,self.reps.grid)
        restored=load_grid(path)
        self.assertTrue(offline._same_grid_evidence(self.reps.grid,restored,'grid.json','grid.npz'))
        altered=copy.deepcopy(restored)
        altered['ego']['velocity'][0]+=1e-10
        self.assertFalse(offline._same_grid_evidence(self.reps.grid,altered,'grid.json','grid.npz'))
        old=copy.deepcopy(self.reps.grid)
        old['diagnostic_only']='changed'
        self.assertFalse(offline._same_grid_evidence(old,self.reps.grid,'grid.json','grid.json'))


if __name__=='__main__':
    unittest.main(verbosity=2)
