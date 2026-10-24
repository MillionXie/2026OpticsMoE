"""Exercise the real probe->formal code without any network, assets, or GPU."""
import json
import shutil
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from LightGenV2.tasks.t19_resnet20_baselines import optimization_stage as stage
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import file_sha256


class TinyPoseData(Dataset):
    def __init__(self,count):
        self.count=count;self.base=SimpleNamespace(training=True,records=[SimpleNamespace(sample_id=f'x{i}') for i in range(count)])
    def __len__(self):return self.count
    def __getitem__(self,i):
        heat=torch.zeros(14,56,56);heat[:,0,0]=1.
        return dict(image=torch.zeros(1,4,4),sample_id=f'x{i}',heatmaps=heat,keypoints=torch.ones(14,2)*1.5,
                    visible=torch.ones(14,dtype=torch.bool),torso_scale=torch.tensor(100.),head_scale=torch.tensor(50.))


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1),requires_grad=False);self.pretrained_provenance={'offline_fixture':True}
        self.eval()
    def train(self,mode=True):super().train(False);return self
    def features(self,x):return (x,)


class TinyPose(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=TinyEncoder();self.heat=nn.Parameter(torch.zeros(14,56,56))
    def forward(self,x):return self.heat[None].expand(len(x),-1,-1,-1)


class TinyProblem:
    def __init__(self):
        self.train=TinyPoseData(33);self.test=TinyPoseData(3);self.batch_size=16
        self.identity={'offline_fixed':True};self.settings=None;self.TEST_calls=0
    def evaluate(self,model,loader,device):
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import PoseMetricAccumulator
        self.TEST_calls+=1;acc=PoseMetricAccumulator();rows=[];model.eval()
        with torch.no_grad():
            for b in loader:
                xy=acc.update(model(b['image'].to(device)),b['keypoints'],b['visible'],b['torso_scale'],b['head_scale'],224)
                rows.extend(dict(sample_id=sid,prediction_xy=x.tolist()) for sid,x in zip(b['sample_id'],xy))
        return acc.compute(),rows


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'task';root.mkdir();(root/'configs').mkdir()
    for name in stage.SOURCES:shutil.copyfile(stage.TASK/name,root/name)
    parent=root/'runs/simulation/parent60';parent.mkdir(parents=True)
    problem=TinyProblem();model=TinyPose()
    torch.save(dict(epoch=60,model=model.state_dict(),manifest={'data_identity':problem.identity}),parent/'last_checkpoint.pt')
    digest=file_sha256(parent/'last_checkpoint.pt')
    plan=dict(probe_steps=2,epochs_per_stage=2,tasks={'lsp':dict(parent='parent60',parent_sha256=digest,
        loss='normalized_Gaussian_spatial_cross_entropy',probe_minimum_train_score=.8,final_epoch=62)})
    path=root/'configs/plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(stage,'TASK',root);monkeypatch.setattr(stage,'COUNTS',{'lsp':(33,3)})
    monkeypatch.setattr(stage,'Problem',lambda task:problem)
    monkeypatch.setattr(stage,'build_model',lambda *a,**k:TinyPose())
    monkeypatch.setattr(stage,'require_published_source',lambda:None)
    monkeypatch.setattr(stage,'git',lambda *a:'offline_published_fixture')
    monkeypatch.setattr(stage,'forward_features',lambda model,task,features,tokens=None:model(features[0]))
    return root,path,problem,parent,digest


def args(root,plan,phase):
    return SimpleNamespace(task='lsp',phase=phase,plan=plan,device='cpu',gpu_uuid='GPU-offline',
        run_dir=root/'runs'/('smoke/probe' if phase=='probe' else 'simulation/formal'),
        probe_result=root/'runs/smoke/probe/result.json')


def test_real_probe_then_fixed_formal_only_once_TEST_and_preserves_parent(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    root,path,problem,parent,digest=fixture(tmp_path,monkeypatch)
    probe=args(root,path,'probe');stage.run(probe)
    result=json.loads((probe.run_dir/'result.json').read_text())
    assert result['gate_passed'] and result['test_evaluations']==problem.TEST_calls==0
    formal=args(root,path,'formal');stage.run(formal)
    result=json.loads((formal.run_dir/'result.json').read_text())
    payload=torch.load(formal.run_dir/'last_checkpoint.pt',weights_only=False)
    assert result['selected_epoch']==payload['epoch']==62
    assert result['test_evaluations']==problem.TEST_calls==1 and result['selection_uses_test'] is False
    assert result['optimizer_steps']==4
    assert all(int(s['step'])==4 for s in payload['optimizer']['state'].values())
    assert file_sha256(parent/'last_checkpoint.pt')==digest==result['parent_SHA']
    assert len(json.loads((formal.run_dir/'fixed_weight_predictions.json').read_text()))==3
    assert len(payload['manifest']['frozen_encoder_audits'])==2


def test_formal_rejects_failed_train_gate_and_preserves_parent(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    root,path,problem,parent,digest=fixture(tmp_path,monkeypatch)
    probe=args(root,path,'probe');stage.run(probe)
    file=probe.run_dir/'result.json';value=json.loads(file.read_text());value['gate_passed']=False;file.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='TRAIN-only probe'):stage.run(args(root,path,'formal'))
    assert problem.TEST_calls==0 and file_sha256(parent/'last_checkpoint.pt')==digest
