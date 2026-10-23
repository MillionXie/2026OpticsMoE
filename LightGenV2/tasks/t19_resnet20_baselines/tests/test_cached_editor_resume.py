import hashlib
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines import train_tasks
from LightGenV2.tasks.t19_resnet20_baselines.queue_remaining import read_plan, worker_command
from LightGenV2.tasks.t19_resnet20_baselines import queue_remaining
from LightGenV2.tasks.t19_resnet20_baselines.train_easy100 import TASK
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import encoder_state_sha256


def test_new_plan_keeps_four_cards_and_separates_last50_extensions(tmp_path):
    plan=read_plan(TASK/'configs/frozen_cache_lgvq50_20261023.json')
    assert plan['max_physical_gpus']==4 and plan['gpu_indices_requested_first']==[2,4,5,6]
    assert len(plan['tasks'])==11
    for target in ('temporal','spatial'):
        task=f'lgvq_{target}_continue50'
        command=worker_command(dict(task=task,phase='formal',name='extension',
            continuation_source=plan['continuation_sources'][task],max_physical_gpus=4,max_jobs_per_gpu=3),
            tmp_path,'GPU-6',plan['backbone'])
        assert command[2].endswith('continue_lgvq50')
        assert command[command.index('--target')+1]==target
        assert '--epochs' not in command and command[command.index('--smoke-batches')+1]=='0'
    settings=plan['engineering_resumes']['abo_conditional_edit']
    assert settings['gpu_uuid']=='GPU-1b963983-7909-af6e-0528-f0f0661ab549'
    assert settings['formal_gate_receipt'] != settings['ready_receipt']
    command=worker_command(dict(task='abo_conditional_edit',phase='formal',name='cached_editor',
        engineering_resume=settings,text_encoder='byte_gru',max_physical_gpus=4,max_jobs_per_gpu=3),
        tmp_path,'GPU-4',plan['backbone'])
    assert command[command.index('--epochs')+1]=='30'
    assert command[command.index('--resume-from')+1].endswith('last_checkpoint.pt')


def test_parent_restores_optimizer_moments_and_rejects_changed_protocol(tmp_path,monkeypatch):
    monkeypatch.setattr(train_tasks,'TASK',tmp_path)
    path=tmp_path/'runs/simulation/parent/last_checkpoint.pt'
    path.parent.mkdir(parents=True)
    model=torch.nn.Linear(2,1)
    model.encoder=torch.nn.Linear(2,2).requires_grad_(False).eval()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    for _ in range(2):
        model(torch.tensor([[1.,2.]])).sum().backward();opt.step();opt.zero_grad()
    identity={'split':'same_original'}
    manifest=dict(task='abo_conditional_edit',backbone='resnet18_imagenet_frozen',seed=42,epochs=30,
        learning_rate=.001,effective_batch=32,microbatch=8,text_encoder='byte_gru',
        training_scope='frozen_visual_trainable_text_and_decoder',data_identity=identity,
        encoder_state_sha256_before=encoder_state_sha256(model.encoder))
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=2,manifest=manifest)
    torch.save(payload,path)
    fresh=torch.nn.Linear(2,1)
    fresh.encoder=torch.nn.Linear(2,2).requires_grad_(False).eval()
    newopt=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    problem=SimpleNamespace(identity=identity,train=list(range(32)))
    parent,audit=train_tasks.load_editor_parent(path,problem,fresh,newopt)
    assert audit['checkpoint_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit['completed_epoch']==2 and parent['epoch']==2
    for left,right in zip(model.parameters(),fresh.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    for old,new in zip(opt.state.values(),newopt.state.values()):
        for key in old:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
    payload['manifest']['seed']=43;torch.save(payload,path)
    with pytest.raises(ValueError,match='protocol'):
        train_tasks.load_editor_parent(path,problem,fresh,newopt)


def test_mapped_cache_batches_preserve_index_order_and_microbatch_boundary():
    data=[{'sample_id':str(i),'image':torch.tensor([float(i)])} for i in range(11)]
    indices=[torch.tensor([4,9,2,1,0,7,8,5]),torch.tensor([10,6,3])]
    loader=train_tasks.CachedIndexBatches(data,indices)
    assert len(loader)==2
    batches=list(loader)
    assert batches[0]['sample_id']==['4','9','2','1','0','7','8','5']
    assert batches[1]['sample_id']==['10','6','3']
    assert len(list(train_tasks.CachedIndexBatches(data,indices,1)))==1


def test_engineering_receipt_cannot_allow_duplicate_editor_formals(tmp_path,monkeypatch):
    monkeypatch.setattr(queue_remaining,'TASK',tmp_path)
    job=dict(task='abo_conditional_edit',phase='formal',status='queued',
             engineering_resume={'formal_gate_receipt':'gate.json'})
    old=dict(task='abo_conditional_edit',phase='formal',status='running')
    assert not queue_remaining.engineering_formal_ready(job,[job])
    (tmp_path/'gate.json').write_text('{}')
    assert not queue_remaining.engineering_formal_ready(job,[job,old])
    old['status']='superseded_for_cached_resume'
    assert queue_remaining.engineering_formal_ready(job,[job,old])
