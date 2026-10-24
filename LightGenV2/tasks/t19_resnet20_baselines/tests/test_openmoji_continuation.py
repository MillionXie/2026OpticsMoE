import copy
import pytest
import torch
from torch import nn
from LightGenV2.tasks.t19_resnet20_baselines.continue_openmoji120 import restore_parent,index_loader,replay_loader,validate_gate
from LightGenV2.tasks.t19_resnet20_baselines.continue_lgvq50 import structured_sha256
from LightGenV2.tasks.t19_resnet20_baselines.frozen_resnet18 import file_sha256


def test_AdamW_resume_matches_uninterrupted_and_parent_immutable(tmp_path):
    torch.manual_seed(42);reference=nn.Linear(2,1);opt=torch.optim.AdamW(reference.parameters(),lr=.001,weight_decay=.0001)
    x=torch.tensor([[1.,2.],[-2.,.5]])
    def step(model,optimizer):
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    for _ in range(3):step(reference,opt)
    parent=tmp_path/'last_checkpoint.pt';torch.save(dict(epoch=90,model=reference.state_dict(),optimizer=opt.state_dict(),
        manifest={'data_identity':{'original':True}}),parent);digest=file_sha256(parent)
    resumed=nn.Linear(2,1);resumed_opt=torch.optim.AdamW(resumed.parameters(),lr=.1)
    restore_parent(resumed,resumed_opt,parent,digest,{'original':True})
    for _ in range(4):step(reference,opt);step(resumed,resumed_opt)
    assert structured_sha256(reference.state_dict())==structured_sha256(resumed.state_dict())
    assert structured_sha256(opt.state_dict())==structured_sha256(resumed_opt.state_dict())
    assert file_sha256(parent)==digest
    assert {int(v['step']) for v in resumed_opt.state.values()}=={7}


@pytest.mark.parametrize('mutation',['sha','epoch','data'])
def test_rejects_wrong_parent_before_changing_live_state(tmp_path,mutation):
    model=nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.001);parent=tmp_path/'last.pt'
    torch.save(dict(epoch=90,model=model.state_dict(),optimizer=opt.state_dict(),manifest={'data_identity':{'fixed':1}}),parent)
    digest=file_sha256(parent);before=structured_sha256(model.state_dict())
    with pytest.raises(ValueError):
        restore_parent(model,opt,parent,'bad' if mutation=='sha' else digest,
            {'fixed':2} if mutation=='data' else {'fixed':1},parent_epoch=91 if mutation=='epoch' else 90)
    assert structured_sha256(model.state_dict())==before


def test_sampler_replay_next_epoch_matches_uninterrupted():
    ids=[f'id{i}' for i in range(67)];reference=index_loader(67);history=[]
    for epoch in range(61,91):
        order=[ids[i] for batch in reference for i in batch.tolist()]
        history.append(dict(epoch=epoch,sample_order_SHA=structured_sha256(order)))
    resumed=index_loader(67);replay_loader(resumed,ids,history)
    assert torch.equal(reference.generator.get_state(),resumed.generator.get_state())
    assert torch.equal(torch.cat(list(reference)),torch.cat(list(resumed)))
    bad=copy.deepcopy(history);bad[7]['sample_order_SHA']='bad'
    with pytest.raises(ValueError,match='sampler replay'):replay_loader(index_loader(67),ids,bad)


def test_gate_requires_same_parent_optimizer_science_and_zero_TEST():
    manifest={k:k+'-fixture' for k in ('profile_SHA','source_sha256','native_source_sha256','parent_SHA','initial_trainable_SHA','optimizer_state_SHA_after_restore')}
    gate=dict(manifest,status='complete',gate_passed=True,test_evaluations=0);validate_gate(gate,manifest)
    for key in manifest:
        bad=dict(gate);bad[key]='changed'
        with pytest.raises(ValueError):validate_gate(bad,manifest)
    for key,value in [('test_evaluations',1),('gate_passed',False),('status','failed')]:
        bad=dict(gate);bad[key]=value
        with pytest.raises(ValueError):validate_gate(bad,manifest)
