import numpy as np
import torch
import json,hashlib,pytest
from pathlib import Path
from LightGenV2.tasks.t17_router_classification.f_router_load_recovery import router_load_objective,replay_order,load_key,prepare_error_replay


def test_error_replay_keeps_every_case_and_repeats_only_fixed_errors():
    order=replay_order(np.random.default_rng(17),[0,1,0,1,0])
    assert np.array_equal(np.bincount(order,minlength=5),[1,3,1,3,1])


def test_modality_balancing_has_finite_gradient_for_unused_port():
    logits=torch.tensor([[1.,2.,5.,-6.]]*6,requires_grad=True);q=logits.softmax(1)
    loss=router_load_objective(q,torch.tensor([0,0,0,1,1,1]));loss.backward()
    assert torch.isfinite(logits.grad).all() and bool((logits.grad[:,3]<0).all())


def test_actual_hard_load_and_both_modalities_determine_warmup_selection():
    a={s:dict(unused_experts=[4],selection_share_CV=.5)for s in ('RGB','SAR')}
    b={s:dict(unused_experts=[],selection_share_CV=.2)for s in ('RGB','SAR')}
    assert load_key(b)<load_key(a)
    b['SAR']['unused_experts']=[4];assert load_key(b)[0]==1


def test_opposite_split_collapses_cannot_be_hidden_by_merged_mean():
    dense=torch.tensor([[.49,.49,.01,.01]]*4+[[.01,.01,.49,.49]]*4,requires_grad=True)
    separate=router_load_objective(dense,torch.tensor([0,0,1,1,2,2,3,3]))
    merged=router_load_objective(dense,torch.tensor([0,0,1,1,0,0,1,1]))
    assert separate>merged


def test_replay_parent_SHA_and_every_ID_label_domain_are_bound(tmp_path):
    folder=tmp_path/'parent';folder.mkdir();(folder/'best_checkpoint.pt').write_bytes(b'identity fixture, not an initialization model')
    raw={s:dict(ids=np.array(['a','b']),labels=np.array([0,1]),domains=np.array([0,1]))for s in ('train','test')};hashes={}
    for split in raw:
        p=folder/f'{split}_predictions.npz';np.savez(p,**raw[split],predictions=np.array([0,0]));hashes[split]=hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=dict(error_replay_parent=dict(run='parent',checkpoint_sha256=hashlib.sha256((folder/'best_checkpoint.pt').read_bytes()).hexdigest(),predictions_sha256=hashes))
    value=prepare_error_replay(raw,cfg,tmp_path);assert np.array_equal(value['wrong']['test'],[False,True]);assert value['record']['evidence']['test']['extra_replay_presentations']==2
    raw['test']['labels'][1]=0
    with pytest.raises(ValueError):prepare_error_replay(raw,cfg,tmp_path)
