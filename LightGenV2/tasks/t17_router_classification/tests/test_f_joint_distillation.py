import json
import numpy as np
import pytest
from LightGenV2.tasks.t17_router_classification.f_joint_distillation import assemble_joint_targets,validate_joint_parent
from LightGenV2.tasks.t17_router_classification.train_figure_revision import metric_key
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def fixture(tmp_path):
    raw={s:dict(ids=np.array([s+'R0',s+'S0',s+'R1',s+'S1']),labels=np.array([0,1,1,0]),domains=np.array([0,1,0,1])) for s in ('train','val','test')}
    cfg=dict(dataset='eurosat_original_f',architecture='LG',main_layers=2,num_classes=2,data={},augmentation='D4',alternating_head={'inner_steps':2},teacher=dict(domain_cache_runs={'RGB':'RGB','SAR':'SAR'}))
    for domain,number in [('RGB',0),('SAR',1)]:
        d=tmp_path/domain;d.mkdir();manifest={}
        for split in ('train','test'):
            identity={k:v[raw[split]['domains']==number] for k,v in raw[split].items()}
            np.savez(d/f'teacher_logits_{split}.npz',**identity,logits=np.full((2,8,2),number+1,dtype=np.float32))
            manifest[f'teacher_logits_{split}.npz']=sha256(d/f'teacher_logits_{split}.npz')
        (d/'best_checkpoint.pt').write_bytes(domain.encode())
        (d/'status.json').write_text(json.dumps(dict(state='complete',opposite_domain_used=False,checkpoint_sha256=sha256(d/'best_checkpoint.pt'))))
        (d/'metadata.json').write_text(json.dumps(dict(training_domain=domain,opposite_domain_used=False,original_data={})))
        (d/'cache_manifest.json').write_text(json.dumps(manifest))
    return raw,cfg


@pytest.mark.parametrize('split',['train','test'])
def test_joint_targets_restore_original_interleaved_order_and_are_not_hard_labels(tmp_path,split):
    raw,cfg=fixture(tmp_path);targets,sources=assemble_joint_targets(raw,cfg,tmp_path,split)
    assert targets.shape==(4,8,2) and targets[:,0,0].tolist()==[1,2,1,2]
    assert set(sources)=={'RGB','SAR'}
    assert raw[split]['labels'].tolist()==[0,1,1,0]


def test_joint_targets_reject_wrong_cache_ids_even_with_updated_sha(tmp_path):
    raw,cfg=fixture(tmp_path);p=tmp_path/'SAR/teacher_logits_test.npz'
    with np.load(p) as z:values={k:z[k] for k in z.files}
    values['ids']=values['ids'][::-1];np.savez(p,**values)
    m=tmp_path/'SAR/cache_manifest.json';manifest=json.loads(m.read_text());manifest[p.name]=sha256(p);m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):assemble_joint_targets(raw,cfg,tmp_path,'test')


def test_joint_parent_rejects_single_domain_and_changed_dataset(tmp_path):
    raw,cfg=fixture(tmp_path)
    validate_joint_parent(dict(metadata=dict(profile=cfg)),cfg)
    for changed in (dict(cfg,training_domain='RGB'),dict(cfg,data={'different':True})):
        with pytest.raises(ValueError,match='parent mismatch'):validate_joint_parent(dict(metadata=dict(profile=changed)),cfg)
    with pytest.raises(ValueError,match='VAL'):assemble_joint_targets(raw,cfg,tmp_path,'val')


def test_joint_min_domain_selection_does_not_trade_sar_for_higher_mean():
    a=dict(domains={'RGB':{'accuracy':.60},'SAR':{'accuracy':.52}},balanced_accuracy=.56,cross_entropy=1.)
    b=dict(domains={'RGB':{'accuracy':.80},'SAR':{'accuracy':.50}},balanced_accuracy=.65,cross_entropy=.8)
    assert metric_key(b)>metric_key(a)
    assert metric_key(a,dict(selection_mode='worst_domain_first'))>metric_key(b,dict(selection_mode='worst_domain_first'))
    assert metric_key(a,{})==metric_key(a)
