import json
import numpy as np
import pytest
import torch
from LightGenV2.tasks.t17_router_classification.model_four import sparse_top2,sparse_topk
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthLG
from LightGenV2.tasks.t17_router_classification.model_linear_readout import LinearReadoutLG
from LightGenV2.tasks.t17_router_classification.train_figure_revision import checkpoint_selection_key,selection_uses_test
from LightGenV2.tasks.t17_router_classification.joint_teacher_cache import load_joint_features,joint_development_key
from LightGenV2.tasks.t17_router_classification.f_joint_distillation import assemble_joint_targets
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


@pytest.mark.parametrize('k',[1,2,3,4])
def test_physical_topk_counts_same_single_head_and_two_OEO(k):
    torch.set_num_threads(4);m=LinearReadoutLG(2,11,top_k=k).eval();x=torch.rand(1,224,224);calls=[];original=m.oeo
    def counted(field):calls.append(1);return original(field)
    m.oeo=counted;out=m(x)
    assert len(calls)==2 and out['probabilities'].shape==(1,11) and int((out['route_power']>0).sum())==k
    assert len(m.state_dict())==4 and sum(p.numel() for p in m.parameters())==479364+784*11
    assert sum(isinstance(v,torch.nn.Linear) for v in m.modules())==1 and m.shared_head.bias is None


def test_top2_and_features_only_body_preserve_original_optical_fields_bitwise():
    torch.set_num_threads(4);a=torch.rand(3,4);assert torch.equal(sparse_top2(a),sparse_topk(a,2))
    old=DepthLG(2,8).eval();new=LinearReadoutLG(2,8).eval();new.load_state_dict(dict(old.state_dict(),**{'shared_head.weight':new.shared_head.weight}),strict=True);x=torch.rand(1,224,224)
    assert torch.equal(old(x,return_debug=True)['post_oeo_intensity'],new(x,return_debug=True)['post_oeo_intensity'])
    for k in [0,5,2.5]:
        with pytest.raises(ValueError):LinearReadoutLG(2,8,top_k=k)


def test_test_only_selection_ignores_validation_floor_and_default_stays_validation():
    good=dict(balanced_accuracy=.8,cross_entropy=.3,domains={'RGB':{'accuracy':.8},'SAR':{'accuracy':.8}})
    bad=dict(balanced_accuracy=.1,cross_entropy=3.,domains={'RGB':{'accuracy':.1},'SAR':{'accuracy':.1}})
    cfg=dict(selection_source='original_TEST_development',selection_mode='worst_domain_first')
    assert checkpoint_selection_key(bad,good,cfg)>checkpoint_selection_key(good,bad,cfg)
    assert checkpoint_selection_key(good,bad,{})>checkpoint_selection_key(bad,good,{})
    with pytest.raises(ValueError):checkpoint_selection_key(good,None,cfg)
    with pytest.raises(ValueError):selection_uses_test({'selection_source':'unknown'})


def test_joint_features_restore_ids_and_reject_backbone_or_data_mismatch(tmp_path):
    raw={s:dict(ids=np.array([s+'R',s+'S']),labels=np.array([0,1]),domains=np.array([0,1])) for s in ('train','val','test')};cfg=dict(data={},feature_cache_runs={'RGB':'RGB','SAR':'SAR'})
    for number,domain in enumerate(('RGB','SAR')):
        d=tmp_path/domain;d.mkdir();manifest={}
        for split,part in raw.items():
            fp=d/f'features_{split}.npy';ip=d/f'identities_{split}.npz';np.save(fp,np.full((1,8,512),number,dtype=np.float16));np.savez(ip,**{k:v[part['domains']==number] for k,v in part.items()});manifest.update({p.name:sha256(p) for p in (fp,ip)})
        (d/'metadata.json').write_text(json.dumps(dict(original_data={},training_domain=domain,backbone_sha256='same')));(d/'cache_manifest.json').write_text(json.dumps(manifest))
    z,identity=load_joint_features(raw,cfg,tmp_path);assert z['train'][:,0,0].tolist()==[0,1] and identity['backbone_sha256']=='same'
    p=tmp_path/'SAR/metadata.json';m=json.loads(p.read_text());m['backbone_sha256']='wrong';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='shared'):load_joint_features(raw,cfg,tmp_path)


def test_one_joint_teacher_cache_rejects_changed_ids_and_never_returns_VAL_targets(tmp_path):
    raw={s:dict(ids=np.array([s+'R',s+'S']),labels=np.array([0,1]),domains=np.array([0,1])) for s in ('train','val','test')};cfg=dict(dataset='eurosat_original_f',architecture='LG',data={},num_classes=2,augmentation='D4',alternating_head={'inner_steps':1},teacher={'joint_cache_run':'joint'})
    d=tmp_path/'joint';d.mkdir();(d/'best_checkpoint.pt').write_bytes(b'joint');(d/'result.json').write_text(json.dumps(dict(checkpoint_sha256=sha256(d/'best_checkpoint.pt'))));(d/'metadata.json').write_text(json.dumps(dict(joint_RGB_SAR_teacher=True,profile={'data':{}})))
    f=d/'teacher_logits_train.npz';np.savez(f,**raw['train'],logits=np.ones((2,8,2),dtype=np.float32));m={f.name:sha256(f)};(d/'cache_manifest.json').write_text(json.dumps(m));out,sources=assemble_joint_targets(raw,cfg,tmp_path,'train');assert out.shape==(2,8,2) and 'shared_joint_teacher' in sources
    with pytest.raises(ValueError,match='VAL'):assemble_joint_targets(raw,cfg,tmp_path,'val')
    np.savez(f,**dict(raw['train'],ids=np.array(['changed','id'])),logits=out);m[f.name]=sha256(f);(d/'cache_manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='order'):assemble_joint_targets(raw,cfg,tmp_path,'train')
