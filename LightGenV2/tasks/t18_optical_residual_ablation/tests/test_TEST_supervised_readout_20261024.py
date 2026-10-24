import ast,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_TEST_supervised_readout_20261024.py'
PROFILE=HERE/'lab_caltech20_TEST_supervised_profile_20261024.json'


def functions():
    tree=ast.parse(ENTRY.read_text());wanted={'validate','supervised_pool'}
    ns={};exec(compile(ast.Module(body=[v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name in wanted],type_ignores=[]),str(ENTRY),'exec'),ns)
    return ns


def test_explicit_TEST_supervision_cannot_be_called_formal_test():
    c=json.loads(PROFILE.read_text());functions()['validate'](c)
    assert c['independent_test_samples']==0 and not c['valid_for_formal_test_table']
    assert not c['replace_original_development_best']
    s=ENTRY.read_text()
    assert 'TEST_gradient=True' in s and 'official_TEST_accuracy=None' in s
    assert 'original_target_evaluable=False' in s and 'independent_test_samples=0' in s
    assert "former_TEST_adaptation" in s and "'test']['accuracy']" not in s


def test_augmented_pool_preserves_origin_labels_and_original_TRAIN():
    c=json.loads(PROFILE.read_text());rng=np.random.default_rng(73)
    x=rng.normal(size=(40,6)).astype(np.float32);z=x[:20].copy();ty=np.arange(40)%20;y=np.arange(20);p=y.copy();p[:3]=(p[:3]+1)%20
    packet,plan=functions()['supervised_pool'](x,ty,np.array(['TRAIN'+str(i) for i in range(40)]),z,y,np.array(['TEST'+str(i) for i in range(20)]),p,c,73)
    assert len(packet['features'])==66 and plan['former_TEST_augmented']==6
    assert np.array_equal(packet['features'][:40],x)
    assert np.array_equal(packet['labels'][40:60],y)
    assert set(packet['source_roles'])=={'original_TRAIN','former_TEST'}
    assert np.isclose(packet['loss_weights'][:40].sum(),.35)
    assert np.isclose(packet['loss_weights'][40:].sum(),.65)
    assert np.array_equal(ty[plan['sameclass_TRAIN_neighbor_indices']],y[:3])
    assert set(packet['augmentation'][60:])=={'sameclass_feature_mix','feature_noise'}
    assert all(str(v).startswith('TEST') for v in packet['source_ids'][40:])


def test_all_correct_pool_does_not_invent_new_samples():
    c=json.loads(PROFILE.read_text());x=np.ones((40,6),np.float32);y=np.arange(20)
    packet,plan=functions()['supervised_pool'](x,np.arange(40)%20,np.arange(40).astype(str),x[:20],y,np.arange(20).astype(str),y,c,73)
    assert len(packet['features'])==60 and plan['former_TEST_augmented']==0


def test_head_only_native_replay_and_original_artifacts_protected():
    s=ENTRY.read_text()
    assert "keys={'linear.weight','linear.bias'}" in s and 'trainable_parameters=5140' in s
    assert 'loop.validate_state(m,native,original,keys)' in s
    assert 'base.identities(m,paths)==lineage' in s and 'maxerr<=3e-6' in s
    assert 'm.sha(protected)==c[' in s and "m.save(protected" not in s
    assert "np.savez_compressed(a.out/" in s and "np.savez_compressed(cap/" not in s
    assert 'former_TEST_gradient_deliberately_nonzero=True' in s
    assert 'new_inference_layers=0' in s and 'SDK_used=False,GPU_used=False' in s
