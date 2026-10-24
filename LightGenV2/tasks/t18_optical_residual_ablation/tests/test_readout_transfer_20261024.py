import ast,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_readout_transfer_20261024.py'


def source_functions():
    tree=ast.parse(ENTRY.read_text());ns={};wanted={'validate','neighborhood_pool','error_augmented_pool'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in wanted],type_ignores=[]),str(ENTRY),'exec'),ns)
    return ns


def test_finite_profile_and_lineage_disclosure():
    c=json.loads((HERE/'lab_caltech20_readout_transfer_profile_20261024.json').read_text());source_functions()['validate'](c)
    s=ENTRY.read_text();assert 'TEST_information_via_teacher=inherited' in s and 'student_direct_original_TEST_gradient=False' in s
    assert 'official_TEST_accuracy=None' in s and 'valid_for_formal_test_table=False' in s and 'original_development_best_replaced=False' in s
    assert "inherited=recipe in('KL_teacher','DKD_TRAIN_neighborhood')" in s
    assert "if inherited else pw" in s and "if inherited else pb" in s


def test_error139_pool_has_only278_derivatives_not_original_TEST():
    rng=np.random.default_rng(73);x=rng.normal(size=(40,6)).astype(np.float32);y=np.arange(40)%20;ids=np.array(['TRAIN'+str(i)for i in range(40)])
    c=json.loads((HERE/'lab_caltech20_readout_transfer_profile_20261024.json').read_text());c['fixed_error_source_samples']=3
    z=x[:3]+.1;p=source_functions()['error_augmented_pool'](x,y,ids,z,y[:3],np.array(['ERR'+str(i)for i in range(3)]),c)
    assert len(p['features'])==46 and np.array_equal(p['features'][:40],x)
    assert list(p['source_ids'][40:])==['ERR0','ERR1','ERR2']*2
    assert set(p['source_roles'][40:])=={'former_TEST_error_derivative'}
    assert 'unmodified' not in set(p['augmentation'][40:])
    assert np.all(np.any(p['features'][40:43]!=z,axis=1))
    assert np.all(np.any(p['features'][43:]!=z,axis=1))


def test_neighborhood_augmentation_has_only_original_TRAIN_sources():
    rng=np.random.default_rng(73);x=rng.normal(size=(40,6)).astype(np.float32);y=np.arange(40)%20;ids=np.array(['TRAIN'+str(i)for i in range(40)])
    c=json.loads((HERE/'lab_caltech20_readout_transfer_profile_20261024.json').read_text());p=source_functions()['neighborhood_pool'](x,y,ids,c)
    assert len(p['features'])==120 and np.array_equal(p['features'][:40],x)
    assert set(p['source_roles'])=={'original_TRAIN'}and set(p['source_ids'])==set(ids)


def test_fitting_has_no_original_TEST_input_or_trainable_teacher():
    tree=ast.parse(ENTRY.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='optimize_pool')
    assert not {a.arg for a in fn.args.args}&{'former','fy','test','TEST','test_labels'}
    s=ENTRY.read_text();assert 'teacher=teacher.detach()' in s and 'target=None if' in s
    assert "keys={'linear.weight','linear.bias'}" in s and 'loop.validate_state(m,native,original,keys)' in s
    assert 'base.identities(m,paths)==lineage' in s and "m.sha(protected)==c['protected_manifest_SHA256']" in s
    assert "m.save(protected" not in s and 'new_capture=False' in s
