"""CPU-only screened20/12-depth locks, MixUp identities and once-test barriers."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import caltech_selected_finalize as f
import audit_caltech_selected_final as audit


def arm_fixture(tmp_path):
    arm=tmp_path/'arm';arm.mkdir()
    profile=f.read(f.PROFILE)
    cfg=dict(profile,classes=profile['selected_classes'],depth=12,residual_rho=.3,candidate='base',requested_epochs=100,
        early_stopping=False,microbatch=1,evaluation_batch_size=1)
    source_path=str(f.PROFILE.relative_to(f.ROOT)).replace('\\','/')
    sources={source_path:f.sha(f.PROFILE)}
    history=[dict(epoch=e,val={'balanced_nll':1. if e>=99 else 2.},
        mixup_sha256=f.expected_mixup_hash(17,e,1372,.2)) for e in range(1,101)]
    f.save(arm/'history.json',history);f.save(arm/'config.json',cfg)
    f.save(arm/'metadata.json',dict(sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=f.ROOT,text=True).strip()))
    (arm/'best_checkpoint.pt').write_bytes(b'fixture checkpoint, not a tensor')
    fit=dict(depth=12,rho=.3,test_evaluations=0,test_used_for_selection=False,selection_split='val',candidate='base',
        epochs_completed=100,config=cfg,stop_reason='maximum_epochs',selected_epoch=99,
        metrics=dict(val=history[98]['val'],train=dict(n=1372)),sources=sources,
        history_sha256=f.sha(arm/'history.json'),checkpoint_sha256=f.sha(arm/'best_checkpoint.pt'),
        orders=['order']*100,transforms=['affine+mix']*100,initial_parameter_sha256='initial',parameters=2649088,data_sha256='fixture')
    f.save(arm/'fit_result.json',fit)
    f.save(arm/'smoke.json',dict(passed=True,depth=12,residual_rho=.3,microbatch=1,expert_batched=True,
        probability_max_error=0.,gradient_max_relative_l2=0.,gradient_min_cosine=1.,gradients_finite_nonzero=True,
        initialization_unchanged=True,router_has_residual=False))
    return arm,fit,history


def test_twelve_layer_first_tie_and_common100(tmp_path):
    arm,fit,_=arm_fixture(tmp_path)
    assert f.verify_arm(arm)['selected_epoch']==99
    fit['selected_epoch']=100;f.save(arm/'fit_result.json',fit)
    with pytest.raises(AssertionError):f.verify_arm(arm)


def test_mixup_corruption_rejected_even_when_history_digest_updated(tmp_path):
    arm,fit,history=arm_fixture(tmp_path)
    history[-1]['mixup_sha256']='wrong';f.save(arm/'history.json',history)
    fit['history_sha256']=f.sha(arm/'history.json');f.save(arm/'fit_result.json',fit)
    with pytest.raises(AssertionError):f.verify_arm(arm)


def test_pair_must_match_mixing_and_only_differ_in_rho(tmp_path):
    _,fit,_=arm_fixture(tmp_path)
    first=copy.deepcopy(fit);first['rho']=first['config']['residual_rho']=0.
    f.verify_pair([first,fit],12,fit['sources'],dict(cache_sha256='fixture'))
    fit['transforms'][-1]='changed mixing'
    with pytest.raises(AssertionError):f.verify_pair([first,fit],12,fit['sources'],dict(cache_sha256='fixture'))


def test_manifest_support_and_class_names_are_dynamic(tmp_path,monkeypatch):
    data=tmp_path/'fixture.npz';data.write_bytes(b'synthetic, no actual test data')
    plan=tmp_path/'plan.json';f.save(plan,dict(fixture=True))
    names=[f'class{k}' for k in range(20)]
    manifest=dict(dataset='fixture20',cache_sha256=f.sha(data),preparation_plan_sha256=f.sha(plan),
        archive_sha256='archive',classes=names,total_records=120,previous_fit_or_validation_images_in_test=0,
        split_ids={s:[f'{s}{k}' for k in range(40)] for s in ('train','val','test')},
        supports={s:[2]*20 for s in ('train','val','test')},test_previously_observed=True,test_development=True)
    f.save(tmp_path/'data_manifest.json',manifest)
    profile=tmp_path/'profile.json'
    f.save(profile,dict(dataset='fixture20',selected_classes=names,depths=list(f.DEPTHS),data_contract=dict(dataset='fixture20',
        cache_sha256=f.sha(data),manifest_sha256=f.sha(tmp_path/'data_manifest.json'),
        preparation_plan_sha256=f.sha(plan),identity_sha256='archive',records=120)))
    monkeypatch.setattr(f,'PROFILE',profile);monkeypatch.setattr(f,'PLAN',plan)
    assert f.verify_data(data)['total_records']==120
    manifest['supports']['test'][0]=0;f.save(tmp_path/'data_manifest.json',manifest)
    cfg=f.read(profile);cfg['data_contract']['manifest_sha256']=f.sha(tmp_path/'data_manifest.json');f.save(profile,cfg)
    with pytest.raises(AssertionError):f.verify_data(data)


def test_all_eight_barrier_before_data_access(tmp_path):
    f.save(tmp_path/'global_selection_lock.json',dict(entries={'L12_rho0':{}}))
    with pytest.raises(AssertionError,match='All eight'):f.verify_global(tmp_path,tmp_path/'missing.npz')
    with pytest.raises(AssertionError):f.claim_test(tmp_path,'L8_rho0','sha')


def test_exclusive_claim_retained_after_failure(tmp_path):
    f.claim_test(tmp_path,'L12_rho03','sha')
    with pytest.raises(FileExistsError):f.claim_test(tmp_path,'L12_rho03','sha')
    assert f.read(tmp_path/'L12_rho03/test_started.json')['global_selection_lock_sha256']=='sha'


def test_worker_claim_before_first_test_array_read(tmp_path,monkeypatch):
    key='L12_rho03';gpu=f.GPUS[0];data=tmp_path/'data.npz';arm=tmp_path/'arm';arm.mkdir()
    f.save(tmp_path/'global_selection_lock.json',dict(fixture=True));f.save(tmp_path/'data_manifest.json',dict(fixture=True))
    cfg=dict(expert_batched=True,evaluation_batch_size=1)
    entry=dict(arm=str(arm),depth=12,rho=.3,selected_epoch=99,checkpoint_sha256='best')
    lock=dict(entries={key:entry},data_sha256='data',scope=f.SCOPE)
    fit=dict(sources={},config=cfg,metrics={},parameters=2649088)
    checkpoint=dict(selected_state_kind='ema',selection_split='val',epoch=99,config=cfg,sources={},data_sha256='data',model={})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',gpu)
    monkeypatch.setattr(f,'verify_global',lambda *a:lock);monkeypatch.setattr(f,'verify_arm',lambda *a:fit)
    model=NS(load_state_dict=lambda *a,**k:None,eval=lambda:None)
    pilot=NS(execution=NS(build=lambda *a,**k:model),torch=NS(set_num_threads=lambda n:None,load=lambda *a,**k:checkpoint),
        o=NS(f=NS(gpu_budget=NS(ALLOWED=set(),configure=lambda *a,**k:{}))))
    monkeypatch.setitem(sys.modules,'medicinal_pilot',pilot)
    def fail(*a):
        assert (tmp_path/key/'test_started.json').exists()
        raise RuntimeError('fixture materialization failure')
    monkeypatch.setattr(f,'load_test_arrays',fail)
    with pytest.raises(RuntimeError,match='fixture materialization'):f.worker(NS(out=tmp_path,data=data,key=key,gpu=gpu))
    with pytest.raises(AssertionError,match='No test retry'):f.worker(NS(out=tmp_path,data=data,key=key,gpu=gpu))


def test_cpu_audit_waits_for_all_tests_and_gpu_children(tmp_path):
    f.save(tmp_path/'status.json',dict(state='selection_locked_test_not_started'))
    with pytest.raises(AssertionError):audit.completion_barrier(tmp_path)
    f.save(tmp_path/'status.json',dict(state='tests_complete_awaiting_cpu_audit'))
    f.save(tmp_path/'current_processes.json',dict(active=[dict(pid=123)]))
    with pytest.raises(AssertionError):audit.completion_barrier(tmp_path)


def test_mixup_identity_includes_tail_and_epoch():
    assert f.expected_mixup_hash(17,1,1372,.2)!=f.expected_mixup_hash(17,2,1372,.2)
    assert f.expected_mixup_hash(17,1,1372,.2)!=f.expected_mixup_hash(17,1,1360,.2)
