"""Screened20 common100: saved-result audits, all-eight locks, once-only test.

This separate entry point leaves historical Caltech20/COIL seals unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import caltech20_finalize as shared

HERE, ROOT = shared.HERE, shared.ROOT
GPUS, ARTIFACTS = shared.GPUS, shared.ARTIFACTS
DEPTHS = (2, 4, 12, 16)
KEYS = [f'L{d}_{rho}' for d in DEPTHS for rho in ('rho0', 'rho03')]
PROFILE = HERE/'caltech_selected_profile.json'
PLAN = HERE/'caltech_selected_plan.json'
read, sha, load_test_arrays, idle = shared.read, shared.sha, shared.load_test_arrays, shared.idle
SCOPE = 'Validation-screened Caltech10120 subset, common100 seed17, regularized; exploratory development, previous test observed; no subject/photographer independence established'


def save(path, value, exclusive=False):
    shared.save(path, value, exclusive=exclusive)
    if exclusive and os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


def source_identity():
    return {p.name:sha(p) for p in (Path(__file__), HERE/'caltech20_finalize.py', HERE/'audit_medicinal_pilot.py')}


def verify_data(data):
    manifest, profile = read(data.parent/'data_manifest.json'), read(PROFILE)
    contract = profile['data_contract']
    assert manifest['dataset'] == profile['dataset'] == contract['dataset']
    assert sha(data) == manifest['cache_sha256'] == contract['cache_sha256']
    assert sha(data.parent/'data_manifest.json') == contract['manifest_sha256']
    assert manifest['preparation_plan_sha256'] == contract['preparation_plan_sha256'] == sha(PLAN)
    assert manifest['archive_sha256'] == contract['identity_sha256']
    assert manifest['classes'] == profile['selected_classes'] and len(set(manifest['classes'])) == 20
    assert profile['depths'] == list(DEPTHS)
    assert manifest['total_records'] == contract['records'] == sum(sum(v) for v in manifest['supports'].values())
    assert manifest['previous_fit_or_validation_images_in_test'] == 0
    all_ids=[]
    for split in ('train','val','test'):
        ids=manifest['split_ids'][split]; support=manifest['supports'][split]
        assert len(support)==20 and min(support)>0 and len(ids)==sum(support)
        assert len(ids)==len(set(ids)); all_ids.extend(ids)
    assert len(all_ids)==len(set(all_ids))
    assert manifest['test_previously_observed'] and manifest['test_development']
    return manifest


def expected_mixup_hash(seed, epoch, samples, alpha):
    import numpy as np
    digest=hashlib.sha256()
    for batch, start in enumerate(range(0,samples,16)):
        n=min(16,samples-start)
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,batch,20261012]))
        lam=float(rng.beta(alpha,alpha));lam=max(lam,1-lam)
        perm=rng.permutation(n)
        part=hashlib.sha256(np.asarray([lam],dtype='<f8').tobytes()+perm.astype('<i8').tobytes()).hexdigest()
        digest.update(part.encode())
    return digest.hexdigest()


def verify_arm(arm):
    fit=shared.verify_arm(arm)
    cfg=fit['config'];history=read(arm/'history.json');meta=read(arm/'metadata.json')
    assert fit['depth'] in DEPTHS and cfg['depth']==fit['depth']
    assert cfg['mixup_alpha']==.2 and cfg['label_smoothing']==.08 and cfg['phase_smooth_weight']==.05
    assert cfg['augmentation']==dict(degrees=20,translation_pixels=5,scale_delta=.1)
    assert cfg['seed']==17 and cfg['test_development'] and cfg['test_previously_observed']
    assert cfg['data_contract']==read(PROFILE)['data_contract']
    assert cfg['classes']==read(PROFILE)['selected_classes']
    assert meta['sources'][str(PROFILE.relative_to(ROOT)).replace('\\','/')]==sha(PROFILE)
    assert len(fit['orders'])==len(fit['transforms'])==100
    for row in history:
        assert row['mixup_sha256']==expected_mixup_hash(cfg['seed'],row['epoch'],fit['metrics']['train']['n'],cfg['mixup_alpha'])
    smoke=read(arm/'smoke.json')
    assert smoke['passed'] and smoke['depth']==fit['depth'] and smoke['residual_rho']==fit['rho']
    assert smoke['microbatch']==cfg['microbatch'] and smoke['expert_batched']==cfg['expert_batched']
    assert smoke['probability_max_error']<=1e-6 and smoke['gradient_max_relative_l2']<=.02 and smoke['gradient_min_cosine']>=.999
    assert smoke['gradients_finite_nonzero'] and smoke['initialization_unchanged'] and not smoke['router_has_residual']
    return fit


def verify_pair(fits, depth, sources, manifest):
    assert [f['depth'] for f in fits]==[depth,depth] and [f['rho'] for f in fits]==[0.,.3]
    assert all(f['sources']==sources and f['data_sha256']==manifest['cache_sha256'] for f in fits)
    a,b=[f['config'] for f in fits]
    assert {k for k in set(a)|set(b) if a.get(k)!=b.get(k)}=={'residual_rho'}
    assert all(fits[0][k]==fits[1][k] for k in ('initial_parameter_sha256','orders','transforms','parameters'))


def seal(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert read(args.base/'status.json')['state']=='fit_complete_awaiting_validation_review'
    assert read(args.base/'current_processes.json')['active']==[]
    manifest=verify_data(args.data);metadata=read(args.base/'metadata.json')
    assert metadata['profile_sha256']==sha(PROFILE) and metadata['plan_sha256']==sha(PLAN)
    assert metadata['data_sha256']==sha(args.data) and metadata['manifest_sha256']==sha(args.data.parent/'data_manifest.json')
    arms=read(args.base/'fit_manifest.json')['arms'];assert set(arms)==set(KEYS),'All eight formal arms required'
    results,audits,choices={},{},{}
    for depth in DEPTHS:
        pair=[Path(arms[f'L{depth}_{name}']) for name in ('rho0','rho03')]
        fits=[verify_arm(arm) for arm in pair];verify_pair(fits,depth,metadata['sources'],manifest)
        assert all(f['config']['microbatch']==metadata['modes'][str(depth)] for f in fits)
        report=args.out/'candidate_audits'/f'L{depth}_base'
        cmd=[sys.executable,str(HERE/'audit_medicinal_pilot.py'),'--run',str(args.base),'--depth',str(depth),
             '--arm0',str(pair[0]),'--arm03',str(pair[1]),'--data',str(args.data),'--out',str(report)]
        with (args.out/f'audit_L{depth}.log').open('x') as log:
            subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(report/'pilot_audit.json')['cpu_checks_passed']
        audits[f'L{depth}_base']=sha(report/'pilot_audit.json')
        choices[str(depth)]=dict(selected_candidate='base',rule='First minimum validation EMA balanced NLL per arm; fixed screened20/common100',
            candidates=dict(base=dict(arms=[str(p.resolve()) for p in pair],epochs=100,
                mean_best_val_balanced_nll=sum(f['metrics']['val']['balanced_nll'] for f in fits)/2)))
        results.update({f'L{depth}_{name}':fit for name,fit in zip(('rho0','rho03'),fits)})
    # No partial pair or local lock is created until every CPU audit has passed.
    entries={}
    for depth in DEPTHS:
        pair_path=args.out/f'pair_lock_L{depth}.json';save(pair_path,choices[str(depth)],exclusive=True)
        for name in ('rho0','rho03'):
            key=f'L{depth}_{name}';fit=results[key];arm=Path(arms[key])
            entry=dict(arm=str(arm.resolve()),depth=depth,rho=fit['rho'],candidate='base',selected_epoch=fit['selected_epoch'],
                selected_state_kind='ema',selection_split='val',test_evaluations_before_lock=0,test_used_for_selection=False,
                pair_lock_sha256=sha(pair_path))
            for field,filename in ARTIFACTS.items():entry[field]=sha(arm/filename)
            save(args.out/key/'selection_lock.json',entry,exclusive=True)
            entries[key]=dict(entry,selection_lock_sha256=sha(args.out/key/'selection_lock.json'))
    save(args.out/'global_selection_lock.json',dict(entries=entries,choices=choices,candidate_audits=audits,
        data_sha256=sha(args.data),manifest_sha256=sha(args.data.parent/'data_manifest.json'),classes=manifest['classes'],
        supports=manifest['supports'],profile_sha256=sha(PROFILE),plan_sha256=sha(PLAN),
        lock_time_unix=time.time(),source_sha256=sha(Path(__file__)),finalizer_sources=source_identity(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scope=SCOPE),exclusive=True)
    save(args.out/'status.json',dict(state='selection_locked_test_not_started',test_evaluations=0))


def verify_global(out,data):
    lock=read(out/'global_selection_lock.json')
    assert set(lock['entries'])==set(KEYS),'All eight choices required before test'
    assert lock['source_sha256']==sha(Path(__file__)) and lock['finalizer_sources']==source_identity()
    assert lock['profile_sha256']==sha(PROFILE) and lock['plan_sha256']==sha(PLAN)
    manifest=verify_data(data)
    assert lock['classes']==manifest['classes'] and lock['supports']==manifest['supports'] and lock['scope']==SCOPE
    assert lock['data_sha256']==sha(data) and lock['manifest_sha256']==sha(data.parent/'data_manifest.json')
    for key,entry in lock['entries'].items():
        depth=entry['depth'];rho=0. if key.endswith('_rho0') else .3
        assert key==f'L{depth}_'+('rho0' if rho==0 else 'rho03') and entry['rho']==rho
        assert entry['selected_state_kind']=='ema' and entry['selection_split']=='val'
        assert entry['test_evaluations_before_lock']==0 and not entry['test_used_for_selection']
        local=out/key/'selection_lock.json';assert sha(local)==entry['selection_lock_sha256']
        assert read(local)=={k:v for k,v in entry.items() if k!='selection_lock_sha256'}
        pair=out/f'pair_lock_L{depth}.json'
        assert sha(pair)==entry['pair_lock_sha256'] and read(pair)==lock['choices'][str(depth)]
        choice=lock['choices'][str(depth)]
        assert choice['selected_candidate']==entry['candidate']=='base' and choice['candidates']['base']['epochs']==100
        assert entry['arm']==choice['candidates']['base']['arms'][0 if rho==0 else 1]
        for field,filename in ARTIFACTS.items():assert sha(Path(entry['arm'])/filename)==entry[field]
        audit=out/'candidate_audits'/f'L{depth}_base'/'pilot_audit.json'
        assert sha(audit)==lock['candidate_audits'][f'L{depth}_base'] and read(audit)['cpu_checks_passed']
    return lock


def claim_test(out,key,global_digest):
    assert key in KEYS
    save(out/key/'test_started.json',dict(global_selection_lock_sha256=global_digest,pid=os.getpid(),time_unix=time.time(),
        policy='Exclusive durable claim before first test materialization; no automatic retries'),exclusive=True)


def worker(args):
    lock=verify_global(args.out,args.data)
    entry=lock['entries'][args.key];arm=Path(entry['arm']);fit=verify_arm(arm)
    assert os.environ['CUDA_VISIBLE_DEVICES']==args.gpu and args.gpu in GPUS
    assert not (args.out/args.key/'test_started.json').exists(),'No test retry'
    assert all(sha(ROOT/p)==digest for p,digest in fit['sources'].items()),'Changed inference dependency'
    import medicinal_pilot as pilot
    o,torch=pilot.o,pilot.torch
    o.f.gpu_budget.ALLOWED.update(GPUS)
    policy=o.f.gpu_budget.configure(torch,args.gpu,shared=False,memory_gib=3)
    torch.set_num_threads(4)
    checkpoint=torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert checkpoint['selected_state_kind']=='ema' and checkpoint['selection_split']=='val'
    assert checkpoint['epoch']==entry['selected_epoch'] and checkpoint['config']==fit['config']
    assert checkpoint['sources']==fit['sources'] and checkpoint['data_sha256']==lock['data_sha256']
    model=pilot.execution.build(entry['depth'],read(PROFILE),fit['config'],batched=fit['config']['expert_batched'])
    model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    claim_test(args.out,args.key,sha(args.out/'global_selection_lock.json'))
    data=load_test_arrays(args.data,read(args.data.parent/'data_manifest.json'))
    metric,rows=pilot.evaluate(model,data,fit['config']['evaluation_batch_size'])
    o.r.csvwrite(args.out/args.key/'test_predictions.csv',rows)
    save(args.out/args.key/'result.json',dict(depth=entry['depth'],rho=entry['rho'],candidate='base',selected_epoch=entry['selected_epoch'],
        selected_state_kind='ema',selection_split='val',metrics=dict(fit['metrics'],test=metric),parameters=fit['parameters'],
        test_evaluations=1,test_used_for_selection=False,global_selection_lock_sha256=sha(args.out/'global_selection_lock.json'),
        checkpoint_sha256=entry['checkpoint_sha256'],data_sha256=lock['data_sha256'],
        test_predictions_sha256=sha(args.out/args.key/'test_predictions.csv'),inference_source_sha256=sha(Path(__file__)),gpu_policy=policy,scope=lock['scope']))


def suite(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    verify_global(args.out,args.data)
    assert not any((args.out/key/'test_started.json').exists() for key in KEYS),'No automatic test retries'
    save(args.out/'suite_started.json',dict(pid=os.getpid(),time_unix=time.time()),exclusive=True)
    pending,active,completed,failures=list(reversed(KEYS)),{},[],[]
    while active or (pending and not failures):
        for gpu,(child,key) in list(active.items()):
            code=child.poll()
            if code is None:continue
            try:
                assert code==0
                result=read(args.out/key/'result.json')
                assert result['test_evaluations']==1 and not result['test_used_for_selection']
                completed.append(key)
            except Exception as error:failures.append(dict(key=key,pid=child.pid,exit_code=code,error=repr(error)))
            del active[gpu]
        if not failures:
            for gpu in GPUS:
                if gpu in active or not pending or not idle(gpu):continue
                key=pending.pop(0)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','worker','--out',str(args.out),
                    '--data',str(args.data),'--key',key,'--gpu',gpu]
                with (args.out/key/'test.log').open('x') as log:
                    child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(child,key)
        save(args.out/'current_processes.json',dict(active=[dict(key=k,gpu_uuid=g,pid=c.pid) for g,(c,k) in active.items()],pending=pending))
        save(args.out/'status.json',dict(state='test_failure_draining' if failures else 'selected_test_running',completed=completed,failures=failures))
        if active or (pending and not failures):time.sleep(3)
    save(args.out/'status.json',dict(state='test_failed_no_automatic_retry' if failures else 'tests_complete_awaiting_cpu_audit',completed=completed,failures=failures))
    assert not failures,'Keep failed claims and artifacts; no automatic retry'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('lock','suite','worker'),required=True)
    p.add_argument('--base',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--key',choices=KEYS);p.add_argument('--gpu',choices=GPUS)
    a=p.parse_args()
    if a.phase=='lock':
        assert a.base is not None
        a.out.mkdir(parents=True,exist_ok=False);seal(a)
    elif a.phase=='suite':suite(a)
    else:worker(a)


if __name__=='__main__':main()
