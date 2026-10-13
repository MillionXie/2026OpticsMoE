"""All-eight common200 locks; once per changed selected EMA, reuse old exact EMA."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import caltech_selected_extend as e
import audit_caltech_selected_extend as audit

ARTIFACTS=dict(fit_result='fit_result.json',best='best_checkpoint.pt',last='last_checkpoint.pt',
    config='config.json',metadata='metadata.json',history='history.json',gradients='gradients.json',
    train_predictions='train_predictions.csv',val_predictions='val_predictions.csv',resume='resume_receipt.json',smoke='smoke.json')


def save(path,value,exclusive=False):e.parent_tools.save(path,value,exclusive=exclusive)


def source_identity():
    return {name:e.sha(e.HERE/name) for name in ('caltech_selected_extend_finalize.py','audit_caltech_selected_extend.py',
        'caltech_selected_extend.py','caltech_selected_finalize.py','caltech20_finalize.py','audit_caltech20_final.py')}


def lock(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    fits=audit.verify_all(a.base,a.data);manifest=e.parent_tools.verify_data(a.data)
    a.out.mkdir(parents=True,exist_ok=False)
    receipt=audit.report(a.base,a.data,a.out/'cpu_fit_review')
    assert receipt['cpu_checks_passed'] and receipt['endpoint_count']==0
    # Every CPU check above passes before the first pair/local/global lock.
    entries={}
    for depth in e.DEPTHS:
        pair=dict(depth=depth,epochs=200,additional_epochs=100,rule=e.extension_config(fits[f'L{depth}_rho0']['config'])['selection'],
            arms=[str((a.base/f'L{depth}_{r}').resolve()) for r in ('rho0','rho03')])
        save(a.out/f'pair_lock_L{depth}.json',pair,exclusive=True)
        for name in ('rho0','rho03'):
            key=f'L{depth}_{name}';arm=a.base/key;fit=fits[key]
            entry=dict(key=key,arm=str(arm.resolve()),depth=depth,rho=fit['rho'],selected_epoch=fit['selected_epoch'],
                selected_state_kind='ema',selection_split='val',test_used_for_selection=False,
                selected_model_state_sha256=fit['selected_model_state_sha256'],checkpoint_sha256=fit['checkpoint_sha256'],
                artifacts={name:e.sha(arm/file) for name,file in ARTIFACTS.items()},pair_lock_sha256=e.sha(a.out/f'pair_lock_L{depth}.json'))
            save(a.out/key/'selection_lock.json',entry,exclusive=True)
            entries[key]=dict(entry,selection_lock_sha256=e.sha(a.out/key/'selection_lock.json'))
    save(a.out/'global_selection_lock.json',dict(entries=entries,base=str(a.base.resolve()),scope=e.SCOPE,
        classes=manifest['classes'],supports=manifest['supports'],data_sha256=e.sha(a.data),manifest_sha256=e.sha(a.data.parent/'data_manifest.json'),
        sources=source_identity(),fit_audit_sha256=e.sha(a.out/'cpu_fit_review/audit.json'),lock_time_unix=time.time(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=e.ROOT,text=True).strip()),exclusive=True)
    save(a.out/'status.json',dict(state='selection_locked_test_not_started'))


def verify_global(out,data):
    lock=e.read(out/'global_selection_lock.json')
    assert set(lock['entries'])==set(e.KEYS) and lock['scope']==e.SCOPE and lock['sources']==source_identity()
    manifest=e.parent_tools.verify_data(data)
    assert lock['classes']==manifest['classes'] and lock['supports']==manifest['supports']
    assert lock['data_sha256']==e.sha(data) and lock['manifest_sha256']==e.sha(data.parent/'data_manifest.json')
    assert lock['fit_audit_sha256']==e.sha(out/'cpu_fit_review/audit.json') and e.read(out/'cpu_fit_review/audit.json')['cpu_checks_passed']
    for key,entry in lock['entries'].items():
        arm=Path(entry['arm']);local=out/key/'selection_lock.json';pair=out/f"pair_lock_L{entry['depth']}.json"
        assert e.sha(local)==entry['selection_lock_sha256'] and e.read(local)=={k:v for k,v in entry.items() if k!='selection_lock_sha256'}
        assert entry['key']==key and entry['selected_state_kind']=='ema' and entry['selection_split']=='val' and not entry['test_used_for_selection']
        assert entry['artifacts']=={name:e.sha(arm/file) for name,file in ARTIFACTS.items()}
        assert e.sha(pair)==entry['pair_lock_sha256'] and e.read(pair)['epochs']==200
        assert str(arm)==e.read(pair)['arms'][0 if entry['rho']==0 else 1]
    return lock


def claim(out,key,lock_digest,reused):
    assert key in e.KEYS
    save(out/key/'test_started.json',dict(global_selection_lock_sha256=lock_digest,pid=os.getpid(),time_unix=time.time(),
        policy='Reuse saved parent test; no inference' if reused else 'Exclusive durable claim before first test materialization; no retry'),exclusive=True)


def reuse_parent(out,key,parent,data,fit,lock_digest):
    e.parent_tools.verify_global(parent.parent/'final',data)
    previous=e.read(parent.parent/'final'/key/'result.json')
    assert previous['test_evaluations']==1 and not previous['test_used_for_selection'] and previous['selection_split']=='val'
    assert previous['data_sha256']==e.sha(data)
    assert previous['checkpoint_sha256']==e.sha(parent/'best_checkpoint.pt')
    previous_audit=e.read(parent.parent/'cpu_report_final/final_audit.json')
    assert previous_audit['cpu_checks_passed']
    prior_receipt=next(r for r in previous_audit['arms'] if r['key']==key)
    assert prior_receipt['result_sha256']==e.sha(parent.parent/'final'/key/'result.json')
    assert previous['test_predictions_sha256']==e.sha(parent.parent/'final'/key/'test_predictions.csv')
    old=e.torch.load(parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert e.state_digest(old['model'])==fit['selected_model_state_sha256']
    claim(out,key,lock_digest,True)
    shutil.copyfile(parent.parent/'final'/key/'test_predictions.csv',out/key/'test_predictions.csv')
    return dict(test=previous['metrics']['test']),dict(test_evaluations=0,test_reused=True,
        reused_parent_test_result_sha256=e.sha(parent.parent/'final'/key/'result.json'))


def worker(a):
    lock=verify_global(a.out,a.data);entry=lock['entries'][a.key];arm=Path(entry['arm'])
    fit=audit.verify_arm(arm,a.data);assert os.environ['CUDA_VISIBLE_DEVICES']==a.gpu
    checkpoint=e.torch.load(arm/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert e.state_digest(checkpoint['model'])==entry['selected_model_state_sha256']==fit['selected_model_state_sha256']
    parent=Path(e.read(arm/'metadata.json')['parent']);old=e.torch.load(parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    global_digest=e.sha(a.out/'global_selection_lock.json')
    if e.state_digest(old['model'])==fit['selected_model_state_sha256']:
        metric,extra=reuse_parent(a.out,a.key,parent,a.data,fit,global_digest)
    else:
        e.o.f.gpu_budget.ALLOWED.update(e.GPUS);policy=e.o.f.gpu_budget.configure(e.torch,a.gpu,shared=False,memory_gib=3);e.torch.set_num_threads(4)
        model=e.pilot.execution.build(entry['depth'],e.read(e.parent_tools.PROFILE),fit['config'],batched=fit['config']['expert_batched'])
        model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        claim(a.out,a.key,global_digest,False)
        data=e.parent_tools.load_test_arrays(a.data,e.read(a.data.parent/'data_manifest.json'))
        test,rows=e.pilot.evaluate(model,data,fit['config']['evaluation_batch_size']);e.r.csvwrite(a.out/a.key/'test_predictions.csv',rows)
        metric=dict(test=test);extra=dict(test_evaluations=1,test_reused=False,gpu_policy=policy)
    save(a.out/a.key/'result.json',dict(depth=entry['depth'],rho=entry['rho'],selected_epoch=entry['selected_epoch'],
        selected_state_kind='ema',selection_split='val',test_used_for_selection=False,metrics=dict(fit['metrics'],**metric),
        selected_model_state_sha256=fit['selected_model_state_sha256'],checkpoint_sha256=fit['checkpoint_sha256'],
        global_selection_lock_sha256=global_digest,data_sha256=e.sha(a.data),test_predictions_sha256=e.sha(a.out/a.key/'test_predictions.csv'),
        scope=e.SCOPE,**extra))


def suite(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    verify_global(a.out,a.data)
    assert not any((a.out/key/'test_started.json').exists() for key in e.KEYS),'No automatic retry'
    save(a.out/'suite_started.json',dict(pid=os.getpid(),time_unix=time.time()),exclusive=True)
    pending,active,completed,failures=list(reversed(e.KEYS)),{},[],[]
    while active or (pending and not failures):
        for gpu,(child,key) in list(active.items()):
            code=child.poll()
            if code is None:continue
            if code==0 and (a.out/key/'result.json').exists():completed.append(key)
            else:failures.append(dict(key=key,pid=child.pid,exit_code=code))
            del active[gpu]
        if not failures:
            for gpu in e.GPUS:
                if gpu in active or not pending or not e.parent_tools.idle(gpu):continue
                key=pending.pop(0);cmd=[sys.executable,str(Path(__file__).resolve()),'--phase','worker','--out',str(a.out),'--data',str(a.data),'--key',key,'--gpu',gpu]
                with (a.out/key/'test.log').open('x') as log:
                    child=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(child,key)
        save(a.out/'current_processes.json',dict(active=[dict(key=k,pid=c.pid,gpu_uuid=g) for g,(c,k) in active.items()],pending=pending))
        save(a.out/'status.json',dict(state='test_failure_draining' if failures else 'selected_test_running',completed=completed,failures=failures))
        if active or (pending and not failures):time.sleep(3)
    save(a.out/'status.json',dict(state='test_failed_no_automatic_retry' if failures else 'tests_complete_awaiting_cpu_audit',completed=completed,failures=failures))
    assert not failures,failures


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('lock','suite','worker'),required=True)
    for name in ('out','data'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--key',choices=e.KEYS);p.add_argument('--gpu',choices=e.GPUS)
    a=p.parse_args()
    if a.phase=='lock':lock(a)
    elif a.phase=='suite':suite(a)
    else:worker(a)
