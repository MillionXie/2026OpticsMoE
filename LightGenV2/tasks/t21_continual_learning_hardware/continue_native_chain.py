"""Finite, equal-budget B continuation and freshly dependent C/D adaptation.

The four-epoch B trial was still improving on full TRAIN. This continuation
preserves its A and selected B, adds the same four B epochs to both models,
then trains C/D from the new selected parent, never splicing old child PTs.
The measured old-memory overfit motivates removing the extra pairwise term
only for old CLEVR replay, identically for both models; TRAIN KD stays fixed.
Capture and the original trainer modules are not modified or monkey-patched.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
import torch

from .model import build_model
from .train_native import (TASKS,sha,put,tensor_sha,frozen,gpu_gate,load_data,
    evaluate,summary,score,reference_TRAIN_objective,reference_TRAIN_cache,
    reference_teacher,clevr_pairwise_loss,load_dataset)


def validate_resume(protocol,history,completion,architecture,cfg):
    if protocol['architecture']!=architecture or protocol['TEST_gradients'] is not False:
        raise ValueError('Wrong original parent or TEST gradients')
    old=protocol['config']
    if old['stage_epochs']!=[12,4,6,4] or cfg['stage_epochs']!=[12,8,6,4]:
        raise ValueError('Only one fixed four-epoch B extension')
    changed={k for k in old if k!='diagnostic_basis' and old[k]!=cfg.get(k)}
    if changed!={'stage_epochs'}:raise ValueError('Unapproved numerical protocol differences: '+repr(changed))
    if set(cfg)-set(old)!={'old_clevr_pairwise_weight'} or cfg['old_clevr_pairwise_weight']!=0:
        raise ValueError('Only fixed removal of old CLEVR pairwise overfit term')
    if history[-1]['epoch']!=4 or completion['phase']!='B' or completion['best_epoch']!=4:
        raise ValueError('Original B not fully complete')
    if len(history)!=5:raise ValueError('Incomplete original B history')


def phase_batches(selection,names,cfg,epoch,phase):
    rng=np.random.default_rng(cfg['seed']+phase*1000+epoch);batches=[]
    for name in names:
        for _ in range(1 if name==names[-1] else cfg['memory_cycles_per_epoch']):
            if name=='eurosat':perm=rng.permutation(len(selection[name]))
            else:
                groups=rng.permutation(len(selection[name])//2)
                perm=(groups[:,None]*2+np.arange(2)).reshape(-1)
            batches.extend((name,perm[s:s+cfg['batch_size']]) for s in range(0,len(perm),cfg['batch_size']))
    rng.shuffle(batches)
    return batches


def paired_weight(task,current,cfg):
    if task=='clevr':return cfg['clevr_pairwise_weight'] if current else cfg['old_clevr_pairwise_weight']
    if task=='physical_binary_raw':return cfg.get('physical_pairwise_weight',0)
    return 0.


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent_run','config','train','test','source_protocols','out'):
        p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--architecture',choices=('moe','d2nn'),required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());parent_protocol=json.loads((a.parent_run/'protocol.json').read_text())
    old_B=json.loads((a.parent_run/'B/completion.json').read_text());old_history=json.loads((a.parent_run/'B/history.json').read_text())
    validate_resume(parent_protocol,old_history,old_B,a.architecture,cfg)
    if sha(a.parent_run/'B/best_checkpoint.pt')!=old_B['best_sha256']:raise ValueError('B parent SHA changed')
    a.out.mkdir(parents=True,exist_ok=False);began=time.time()
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0')
        train=load_data(a.train,'train');test=load_data(a.test,'test')
        memories=json.loads((a.parent_run/'memory_manifest.json').read_text())
        for name in TASKS:
            if memories[name]['source_manifest']!=train[name]['manifest']['identity_sha256'] or memories[name]['count']!=300:
                raise ValueError('Old TRAIN memory identity changed')
        paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
        identity=json.loads(Path(__file__).with_name('source_identity.json').read_text())
        model=build_model(a.architecture).to(device)
        model.load_state_dict(torch.load(a.parent_run/'B/best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True)
        # Only run evidence/data are copied; there is no source/engineering clone.
        shutil.copytree(a.parent_run/'A',a.out/'A')
        for name in ('memory_manifest.json','gain_calibration.json'):
            shutil.copy2(a.parent_run/name,a.out/name)
        protocol=dict(parent_protocol,config=cfg,config_sha256=sha(a.config),gpu=gpu,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={str(q.relative_to(Path(__file__).parents[3])):sha(q) for q in Path(__file__).parent.glob('*.py')},
            command=__import__('sys').argv,parent_run=str(a.parent_run),parent_protocol_sha256=sha(a.parent_run/'protocol.json'),
            warm_B_checkpoint_sha256=old_B['best_sha256'],optimizer_reset_at_B_continuation=True,
            added_epochs={'B':4},C_D_retrained_from_new_parent=True,old_C_D_not_spliced=True,
            hardware_admitted=False,automatic_capture=False)
        put(a.out/'protocol.json',protocol)
        chain=[json.loads((a.out/'A/completion.json').read_text())];put(a.out/'chain.json',chain)
        for phase in (2,3,4):
            name=chr(64+phase);out=a.out/name;out.mkdir();names=TASKS[:phase]
            model.configure_stage(phase);immutable=frozen(model);gain_sha=tensor_sha(model.display_gains)
            expected=identity['models'][3]['config']['protocol_sha256'][names[-1]]
            if sha(paths[names[-1]])!=expected or sha(paths['vision'])!=train[names[-1]]['manifest']['vision_checkpoint_sha256']:
                raise ValueError('Full TRAIN or frozen frontend changed')
            put(a.out/'status.json',dict(state='loading_full_current_TRAIN',phase=name,updated=time.time()))
            current=load_dataset(paths,names[-1],'train',vision_checkpoint=paths['vision'],device=device)
            if len(current)!=(15998,140000,12526,140800)[phase-1]:raise ValueError('Full TRAIN count changed')
            selection={n:(np.arange(len(current)) if n==names[-1] else np.asarray(memories[n]['indices'])) for n in names}
            cache_path=a.parent_run/name/'TRAIN_reference_teacher_logits.pt'
            teacher_id=a.parent_run/name/'teacher_identity.json'
            if cache_path.exists() and teacher_id.exists():
                cached=torch.load(cache_path,map_location='cpu',weights_only=False)
                reference=identity['models'][(0 if a.architecture=='moe' else 4)+phase-1]
                if cached['teacher_sha256']!=reference['sha256'] or sha(reference['checkpoint'])!=reference['sha256']:
                    raise ValueError('Original stage TRAIN teacher changed')
                if cached['TEST_inputs'] is not False or cached['source_scope']!='TRAIN only':raise ValueError('Not TRAIN cache')
                if set(cached['selection'])!=set(selection) or any(not np.array_equal(cached['selection'][n],ix) for n,ix in selection.items()):
                    raise ValueError('Teacher TRAIN record identity differs')
                logits_cache=cached['logits'];shutil.copy2(cache_path,out/cache_path.name);shutil.copy2(teacher_id,out/teacher_id.name)
            else:
                teacher,reference=reference_teacher(identity,a.architecture,phase,device)
                logits_cache=reference_TRAIN_cache(teacher,train,current,selection,names,device,cfg['batch_size'])
                torch.save(dict(logits=logits_cache,teacher_sha256=reference['sha256'],source_scope='TRAIN only',
                    selection={n:ix.tolist() for n,ix in selection.items()},TEST_inputs=False),out/'TRAIN_reference_teacher_logits.pt')
                put(out/'teacher_identity.json',dict(checkpoint_SHA256=reference['sha256'],config_SHA256=reference['config_sha256'],TRAIN_only=True))
                del teacher;torch.cuda.empty_cache()
            baseline=evaluate(model,test,names,device,cfg['batch_size']);best=score(baseline)
            if phase==2 and abs(best-np.mean([v['balanced_accuracy'] for v in old_B['test'].values()]))>1e-7:
                raise ValueError('Warm B baseline replay differs from original selected state')
            best_epoch=4 if phase==2 else 0
            history=list(old_history) if phase==2 else [dict(epoch=0,selection_split='test',selection_metrics=summary(baseline),selection_score=best)]
            def save(filename,epoch):
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=a.architecture,
                    stage=phase,epoch=epoch,best_epoch=best_epoch,config_sha256=sha(a.config),selection='TEST-selected development',
                    TEST_gradients=False,TEST_selected_development=True,parent_checkpoint_sha256=chain[-1]['best_sha256'],
                    warm_B_checkpoint_sha256=old_B['best_sha256'] if phase==2 else None),out/filename)
            save('best_checkpoint.pt',best_epoch)
            parameters=[v for v in model.parameters() if v.requires_grad]
            optical=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('shared_head.')]
            optimizer=torch.optim.AdamW([dict(params=optical,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
            start_epoch=5 if phase==2 else 1;fit_started=time.monotonic()
            for epoch in range(start_epoch,cfg['stage_epochs'][phase-1]+1):
                started=time.time();model.train();loss_sum=0.;seen=0;correct=0
                for task,relative in phase_batches(selection,names,cfg,epoch,phase):
                    ix=selection[task][relative]
                    if task==names[-1]:x=current.get_batch(ix,device);y=torch.as_tensor(np.asarray(current.labels)[ix],device=device,dtype=torch.long)
                    else:x=train[task]['x'][ix].to(device);y=train[task]['y'][ix].to(device)
                    optimizer.zero_grad(set_to_none=True);logits=model(x)['logits']
                    loss=reference_TRAIN_objective(logits,y,logits_cache[task][relative].to(device),cfg)
                    pair=paired_weight(task,task==names[-1],cfg)
                    if pair:loss=loss+pair*clevr_pairwise_loss(logits,y)
                    if a.architecture=='moe' and task!=names[-1]:loss=loss*cfg.get('moe_old_task_loss_weight',1.)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
                    loss_sum+=float(loss.detach())*len(y);seen+=len(y);correct+=int((logits.argmax(1)==y).sum())
                    if seen%cfg['progress_every_records']<len(y):put(a.out/'status.json',dict(state='training',phase=name,epoch=epoch,TRAIN_seen=seen,running_loss=loss_sum/seen,updated=time.time(),gpu=gpu))
                metrics=evaluate(model,test,names,device,cfg['batch_size']);value=score(metrics)
                if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain_sha:raise RuntimeError('Frozen old experts/gains changed')
                if value>best+1e-10:best=value;best_epoch=epoch;save('best_checkpoint.pt',epoch)
                save('last_checkpoint.pt',epoch)
                row=dict(epoch=epoch,loss=loss_sum/seen,TRAIN_accuracy=correct/seen,TRAIN_records=seen,selection_split='test',
                    selection_metrics=summary(metrics),selection_score=value,best_epoch=best_epoch,best_score=best,seconds=time.time()-started)
                history.append(row);put(out/'history.json',history);put(a.out/'status.json',dict(state='training',phase=name,**row,updated=time.time(),gpu=gpu))
                print(json.dumps(dict(phase=name,**row)),flush=True)
                if time.monotonic()-fit_started>=cfg['max_phase_fit_seconds']:raise RuntimeError('Finite phase budget reached; partial evidence retained, no admission')
            model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True)
            selected=evaluate(model,test,names,device,cfg['batch_size'])
            put(out/'selected_test.json',dict(scope='adapted simulation, not hardware',TEST_gradients=False,TEST_selection=True,
                independent_generalization_claim=False,selected_epoch=best_epoch,scores=selected))
            put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,fixed_gain_sha256=gain_sha,old_experts_bitwise_frozen=True))
            completion=dict(phase=name,best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),
                selected_test_sha256=sha(out/'selected_test.json'),test=summary(selected),best_is_untrained_parent=(best_epoch==0),
                B_no_new_improvement=(phase==2 and best_epoch<=4))
            put(out/'completion.json',completion);chain.append(completion);put(a.out/'chain.json',chain)
            del current,logits_cache;torch.cuda.empty_cache()
        put(a.out/'return.json',dict(returncode=0,completed=time.time(),seconds=time.time()-began,chain=chain))
        put(a.out/'status.json',dict(state='complete',updated=time.time()))
    except BaseException:
        put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
