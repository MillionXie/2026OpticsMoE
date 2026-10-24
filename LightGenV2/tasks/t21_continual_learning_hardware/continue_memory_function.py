"""Finite C+2 warm continuation and freshly dependent D, equal300 memory.

Old batches alternate original anchors and fixed20% amplitude mixtures of
the SAME300 old TRAIN sources. Original stage teacher supplies mixture soft
targets, not a claim that mixed inputs possess the original hard labels.
Both architectures use identical admission, update counts and objectives.
"""
import argparse,json,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_native import (TASKS,sha,put,tensor_sha,frozen,gpu_gate,load_data,evaluate,
    summary,score,reference_TRAIN_objective,reference_teacher,load_dataset,clevr_pairwise_loss)
from .continue_native_chain import phase_batches,paired_weight
from .model import build_model


def partner_bank(task,count=300):
    if count!=300:raise ValueError('Exactly300 unique old TRAIN source records')
    unit=1 if task=='eurosat' else 2
    rng=np.random.default_rng(20261025+TASKS.index(task));rows=[]
    for _ in range(8):
        groups=rng.permutation(count//unit)
        rows.append((groups[:,None]*unit+np.arange(unit)).reshape(-1))
    return np.stack(rows)


def mixture(x,partners):
    if x.shape!=partners.shape or x.shape[-2:]!=(224,224):raise ValueError('Preserve paired encoded amplitude geometry')
    return .8*x+.2*partners


def validate_config(old,new):
    if old['stage_epochs']!=[12,8,6,4] or new['stage_epochs']!=[12,8,8,4]:raise ValueError('Only fixed C+2 then fresh D4')
    if any(old[k]!=new.get(k) for k in old if k not in ('stage_epochs','diagnostic_basis')):raise ValueError('Unexpected base numerical change')
    extra=set(new)-set(old)
    if extra!={'old_memory_function_augmentation'}:raise ValueError('Unexpected additional numerical controls')
    if new['old_memory_function_augmentation']!={'mixture_fraction':.2,'bank_size':8,'old_batch_alternation':2,'teacher_target':'original corresponding stage on derived TRAIN amplitudes, KL only','hard_label_CE_on_mixtures':False}:
        raise ValueError('Only fixed equal-memory functional augmentation admitted')


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent_run','config','train','test','source_protocols','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--architecture',choices=('moe','d2nn'),required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    cfg=json.loads(a.config.read_text());prior=json.loads((a.parent_run/'protocol.json').read_text());validate_config(prior['config'],cfg)
    if prior['architecture']!=a.architecture or prior['TEST_gradients'] is not False:raise ValueError('Wrong immutable parent protocol')
    cparent=json.loads((a.parent_run/'C/completion.json').read_text())
    if sha(a.parent_run/'C/best_checkpoint.pt')!=cparent['best_sha256']:raise ValueError('C parent SHA changed')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0')
    train=load_data(a.train,'train');memories=json.loads((a.parent_run/'memory_manifest.json').read_text())
    for name in TASKS:
        if memories[name]['source_manifest']!=train[name]['manifest']['identity_sha256'] or memories[name]['count']!=300:raise ValueError('Memory source changed')
    identity=json.loads(Path(__file__).with_name('source_identity.json').read_text())
    model=build_model(a.architecture).to(device);model.load_state_dict(torch.load(a.parent_run/'C/best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True);model.configure_stage(3)
    if a.preflight_only:
        before=frozen(model);gain=tensor_sha(model.display_gains);teacher,_=reference_teacher(identity,a.architecture,3,device)
        ix=np.asarray(memories['eurosat']['indices'][:2]);x=train['eurosat']['x'][ix].to(device);mixed=mixture(x,x.flip(0))
        with torch.no_grad():target=teacher(mixed)['logits']
        model.zero_grad(set_to_none=True);logits=model(mixed)['logits'];loss=F.kl_div(F.log_softmax(logits/2,1),F.softmax(target/2,1),reduction='batchmean')*4;loss.backward()
        if frozen(model)!=before or tensor_sha(model.display_gains)!=gain or not torch.isfinite(loss):raise RuntimeError('Real TRAIN preflight failed')
        if not any(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('No real TRAIN gradient')
        print(json.dumps(dict(passed=True,architecture=a.architecture,TRAIN_only=True,TEST_used=False,SDK=False,run_written=False,loss=float(loss.detach()),frozen_old_experts=True,gains_unchanged=True)));return
    a.out.mkdir(parents=True,exist_ok=False);began=time.time()
    try:
        test=load_data(a.test,'test');paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
        for phase in ('A','B'):shutil.copytree(a.parent_run/phase,a.out/phase)
        for name in ('memory_manifest.json','gain_calibration.json'):shutil.copy2(a.parent_run/name,a.out/name)
        protocol=dict(prior,config=cfg,config_sha256=sha(a.config),gpu=gpu,parent_run=str(a.parent_run),
            parent_protocol_sha256=sha(a.parent_run/'protocol.json'),warm_C_checkpoint_SHA256=cparent['best_sha256'],
            optimizer_reset_at_C_continuation=True,added_epochs={'C':2},D_retrained_from_new_C=True,old_D_not_spliced=True,
            unique_old_TRAIN_sources_per_task=300,derived_old_TRAIN_bank_per_task=2400,old_batches_half_anchors_half_mixtures=True,
            mixture_hard_labels_not_used=True,teacher_TEST_inputs=False,TEST_gradients=False,automatic_capture=False,hardware_admitted=False,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={str(q.relative_to(Path(__file__).parents[3])):sha(q) for q in Path(__file__).parent.glob('*.py')})
        put(a.out/'protocol.json',protocol);chain=[json.loads((a.out/n/'completion.json').read_text()) for n in ('A','B')];put(a.out/'chain.json',chain)
        for phase in (3,4):
            name=chr(64+phase);out=a.out/name;out.mkdir();names=TASKS[:phase];model.configure_stage(phase);immutable=frozen(model);gain=tensor_sha(model.display_gains)
            expected=identity['models'][3]['config']['protocol_sha256'][names[-1]]
            if sha(paths[names[-1]])!=expected or sha(paths['vision'])!=train[names[-1]]['manifest']['vision_checkpoint_sha256']:
                raise ValueError('Full current TRAIN or frozen frontend source changed')
            put(a.out/'status.json',dict(state='loading_full_current_TRAIN',phase=name,updated=time.time()))
            current=load_dataset(paths,names[-1],'train',vision_checkpoint=paths['vision'],device=device)
            if len(current)!=(12526 if phase==3 else 140800):raise ValueError('Full current TRAIN count changed')
            selection={n:(np.arange(len(current)) if n==names[-1] else np.asarray(memories[n]['indices'])) for n in names}
            source=a.parent_run/name/'TRAIN_reference_teacher_logits.pt';cached=torch.load(source,map_location='cpu',weights_only=False)
            reference=identity['models'][(0 if a.architecture=='moe' else 4)+phase-1]
            if cached['TEST_inputs'] is not False or cached['teacher_sha256']!=reference['sha256'] or sha(reference['checkpoint'])!=reference['sha256']:
                raise ValueError('Original TRAIN teacher identity changed')
            if any(not np.array_equal(selection[n],cached['selection'][n]) for n in names):raise ValueError('TRAIN teacher source order changed')
            shutil.copy2(source,out/source.name);logits_cache=cached['logits'];teacher,teacher_identity=reference_teacher(identity,a.architecture,phase,device)
            banks={n:partner_bank(n) for n in names[:-1]};aug_targets={}
            with torch.no_grad():
                for task in names[:-1]:
                    base=train[task]['x'][selection[task]];targets=[]
                    for variant in range(8):
                        rows=[]
                        for start in range(0,300,8):
                            ix=np.arange(start,min(start+8,300));mixed=mixture(base[ix].to(device),base[banks[task][variant,ix]].to(device))
                            rows.append(teacher(mixed)['logits'].cpu())
                        targets.append(torch.cat(rows));put(a.out/'status.json',dict(state='original_teacher_derived_TRAIN_only',phase=name,task=task,variant=variant,updated=time.time()))
                    aug_targets[task]=torch.stack(targets)
            torch.save(dict(targets=aug_targets,partners=banks,teacher_SHA256=teacher_identity['sha256'],unique_sources=300,TEST_inputs=False),out/'TRAIN_derived_teacher_targets.pt')
            del teacher;torch.cuda.empty_cache()
            baseline=evaluate(model,test,names,device,cfg['batch_size']);best=score(baseline);best_epoch=cparent['best_epoch'] if phase==3 else 0;selected_new=False
            history=[dict(epoch=0,warm_from_C_epoch=cparent['best_epoch'] if phase==3 else None,selection_metrics=summary(baseline),selection_score=best)]
            def save(filename,epoch):
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=a.architecture,stage=phase,
                    epoch=epoch,best_epoch=best_epoch,config_sha256=sha(a.config),parent_checkpoint_sha256=chain[-1]['best_sha256'],
                    warm_C_checkpoint_sha256=cparent['best_sha256'] if phase==3 else None,TEST_gradients=False,TEST_selected_development=True),out/filename)
            save('best_checkpoint.pt',best_epoch);parameters=[v for v in model.parameters() if v.requires_grad]
            optical=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('shared_head.')]
            optimizer=torch.optim.AdamW([dict(params=optical,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay']);fit_started=time.monotonic()
            for epoch in (range(7,9) if phase==3 else range(1,5)):
                started=time.time();model.train();seen=0;loss_sum=0.;anchor_seen=0;anchor_correct=0;aug_seen=0;aug_agree=0;counters={n:0 for n in names[:-1]}
                for task,relative in phase_batches(selection,names,cfg,epoch,phase):
                    ix=selection[task][relative];is_aug=False
                    if task==names[-1]:x=current.get_batch(ix,device);y=torch.as_tensor(np.asarray(current.labels)[ix],device=device,dtype=torch.long)
                    else:
                        counters[task]+=1;is_aug=counters[task]%2==0;x=train[task]['x'][ix].to(device);y=train[task]['y'][ix].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    if is_aug:
                        variant=(epoch+counters[task]//2)%8;partner=selection[task][banks[task][variant,relative]]
                        x=mixture(x,train[task]['x'][partner].to(device));target=aug_targets[task][variant,relative].to(device)
                        logits=model(x)['logits'];loss=F.kl_div(F.log_softmax(logits/2,1),F.softmax(target/2,1),reduction='batchmean')*4*cfg['reference_KD_weight']
                        aug_seen+=len(y);aug_agree+=int((logits.argmax(1)==target.argmax(1)).sum())
                    else:
                        logits=model(x)['logits'];loss=reference_TRAIN_objective(logits,y,logits_cache[task][relative].to(device),cfg)
                        pair=paired_weight(task,task==names[-1],cfg)
                        if pair:loss+=pair*clevr_pairwise_loss(logits,y)
                        anchor_seen+=len(y);anchor_correct+=int((logits.argmax(1)==y).sum())
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN functional augmentation loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();seen+=len(y);loss_sum+=float(loss.detach())*len(y)
                    if seen%4000<len(y):put(a.out/'status.json',dict(state='TRAIN_functional_memory',phase=name,epoch=epoch,TRAIN_seen=seen,running_loss=loss_sum/seen,updated=time.time()))
                metrics=evaluate(model,test,names,device,cfg['batch_size']);value=score(metrics)
                if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain:raise RuntimeError('Old experts or gains changed')
                if value>best+1e-10:best=value;best_epoch=epoch;selected_new=True;save('best_checkpoint.pt',epoch)
                save('last_checkpoint.pt',epoch)
                row=dict(epoch=epoch,loss=loss_sum/seen,TRAIN_records=seen,TRAIN_anchor_records=anchor_seen,TRAIN_anchor_accuracy=anchor_correct/anchor_seen,
                    TRAIN_derived_records=aug_seen,TRAIN_derived_teacher_agreement=aug_agree/max(1,aug_seen),selection_metrics=summary(metrics),selection_score=value,best_epoch=best_epoch,best_score=best,seconds=time.time()-started)
                history.append(row);put(out/'history.json',history);put(a.out/'status.json',dict(state='training',phase=name,**row,updated=time.time()));print(json.dumps(dict(phase=name,**row)),flush=True)
                if time.monotonic()-fit_started>cfg['max_phase_fit_seconds']:raise RuntimeError('Finite phase fit limit reached, preserve partial result')
            model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True);selected=evaluate(model,test,names,device,cfg['batch_size'])
            put(out/'selected_test.json',dict(scores=selected,scope='adapted simulation, not hardware',TEST_gradients=False,TEST_selection=True,independent_generalization_claim=False))
            put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,fixed_gain_sha256=gain,old_experts_bitwise_frozen=True))
            completion=dict(phase=name,best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),selected_test_sha256=sha(out/'selected_test.json'),test=summary(selected),no_new_C_improvement=(phase==3 and not selected_new),best_is_untrained_parent=(phase==4 and best_epoch==0))
            put(out/'completion.json',completion);chain.append(completion);put(a.out/'chain.json',chain);del current,logits_cache,aug_targets;torch.cuda.empty_cache()
        put(a.out/'return.json',dict(returncode=0,completed=time.time(),seconds=time.time()-began,chain=chain));put(a.out/'status.json',dict(state='complete',updated=time.time()))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
