"""LG-only warm B retention repair followed by genuinely dependent C and D.

A is immutable. Original corresponding-stage teachers consume TRAIN only.
Keep exactly300 old sources and50 replay cycles; change retention weighting,
lower optical LR and select one whole-stage PT by worst original-score deficit.
This is a disclosed additional optimization budget, not an equal-budget D2NN comparison.
"""
import argparse,json,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .train_native import (TASKS,sha,put,tensor_sha,frozen,gpu_gate,load_data,load_dataset,
    evaluate,summary,reference_TRAIN_objective,reference_TRAIN_cache,reference_teacher,clevr_pairwise_loss)
from .continue_native_chain import phase_batches,paired_weight
from .model import build_model


def retention_selection(scores,reference):
    if set(scores)!=set(reference):raise ValueError('Whole seen-task stage required')
    values=[scores[n]['balanced_accuracy'] for n in sorted(scores)]
    return (min(scores[n]['balanced_accuracy']-reference[n] for n in scores),float(np.mean(values)))


def validate_profile(c):
    if c['schema']!='T21_LG_retention_original_function_v1' or c['architecture']!='moe':raise ValueError('LG-only protocol')
    if (c['memory_records_per_old_task']!=300 or c['memory_cycles_per_epoch']!=50 or
        c['additional_B_epochs']!=2 or c['new_C_D_epochs']!=[8,6] or c['old_task_weight']!=4):raise ValueError('Finite declared retention budget only')
    if c['TEST_gradients'] is not False or c['freeze_A'] is not True:raise ValueError('Immutable A and TRAIN-only gradients required')


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('parent_run','config','train','test','source_protocols','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    cfg=json.loads(a.config.read_text());validate_profile(cfg);prior=json.loads((a.parent_run/'protocol.json').read_text())
    if prior['architecture']!='moe' or prior['TEST_gradients'] is not False:raise ValueError('Validated LG parent required')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');train=load_data(a.train,'train')
    memories=json.loads((a.parent_run/'memory_manifest.json').read_text());identity=json.loads(Path(__file__).with_name('source_identity.json').read_text())
    for n in TASKS:
        if memories[n]['count']!=300 or memories[n]['source_manifest']!=train[n]['manifest']['identity_sha256']:raise ValueError('Changed old memory')
    parent=json.loads((a.parent_run/'B/completion.json').read_text());parent_pt=a.parent_run/'B/best_checkpoint.pt'
    if sha(parent_pt)!=parent['best_sha256']:raise ValueError('B parent changed')
    model=build_model('moe').to(device);model.load_state_dict(torch.load(parent_pt,map_location=device,weights_only=False)['model'],strict=True);model.configure_stage(2)
    if a.preflight_only:
        immutable=frozen(model);gain=tensor_sha(model.display_gains);teacher,_=reference_teacher(identity,'moe',2,device)
        ix=np.asarray(memories['eurosat']['indices'][:2]);x=train['eurosat']['x'][ix].to(device);y=train['eurosat']['y'][ix].to(device)
        with torch.no_grad():target=teacher(x)['logits']
        loss=reference_TRAIN_objective(model(x)['logits'],y,target,cfg)*cfg['old_task_weight'];loss.backward()
        if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain or not torch.isfinite(loss):raise RuntimeError('Real TRAIN preflight failed')
        if not any(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Missing TRAIN gradient')
        print(json.dumps(dict(passed=True,loss=float(loss.detach()),old_experts_and_gains_unchanged=True,TEST_used=False,SDK=False)));return
    a.out.mkdir(parents=True,exist_ok=False);started=time.time()
    try:
        test=load_data(a.test,'test');paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
        shutil.copytree(a.parent_run/'A',a.out/'A')
        for n in ('memory_manifest.json','gain_calibration.json'):shutil.copy2(a.parent_run/n,a.out/n)
        a_sha=sha(a.out/'A/best_checkpoint.pt');chain=[json.loads((a.out/'A/completion.json').read_text())]
        put(a.out/'protocol.json',dict(architecture='moe',config=cfg,config_sha256=sha(a.config),parent_run=str(a.parent_run),
            warm_B_checkpoint_SHA256=parent['best_sha256'],A_bitwise_copied=True,D2NN_not_retrained=True,
            C_D_from_selected_new_parent=True,teacher_TEST_inputs=False,TEST_gradients=False,SDK=False,gpu=gpu,
            selection='lexicographic maximum worst seen-task deficit to fixed original same1000 reference, then mean; one shared PT',
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),command=__import__('sys').argv,
            TEST_selected_development_metric=True,independent_generalization_claim=False))
        for phase,epochs in ((2,2),(3,8),(4,6)):
            letter=chr(64+phase);names=TASKS[:phase];out=a.out/letter;out.mkdir();model.configure_stage(phase);immutable=frozen(model);gain=tensor_sha(model.display_gains)
            reference={n:cfg['original_same1000'][letter][n] for n in names}
            put(a.out/'status.json',dict(state='loading_full_current_TRAIN',phase=letter,updated=time.time()))
            if (sha(paths[names[-1]])!=identity['models'][3]['config']['protocol_sha256'][names[-1]] or
                sha(paths['vision'])!=train[names[-1]]['manifest']['vision_checkpoint_sha256']):raise ValueError('Original TRAIN protocol or frozen frontend changed')
            current=load_dataset(paths,names[-1],'train',vision_checkpoint=paths['vision'],device=device)
            if len(current)!=(140000 if phase==2 else 12526 if phase==3 else 140800):raise ValueError('Full current TRAIN required')
            selection={n:(np.arange(len(current)) if n==names[-1] else np.asarray(memories[n]['indices'])) for n in names}
            source=a.parent_run/letter/'TRAIN_reference_teacher_logits.pt'
            if source.exists():
                cached=torch.load(source,map_location='cpu',weights_only=False);original=identity['models'][phase-1]
                if cached['teacher_sha256']!=original['sha256'] or cached['TEST_inputs'] is not False or sha(original['checkpoint'])!=original['sha256']:raise ValueError('TRAIN teacher changed')
                if any(not np.array_equal(selection[n],cached['selection'][n]) for n in names):raise ValueError('Teacher TRAIN IDs differ')
                targets=cached['logits'];shutil.copy2(source,out/source.name)
            else:
                teacher,original=reference_teacher(identity,'moe',phase,device);targets=reference_TRAIN_cache(teacher,train,current,selection,names,device,8)
                torch.save(dict(logits=targets,teacher_sha256=original['sha256'],selection={n:ix.tolist() for n,ix in selection.items()},TEST_inputs=False),out/'TRAIN_reference_teacher_logits.pt');del teacher;torch.cuda.empty_cache()
            baseline=evaluate(model,test,names,device,8);best=retention_selection(baseline,reference);best_epoch=0;history=[]
            def save(role,epoch):
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture='moe',stage=phase,epoch=epoch,best_epoch=best_epoch,
                    config_sha256=sha(a.config),parent_checkpoint_sha256=chain[-1]['best_sha256'],warm_B_checkpoint_sha256=parent['best_sha256'] if phase==2 else None,
                    TEST_gradients=False,TEST_selected_development=True),out/(role+'_checkpoint.pt'))
            save('best',0);params=[v for v in model.parameters() if v.requires_grad]
            optical=[v for k,v in model.named_parameters() if v.requires_grad and not k.startswith('shared_head.')]
            optimizer=torch.optim.AdamW([dict(params=optical,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay']);fit_start=time.monotonic()
            for epoch in range(1,epochs+1):
                model.train();seen=0;loss_sum=0.;by_task={n:dict(seen=0,correct=0,loss_sum=0.) for n in names}
                for task,relative in phase_batches(selection,names,cfg,epoch,phase):
                    ix=selection[task][relative]
                    if task==names[-1]:x=current.get_batch(ix,device);y=torch.as_tensor(np.asarray(current.labels)[ix],device=device,dtype=torch.long)
                    else:x=train[task]['x'][ix].to(device);y=train[task]['y'][ix].to(device)
                    optimizer.zero_grad(set_to_none=True);logits=model(x)['logits'];loss=reference_TRAIN_objective(logits,y,targets[task][relative].to(device),cfg)
                    pair=paired_weight(task,task==names[-1],cfg)
                    if pair:loss+=pair*clevr_pairwise_loss(logits,y)
                    if task!=names[-1]:loss=loss*cfg['old_task_weight']
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();seen+=len(y);loss_sum+=float(loss.detach())*len(y)
                    row=by_task[task];row['seen']+=len(y);row['correct']+=int((logits.argmax(1)==y).sum());row['loss_sum']+=float(loss.detach())*len(y)
                    if seen%4000<len(y):put(a.out/'status.json',dict(state='training',phase=letter,epoch=epoch,TRAIN_seen=seen,running_loss=loss_sum/seen,updated=time.time()))
                metrics=evaluate(model,test,names,device,8);value=retention_selection(metrics,reference)
                if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain or sha(a.out/'A/best_checkpoint.pt')!=a_sha:raise RuntimeError('Frozen expert/gain/A changed')
                if value>best:best=value;best_epoch=epoch;save('best',epoch)
                save('last',epoch);history.append(dict(epoch=epoch,loss=loss_sum/seen,TRAIN_by_task=by_task,selection_metrics=summary(metrics),selection_score=value,best_epoch=best_epoch,best_score=best))
                put(out/'history.json',history);put(a.out/'status.json',dict(state='training',phase=letter,**history[-1],updated=time.time()));print(json.dumps(dict(phase=letter,**history[-1])),flush=True)
                if time.monotonic()-fit_start>cfg['max_phase_fit_seconds']:raise RuntimeError('Finite phase fit bound reached')
            model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True);selected=evaluate(model,test,names,device,8)
            put(out/'selected_test.json',dict(scores=selected,TEST_gradients=False,TEST_selected_development_metric=True))
            put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,fixed_gain_sha256=gain,A_checkpoint_SHA256=a_sha))
            gate={n:dict(reference=reference[n],adapted=selected[n]['balanced_accuracy'],within_requested_band=reference[n]-.01-1e-9<=selected[n]['balanced_accuracy']<=reference[n]+.06+1e-9) for n in names}
            completion=dict(phase=letter,best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),test=summary(selected),original_band=gate,all_tasks_within_band=all(v['within_requested_band'] for v in gate.values()))
            put(out/'completion.json',completion);chain.append(completion);put(a.out/'chain.json',chain);del current,targets;torch.cuda.empty_cache()
        put(a.out/'return.json',dict(returncode=0,completed=time.time(),seconds=time.time()-started));put(a.out/'status.json',dict(state='complete',updated=time.time()))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
