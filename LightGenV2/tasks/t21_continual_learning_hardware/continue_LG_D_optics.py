"""Finite provisional D from an exact new C; never a spliced final chain.

Only original full current TRAIN and exactly300 old memories/task. C admission
is recorded, not invented. Even a good D is not hardware-admitted until its
actual C parent also passes. A later different C requires newly trained D.
"""
import argparse,json,subprocess,time,shutil,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .train_native import TASKS,sha,put,tensor_sha,frozen,gpu_gate,load_data,load_dataset,evaluate,summary
from .continue_native_chain import phase_batches
from .continue_LG_C_optics import continuation_objective,feature_retention_loss
from .hardware_bridge import load_checkpoint
from .refine_LG_retention import retention_selection


def validate(cfg):
    names=('config_LG_D_dependent_C_20261025.json','config_LG_D_native_C_retention_20261025.json')
    if not any(cfg==json.loads(Path(__file__).with_name(n).read_text()) for n in names):raise ValueError('Only finite published D profiles')


def native_C_retention(result,teacher,labels,cfg):
    """Old300 TRAIN function/route retention; no additional inference module."""
    temperature=cfg['native_C_temperature'];target=teacher['logits'].detach()
    rows=F.kl_div(F.log_softmax(result['logits']/temperature,1),F.softmax(target/temperature,1),reduction='none').sum(1)*temperature**2
    kd=(rows*(target.argmax(1)==labels)).mean()
    route=teacher['route_power'].detach();q=result['route_power'].clamp_min(1e-12)
    route_loss=(route*(route.clamp_min(1e-12).log()-q.log())).sum(1).mean()
    features=feature_retention_loss(result['ccd_features'],teacher['ccd_features'])
    return cfg['native_C_logit_weight']*kd+cfg['native_C_route_weight']*route_loss+cfg['native_C_feature_weight']*features


def configure_D(model):
    if model.architecture!='moe':raise ValueError('Original LG only')
    for value in model.parameters():value.requires_grad_(True)
    model.configure_stage(4)
    expected={'router_phase','global_phase','shared_head.weight'}|{'first_phase.'+str(i) for i in model.active_indices[12:16].tolist()}
    actual={n:v.numel() for n,v in model.named_parameters() if v.requires_grad}
    if set(actual)!=expected:raise RuntimeError('D trainability contract changed')
    return actual


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('chain','parent_c','profile','train','test','source_protocols','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--warm-d',type=Path);p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg)
    if bool(a.warm_d)!=bool(cfg.get('native_C_retention')):raise ValueError('Warm D and native C retention must be declared together')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');receipt=json.loads((a.parent_c/'C/completion.json').read_text());parent=a.parent_c/'C/best_checkpoint.pt'
    source=json.loads((a.parent_c/'protocol.json').read_text());origin=json.loads((a.chain/'C/completion.json').read_text())
    if source.get('parent_C_sha256',source.get('original_chain_C_sha256'))!=origin['best_sha256']:raise ValueError('Exact new C from same A/B chain required')
    model,cp=load_checkpoint(parent,receipt['best_sha256']);native_model=model;warm_receipt=None
    if cp['architecture']!='moe' or cp['stage']!=3 or cp['TEST_gradients'] is not False:raise ValueError('TRAIN-only C parent required')
    if a.warm_d:
        warm_protocol=json.loads((a.warm_d/'protocol.json').read_text());warm_receipt=json.loads((a.warm_d/'D/completion.json').read_text())
        audit=json.loads((a.warm_d/'CPU_deployment_D_20261025/report.json').read_text())
        if warm_protocol['parent_C_sha256']!=receipt['best_sha256'] or not audit['passed'] or not audit['strict_CPU']:raise ValueError('Strict CPU-audited D with the identical C parent required')
        model,cp=load_checkpoint(a.warm_d/'D/best_checkpoint.pt',warm_receipt['best_sha256'])
        if cp['architecture']!='moe' or cp['stage']!=4 or cp['TEST_gradients'] is not False or cp['parent_checkpoint_sha256']!=receipt['best_sha256']:raise ValueError('Warm D identity changed')
    trainable=configure_D(model);model.to(device);immutable=frozen(model);gains=tensor_sha(model.display_gains)
    train=load_data(a.train,'train');memories=json.loads((a.chain/'memory_manifest.json').read_text());cache=a.chain/'D/TRAIN_reference_teacher_logits.pt';targets=torch.load(cache,map_location='cpu',weights_only=False)
    identity=json.loads(Path(__file__).with_name('source_identity.json').read_text());teacher=identity['models'][3]
    if targets['TEST_inputs'] is not False or targets['teacher_sha256']!=teacher['sha256'] or sha(teacher['checkpoint'])!=teacher['sha256']:raise ValueError('Original D TRAIN teacher changed')
    for n in TASKS[:-1]:
        if len(memories[n]['indices'])!=300 or not np.array_equal(memories[n]['indices'],targets['selection'][n]) or memories[n]['source_manifest']!=train[n]['manifest']['identity_sha256']:raise ValueError('Old300 source identity changed')
    if not np.array_equal(targets['selection'][TASKS[-1]],np.arange(140800)):raise ValueError('Full Physical TRAIN identity required')
    native={};native_provenance=None
    if cfg.get('native_C_retention'):
        native_model.to(device).eval()
        for value in native_model.parameters():value.requires_grad_(False)
        with torch.no_grad():
            for name in TASKS[:-1]:
                indices=np.asarray(memories[name]['indices']);chunks={k:[] for k in ('logits','route_power','ccd_features')}
                for ix in np.array_split(indices,38):
                    result=native_model(train[name]['x'][ix].to(device))
                    for key in chunks:chunks[key].append(result[key].cpu())
                native[name]={key:torch.cat(rows) for key,rows in chunks.items()}
        native_provenance=dict(checkpoint_sha256=receipt['best_sha256'],TRAIN_indices={n:memories[n]['indices'] for n in native},TEST_inputs=False,unique_records_per_old_task=300)
        del native_model;torch.cuda.empty_cache()
    if a.preflight_only:
        n=TASKS[0];ix=np.asarray(memories[n]['indices'][:2]);labels=train[n]['y'][ix].to(device);result=model(train[n]['x'][ix].to(device));loss=continuation_objective(result['logits'],labels,targets['logits'][n][:2].to(device),cfg)
        if native:loss+=native_C_retention(result,{k:v[:2].to(device) for k,v in native[n].items()},labels,cfg)
        loss.backward()
        gradients={k:float(v.grad.norm()) for k,v in model.named_parameters() if v.requires_grad and v.grad is not None}
        if not torch.isfinite(loss) or set(gradients)!=set(trainable) or any(not np.isfinite(v) or v<=0 for v in gradients.values()) or frozen(model)!=immutable:raise RuntimeError('Real TRAIN gradient/frozen audit failed')
        print(json.dumps(dict(passed=True,loss=float(loss.detach()),trainable=trainable,gradient_norms=gradients,old_experts_frozen=True,TEST_inputs=False,SDK=False)));return
    a.out.mkdir(parents=True,exist_ok=False);out=a.out/'D';out.mkdir();began=time.time()
    try:
        test=load_data(a.test,'test');paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
        for n in ('memory_manifest.json','gain_calibration.json'):shutil.copy2(a.chain/n,a.out/n)
        if native:torch.save(dict(targets=native,provenance=native_provenance),out/'native_C_old300_TRAIN_teacher.pt')
        put(a.out/'protocol.json',dict(architecture='moe',config=cfg,config_sha256=sha(a.profile),chain=str(a.chain),parent_C_run=str(a.parent_c),parent_C_sha256=receipt['best_sha256'],
            warm_D_sha256=warm_receipt['best_sha256'] if warm_receipt else None,native_C_TRAIN_teacher=native_provenance,
            parent_C_all_tasks_within_band=receipt['all_tasks_within_band'],provisional_not_hardware_admitted=True,old_D_not_spliced=True,C_must_match_final_parent=True,
            TRAIN_teacher_sha256=targets['teacher_sha256'],TRAIN_teacher_cache_sha256=sha(cache),TEST_gradients=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
            trainable=trainable,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),gpu=gpu,SDK=False,new_inference_layers=0))
        if sha(paths[TASKS[-1]])!=identity['models'][3]['config']['protocol_sha256'][TASKS[-1]] or sha(paths['vision'])!=train[TASKS[-1]]['manifest']['vision_checkpoint_sha256']:raise ValueError('Full TRAIN protocol/frozen frontend differs')
        put(a.out/'status.json',dict(state='loading_full_current_TRAIN',updated=time.time()))
        current=load_dataset(paths,TASKS[-1],'train',vision_checkpoint=paths['vision'],device=device)
        if len(current)!=140800:raise ValueError('Full Physical TRAIN required')
        selection={n:np.arange(len(current)) if n==TASKS[-1] else np.asarray(memories[n]['indices']) for n in TASKS}
        baseline=evaluate(model,test,TASKS,device,8);best=retention_selection(baseline,cfg['reference']);best_epoch=0;history=[]
        def save(role,epoch):
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture='moe',stage=4,epoch=epoch,best_epoch=best_epoch,parent_checkpoint_sha256=receipt['best_sha256'],
                config_sha256=sha(a.profile),TEST_gradients=False,TEST_selected_development=True,provisional_not_hardware_admitted=True),out/(role+'_checkpoint.pt'))
        save('best',0);params=[v for v in model.parameters() if v.requires_grad];optical=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')]
        optimizer=torch.optim.AdamW([dict(params=optical,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay']);started=time.monotonic()
        for epoch in range(1,cfg['epochs']+1):
            model.train();seen=0;total=0.;by_task={n:dict(seen=0,correct=0) for n in TASKS}
            for task,relative in phase_batches(selection,TASKS,cfg,epoch,4):
                ix=selection[task][relative]
                if task==TASKS[-1]:x=current.get_batch(ix,device);y=torch.as_tensor(np.asarray(current.labels)[ix],device=device,dtype=torch.long)
                else:x=train[task]['x'][ix].to(device);y=train[task]['y'][ix].to(device)
                optimizer.zero_grad(set_to_none=True);result=model(x);logits=result['logits'];loss=continuation_objective(logits,y,targets['logits'][task][relative].to(device),cfg)
                if task!=TASKS[-1]:loss*=cfg['old_task_weight']
                if task in native:loss+=native_C_retention(result,{k:v[relative].to(device) for k,v in native[task].items()},y,cfg)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();seen+=len(y);total+=float(loss.detach())*len(y);by_task[task]['seen']+=len(y);by_task[task]['correct']+=int((logits.argmax(1)==y).sum())
                if seen%4000<len(y):put(a.out/'status.json',dict(state='training',phase='D',epoch=epoch,TRAIN_seen=seen,running_loss=total/seen,updated=time.time()))
            scores=evaluate(model,test,TASKS,device,8);key=retention_selection(scores,cfg['reference'])
            if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gains:raise RuntimeError('Frozen old experts/gains changed')
            if key>best:best=key;best_epoch=epoch;save('best',epoch)
            save('last',epoch);history.append(dict(epoch=epoch,loss=total/seen,TRAIN_by_task=by_task,scores=summary(scores),best_epoch=best_epoch));put(out/'history.json',history);put(a.out/'status.json',dict(state='epoch_complete',**history[-1],updated=time.time()))
            if time.monotonic()-started>=cfg['max_fit_seconds']:break
        model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True);selected=evaluate(model,test,TASKS,device,8)
        put(out/'selected_test.json',dict(scores=selected,TEST_gradients=False,TEST_selected_development_metric=True));put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,fixed_gain_sha256=gains))
        gate={n:dict(reference=cfg['reference'][n],adapted=selected[n]['balanced_accuracy'],within_requested_band=cfg['reference'][n]-.01-1e-9<=selected[n]['balanced_accuracy']<=cfg['reference'][n]+.06+1e-9) for n in TASKS}
        put(out/'completion.json',dict(phase='D',best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),selected_test_sha256=sha(out/'selected_test.json'),test=summary(selected),original_band=gate,all_tasks_within_band=all(v['within_requested_band'] for v in gate.values()),parent_C_all_tasks_within_band=receipt['all_tasks_within_band'],hardware_admitted=False))
        put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()));put(a.out/'fit_return.json',dict(returncode=0,seconds=time.time()-began))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
