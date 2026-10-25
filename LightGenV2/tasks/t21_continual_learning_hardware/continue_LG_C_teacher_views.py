"""Actual frozen-teacher responses to TRAIN views, not mixed optical logits.

Eight weak input-amplitude views of exactly300 old sources/task, cached once.
Original C inference graph, gains, frozen eight old experts and A/B unchanged.
No TEST teacher inputs, labels or gradients. A new C still requires new D.
"""
import argparse,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from .train_native import sha,put,tensor_sha,frozen,gpu_gate,load_data,load_dataset,evaluate,summary,reference_teacher
from .continue_native_chain import phase_batches
from .continue_LG_C_optics import continuation_objective,configure_simulation_trainability
from .hardware_bridge import load_checkpoint
from .refine_LG_retention import retention_selection


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_LG_C_teacher_views_20261025.json').read_text()):raise ValueError('Only published finite actual-teacher TRAIN views')


def amplitude_view(fields,cfg,view):
    """CPU TRAIN-only photometric perturbation; no inference preprocessing."""
    if not 0<=view<cfg['views_per_old_source']:raise ValueError('Unknown declared view')
    dims=tuple(range(1,fields.ndim));scale=fields.std(dims,keepdim=True,unbiased=False)
    noise=torch.randn(fields.shape,generator=torch.Generator().manual_seed(cfg['seed']+view),dtype=fields.dtype)
    gain=1+cfg['view_gain_delta']*(-1 if view%2 else 1)
    return (fields*gain+noise*scale*cfg['view_noise_TRAIN_std']).clamp(0,1)


@torch.no_grad()
def teacher_views(teacher,fields,cfg,device):
    """Compute actual nonlinear responses, never interpolate source logits."""
    inputs=[];targets=[]
    for view in range(cfg['views_per_old_source']):
        x=amplitude_view(fields,cfg,view);rows=[]
        for start in range(0,len(x),8):rows.append(teacher(x[start:start+8].to(device))['logits'].cpu())
        inputs.append(x);targets.append(torch.cat(rows))
    return torch.stack(inputs),torch.stack(targets)


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('chain','warm_run','profile','train','test','source_protocols','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');receipt=json.loads((a.warm_run/'C/completion.json').read_text());origin=json.loads((a.chain/'C/completion.json').read_text());protocol=json.loads((a.warm_run/'protocol.json').read_text())
    if protocol.get('parent_C_sha256',protocol.get('original_chain_C_sha256'))!=origin['best_sha256']:raise ValueError('Exact same-chain warm C required')
    model,cp=load_checkpoint(a.warm_run/'C/best_checkpoint.pt',receipt['best_sha256']);trainable=configure_simulation_trainability(model);model.to(device);immutable=frozen(model);gains=tensor_sha(model.display_gains)
    train=load_data(a.train,'train');names=tuple(cfg['reference']);memories=json.loads((a.chain/'memory_manifest.json').read_text());target=torch.load(a.chain/'C/TRAIN_reference_teacher_logits.pt',map_location='cpu',weights_only=False);identity=json.loads(Path(__file__).with_name('source_identity.json').read_text())
    if target['TEST_inputs'] is not False or target['teacher_sha256']!=identity['models'][2]['sha256']:raise ValueError('Original TRAIN teacher cache differs')
    old={}
    for name in names[:-1]:
        indices=np.asarray(memories[name]['indices'])
        if len(indices)!=300 or not np.array_equal(indices,target['selection'][name]) or memories[name]['source_manifest']!=train[name]['manifest']['identity_sha256']:raise ValueError('Exact old300 source identity required')
        old[name]=dict(fields=train[name]['x'][indices],labels=train[name]['y'][indices])
    if a.preflight_only:
        n=names[0];x=amplitude_view(old[n]['fields'][:2],cfg,0).to(device);loss=continuation_objective(model(x)['logits'],old[n]['labels'][:2].to(device),target['logits'][n][:2].to(device),cfg);loss.backward()
        if not torch.isfinite(loss) or frozen(model)!=immutable or any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Real augmented TRAIN gradient/frozen preflight failed')
        print(json.dumps(dict(passed=True,loss=float(loss.detach()),trainable=trainable,original_sources_per_old_task=300,TEST_inputs=False,SDK=False)));return
    a.out.mkdir(parents=True,exist_ok=False);out=a.out/'C';out.mkdir();began=time.time()
    try:
        put(a.out/'protocol.json',dict(config=cfg,config_sha256=sha(a.profile),architecture='moe',original_chain_C_sha256=origin['best_sha256'],warm_C_sha256=receipt['best_sha256'],
            candidate_scope='C only; changed C requires newly dependent D',old_A_B_unmodified=True,TEST_gradients=False,TEST_teacher_inputs=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
            augmentation_scope='Only fixed300 original TRAIN sources per old task; eight weak amplitude views; actual frozen original C teacher forward',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),gpu=gpu,trainable=trainable,SDK=False,new_inference_layers=0))
        teacher,teacher_identity=reference_teacher(identity,'moe',3,device);views={}
        for name in names[:-1]:
            put(a.out/'status.json',dict(state='actual_TRAIN_teacher_views',task=name,updated=time.time()));fields,logits=teacher_views(teacher,old[name]['fields'],cfg,device);views[name]=dict(fields=fields,logits=logits)
        del teacher;torch.cuda.empty_cache();cache=a.out/'actual_TRAIN_teacher_views.pt';torch.save(views,cache)
        put(a.out/'TRAIN_view_manifest.json',dict(teacher_checkpoint_SHA256=teacher_identity['sha256'],cache_SHA256=sha(cache),TRAIN_source_indices={n:memories[n]['indices'] for n in names[:-1]},views_per_source=8,TEST_inputs=False,interpolated_logits=False))
        test=load_data(a.test,'test');paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
        if sha(paths['speech_binary'])!=identity['models'][3]['config']['protocol_sha256']['speech_binary'] or sha(paths['vision'])!=train['speech_binary']['manifest']['vision_checkpoint_sha256']:raise ValueError('Frozen full TRAIN frontend/protocol differs')
        current=load_dataset(paths,'speech_binary','train',vision_checkpoint=paths['vision'],device=device)
        if len(current)!=12526 or not np.array_equal(target['selection']['speech_binary'],np.arange(12526)):raise ValueError('Full Speech TRAIN required')
        selections={n:np.arange(12526) if n=='speech_binary' else np.asarray(memories[n]['indices']) for n in names}
        baseline=evaluate(model,test,names,device,8);best=retention_selection(baseline,cfg['reference']);best_epoch=0;history=[]
        def save(role,epoch):
            torch.save(dict(model={n:v.detach().cpu() for n,v in model.state_dict().items()},architecture='moe',stage=3,epoch=epoch,best_epoch=best_epoch,parent_checkpoint_sha256=receipt['best_sha256'],config_sha256=sha(a.profile),TEST_gradients=False,TEST_selected_development=True),out/(role+'_checkpoint.pt'))
        save('best',0);params=[v for v in model.parameters() if v.requires_grad];optical=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')];optimizer=torch.optim.AdamW([dict(params=optical,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay']);started=time.monotonic()
        for epoch in range(1,cfg['epochs']+1):
            model.train();seen=0;total=0.;by_task={n:dict(seen=0,correct=0,augmented_seen=0) for n in names};view_counter={n:0 for n in names[:-1]}
            for task,relative in phase_batches(selections,names,cfg,epoch,3):
                if task=='speech_binary':x=current.get_batch(relative,device);y=torch.as_tensor(np.asarray(current.labels)[relative],device=device,dtype=torch.long)
                else:x=old[task]['fields'][relative].to(device);y=old[task]['labels'][relative].to(device)
                optimizer.zero_grad(set_to_none=True);logits=model(x)['logits'];loss=continuation_objective(logits,y,target['logits'][task][relative].to(device),cfg)
                if task!='speech_binary':
                    view=(epoch+view_counter[task])%8;view_counter[task]+=1;extra=model(views[task]['fields'][view,relative].to(device))['logits'];loss=cfg['old_task_weight']*(loss+cfg['actual_teacher_view_weight']*continuation_objective(extra,y,views[task]['logits'][view,relative].to(device),cfg));by_task[task]['augmented_seen']+=len(y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite actual-view TRAIN objective')
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();seen+=len(y);total+=float(loss.detach())*len(y);by_task[task]['seen']+=len(y);by_task[task]['correct']+=int((logits.argmax(1)==y).sum())
                if seen%4000<len(y):put(a.out/'status.json',dict(state='training',epoch=epoch,TRAIN_seen=seen,running_loss=total/seen,updated=time.time()))
            scores=evaluate(model,test,names,device,8);key=retention_selection(scores,cfg['reference'])
            if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gains:raise RuntimeError('Old experts/gains changed')
            if key>best:best=key;best_epoch=epoch;save('best',epoch)
            save('last',epoch);history.append(dict(epoch=epoch,loss=total/seen,TRAIN_by_task=by_task,scores=summary(scores),best_epoch=best_epoch));put(out/'history.json',history)
            if time.monotonic()-started>=cfg['max_fit_seconds']:break
        model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True);selected=evaluate(model,test,names,device,8);put(out/'selected_test.json',dict(scores=selected,TEST_gradients=False));put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,fixed_gain_sha256=gains))
        gate={n:dict(reference=cfg['reference'][n],adapted=selected[n]['balanced_accuracy'],within_requested_band=cfg['reference'][n]-.01-1e-9<=selected[n]['balanced_accuracy']<=cfg['reference'][n]+.06+1e-9) for n in names}
        put(out/'completion.json',dict(phase='C',best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),selected_test_sha256=sha(out/'selected_test.json'),test=summary(selected),original_band=gate,all_tasks_within_band=all(v['within_requested_band'] for v in gate.values())))
        put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()));put(a.out/'fit_return.json',dict(returncode=0,seconds=time.time()-began))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),updated=time.time()));raise


if __name__=='__main__':main()
