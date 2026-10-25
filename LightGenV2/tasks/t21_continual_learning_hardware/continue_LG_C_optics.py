"""Two full TRAIN epochs from fixed-optics C head candidate, no SDK.

Original model, old experts, gains, memory identities and A/B preserved.
Any accepted new C requires a newly dependent D; never splice the old D.
"""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from .train_native import (sha,put,tensor_sha,frozen,gpu_gate,load_data,load_dataset,
    evaluate,summary,reference_TRAIN_objective)
from .hardware_bridge import load_checkpoint
from .continue_native_chain import phase_batches
from .refine_LG_retention import retention_selection


def validate(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_LG_C_optical_continuation_20261025.json').read_text()):raise ValueError('Only finite two-epoch C continuation')


def main():
    p=argparse.ArgumentParser(__doc__)
    for n in ('chain','warm_run','profile','train','test','source_protocols','out'):p.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();cfg=json.loads(a.profile.read_text());validate(cfg)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');receipt=json.loads((a.warm_run/'C/completion.json').read_text());parent=a.warm_run/'C/best_checkpoint.pt'
    model,cp=load_checkpoint(parent,receipt['best_sha256']);model.configure_stage(3);model.to(device);immutable=frozen(model);gain=tensor_sha(model.display_gains)
    source=json.loads((a.warm_run/'protocol.json').read_text());original=json.loads((a.chain/'C/completion.json').read_text())
    if cp['architecture']!='moe' or cp['stage']!=3 or source['parent_C_sha256']!=original['best_sha256']:raise ValueError('Exact same-chain warm C required')
    names=list(cfg['reference']);train=load_data(a.train,'train');memories=json.loads((a.chain/'memory_manifest.json').read_text());cache=a.chain/'C/TRAIN_reference_teacher_logits.pt';target=torch.load(cache,map_location='cpu',weights_only=False)
    identity=json.loads(Path(__file__).with_name('source_identity.json').read_text());ref=identity['models'][2]
    if target['TEST_inputs'] is not False or target['teacher_sha256']!=ref['sha256'] or sha(ref['checkpoint'])!=ref['sha256']:raise ValueError('Original TRAIN teacher changed')
    for n in names[:-1]:
        if len(memories[n]['indices'])!=300 or not np.array_equal(memories[n]['indices'],target['selection'][n]):raise ValueError('Old300 identity changed')
    if a.preflight_only:
        n=names[0];ix=np.asarray(memories[n]['indices'][:2]);logits=model(train[n]['x'][ix].to(device))['logits'];loss=reference_TRAIN_objective(logits,train[n]['y'][ix].to(device),target['logits'][n][:2].to(device),cfg);loss.backward()
        if not torch.isfinite(loss) or frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain:raise RuntimeError('Real TRAIN/freeze preflight failed')
        print(json.dumps(dict(passed=True,loss=float(loss.detach()),TRAIN_only=True,old_experts_gains_frozen=True,SDK=False)));return
    paths={k:Path(v) for k,v in json.loads(a.source_protocols.read_text()).items()}
    if sha(paths['speech_binary'])!=identity['models'][3]['config']['protocol_sha256']['speech_binary'] or sha(paths['vision'])!=train['speech_binary']['manifest']['vision_checkpoint_sha256']:raise ValueError('Original current TRAIN protocol/frontend differs')
    current=load_dataset(paths,'speech_binary','train',vision_checkpoint=paths['vision'],device=device);selection={n:np.arange(len(current)) if n==names[-1] else np.asarray(memories[n]['indices']) for n in names}
    if len(current)!=12526 or not np.array_equal(selection['speech_binary'],target['selection']['speech_binary']):raise ValueError('Full current TRAIN teacher order changed')
    test=load_data(a.test,'test');a.out.mkdir(parents=True,exist_ok=False);out=a.out/'C';out.mkdir();started=time.monotonic()
    put(a.out/'protocol.json',dict(architecture='moe',config=cfg,config_sha256=sha(a.profile),warm_C_checkpoint_sha256=receipt['best_sha256'],original_chain_C_sha256=original['best_sha256'],
        original_A_B_unchanged=True,D_requires_retrain_from_new_C=True,no_old_D_splicing=True,TRAIN_teacher_cache_sha256=sha(cache),TEST_gradients=False,TEST_selected_development_metric=True,independent_generalization_claim=False,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(__file__),gpu=gpu,SDK=False))
    baseline=evaluate(model,test,names,device,8);best=retention_selection(baseline,cfg['reference']);best_epoch=0;history=[]
    def save(role,epoch):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture='moe',stage=3,epoch=epoch,best_epoch=best_epoch,parent_checkpoint_sha256=cp['parent_checkpoint_sha256'],warm_C_checkpoint_sha256=receipt['best_sha256'],TEST_gradients=False,TEST_selected_development=True),out/(role+'_checkpoint.pt'))
    save('best',0);params=[v for v in model.parameters() if v.requires_grad];optical=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')]
    optimizer=torch.optim.AdamW([dict(params=optical,lr=cfg['optical_lr']),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    for epoch in range(1,cfg['epochs']+1):
        model.train();seen=0;loss_sum=0.;task_stats={n:dict(seen=0,correct=0) for n in names}
        for n,relative in phase_batches(selection,names,cfg,epoch,3):
            ix=selection[n][relative];x=current.get_batch(ix,device) if n==names[-1] else train[n]['x'][ix].to(device);y=torch.as_tensor(np.asarray(current.labels)[ix],device=device,dtype=torch.long) if n==names[-1] else train[n]['y'][ix].to(device)
            optimizer.zero_grad(set_to_none=True);logits=model(x)['logits'];loss=reference_TRAIN_objective(logits,y,target['logits'][n][relative].to(device),cfg)*(cfg['old_task_weight'] if n!=names[-1] else 1)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite original TRAIN objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();seen+=len(y);loss_sum+=float(loss.detach())*len(y);task_stats[n]['seen']+=len(y);task_stats[n]['correct']+=int((logits.argmax(1)==y).sum())
            if seen%4000<len(y):put(a.out/'status.json',dict(state='training',epoch=epoch,TRAIN_seen=seen,loss=loss_sum/seen,updated=time.time()))
        scores=evaluate(model,test,names,device,8);value=retention_selection(scores,cfg['reference'])
        if frozen(model)!=immutable or tensor_sha(model.display_gains)!=gain:raise RuntimeError('Old expert/gain changed')
        if value>best:best=value;best_epoch=epoch;save('best',epoch)
        save('last',epoch);history.append(dict(epoch=epoch,loss=loss_sum/seen,TRAIN_by_task=task_stats,scores=summary(scores),best_epoch=best_epoch));put(out/'history.json',history);put(a.out/'status.json',dict(state='training',**history[-1],updated=time.time()))
        if time.monotonic()-started>cfg['max_fit_seconds']:raise RuntimeError('Finite full-epoch fit bound reached')
    model.load_state_dict(torch.load(out/'best_checkpoint.pt',map_location=device,weights_only=False)['model'],strict=True);selected=evaluate(model,test,names,device,8)
    put(out/'selected_test.json',dict(scores=selected,TEST_gradients=False,TEST_selected_development_metric=True));put(out/'frozen_audit.json',dict(pass_all=True,frozen_parameter_SHA256=immutable,fixed_gain_sha256=gain))
    gate={n:dict(reference=cfg['reference'][n],adapted=v['accuracy'],within_requested_band=cfg['reference'][n]-.01-1e-9<=v['accuracy']<=cfg['reference'][n]+.06+1e-9) for n,v in selected.items()}
    put(out/'completion.json',dict(phase='C',best_epoch=best_epoch,best_sha256=sha(out/'best_checkpoint.pt'),last_sha256=sha(out/'last_checkpoint.pt'),selected_test_sha256=sha(out/'selected_test.json'),test=summary(selected),original_band=gate,all_tasks_within_band=all(v['within_requested_band'] for v in gate.values())))
    put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
