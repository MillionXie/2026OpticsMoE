"""Evidence-led, finite continuation: same native T21 graph and original input.

TRAIN-only conditioned fit is folded into the existing no-bias 7840 head.
Optical and head gradient clipping are separate; per-domain route balance avoids
accepting global means which hide a sensor collapse. Never reads TEST in fitting.
"""
import argparse,json,subprocess,time,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .unified_native import build_unified,CONTRACT,FRONTEND_SHA256
from .train_unified_static import load_arrays,cache_frontend,batch_amplitude,balanced_batches,evaluate,selection
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import gpu_gate,put,sha,tensor_sha


def validate_profile(cfg):
    if cfg!=json.loads(Path(__file__).with_name('config_unified_refine_20261025.json').read_text()):raise ValueError('Exact finite published continuation profile required')


def conditioned_inputs(train,val,ridge):
    """TRAIN covariance only. Return transform mapping original x -> x@T."""
    x=torch.as_tensor(train,dtype=torch.float64);v=torch.as_tensor(val,dtype=torch.float64)
    scale=x.square().mean(0).sqrt().clamp_min(1e-6);z=x/scale
    eigen,vectors=torch.linalg.eigh(z.T@z/len(z));transform=(vectors/(eigen.clamp_min(0)+ridge).sqrt()[None,:])/scale[:,None]
    return x@transform,v@transform,transform


def scores_from_logits(logits,data):
    logits=torch.as_tensor(logits);y=torch.as_tensor(data['labels']);d=np.asarray(data['domains']);loss=F.cross_entropy(logits,y,reduction='none');pred=logits.argmax(1).numpy();scores={}
    for domain,name in ((0,'RGB'),(1,'SAR')):
        ix=d==domain;cm=np.zeros((10,10),dtype=int);np.add.at(cm,(data['labels'][ix],pred[ix]),1)
        scores[name]=dict(n=int(ix.sum()),correct=int((pred[ix]==data['labels'][ix]).sum()),accuracy=float((pred[ix]==data['labels'][ix]).mean()),loss=float(loss[ix].mean()),confusion_matrix=cm.tolist())
    return dict(scores=scores)


def domain_balance(route,domain,indices):
    active=route.index_select(1,indices);values=[]
    for d in torch.unique(domain):values.append((active[domain==d].mean(0)-1/len(indices)).square().sum())
    return torch.stack(values).mean()


@torch.no_grad()
def cache_final(model,data,front,device,out,split):
    rows=[]
    for start in range(0,len(data['labels']),8):
        ix=np.arange(start,min(start+8,len(data['labels'])));rows.append(model(batch_amplitude(data,front,ix,device))['ccd_features'].cpu().numpy())
        if start%800==0:put(out/'status.json',dict(state='readout_cache',split=split,seen=int(ix[-1]+1),updated=time.time()))
    return np.concatenate(rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('parent','data','frontend','profile','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    cfg=json.loads(a.profile.read_text());validate_profile(cfg);torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=gpu_gate(cfg,a.gpu_uuid);device=torch.device('cuda:0');parent=json.loads((a.parent/'completion.json').read_text());old_protocol=json.loads((a.parent/'protocol.json').read_text());path=a.parent/'best_checkpoint.pt';parent_sha=sha(path)
    if parent_sha!=parent['best_sha256'] or old_protocol['selection_uses_TEST']:raise ValueError('Locked validation-selected parent required')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['contract']!=CONTRACT or cp['frontend_sha256']!=FRONTEND_SHA256:raise ValueError('Changed original graph/encoding')
    architecture=cp['architecture'];domains=old_protocol['training_domains'];case=old_protocol['case'];model=build_unified(architecture).to(device);model.load_state_dict(cp['model'],strict=True)
    fixed_gain=tensor_sha(model.display_gains);frozen={n:tensor_sha(v) for n,v in model.named_parameters() if not v.requires_grad};train=load_arrays(a.data,'train');front=cache_frontend(train,a.frontend)
    optics=[v for n,v in model.named_parameters() if v.requires_grad and not n.startswith('shared_head.')]
    ix=next(balanced_batches(train,domains,cfg['batch_size'],cfg['seed']));r=model(batch_amplitude(train,front,ix,device));loss=F.cross_entropy(r['logits'],torch.as_tensor(train['labels'][ix],device=device));loss.backward()
    if any(v.grad is None or not torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad):raise RuntimeError('Continuation finite gradient preflight failed')
    if a.preflight_only:
        print(json.dumps(dict(passed=True,case=case,parent_SHA256=parent_sha,loss=float(loss.detach()),TEST_loaded=False,SDK=False)));return
    model.zero_grad(set_to_none=True);model.eval();val=load_arrays(a.data,'validation');vf=cache_frontend(val,a.frontend);a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    numerical=[Path(__file__),Path(__file__).with_name('unified_native.py'),Path(__file__).with_name('train_unified_static.py'),Path('LightGenV2/tasks/t21_continual_learning_hardware/model.py'),Path(__file__).with_name('hardware_aware.py')];source_sha={str(v):sha(v) for v in numerical}
    put(a.out/'protocol.json',dict(contract=CONTRACT,case=case,architecture=architecture,config=cfg,config_sha256=sha(a.profile),training_domains=domains,parent_SHA256=parent_sha,parent_run=str(a.parent),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=source_sha,frontend_sha256=FRONTEND_SHA256,fixed_gain_sha256=fixed_gain,dataset_manifest_sha256=sha(a.data/'manifest.json'),input_contract=old_protocol['input_contract'],physical_graph=old_protocol['physical_graph'],selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,SDK=False,gpu=gpu))
    tx=cache_final(model,train,front,device,a.out,'train');vx=cache_final(model,val,vf,device,a.out,'validation');current=model.shared_head.weight.detach().cpu().clone();baseline=scores_from_logits(vx@current.numpy().T,val);put(a.out/'baseline_validation.json',baseline);best=selection(baseline,domains);best_epoch=0;best_readout_step=0;history=[]
    def guard():
        if tensor_sha(model.display_gains)!=fixed_gain or any(tensor_sha(dict(model.named_parameters())[n])!=v for n,v in frozen.items()):raise RuntimeError('Frozen gain/inactive phases changed')
        if any(sha(v)!=source_sha[str(v)] for v in numerical):raise RuntimeError('Numerical source changed during run')
        if sha(path)!=parent_sha:raise RuntimeError('Locked parent changed')
    def save(role,epoch,step=0):
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},architecture=architecture,stage=1,epoch=epoch,readout_step=step,contract=CONTRACT,case=case,config_sha256=sha(a.profile),frontend_sha256=FRONTEND_SHA256,parent_SHA256=parent_sha,TEST_gradients=False),a.out/(role+'_checkpoint.pt'))
    def consider(report,epoch,step=0):
        nonlocal best,best_epoch,best_readout_step
        score=selection(report,domains)
        if score>best:
            best=score;best_epoch=epoch;best_readout_step=step;save('best',epoch,step);put(a.out/'selected_validation.json',report);return True
        return False
    save('best',0);save('last',0)
    mask=np.isin(train['domains'],domains);z,zv,transform=conditioned_inputs(tx[mask],vx,cfg['conditioner_ridge']);labels=torch.as_tensor(train['labels'][mask]);param=torch.nn.Parameter(torch.linalg.solve(transform,current.double().T).T)
    torch.testing.assert_close(zv@param.detach().T,torch.as_tensor(vx,dtype=torch.float64)@current.double().T,atol=1e-7,rtol=1e-7)
    opt=torch.optim.LBFGS([param],lr=1.,max_iter=cfg['readout_inner_iterations'],history_size=10,line_search_fn='strong_wolfe')
    for step in range(1,cfg['readout_fit_steps']+1):
        def closure():
            opt.zero_grad();cost=F.cross_entropy(z@param.T,labels,label_smoothing=cfg['label_smoothing'])+cfg['readout_regularization']*param.square().sum()/10;cost.backward();return cost
        fit_loss=float(opt.step(closure).detach());weight=(param.detach()@transform.T).float()
        if not math.isfinite(fit_loss) or not torch.isfinite(weight).all():raise RuntimeError('Nonfinite conditioned readout')
        with torch.no_grad():model.shared_head.weight.copy_(weight.to(device))
        report=scores_from_logits(vx@weight.numpy().T,val);consider(report,0,step);history.append(dict(stage='conditioned_readout',step=step,TRAIN_loss=fit_loss,validation=report['scores']));put(a.out/'history.json',history);put(a.out/'status.json',dict(state='conditioned_readout',step=step,validation=report['scores'],updated=time.time()))
    selected=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(selected['model'],strict=True);model.zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW([dict(params=optics,lr=cfg['optical_lr'],weight_decay=0.),dict(params=model.shared_head.parameters(),lr=cfg['head_lr'],weight_decay=cfg['head_weight_decay'])]);last_improvement=0
    for epoch in range(1,cfg['epochs']+1):
        model.train();total=0.;seen=0;correct={0:0,1:0};count={0:0,1:0};factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,cfg['epochs']-1)))
        for group,lr in zip(optimizer.param_groups,(cfg['optical_lr'],cfg['head_lr'])):group['lr']=lr*factor
        for ix in balanced_batches(train,domains,cfg['batch_size'],cfg['seed']+epoch):
            optimizer.zero_grad(set_to_none=True);r=model(batch_amplitude(train,front,ix,device));y=torch.as_tensor(train['labels'][ix],device=device);d=torch.as_tensor(train['domains'][ix],device=device);cost=F.cross_entropy(r['logits'],y,label_smoothing=cfg['label_smoothing'])
            if architecture=='moe':
                active=model.active_indices[:4];q=r['route_power'];cost+=cfg['global_balance_weight']*(q.index_select(1,active).mean(0)-.25).square().sum()+cfg['domain_balance_weight']*domain_balance(q,d,active)
            if not torch.isfinite(cost):raise RuntimeError('Nonfinite continuation loss')
            cost.backward();torch.nn.utils.clip_grad_norm_(optics,cfg['optical_clip_norm']);torch.nn.utils.clip_grad_norm_(model.shared_head.parameters(),cfg['head_clip_norm']);optimizer.step();total+=float(cost.detach())*len(y);seen+=len(y)
            for domain in domains:
                maskd=d==domain;correct[domain]+=int(((r['logits'].argmax(1)==y)&maskd).sum());count[domain]+=int(maskd.sum())
            if seen%1000<len(y):put(a.out/'status.json',dict(state='optical_continuation',epoch=epoch,TRAIN_seen=seen,loss=total/seen,updated=time.time()))
        guard();report=evaluate(model,val,vf,device)
        if consider(report,epoch):last_improvement=epoch
        save('last',epoch);row=dict(stage='optical_continuation',epoch=epoch,TRAIN_loss=total/seen,TRAIN_accuracy_by_domain={str(d):correct[d]/count[d] for d in domains},validation=report['scores'],routing=report['routing'],best_epoch=best_epoch,best_readout_step=best_readout_step);history.append(row);put(a.out/'history.json',history);put(a.out/'status.json',dict(state='optical_continuation',**row,updated=time.time()))
        if time.monotonic()-started>cfg['max_case_seconds']:raise RuntimeError('Finite continuation wall bound reached')
        if epoch>=cfg['minimum_epochs'] and epoch-last_improvement>=cfg['patience']:break
    guard();locked=dict(best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),TEST_loaded=False);put(a.out/'locked_before_TEST.json',locked)
    cp=torch.load(a.out/'best_checkpoint.pt',map_location=device,weights_only=False);model.load_state_dict(cp['model'],strict=True);test=load_arrays(a.data,'test');tf=cache_frontend(test,a.frontend);report=evaluate(model,test,tf,device);put(a.out/'selected_test.json',report)
    if sha(a.out/'best_checkpoint.pt')!=locked['best_sha256']:raise RuntimeError('Locked selected weights changed')
    put(a.out/'completion.json',dict(passed=True,case=case,architecture=architecture,best_epoch=best_epoch,best_readout_step=best_readout_step,epochs_completed=epoch,test=report['scores'],best_sha256=locked['best_sha256'],last_sha256=locked['last_sha256'],SDK=False,selection_uses_TEST=False,TEST_gradients=False,TEST_teacher=False,independent_new_blind_TEST_claim=False,original_graph_CPU_audit_pending=True,seconds=time.monotonic()-started));put(a.out/'status.json',dict(state='complete_CPU_audit_pending',updated=time.time()))


if __name__=='__main__':main()
