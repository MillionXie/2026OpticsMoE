"""Explicit pretrained training-only teacher diagnostic; no TEST access or student changes."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import caltech10_equal60_hybrid as m
OUT=m.HERE/'runs/smoke/caltech10_fixed10_resnet18_trainval_teacher_20261017'
CPU=m.HERE/'runs/smoke/caltech10_fixed10_resnet18_teacher_cpu_20261017'

def self_guard(live,pid,gpu,authorized_parallel=()):
    assert gpu in m.GPU.values() and live['total']<=4
    own=[x for x in live['jobs'] if x['pid']==pid]
    assert own==[dict(pid=pid,gpu_uuid=gpu)] and live['counts'][gpu]==1,live
    assert all(x['gpu_uuid'] in m.GPU.values() for x in live['jobs'])
    assert all(x['pid']==pid or (x['pid'],x['gpu_uuid']) in authorized_parallel for x in live['jobs']),live

def parallel_pilot():
    if not (m.OUT/'status.json').exists():return ()
    meta=m.s.read(m.OUT/'metadata.json');assert meta['sources']==m.sources() and meta['config']==m.config();authorized=[]
    for job in m.s.read(m.OUT/'status.json')['active']:
        path=Path('/proc',str(job['pid']),'cmdline')
        if not path.exists():continue
        cmd=path.read_bytes();assert str(m.OUT).encode() in cmd and b'caltech10_equal60_hybrid.py' in cmd
        authorized.append((job['pid'],job['gpu']))
    return tuple(authorized)

def cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and m.slots()['total']==0 and not CPU.exists();m.claim(CPU.with_name(CPU.name+'.claim.json'),'teacher_analytic_CPU')
    cfg=m.config();assert m.s.sha(cfg['teacher_checkpoint'])==cfg['teacher_checkpoint_sha256']
    rng=np.random.default_rng(17);x=rng.normal(size=(6,4));w=rng.normal(size=(3,4));y=np.arange(6)%3;xt=torch.tensor(x,dtype=torch.float64);wt=torch.tensor(w,dtype=torch.float64,requires_grad=True)
    loss=torch.nn.functional.cross_entropy(xt@wt.T,torch.tensor(y));g=torch.autograd.grad(loss,wt)[0].numpy();z=x@w.T;p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);p[np.arange(6),y]-=1;ref=p.T@x/6;err=float(np.abs(g-ref).max());assert err<1e-12
    gpu=m.GPU[4];live=dict(total=1,jobs=[dict(pid=123,gpu_uuid=gpu)],counts={g:int(g==gpu) for g in m.GPU.values()});self_guard(live,123,gpu)
    rejected=0
    for bad in [dict(total=0,jobs=[],counts=live['counts']),dict(total=1,jobs=[dict(pid=124,gpu_uuid=gpu)],counts=live['counts']),dict(total=2,jobs=live['jobs']+[dict(pid=124,gpu_uuid=gpu)],counts={**live['counts'],gpu:2}),dict(total=2,jobs=live['jobs']+[dict(pid=124,gpu_uuid='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55')],counts=live['counts'])]:
        try:self_guard(bad,123,gpu)
        except AssertionError:rejected+=1
    assert rejected==4;CPU.mkdir();m.s.save(CPU/'checks.json',dict(m.identity(),passed=True,analytic_CE_gradient_maxabs=err,resource_guard_rejected_cases=rejected,real_model_objects=0,real_forward_calls=0,TEST_images_read=0,backbone_optimizer_steps=0),exclusive=True)

@torch.no_grad()
def extract(model,transform,data,device,batch):
    # Exact torchvision prescribed RGB preprocessing; no augmentation/TEST.
    arrays=[]
    for start in range(0,len(data[1]),batch):
        rgb=data[0][start:start+batch].to(device);features=model(transform(rgb));assert features.shape==(len(rgb),512) and torch.isfinite(features).all();arrays.append(features.cpu().numpy())
    return np.concatenate(arrays).astype(np.float32)

def run(gpu):
    assert not OUT.exists() and gpu==m.GPU[4] and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu
    proof=m.s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==m.sources();m.engine.verify_sources(proof['sources'],proof['git_commit']);self_guard(m.slots(),os.getpid(),gpu,parallel_pilot())
    m.claim(OUT.with_name(OUT.name+'.claim.json'),'pretrained_teacher_TRAIN_VAL_only');cfg=m.config();assert m.s.sha(cfg['teacher_checkpoint'])==cfg['teacher_checkpoint_sha256']
    m.s.gpu_budget.ALLOWED.update(m.GPU.values());policy=m.s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2);torch.set_num_threads(4);m.engine.seed(17)
    from torchvision.models import resnet18,ResNet18_Weights
    OUT.mkdir();m.s.save(OUT/'metadata.json',dict(m.identity(),pid=os.getpid(),config=cfg,policy=policy,gpu_uuid=gpu,checkpoint_sha256=cfg['teacher_checkpoint_sha256'],TEST_images_read=0),exclusive=True)
    model=resnet18(weights=None);model.load_state_dict(torch.load(cfg['teacher_checkpoint'],map_location='cpu',weights_only=True),strict=True);model.fc=nn.Identity();model.eval().requires_grad_(False);before=m.s.digest(model.state_dict());model.cuda()
    data={role:m.load(role) for role in ('train','val')};features={role:extract(model,ResNet18_Weights.IMAGENET1K_V1.transforms(),data[role],'cuda',cfg['teacher_feature_batch']) for role in data};after=m.s.digest(model.state_dict());assert before==after and all(p.grad is None and not p.requires_grad for p in model.parameters());del model;torch.cuda.empty_cache()
    for role in data:m.save_np(OUT/(role+'_features.npz'),dict(features=features[role],labels=data[role][1].numpy(),ids=data[role][2]))
    mean=features['train'].astype(np.float64).mean(0);std=features['train'].astype(np.float64).std(0).clip(1e-3);x={role:torch.from_numpy(((features[role]-mean)/std).astype(np.float32)) for role in data};ys={role:data[role][1] for role in data};m.engine.seed(17);head=nn.Linear(512,10);optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['teacher_learning_rate'],weight_decay=cfg['teacher_weight_decay']);history=[];best=float('inf')
    for epoch in range(1,cfg['teacher_epochs']+1):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(head(x['train']),ys['train']);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():metrics={role:m.metrics(ys[role].numpy(),torch.softmax(head(x[role]),1).numpy().astype(np.float64)) for role in data}
        history.append(dict(epoch=epoch,train=metrics['train'],val=metrics['val'],loss=float(loss.detach())))
        if metrics['val']['nll']<best:best=metrics['val']['nll'];selected=copy.deepcopy(head.state_dict());selected_epoch=epoch;selected_metrics=metrics
    head.load_state_dict(selected,strict=True);m.engine.save_pt(OUT/'selected_head.pt',dict(model=selected,mean=mean,std=std,epoch=selected_epoch,metrics=selected_metrics,config=cfg,sources=m.sources()))
    for role in data:
        with torch.no_grad():p=torch.softmax(head(x[role]),1).numpy().astype(np.float64)
        m.save_np(OUT/('selected_'+role+'.npz'),dict(probabilities=p,labels=ys[role].numpy(),ids=data[role][2]))
    m.s.save(OUT/'history.json',history,exclusive=True);m.s.save(OUT/'fit_result.json',dict(passed=True,actual_epochs=cfg['teacher_epochs'],selected_epoch=selected_epoch,metrics=selected_metrics,backbone_before=before,backbone_after=after,backbone_optimizer_steps=0,linear_optimizer_steps=cfg['teacher_epochs'],TEST_images_read=0,teacher_only=True,student_modified=False,teacher_checkpoint_sha256=cfg['teacher_checkpoint_sha256'],head_sha256=m.s.sha(OUT/'selected_head.pt'),feature_sha256={role:m.s.sha(OUT/(role+'_features.npz')) for role in data}),exclusive=True)
    print(json.dumps(dict(passed=True,actual_epochs=cfg['teacher_epochs'],selected_epoch=selected_epoch,train=selected_metrics['train']['accuracy'],val=selected_metrics['val']['accuracy'],TEST_images_read=0,student_modified=False)),flush=True)

def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'science').exists();m.claim(OUT/'audit_claim.json','independent_teacher_saved_linear_arithmetic')
    meta=m.s.read(OUT/'metadata.json');f=m.s.read(OUT/'fit_result.json');ex=m.s.read(OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid'])).exists();assert meta['sources']==m.sources();m.engine.verify_sources(meta['sources'],meta['git_commit'])
    cfg=m.config();assert meta['config']==cfg and m.s.sha(cfg['teacher_checkpoint'])==f['teacher_checkpoint_sha256']==cfg['teacher_checkpoint_sha256'];assert f['backbone_before']==f['backbone_after'] and f['backbone_optimizer_steps']==0 and f['TEST_images_read']==0
    state=torch.load(OUT/'selected_head.pt',map_location='cpu',weights_only=False);assert m.s.sha(OUT/'selected_head.pt')==f['head_sha256'] and state['sources']==m.sources() and state['config']==cfg;hist=m.s.read(OUT/'history.json');assert len(hist)==100 and [v['epoch'] for v in hist]==list(range(1,101));chosen=min(hist,key=lambda v:v['val']['nll']);assert chosen['epoch']==state['epoch']==f['selected_epoch']
    arrays={}
    for role in ('train','val'):
        assert f['feature_sha256'][role]==m.s.sha(OUT/(role+'_features.npz'))
        with np.load(OUT/(role+'_features.npz'),allow_pickle=False) as z:arrays[role]={k:z[k] for k in z.files}
    assert np.array_equal(state['mean'],arrays['train']['features'].astype(np.float64).mean(0)) and np.array_equal(state['std'],arrays['train']['features'].astype(np.float64).std(0).clip(1e-3));checks={}
    for role,z in arrays.items():
        assert z['ids'].tolist()==m.manifest()['split_ids'][role] and np.bincount(z['labels'],minlength=10).tolist()==m.manifest()['supports'][role]
        x=((z['features']-state['mean'])/state['std']).astype(np.float32).astype(np.float64);logits=x@state['model']['weight'].numpy().astype(np.float64).T+state['model']['bias'].numpy().astype(np.float64);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        with np.load(OUT/('selected_'+role+'.npz'),allow_pickle=False) as saved:
            assert np.array_equal(saved['ids'],z['ids']) and np.array_equal(saved['labels'],z['labels']);error=float(np.abs(p-saved['probabilities']).max());assert error<3e-6,error;metrics=m.metrics(z['labels'],saved['probabilities'])
        for field in ('accuracy','balanced_accuracy','nll','balanced_nll'):assert abs(metrics[field]-f['metrics'][role][field])<1e-8 and abs(metrics[field]-chosen[role][field])<1e-8
        checks[role]=dict(linear_numpy_maxabs=error,prediction_sha256=m.s.sha(OUT/('selected_'+role+'.npz')),metrics=metrics)
    (OUT/'science').mkdir();m.s.save(OUT/'science/audit.json',dict(passed=True,sources=m.sources(),checks=checks,model_objects=0,forward_calls=0,TEST_images_read=0,pretraining='ImageNet1K public checkpoint, training-only teacher, not student architecture'),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','run','audit'],required=True);p.add_argument('--gpu');a=p.parse_args()
    try:run(a.gpu) if a.phase=='run' else globals()[a.phase]()
    except BaseException as e:
        target=CPU if a.phase=='cpu' else OUT;mark=target/'failure.json' if target.exists() else target.with_name(target.name+'.failure.json')
        if not mark.exists():m.s.save(mark,dict(error=repr(e),automatic_retry=False,phase=a.phase),exclusive=True)
        raise
