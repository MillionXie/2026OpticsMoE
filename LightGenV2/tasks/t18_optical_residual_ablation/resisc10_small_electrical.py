"""490-parameter CPU-only electrical side branch on audited frozen optical probabilities."""
import argparse,copy,csv,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import resisc10 as study
from audit_caltech_repair234 import exited
s=study.s;HERE=study.HERE;ROOT=study.ROOT;PROFILE=HERE/'resisc10_small_electrical_profile.json'
class SmallElectrical(torch.nn.Module):
    def __init__(self,alpha=.8):
        super().__init__();assert .6<alpha<=1.;self.alpha=float(alpha);self.linear=torch.nn.Linear(48,10)
        torch.nn.init.zeros_(self.linear.weight);torch.nn.init.zeros_(self.linear.bias)
        assert sum(p.numel() for p in self.parameters())==490
    def forward(self,optical,features):
        assert .6<self.alpha<=1. and optical.shape==(len(features),10) and features.shape[1]==48
        assert torch.isfinite(optical).all() and torch.allclose(optical.sum(1),torch.ones(len(features)),atol=1e-5)
        return self.alpha*optical+(1-self.alpha)*self.linear(features).softmax(1)
def features(images):
    assert images.dtype==torch.uint8 and images.shape[1:]==(3,150,150)
    x=F.interpolate(images.float()/255,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    return F.avg_pool2d(x,25).flatten(1)
def independent_features(images):
    x=F.interpolate(images.float()/255,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).numpy().astype(np.float64)
    return x.reshape(len(x),3,4,25,4,25).mean((3,5)).reshape(len(x),48)
def orders(epoch,n,seed):return np.random.default_rng(np.random.SeedSequence([seed,epoch,20261015,490])).permutation(n).astype('<i8')
def identity(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
def parent(base):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    proof=s.read(base/'cpu_report/audit.json');meta=s.read(base/'metadata.json');state=s.read(base/'status.json');done=s.read(base/'saved_cpu_audit_dispatch_completed_20261015.json')
    assert proof['passed'] and done['passed'] and done['audit_sha256']==s.sha(base/'cpu_report/audit.json') and proof['class_count']==10 and proof['test_evaluations']==0
    assert exited(meta['pid']) and state['active']==state['pending']==0 and not state['failures'] and len(state['completed'])==2 and all(exited(j['pid']) and j['exit_code']==0 for j in state['completed'])
    s.engine.verify_sources(meta['sources'],meta['git_commit']);assert not (base/'cpu_report_failure.json').exists()
    candidates=[(k,f) for k,f in proof['fits'].items() if f['rho']==.3];assert len(candidates)==1;key,f=candidates[0];cfg=s.read(PROFILE)
    assert f['config']['study_case']==cfg['parent_required_study_case'] and f['metrics']['val']['accuracy']<cfg['parent_target_validation_accuracy']
    assert f['config']['classes']==study.plan()['classes'] and f['checkpoint_sha256']==s.sha(base/key/'best_checkpoint.pt')
    assert min(s.read(base/key/'history.json'),key=lambda h:h['val']['balanced_nll'])['epoch']==f['selected_epoch']
    return proof,meta,key,f

def data(base,key):
    m=study.manifest(Path(study.plan()['parent_data']));result={}
    for role in ('train','val'):
        x,y,ids=study.load_fit(Path(study.plan()['parent_data']),role,m)
        rows=list(csv.DictReader((base/key/(role+'_predictions.csv')).open()))
        assert [v['sample_id'] for v in rows]==ids.tolist() and np.array_equal([int(v['label_true']) for v in rows],y.numpy())
        p=np.asarray([[float(v['score'+str(k)]) for k in range(10)] for v in rows],np.float32)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5) and np.array_equal(p.argmax(1),[int(v['label_pred']) for v in rows])
        result[role]=dict(images=x,y=y,ids=ids,p=torch.from_numpy(p),features=features(x))
    return result
@torch.no_grad()
def evaluate(model,d):
    p=model(d['p'],d['features']).numpy().astype(np.float64);v=study.metrics(d['y'].numpy(),p)
    rows=[dict(sample_id=str(sid),label_true=int(y),label_pred=int(row.argmax()),**{f'score{k}':float(row[k]) for k in range(10)}) for sid,y,row in zip(d['ids'],d['y'],p)]
    return v,rows

def fit(a):
    proof,meta,key,parent_fit=parent(a.parent);cfg=s.read(PROFILE);assert not a.out.exists() and .6<cfg['optical_alpha']<=1. and cfg['electronic_parameters']==490
    assert a.out.name==cfg['run'];torch.set_num_threads(4);torch.manual_seed(cfg['seed']);src=dict(meta['sources'])
    for name in ('resisc10_small_electrical.py','resisc10_small_electrical_profile.json'):src[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();s.engine.verify_sources(src,commit)
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'claim.json',dict(pid=os.getpid(),started_unix=time.time(),sources=src,git_commit=commit,parent=str(a.parent),parent_audit_sha256=s.sha(a.parent/'cpu_report/audit.json'),parent_checkpoint_sha256=parent_fit['checkpoint_sha256'],config=cfg,automatic_retry=False),exclusive=True)
    d=data(a.parent,key);n=len(d['train']['y']);assert n==5600 and len(d['val']['y'])==700
    model=SmallElectrical(cfg['optical_alpha']);ema=copy.deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0.);history=[];best=float('inf');significant=float('inf');stale=0;started=time.monotonic();steps=0;reason='maximum_epochs'
    for epoch in range(1,cfg['maximum_epochs']+1):
        if time.monotonic()-started>=cfg['maximum_seconds']:reason='time_budget';break
        order=orders(epoch,n,cfg['seed']);first_norm=None
        for idx in np.array_split(order,math.ceil(n/cfg['batch_size'])):
            opt.zero_grad(set_to_none=True);p=model(d['train']['p'][idx],d['train']['features'][idx]);loss=F.nll_loss(p.clamp_min(1e-12).log(),d['train']['y'][idx]);assert torch.isfinite(loss);loss.backward()
            norms={name:float(v.grad.norm()) for name,v in model.named_parameters()};assert all(np.isfinite(v) and v>0 for v in norms.values())
            if first_norm is None:first_norm=norms
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,1-cfg['ema_decay'])
        vm,rows=evaluate(ema,d['val']);h=dict(epoch=epoch,val=vm,order_sha256=identity(order),optimizer_steps=steps,gradient_norms=first_norm,seconds=time.monotonic()-started);history.append(h)
        if vm['balanced_nll']<best:
            best=vm['balanced_nll'];s.engine.save_pt(a.out/'best_checkpoint.pt',dict(model=ema.state_dict(),epoch=epoch,config=cfg,sources=src,parent_checkpoint_sha256=parent_fit['checkpoint_sha256'],validation=vm));s.engine.csvwrite(a.out/'val_predictions.csv',rows)
        if vm['balanced_nll']<significant-cfg['early_min_delta']:significant=vm['balanced_nll'];stale=0
        else:stale+=1
        s.save(a.out/'history.json',history);s.engine.save_pt(a.out/'last_checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),epoch=epoch,config=cfg,sources=src,optimizer_steps=steps))
        s.save(a.out/'status.json',dict(state='head_training',epoch=epoch,val_accuracy=vm['accuracy'],optical_alpha=cfg['optical_alpha'],electronic_parameters=490,test_evaluations=0))
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience']:reason='validation_plateau';break
    ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(ck['model']);tm,rows=evaluate(ema,d['train']);s.engine.csvwrite(a.out/'train_predictions.csv',rows)
    assert parent_fit['checkpoint_sha256']==s.sha(a.parent/key/'best_checkpoint.pt')
    f=dict(passed=True,kind='hybrid_optical_fixed_plus_tiny_electrical',config=cfg,sources=src,git_commit=commit,parent=str(a.parent),parent_key=key,parent_audit_sha256=s.sha(a.parent/'cpu_report/audit.json'),parent_checkpoint_sha256=parent_fit['checkpoint_sha256'],parent_selected_epoch=parent_fit['selected_epoch'],optical_alpha=cfg['optical_alpha'],electrical_weight=1-cfg['optical_alpha'],electronic_parameters=490,optical_parameters=908552,optical_weights_frozen=True,optical_model_constructed=0,optical_forward_calls=0,electrical_model_constructed=1,cpu_only=True,epochs_completed=len(history),selected_epoch=ck['epoch'],stop_reason=reason,optimizer_steps=steps,metrics=dict(train=tm,val=ck['validation']),checkpoint_sha256=s.sha(a.out/'best_checkpoint.pt'),history_sha256=s.sha(a.out/'history.json'),test_arrays_read=0,test_evaluations=0,selection='first minimum validation EMA balancedNLL',target80_achieved=ck['validation']['accuracy']>=.80)
    s.save(a.out/'fit_result.json',f,exclusive=True);s.save(a.out/'status.json',dict(state='head_fit_complete',test_evaluations=0))

def audit(a):
    assert not a.out.exists();f=s.read(a.base/'fit_result.json');cfg=s.read(PROFILE);claim=s.read(a.base/'claim.json');assert exited(claim['pid']) and f['config']==cfg and .6<f['optical_alpha']<=1 and f['electronic_parameters']==490
    base=Path(f['parent']);proof,meta,key,parent_fit=parent(base);assert key==f['parent_key'] and f['parent_audit_sha256']==s.sha(base/'cpu_report/audit.json') and parent_fit['checkpoint_sha256']==f['parent_checkpoint_sha256'];s.engine.verify_sources(f['sources'],f['git_commit'])
    d=data(base,key);h=s.read(a.base/'history.json');ck=torch.load(a.base/'best_checkpoint.pt',map_location='cpu',weights_only=False);last=torch.load(a.base/'last_checkpoint.pt',map_location='cpu',weights_only=False)
    assert f['checkpoint_sha256']==s.sha(a.base/'best_checkpoint.pt') and f['history_sha256']==s.sha(a.base/'history.json') and f['selected_epoch']==ck['epoch']==min(h,key=lambda x:x['val']['balanced_nll'])['epoch']
    assert len(h)==f['epochs_completed']<=120 and last['epoch']==len(h) and f['optimizer_steps']==350*len(h)==last['optimizer_steps']
    assert {int(x['step']) for x in last['optimizer']['state'].values()}=={350*len(h)} and sum(v.numel() for v in ck['model'].values())==490
    significant=float('inf');stale=0;prior=0.
    for epoch,row in enumerate(h,1):
        assert row['epoch']==epoch and row['order_sha256']==identity(orders(epoch,5600,cfg['seed'])) and row['optimizer_steps']==350*epoch and prior<7200
        assert all(np.isfinite(v) and v>0 for v in row['gradient_norms'].values()) and set(row['gradient_norms'])=={'linear.weight','linear.bias'}
        if row['val']['balanced_nll']<significant-cfg['early_min_delta']:significant=row['val']['balanced_nll'];stale=0
        else:stale+=1
        if epoch<len(h):assert not (epoch>=cfg['minimum_epochs'] and stale>=cfg['early_patience'])
        prior=row['seconds']
    w=ck['model']['linear.weight'].numpy().astype(np.float64);bias=ck['model']['linear.bias'].numpy().astype(np.float64);assert w.shape==(10,48) and bias.shape==(10,)
    errors={}
    for role in ('train','val'):
        x=independent_features(d[role]['images']);assert np.max(np.abs(x-d[role]['features'].numpy()))<2e-6
        z=x@w.T+bias;exp=np.exp(z-z.max(1,keepdims=True));electric=exp/exp.sum(1,keepdims=True);p=cfg['optical_alpha']*d[role]['p'].numpy()+(1-cfg['optical_alpha'])*electric
        rows=list(csv.DictReader((a.base/(role+'_predictions.csv')).open()));assert [v['sample_id'] for v in rows]==d[role]['ids'].tolist() and np.array_equal([int(v['label_true']) for v in rows],d[role]['y'].numpy())
        saved=np.asarray([[float(v['score'+str(k)]) for k in range(10)] for v in rows]);error=float(np.abs(p-saved).max());assert error<=1e-5 and np.array_equal(saved.argmax(1),[int(v['label_pred']) for v in rows])
        metric=study.metrics(d[role]['y'].numpy(),saved);assert metric==f['metrics'][role];errors[role]=error
    a.out.mkdir(parents=True,exist_ok=False);s.save(a.out/'audit.json',dict(passed=True,kind=f['kind'],parent_audit_sha256=f['parent_audit_sha256'],fit_result_sha256=s.sha(a.base/'fit_result.json'),sources=f['sources'],fixed_optical_alpha=cfg['optical_alpha'],alpha_strictly_greater_than_point6=True,electronic_parameters=490,optical_weights_frozen=True,independent_numpy_probability_errors=errors,all_saved_ID_label_CM_NLL_verified=True,epochs=len(h),actual_Adam_steps=350*len(h),target80_achieved=f['target80_achieved'],optical_model_constructed=0,optical_forward_calls=0,test_arrays_read=0,test_evaluations=0,pure_optical=False),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','audit'],required=True);p.add_argument('--parent',type=Path);p.add_argument('--base',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        if a.out.exists() and not (a.out/'failure.json').exists():s.save(a.out/'failure.json',dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
