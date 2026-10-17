"""CPU TRAIN/VAL-only teacher preparation while paired50 uses four GPUs."""
import argparse,copy,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import caltech20_rich_pair50_alpha as student
from audit_caltech20_equal60_logrepair import metrics,agree
HERE,ROOT,s,engine=student.HERE,student.ROOT,student.s,student.engine
FILES=('caltech20_rich_pair50_teacher.py','caltech20_rich_pair50_teacher_profile.json','CALTECH20_RICH_PAIR50_TEACHER_PROTOCOL_20261017.md')
CPU=HERE/'runs/smoke/caltech20_rich_pair50_teacher_CPUproof_20261017'
OUT=HERE/'runs/smoke/caltech20_rich_pair50_resnet18_TRAIN_VAL_teacher_20261017'
slots,manifest=student.slots,student.manifest

def config():
    cfg=s.read(HERE/FILES[1]);cfg.update(classes=student.config()['classes'],cache_sha256=student.config()['cache_sha256'],manifest_sha256=student.config()['manifest_sha256'],data=student.config()['data']);assert cfg['teacher_device']=='cpu' and cfg['teacher_epochs']==100 and cfg['TEST_images_read']==0;return cfg
def sources():
    src=student.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src
def identity():
    z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def cpu():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not CPU.exists();slots();claim(CPU.with_name(CPU.name+'.claim.json'),'own_rich20_teacher_analytic_CPU');cfg=config();assert s.sha(cfg['teacher_checkpoint'])==cfg['teacher_checkpoint_sha256']
    rng=np.random.default_rng(17);x=rng.normal(size=(6,4));w=rng.normal(size=(3,4));y=np.arange(6)%3;xt=torch.tensor(x,dtype=torch.float64);wt=torch.tensor(w,dtype=torch.float64,requires_grad=True)
    value=torch.nn.functional.cross_entropy(xt@wt.T,torch.tensor(y));g=torch.autograd.grad(value,wt)[0].numpy();z=x@w.T;p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);p[np.arange(6),y]-=1;err=float(np.abs(g-p.T@x/6).max());assert err<1e-12
    CPU.mkdir();s.save(CPU/'checks.json',dict(identity(),passed=True,analytic_CE_gradient_maxabs=err,real_models=0,real_forward_calls=0,real_images_read=0,TEST_images_read=0,GPU_children_added=0,official_checkpoint_sha256=cfg['teacher_checkpoint_sha256']),exclusive=True)

def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();engine.verify_sources(proof['sources'],proof['git_commit']);slots();claim(OUT.with_name(OUT.name+'.claim.json'),'CPU_pretrained_TRAIN_VAL_only_no_student_modification');cfg=config();assert s.sha(cfg['teacher_checkpoint'])==cfg['teacher_checkpoint_sha256'];torch.set_num_threads(4);engine.seed(17)
    from torchvision.models import resnet18,ResNet18_Weights
    OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),pid=os.getpid(),config=cfg,device='cpu',GPU_children_added=0,TEST_images_read=0),exclusive=True)
    model=resnet18(weights=None);model.load_state_dict(torch.load(cfg['teacher_checkpoint'],map_location='cpu',weights_only=True),strict=True);model.fc=nn.Identity();model.eval().requires_grad_(False);before=s.digest(model.state_dict());transform=ResNet18_Weights.IMAGENET1K_V1.transforms();data={r:student.load(r) for r in ('train','val')};features={}
    for role,d in data.items():
        parts=[]
        with torch.no_grad():
            for off in range(0,len(d[1]),cfg['teacher_feature_batch']):
                rgb=d[0][off:off+cfg['teacher_feature_batch']];f=model(transform(rgb));assert f.shape==(len(rgb),512) and torch.isfinite(f).all();parts.append(f.numpy())
                s.save(OUT/'status.json',dict(state='CPU_feature_extraction',role=role,completed=min(off+len(rgb),len(d[1])),total=len(d[1]),GPU_children_added=0,TEST_images_read=0))
        features[role]=np.concatenate(parts).astype(np.float32);student.save_np(OUT/(role+'_features.npz'),dict(features=features[role],labels=d[1].numpy(),ids=d[2]));s.save(OUT/(role+'_input_identity.json'),dict(input_array_sha256=s.digest(d[0].numpy()),count=len(d[1]),role=role),exclusive=True)
    after=s.digest(model.state_dict());assert before==after and all(p.grad is None and not p.requires_grad for p in model.parameters());del model
    mean=features['train'].astype(np.float64).mean(0);std=features['train'].astype(np.float64).std(0).clip(1e-3);x={r:torch.tensor(((features[r]-mean)/std).astype(np.float32)) for r in data};engine.seed(17);head=nn.Linear(512,20);optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['teacher_learning_rate'],weight_decay=cfg['teacher_weight_decay']);history=[];best=float('inf')
    for epoch in range(1,101):
        optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(head(x['train']),data['train'][1]);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():met={r:student.engine.metrics(data[r][1].numpy(),torch.softmax(head(x[r]),1).numpy().astype(np.float64)) for r in data}
        history.append(dict(epoch=epoch,train=met['train'],val=met['val'],loss=float(loss.detach())))
        if met['val']['nll']<best:best=met['val']['nll'];selected=copy.deepcopy(head.state_dict());selected_epoch=epoch;selected_metrics=met
    head.load_state_dict(selected);engine.save_pt(OUT/'selected_head.pt',dict(model=selected,mean=mean,std=std,epoch=selected_epoch,metrics=selected_metrics,config=cfg,sources=sources()))
    for role in data:
        with torch.no_grad():p=torch.softmax(head(x[role]),1).numpy().astype(np.float64)
        student.save_np(OUT/('selected_'+role+'.npz'),dict(probabilities=p,labels=data[role][1].numpy(),ids=data[role][2]))
    s.save(OUT/'history.json',history,exclusive=True);s.save(OUT/'fit_result.json',dict(passed=True,actual_epochs=100,selected_epoch=selected_epoch,metrics=selected_metrics,backbone_before=before,backbone_after=after,backbone_optimizer_steps=0,new_forwarded_images={r:len(data[r][1]) for r in data},reused_features=0,linear_optimizer_steps=100,TEST_images_read=0,GPU_children_added=0,teacher_only=True,student_modified=False,teacher_checkpoint_sha256=cfg['teacher_checkpoint_sha256'],head_sha256=s.sha(OUT/'selected_head.pt'),feature_sha256={r:s.sha(OUT/(r+'_features.npz')) for r in data}),exclusive=True)

def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'science').exists();claim(OUT/'audit_claim.json','independent_CPU_teacher_saved_linear_math');torch.set_num_threads(4)
    meta=s.read(OUT/'metadata.json');f=s.read(OUT/'fit_result.json');ex=s.read(OUT/'producer_exit.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid'])).exists();assert meta['sources']==sources() and meta['config']==config();engine.verify_sources(meta['sources'],meta['git_commit']);cfg=config();assert s.sha(cfg['teacher_checkpoint'])==f['teacher_checkpoint_sha256']==cfg['teacher_checkpoint_sha256']
    assert f['backbone_before']==f['backbone_after']=='de620ffcedf6c2818d8b6575ad3b3fc89a41351a1588ef8c0d904defc28ebf8e' and f['backbone_optimizer_steps']==f['TEST_images_read']==f['GPU_children_added']==0 and f['new_forwarded_images']==dict(train=3141,val=383) and f['reused_features']==0 and not f['student_modified']
    state=torch.load(OUT/'selected_head.pt',map_location='cpu',weights_only=False);assert s.sha(OUT/'selected_head.pt')==f['head_sha256'] and state['sources']==sources() and state['config']==cfg;hist=s.read(OUT/'history.json');assert len(hist)==100 and [v['epoch'] for v in hist]==list(range(1,101));chosen=min(hist,key=lambda v:v['val']['nll']);assert chosen['epoch']==state['epoch']==f['selected_epoch'];arrays={r:np.load(OUT/(r+'_features.npz'),allow_pickle=False) for r in ('train','val')}
    assert np.array_equal(state['mean'],arrays['train']['features'].astype(np.float64).mean(0)) and np.array_equal(state['std'],arrays['train']['features'].astype(np.float64).std(0).clip(1e-3));checks={}
    for role,z in arrays.items():
        assert f['feature_sha256'][role]==s.sha(OUT/(role+'_features.npz')) and z['ids'].tolist()==manifest()['split_ids'][role] and np.bincount(z['labels'],minlength=20).tolist()==manifest()['supports'][role]
        d=student.load(role);assert s.read(OUT/(role+'_input_identity.json'))['input_array_sha256']==s.digest(d[0].numpy()) and np.array_equal(z['labels'],d[1].numpy())
        x=((z['features']-state['mean'])/state['std']).astype(np.float32).astype(np.float64);logits=x@state['model']['weight'].double().numpy().T+state['model']['bias'].double().numpy();p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        with np.load(OUT/('selected_'+role+'.npz'),allow_pickle=False) as saved:
            assert np.array_equal(saved['ids'],z['ids']) and np.array_equal(saved['labels'],z['labels']);err=float(np.abs(p-saved['probabilities']).max());assert err<3e-6;actual=metrics(saved['labels'],saved['probabilities'])
        agree(actual,f['metrics'][role]);agree(actual,chosen[role]);agree(actual,state['metrics'][role]);checks[role]=dict(linear_numpy_maxabs=err,metrics=actual,prediction_sha256=s.sha(OUT/('selected_'+role+'.npz')));z.close()
    (OUT/'science').mkdir();s.save(OUT/'science/audit.json',dict(identity(),passed=True,checks=checks,model_objects=0,forward_calls=0,TEST_images_read=0,teacher_only=True,GPU_children_added=0,pretraining='public official ResNet18 training-only teacher, not deployed student architecture'),exclusive=True)

def supervise():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists();claim(OUT.with_name(OUT.name+'.supervisor_claim.json'),'single_CPU_teacher_supervisor_no_retry')
    with OUT.with_name(OUT.name+'.producer.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','run'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT);s.save(OUT.with_name(OUT.name+'.started.json'),dict(supervisor_pid=os.getpid(),producer_pid=p.pid),exclusive=True);rc=p.wait()
    assert OUT.exists();s.save(OUT/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0
    with (OUT/'audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--phase','audit'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(OUT/'science/audit.json')['passed'];s.save(OUT/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/'science/audit.json')),exclusive=True);s.save(OUT/'status.json',dict(state='teacher_TRAIN_VAL_complete_no_TEST',student_modified=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','run','audit','supervise'],required=True);a=p.parse_args()
    try:globals()[a.phase]()
    except BaseException as e:
        root=CPU if a.phase=='cpu' else OUT;mark=root/'failure.json' if root.exists() else root.with_name(root.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False,phase=a.phase),exclusive=True)
        raise
