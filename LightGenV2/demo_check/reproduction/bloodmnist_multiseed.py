"""Predeclared 3-seed comparison with an illuminated-aperture D2NN control.

Reuse the exact earlier trainer. A new forward geometry alone warrants a wrapper:
each RGB/mean tile is enlarged separately, then total incident power is restored.
"""
import argparse,concurrent.futures,copy,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import bloodmnist_experiment as b
from bloodmnist_experiment import r,torch,np,F
from optical_reference.optics import PhaseLayer

BASE=Path(__file__).with_name('bloodmnist_profile.json')
_build=b.build
_forward=b.forward

def build(arch,depth,cfg):
    m=_build('d2nn' if arch=='d2nn_wide' else arch,depth,cfg)
    if arch=='d2nn_wide':
        m.input_size=m.phases[0].phase.raw_phase.shape[-1]
        m.pad=(m.canvas-m.input_size)//2
    return m

def enlarge(x,side):
    assert x.shape[1:]==(1,100,100) and side%2==0
    rows=[]
    for i in range(2):
        rows.append(torch.cat([F.interpolate(x[:,:,i*50:(i+1)*50,j*50:(j+1)*50],size=(side//2,side//2),mode='bilinear',align_corners=False,antialias=True) for j in range(2)],3))
    y=torch.cat(rows,2)
    scale=(x.square().sum((1,2,3),keepdim=True)/y.square().sum((1,2,3),keepdim=True).clamp_min(1e-12)).sqrt()
    return y*scale

def forward(m,x,arch):
    return _forward(m,enlarge(x,m.input_size),'d2nn') if arch=='d2nn_wide' else _forward(m,x,arch)

b.build=build
b.forward=forward

def sources():
    return dict(b.sources(BASE),**{str(Path(__file__).relative_to(b.TASK)).replace('\\','/'):r.sha(__file__)})

def environment():
    import sklearn
    return dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=np.__version__,sklearn=sklearn.__version__,gpu=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))

def pixel_audit(m,arch,data,val=None):
    """No optimizer: input-power maps and classification-only gradient support."""
    m.eval();modules={n:q for n,q in m.named_modules() if isinstance(q,PhaseLayer)};powers={};counts={};handles=[]
    def pre(name):
        def hook(module,args):
            x=args[0].detach().abs().square();assert x.ndim==3 and x.shape[-2:]==module.raw_phase.shape,(name,x.shape,module.raw_phase.shape)
            v=x.sum(0);powers[name]=powers.get(name,torch.zeros_like(v))+v;counts[name]=counts.get(name,0)+len(x)
        return hook
    for n,q in modules.items():handles.append(q.register_forward_pre_hook(pre(n)))
    labels=data[1].cpu().numpy();grad={n:torch.zeros_like(q.raw_phase) for n,q in modules.items()}
    try:
        for offset in [0,2]:
            idx=np.concatenate([np.flatnonzero(labels==k)[offset:offset+2] for k in range(8)])
            m.zero_grad(set_to_none=True);prob,_,_=forward(m,b.encode(data[0][idx]),arch)
            loss=-prob[torch.arange(len(idx)),data[1][idx]].clamp_min(1e-12).log().mean();loss.backward()
            for n,q in modules.items():
                assert q.raw_phase.grad is not None and torch.isfinite(q.raw_phase.grad).all()
                grad[n]+=q.raw_phase.grad.detach().abs()
        if val is not None:
            powers.clear();counts.clear();b.evaluate(m,val,arch,16)
    finally:
        for h in handles:h.remove()
    report={};maps={}
    for i,(n,q) in enumerate(modules.items()):
        pw=powers[n]/counts[n];g=grad[n];sat=q.raw_phase.detach().sigmoid()
        report[n]=dict(shape=list(pw.shape),pixels=pw.numel(),input_samples=counts[n],illuminated_exact_fraction=float((pw>0).float().mean()),illuminated_relative_1e6_fraction=float((pw>pw.max()*1e-6).float().mean()),classification_gradient_exact_fraction=float((g>0).float().mean()),classification_gradient_relative_1e6_fraction=float((g>g.max()*1e-6).float().mean()) if float(g.max())>0 else 0.,classification_gradient_norm=float(g.norm()),saturation_fraction=float(((sat<.01)|(sat>.99)).float().mean()),map_key=f'phase{i}')
        maps[f'phase{i}_power']=pw.cpu().numpy();maps[f'phase{i}_class_gradient']=g.cpu().numpy()
    m.zero_grad(set_to_none=True)
    return report,maps

def smoke(a):
    a.out.mkdir(parents=True,exist_ok=False);cfg=r.read(BASE);torch.set_num_threads(4);data=b.load_data(a.data,'train');results=[]
    r.save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources(),environment=environment(),data_sha256=r.sha(a.data),time=r.now()))
    for depth in [2,4,6]:
        reference=None
        for arch in ['moe','d2nn','d2nn_wide']:
            r.setseed(17);m=build(arch,depth,cfg);x=b.encode(data[0][:16]);p,c,_=forward(m,x,arch);assert p.shape==(16,8) and torch.isfinite(p).all()
            if arch=='d2nn':reference={n:q.detach().clone() for n,q in m.named_parameters()}
            power_error=None
            if arch=='d2nn_wide':
                assert all(torch.equal(q,reference[n]) for n,q in m.named_parameters());wide=enlarge(x,m.input_size)
                power_error=float(((wide.square().sum((1,2,3))-x.square().sum((1,2,3))).abs()/x.square().sum((1,2,3))).max());assert power_error<1e-6
            audit,maps=pixel_audit(m,arch,data);assert all(x['classification_gradient_norm']>0 for x in audit.values())
            if arch.startswith('d2nn'):
                first=audit['phases.0.phase'];expected=1 if arch=='d2nn_wide' else (100/m.input_size)**2
                if arch=='d2nn':expected=10000/m.phases[0].phase.raw_phase.numel()
                assert abs(first['illuminated_exact_fraction']-expected)<1e-6
            name=f'{arch}_L{depth}';r.save(a.out/(name+'.json'),audit);np.savez_compressed(a.out/(name+'_maps.npz'),**maps)
            results.append(dict(arch=arch,depth=depth,parameters=sum(q.numel() for q in m.parameters()),model_input_side=m.input_size if arch.startswith('d2nn') else 100,power_relative_error=power_error,phase_audit=audit));del m
            if arch=='d2nn_wide':reference=None
            torch.cuda.empty_cache()
    r.save(a.out/'smoke.json',results);r.save(a.out/'status.json',dict(state='complete',time=r.now()))

def train_one(a):
    a.out.mkdir(parents=True,exist_ok=False);cfg=r.read(BASE);torch.set_num_threads(4);r.setseed(a.seed);src=sources();data=b.load_data(a.data,'train');val=b.load_data(a.data,'val')
    r.save(a.out/'metadata.json',dict(command=sys.argv,config=cfg,arch=a.arch,depth=a.depth,seed=a.seed,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=src,environment=environment(),data_sha256=r.sha(a.data),time=r.now(),test_read=False))
    result=b.train(a.arch,a.depth,a.seed,data,val,cfg,a.out,src);r.save(a.out/'result.json',result);r.save(a.out/'status.json',dict(state='complete',time=r.now()))

def suite(a):
    assert len(a.gpus)<=2 and len(a.gpus)==len(set(a.gpus));a.out.mkdir(parents=True,exist_ok=False)
    old=r.read(a.reuse/'validation_results.json');assert r.read(a.reuse/'metadata.json')['config']==r.read(BASE);old_sources=r.read(a.reuse/'test_lock.json')['sources'];assert old_sources==b.sources(BASE)
    assert r.sha(a.data)==r.read(a.reuse/'metadata.json')['data_sha256'];reused=[]
    for x in old:
        if x['arch']=='cnn':continue
        folder=a.reuse/x['name'];assert r.sha(folder/'best_checkpoint.pt')==x['checkpoint_sha256'];reused.append(dict(result=x,folder=str(folder.resolve()),reused=True))
    jobs=[dict(arch=arch,depth=d,seed=s) for s in [17,27,37] for d in [2,4,6] for arch in ['moe','d2nn','d2nn_wide'] if s!=17 or arch=='d2nn_wide']
    # Longest jobs first, without inspecting metrics. Two workers share the queue.
    jobs.sort(key=lambda j:(j['arch']=='moe',j['depth']),reverse=True)
    for j in jobs:j['name']=f"{j['arch']}_L{j['depth']}_seed{j['seed']}"
    src=sources();r.save(a.out/'metadata.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=src,config=r.read(BASE),data_sha256=r.sha(a.data),time=r.now(),seeds=[17,27,37],depths=[2,4,6],architectures=['moe','d2nn','d2nn_wide'],gpus=a.gpus,jobs=jobs,reused=reused,test_previously_observed=True,scope='Geometry-driven control; old seed17 test previously observed. No new tuning or test-based selection.'))
    (a.out/'jobs').mkdir();(a.out/'logs').mkdir();import queue
    q=queue.Queue()
    for j in jobs:q.put(j)
    def worker(gpu):
        done=[]
        while True:
            try:j=q.get_nowait()
            except queue.Empty:return done
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);dest=a.out/'jobs'/j['name'];cmd=[sys.executable,'-u',__file__,'--phase','train-one','--data',str(a.data),'--out',str(dest),'--arch',j['arch'],'--depth',str(j['depth']),'--seed',str(j['seed'])]
            r.save(a.out/f'worker_gpu{gpu}.json',dict(state='training',job=j,command=cmd,time=r.now()))
            with (a.out/'logs'/(j['name']+'.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=r.read(dest/'result.json');done.append(dict(result=result,folder=str((dest/j['name']).resolve()),reused=False))
            r.save(a.out/f'worker_gpu{gpu}.json',dict(state='job_complete',job=j,time=r.now()));q.task_done()
    r.save(a.out/'status.json',dict(state='training',time=r.now()))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as ex:
            futures=[ex.submit(worker,g) for g in a.gpus];entries=reused+[x for f in futures for x in f.result()]
        assert len(entries)==27
        # Selection was made by the unchanged per-model validation criterion.
        r.save(a.out/'selection_lock.json',dict(entries=entries,sources=src,data_sha256=r.sha(a.data),time=r.now(),selection='minimum validation balanced NLL with min_delta=0.0005'))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpus[0]);r.save(a.out/'status.json',dict(state='locked_evaluation',time=r.now()))
        subprocess.run([sys.executable,'-u',__file__,'--phase','evaluate','--data',str(a.data),'--out',str(a.out)],env=env,check=True)
        r.save(a.out/'status.json',dict(state='complete',time=r.now()))
    except Exception:
        r.save(a.out/'status.json',dict(state='failed',traceback=traceback.format_exc(),time=r.now()));raise

def evaluate(a):
    lock=r.read(a.out/'selection_lock.json');assert lock['sources']==sources() and lock['data_sha256']==r.sha(a.data);cfg=r.read(BASE);torch.set_num_threads(4);val=b.load_data(a.data,'val');data=b.load_data(a.data,'train')
    entries=sorted(lock['entries'],key=lambda x:(x['result']['seed'],x['result']['depth'],x['result']['arch']))
    dest=a.out/'evaluation';dest.mkdir(exist_ok=False);results=[]
    # Every validation score is exactly replayed before this new evaluation opens test.
    for entry in entries:
        x=entry['result'];folder=Path(entry['folder']);assert r.sha(folder/'best_checkpoint.pt')==x['checkpoint_sha256'];r.setseed(x['seed']);m=build(x['arch'],x['depth'],cfg);initial={n:p.detach().clone() for n,p in m.named_parameters()};ck=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);vm,_=b.evaluate(m,val,x['arch'],cfg['batch_size']);assert vm==x['metrics']['val'],x['name']
        audit,maps=pixel_audit(m,x['arch'],data,val)
        for n,q in m.named_modules():
            if isinstance(q,PhaseLayer):
                change=q.raw_phase.detach()-initial[n+'.raw_phase'];audit[n]['changed_fraction']=float((change.abs()>1e-7).float().mean());audit[n]['rms_parameter_update']=float(change.square().mean().sqrt())
        out=dest/x['name'];out.mkdir();r.save(out/'phase_audit.json',audit);np.savez_compressed(out/'phase_maps.npz',**maps);del m,ck,initial;torch.cuda.empty_cache()
    test=b.load_data(a.data,'test')
    with np.load(a.data,allow_pickle=False) as z:
        seen={hashlib.sha256(img.tobytes()).hexdigest() for split in ['train','val'] for img in z[split+'_images']};clean=[]
        for i,img in enumerate(z['test_images']):
            h=hashlib.sha256(img.tobytes()).hexdigest()
            if h not in seen:clean.append(i);seen.add(h)
    r.save(a.out/'test_clean_indices.json',clean)
    for entry in entries:
        x=entry['result'];m=build(x['arch'],x['depth'],cfg);ck=torch.load(Path(entry['folder'])/'best_checkpoint.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);tm,rows=b.evaluate(m,test,x['arch'],cfg['batch_size']);out=dest/x['name'];r.csvwrite(out/'test_predictions.csv',rows);cr=[rows[i] for i in clean];r.csvwrite(out/'test_clean_predictions.csv',cr);cm=b.metrics(np.array([v['label_true'] for v in cr]),np.array([[v[f'score{k}'] for k in range(8)] for v in cr]));result=dict(entry, test=tm,clean_test=cm,validation_replayed=True);results.append(result);r.save(out/'result.json',result);del m,ck;torch.cuda.empty_cache()
    r.save(a.out/'results.json',results);r.save(a.out/'evaluation_identity.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),environment=environment(),sources=sources(),selection_lock_sha256=r.sha(a.out/'selection_lock.json'),time=r.now()));print(json.dumps(dict(evaluated=len(results),clean_test_n=len(clean))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','suite','train-one','evaluate'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reuse',type=Path);p.add_argument('--gpus',type=int,nargs='+',default=[0]);p.add_argument('--arch',choices=['moe','d2nn','d2nn_wide']);p.add_argument('--depth',type=int);p.add_argument('--seed',type=int);a=p.parse_args();assert hashlib.md5(a.data.read_bytes()).hexdigest()==r.read(BASE)['md5']
    {'smoke':smoke,'suite':suite,'train-one':train_one,'evaluate':evaluate}[a.phase](a)

if __name__=='__main__':main()
