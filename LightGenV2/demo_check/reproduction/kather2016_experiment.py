"""Equal-budget validation optimization then locked 3-seed OEO/geometry comparison."""
import argparse,copy,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import bloodmnist_multiseed as m
from bloodmnist_multiseed import b,r,torch,np

BASE=Path(__file__).with_name('bloodmnist_profile.json')
PROFILES=Path(__file__).with_name('kather2016_profiles.json')
SPEC=r.read(PROFILES)

def config(name):
    c=r.read(BASE);c.update(SPEC['candidates'][name]);c.pop('url');c.pop('md5');c.update(dataset=SPEC['dataset'],classes=SPEC['classes'],scope=SPEC['scope'],encoding=SPEC['encoding'],deduplication=SPEC['data_split']);return c

def sources():
    d=b.sources(BASE)
    for p in [Path(m.__file__),Path(__file__),PROFILES,Path(__file__).with_name('prepare_kather2016.py')]:d[str(p.relative_to(b.TASK)).replace('\\','/')]=r.sha(p)
    return d

def load_data(path,split):
    manifest=r.read(path.parent/'data_manifest.json');assert r.sha(path)==manifest['cache_sha256']
    with np.load(path,allow_pickle=False) as z:x=z[split+'_images'].copy();y=z[split+'_labels'].copy();ids=z[split+'_ids'].copy()
    assert x.dtype==np.uint8 and x.shape==(len(y),150,150,3);assert np.bincount(y,minlength=8).tolist()==manifest['supports'][split];assert ids.tolist()==manifest['split_ids'][split]
    return torch.from_numpy(x.transpose(0,3,1,2).copy()).float().cuda()/255,torch.from_numpy(y).long().cuda(),ids

b.load_data=load_data
m.sources=sources

def train_one(a):
    a.out.mkdir(parents=True,exist_ok=False);cfg=config(a.candidate);torch.set_num_threads(4);r.setseed(a.seed);src=sources();data=load_data(a.data,'train');val=load_data(a.data,'val')
    r.save(a.out/'metadata.json',dict(command=sys.argv,config=cfg,arch=a.arch,depth=a.depth,seed=a.seed,candidate=a.candidate,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=src,environment=m.environment(),data_sha256=r.sha(a.data),time=r.now(),test_read=False))
    result=b.train(a.arch,a.depth,a.seed,data,val,cfg,a.out,src);r.save(a.out/'result.json',result);r.save(a.out/'status.json',dict(state='complete',time=r.now()))

def job(candidate,arch,depth,seed):return dict(candidate=candidate,arch=arch,depth=depth,seed=seed,name=f'{candidate}_{arch}_L{depth}_seed{seed}')

def worker(a):
    jobs=r.read(a.out/'jobs.json')
    for j in jobs:
        claim=a.out/'claims'/j['name']
        try:claim.mkdir()
        except FileExistsError:continue
        r.save(claim/'worker.json',dict(pid=os.getpid(),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),time=r.now()))
        cmd=[sys.executable,'-u',__file__,'--phase','train-one','--data',str(a.data),'--out',str(a.out/'jobs'/j['name']),'--candidate',j['candidate'],'--arch',j['arch'],'--depth',str(j['depth']),'--seed',str(j['seed'])]
        r.save(a.out/f"worker_gpu{os.environ['CUDA_VISIBLE_DEVICES']}.json",dict(state='training',job=j,time=r.now()))
        try:
            with (a.out/'logs'/(j['name']+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            r.save(claim/'status.json',dict(state='complete',time=r.now()))
        except Exception:r.save(claim/'status.json',dict(state='failed',traceback=traceback.format_exc(),time=r.now()));raise

def run_jobs(a,jobs):
    jobs.sort(key=lambda j:(j['arch'].startswith('moe'),j['depth']),reverse=True)
    for folder in ['jobs','claims','logs']:(a.out/folder).mkdir()
    r.save(a.out/'jobs.json',jobs);r.save(a.out/'gpu_schedule.json',a.gpus);running={};used=set()
    try:
        while True:
            schedule=r.read(a.out/'gpu_schedule.json');assert len(schedule)<=5 and len(schedule)==len(set(schedule))
            for gpu in schedule:
                if gpu in used:continue
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu);cmd=[sys.executable,'-u',__file__,'--phase','worker','--data',str(a.data),'--out',str(a.out)];log=(a.out/'logs'/f'worker_gpu{gpu}.log').open('w');proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[gpu]=(proc,log);used.add(gpu)
                r.save(a.out/f'process_gpu{gpu}.json',dict(pid=proc.pid,command=cmd,time=r.now()))
            for gpu,(proc,_) in running.items():
                if proc.poll() is not None:assert proc.returncode==0,(gpu,proc.returncode)
            completed=sum((a.out/'jobs'/j['name']/'result.json').exists() for j in jobs);r.save(a.out/'status.json',dict(state='training',completed=completed,total=len(jobs),gpus=list(used),time=r.now()))
            if completed==len(jobs) and all(p.poll() is not None for p,_ in running.values()):break
            time.sleep(5)
    finally:
        for proc,log in running.values():
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait()
            log.close()
    entries=[]
    for j in jobs:
        dest=a.out/'jobs'/j['name'];x=r.read(dest/'result.json');entries.append(dict(result=x,folder=str((dest/x['name']).resolve()),candidate=j['candidate'],reused=False))
    return entries

def coordinator(a):
    a.out.mkdir(parents=True,exist_ok=False);assert a.gpus and len(a.gpus)<=5
    metadata=dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources(),data_sha256=r.sha(a.data),data_manifest_sha256=r.sha(a.data.parent/'data_manifest.json'),specification=SPEC,time=r.now(),test_read=False)
    r.save(a.out/'metadata.json',metadata)
    try:
        if a.phase=='pilot':
            jobs=[job(c,arch,4,17) for c in SPEC['candidate_order'] for arch in SPEC['architectures']];entries=run_jobs(a,jobs)
            means={c:float(np.mean([e['result']['metrics']['val']['balanced_nll'] for e in entries if e['candidate']==c])) for c in SPEC['candidate_order']};chosen=min(SPEC['candidate_order'],key=lambda c:means[c]);r.save(a.out/'validation_results.json',entries)
            r.save(a.out/'candidate_selection.json',dict(chosen=chosen,mean_validation_balanced_nll=means,criterion=SPEC['selection'],entries=entries,sources=sources(),data_sha256=r.sha(a.data),time=r.now(),test_read=False));r.save(a.out/'status.json',dict(state='pilot_complete_test_not_read',chosen=chosen,time=r.now()));return
        selection=r.read(a.pilot/'candidate_selection.json');assert selection['sources']==sources() and selection['data_sha256']==r.sha(a.data);chosen=selection['chosen'];reused=[dict(e,reused=True) for e in selection['entries'] if e['candidate']==chosen]
        jobs=[job(chosen,arch,d,s) for s in SPEC['seeds'] for d in SPEC['depths'] for arch in SPEC['architectures'] if not (s==17 and d==4)];entries=reused+run_jobs(a,jobs);assert len(entries)==36
        r.save(a.out/'selection_lock.json',dict(entries=entries,config=config(chosen),candidate=chosen,pilot_selection_sha256=r.sha(a.pilot/'candidate_selection.json'),sources=sources(),data_sha256=r.sha(a.data),time=r.now()))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpus[0]);r.save(a.out/'status.json',dict(state='locked_evaluation',time=r.now()))
        subprocess.run([sys.executable,'-u',__file__,'--phase','evaluate','--data',str(a.data),'--out',str(a.out)],env=env,check=True);r.save(a.out/'status.json',dict(state='complete',time=r.now()))
    except Exception:r.save(a.out/'status.json',dict(state='failed',traceback=traceback.format_exc(),time=r.now()));raise

def evaluate(a):
    import csv
    lock=r.read(a.out/'selection_lock.json');assert lock['sources']==sources() and lock['data_sha256']==r.sha(a.data);cfg=lock['config'];torch.set_num_threads(4);val=load_data(a.data,'val');data=load_data(a.data,'train');dest=a.out/'evaluation';dest.mkdir(exist_ok=False);results=[];replays=[]
    for entry in lock['entries']:
        x=entry['result'];folder=Path(entry['folder']);assert r.sha(folder/'best_checkpoint.pt')==x['checkpoint_sha256'];r.setseed(x['seed']);model=m.build(x['arch'],x['depth'],cfg);initial={n:p.detach().clone() for n,p in model.named_parameters()};ck=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);vm,rows=b.evaluate(model,val,x['arch'],cfg['batch_size'])
        with (folder/'val_predictions.csv').open() as f:old=list(csv.DictReader(f))
        assert [z['sample_id'] for z in rows]==[z['sample_id'] for z in old]
        assert [z['label_pred'] for z in rows]==[int(z['label_pred']) for z in old]
        error=max(abs(z[f'score{k}']-float(o[f'score{k}'])) for z,o in zip(rows,old) for k in range(8));assert error<=2e-6,(x['name'],error)
        for k in ['accuracy','balanced_accuracy','macro_f1','macro_ovr_auroc','balanced_nll']:assert abs(vm[k]-x['metrics']['val'][k])<=2e-6,(x['name'],k)
        audit,maps=m.pixel_audit(model,x['arch'],data,val)
        for n,q in model.named_modules():
            if isinstance(q,m.PhaseLayer):
                delta=q.raw_phase.detach()-initial[n+'.raw_phase'];audit[n].update(changed_fraction=float((delta.abs()>1e-7).float().mean()),rms_parameter_update=float(delta.square().mean().sqrt()))
        out=dest/x['name'];out.mkdir();r.save(out/'phase_audit.json',audit);np.savez_compressed(out/'phase_maps.npz',**maps);replays.append(dict(model=x['name'],maximum_validation_score_error=error));del model,initial,ck;torch.cuda.empty_cache()
    r.save(a.out/'validation_replay.json',replays);test=load_data(a.data,'test')
    for entry in lock['entries']:
        x=entry['result'];model=m.build(x['arch'],x['depth'],cfg);ck=torch.load(Path(entry['folder'])/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);tm,rows=b.evaluate(model,test,x['arch'],cfg['batch_size']);out=dest/x['name'];r.csvwrite(out/'test_predictions.csv',rows);result=dict(entry,test=tm,validation_replayed=True);r.save(out/'result.json',result);results.append(result);del model,ck;torch.cuda.empty_cache()
    r.save(a.out/'results.json',results);r.save(a.out/'evaluation_identity.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),environment=m.environment(),sources=sources(),lock_sha256=r.sha(a.out/'selection_lock.json'),time=r.now()));print(json.dumps(dict(evaluated=len(results))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','pilot','suite','worker','train-one','evaluate'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',type=Path);p.add_argument('--gpus',type=int,nargs='+',default=[2]);p.add_argument('--candidate',choices=SPEC['candidate_order'],default='base');p.add_argument('--arch',choices=SPEC['architectures']);p.add_argument('--depth',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.phase=='smoke':m.smoke(a)
    elif a.phase in ['pilot','suite']:coordinator(a)
    else:{'worker':worker,'train-one':train_one,'evaluate':evaluate}[a.phase](a)
if __name__=='__main__':main()
