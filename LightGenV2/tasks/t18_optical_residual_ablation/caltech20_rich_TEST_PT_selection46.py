"""Human-requested direct TEST selection of all available retained/augmentation PTs."""
import argparse,os,sys,subprocess,copy,time
from functools import lru_cache
from pathlib import Path
import numpy as np,torch
import caltech20_rich_zoomout_snapshot_average46 as average
import caltech20_rich_currentKD46 as kd
import caltech20_rich_currentKD_strength46 as strength
import caltech20_rich_sameclass_patch46 as patch
import caltech20_rich_overallCE46 as ordinary
import caltech20_rich_appearance_mix46 as appearance
from audit_caltech20_rich_pair50_alpha import metrics,agree,verify_predictions
HERE,ROOT,s,base,engine,resources=average.HERE,average.ROOT,average.s,average.base,average.engine,average.resources
FILES=('caltech20_rich_TEST_PT_selection46.py','check_caltech20_rich_TEST_PT_selection46.py','audit_caltech20_rich_TEST_PT_selection46.py','CALTECH20_RICH_TEST_PT_SELECTION46_PROTOCOL_20261018.md')
CPU=HERE/'runs/smoke/caltech20_rich_TEST_PT_selection46_CPU_20261018';OUT=HERE/'runs/simulation/caltech20_rich20_direct_TEST_PT_selection46_20261018'
def config():return dict(depths=[4,6],selection='maximum_TEST_correct_then_earliest_in_bank',VALID_used_for_selection=False,TRAIN_used_for_selection=False,L2_FINAL_untouched=True,new_optimizer_steps=0,new_training_epochs=0,new_inference_parameters=0,TEST_informed_development=True,independent_generalization=False,max_owned_GPUs=5)
def sources():
    z=average.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def identity():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();z=sources();engine.verify_sources(z,commit);return dict(sources=z,git_commit=commit)
def claim(p,phase):s.save(p,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def read(p):return s.read(p) if p.exists() else None
def npz(path):return kd.load_predictions(path)
def completed(module,key):
    f=module.OUT/key;ex=s.read(module.OUT/(key+'.producer_exit.json'));rec=s.read(f/'audit_completed.json');meta=s.read(f/'metadata.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(f/'science/audit.json') and s.read(f/'science/audit.json')['passed'];assert meta['sources']==module.sources();engine.verify_sources(meta['sources'],meta['git_commit']);return f,rec
@lru_cache(None)
def bank():
    rows=[];states={};caches={};role_caches={};seen={}
    def add(d,tag,state,train,val,origin,test=None,roles=None):
        state={n:v.detach().cpu().clone() for n,v in state.items()};digest=s.digest(state);token=(d,digest)
        if token in seen:
            key=seen[token]
            if test is not None:caches[key]=test
            if roles is not None:role_caches[key]=roles
            return
        key=f'L{d}_{len(rows):03d}';seen[token]=key;states[key]=state;rows.append(dict(key=key,depth=d,tag=tag,model_digest=digest,train=train,val=val,origin=origin));assert sum(v.numel() for n,v in state.items() if n.startswith('linear.'))==5140
        if test is not None:caches[key]=test
        if roles is not None:role_caches[key]=roles
    for d in (4,6):
        q,pt,p=kd.incumbent(d);add(d,'earliest_retained_canonical',pt['model'],metrics(p['train']['labels'],p['train']['probabilities']),metrics(p['val']['labels'],p['val']['probabilities']),dict(path=q['path'],sha256=q['sha256'],state_key='model'),p['test'],p)
    for module in (kd,strength,patch,ordinary,appearance,average.train):
        for key,case in module.config()['cases'].items():
            d=case['depth'];assert d in (4,6);f,rec=completed(module,key);fit=s.read(f/'fit_result.json');path=f/'last.pt';assert fit['last_sha256']==s.sha(path);pt=torch.load(path,map_location='cpu',weights_only=False);h=s.read(f/'history.json')[-1];assert pt['tune_epoch']==5 and s.digest(pt['ema'])==h['ema_digest'];add(d,module.__name__+':'+key,pt['ema'],h['train'],h['val'],dict(path=str(path),sha256=s.sha(path),state_key='ema',science=rec['audit_sha256']),npz(f/'test_epoch05.npz'))
    for phase in ('fit','test'):
        rec=s.read(average.OUT/(phase+'_audit_completed.json'));assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(average.OUT/(phase+'_science/audit.json')) and s.read(average.OUT/(phase+'_science/audit.json'))['passed']
    for d in (4,6):
        f=average.OUT/f'L{d}';fit=s.read(f/'fit_result.json');p=torch.load(f/'candidate.pt',map_location='cpu',weights_only=False);assert fit['candidate_SHA256']==s.sha(f/'candidate.pt');roles={r:npz(f/('candidate_'+r+'.npz')) for r in ('train','val','test')};add(d,'source487_selected_available_average',p['model'],fit['metrics']['train'],fit['metrics']['val'],dict(path=str(f/'candidate.pt'),sha256=s.sha(f/'candidate.pt'),state_key='model'),roles['test'],roles)
        _,parents=average.bank(d)
        for item in parents[1]['all_available_candidates']:
            p=torch.load(item['path'],map_location='cpu',weights_only=False);assert item['sha256']==s.sha(Path(item['path']));add(d,'preserved_actual_local'+str(item['local_epoch']),p[item['state_key']],item['train'],item['val'],dict(path=item['path'],sha256=item['sha256'],state_key=item['state_key'],science=item['science_SHA256']))
        for i,fraction in enumerate(average.config()['mixture_fractions']):
            state=average.state(d,fraction);row=fit['records'][i];assert row['fraction']==fraction and row['model_digest']==s.digest(state);add(d,'already_VALID_evaluated_average_fraction'+str(fraction),state,None,row['val'],dict(source='source487_independent_weight_arithmetic',depth=d,fraction=fraction,fit_science=s.read(average.OUT/'fit_audit_completed.json')['audit_sha256']))
    for row in rows:row['cached_TEST']=row['key'] in caches;row['cached_TRAIN_VALID']=row['key'] in role_caches
    return rows,states,caches,role_caches
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['independent_TEST_ranking_and_earliest_ties'] and p['sources']==sources() and p['config']==config() and p['candidates']==bank()[0];identity();kd.old.controls()
def configure(gpu):
    assert gpu in resources.GPU.values() and os.environ['CUDA_VISIBLE_DEVICES']==gpu;s.gpu_budget.ALLOWED.update(resources.GPU.values());return s.gpu_budget.configure(torch,gpu,shared=True,memory_gib=2)
def initialize(depth,state):
    model=average.previous.physics.Alpha40MoE(depth,.3);model.load_state_dict(state);model.requires_grad_(False);return model.eval()
def evaluate(key,gpu):
    ready();row=next(q for q in bank()[0] if q['key']==key);assert not row['cached_TEST'];f=OUT/key;assert not f.exists();claim(f.with_name(f.name+'.claim.json'),'single_new_digest_TEST');f.mkdir();policy=configure(gpu);torch.set_num_threads(4);state=bank()[1][key];model=initialize(row['depth'],state).cuda();reference=average.train.Alpha40MoE(row['depth'],.3).cuda().eval();reference.load_state_dict(state);data=base.load('train');idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);pe=ce=0.
    with torch.no_grad():
        for off in range(0,20,8):x=base.encode(data[0][idx[off:off+8]].cuda(),None,100);p,c=model(x);q,d=reference(x);pe=max(pe,float((p-q).abs().max()));ce=max(ce,float((c-d).abs().max()))
    assert pe<=3e-6 and ce<=3e-6;del reference,data;torch.cuda.empty_cache();before=s.digest(model.state_dict());metric,p=base.evaluate(model,base.load('test'));assert before==s.digest(model.state_dict())==row['model_digest'];base.save_np(f/'test_predictions.npz',p);s.save(f/'result.json',dict(identity(),passed=True,candidate=row,metrics=metric,new_TEST_models=1,cached=False,policy=policy,numeric_gate=dict(all20_TRAIN=True,probability_maxabs=pe,capture_maxabs=ce),optimizer_steps=0,VALID_images_read=0),exclusive=True)
def selected(depth,gpu):
    ready();row=s.read(OUT/'selection.json')[str(depth)];key=row['key'];f=OUT/f'L{depth}_selected';assert not f.exists();claim(f.with_name(f.name+'.claim.json'),'winner_metrics_no_TEST');f.mkdir();state=bank()[1][key];configure(gpu);torch.set_num_threads(4)
    if key in bank()[3]:p=bank()[3][key];results={r:metrics(p[r]['labels'],p[r]['probabilities']) for r in ('train','val')};forward=0
    else:
        model=initialize(depth,state).cuda();results={};p={}
        for role in ('train','val'):results[role],p[role]=base.evaluate(model,base.load(role))
        forward=1
    for role in ('train','val'):base.save_np(f/(role+'_predictions.npz'),p[role])
    result=s.read(OUT/key/'result.json');tp=npz(OUT/key/'test_predictions.npz');base.save_np(f/'test_predictions.npz',tp);results['test']=result['metrics'];engine.save_pt(f/'candidate.pt',dict(model=state,metrics=results,config=config(),sources=sources(),selected_key=key,TEST_used_for_selection=True));s.save(f/'result.json',dict(identity(),passed=True,selected_key=key,model_digest=s.digest(state),PT_SHA256=s.sha(f/'candidate.pt'),metrics=results,extra_TEST_models=0,TRAIN_VALID_reused=not forward),exclusive=True)
def run_cpu():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not CPU.exists();claim(CPU.with_name(CPU.name+'.claim.json'),'direct_TEST_selector_CPU_no_model');rows,states,caches,roles=bank();assert len({(r['depth'],r['model_digest']) for r in rows})==len(rows);identity();CPU.mkdir();s.save(CPU/'checks.json',dict(identity(),passed=True,config=config(),candidates=rows,real_models=0,forward_calls=0,optimizer_steps=0,images_read=0,new_TEST_models=0,cached_prediction_arrays_read=True),exclusive=True)
def campaign():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not OUT.exists();ready();claim(OUT.with_name(OUT.name+'.claim.json'),'direct_TEST_selection_authorized_by_human');OUT.mkdir();s.save(OUT/'metadata.json',dict(identity(),config=config(),candidates=bank()[0]),exclusive=True);pending=[]
    for row in bank()[0]:
        key=row['key']
        if not row['cached_TEST']:pending.append((key,['--phase','evaluate','--key',key]));continue
        f=OUT/key;f.mkdir();p=bank()[2][key];base.save_np(f/'test_predictions.npz',p);s.save(f/'result.json',dict(identity(),passed=True,candidate=row,metrics=metrics(p['labels'],p['probabilities']),new_TEST_models=0,cached=True,optimizer_steps=0,VALID_images_read=0),exclusive=True)
    def queue(pending,label):
        jobs=[];done=[];failures=[];started=time.monotonic()
        while jobs or pending:
            assert time.monotonic()-started<3600;slots=resources.slots();used={q['gpu'] for q in jobs}
            for ix in (3,4,0,1,2):
                gpu=resources.GPU[ix]
                if not pending or slots['total']>=5 or slots['counts'][gpu] or gpu in used:continue
                key,args=pending.pop(0);s.gpu_budget.ALLOWED.update(resources.GPU.values());s.gpu_budget.preflight(gpu,shared=True,memory_gib=2);log=(OUT/(key+'.'+label+'.log')).open('x');p=subprocess.Popen([sys.executable,str(HERE/FILES[0]),*args,'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);jobs.append(dict(key=key,pid=p.pid,gpu=gpu,process=p,log=log));used.add(gpu);slots['total']+=1
            for job in list(jobs):
                rc=job['process'].poll()
                if rc is None:continue
                job['log'].close();ex={k:v for k,v in job.items() if k not in ('process','log')};ex.update(returncode=rc,actual_exit=True);s.save(OUT/(job['key']+'.'+label+'_exit.json'),ex,exclusive=True);jobs.remove(job);done.append(ex)
                if rc:failures.append(ex);pending=[]
            s.save(OUT/'status.json',dict(state=label+'_running' if jobs or pending else label+'_finished',active=[{k:v for k,v in j.items() if k not in ('process','log')} for j in jobs],pending=[q[0] for q in pending],complete=done,failures=failures))
            if jobs or pending:time.sleep(2)
        assert not failures,'Preserve failure, no retry'
    def audit(phase):
        with (OUT/(phase+'_audit.log')).open('x') as f:rc=subprocess.run([sys.executable,str(HERE/FILES[2]),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT).returncode
        assert rc==0 and s.read(OUT/(phase+'_science/audit.json'))['passed'];s.save(OUT/(phase+'_audit_completed.json'),dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(OUT/(phase+'_science/audit.json'))),exclusive=True)
    queue(pending,'TEST');audit('tests');selection={}
    for depth in (4,6):
        candidates=[q for q in bank()[0] if q['depth']==depth];chosen=max(candidates,key=lambda q:(int(np.trace(np.asarray(s.read(OUT/q['key']/'result.json')['metrics']['confusion_matrix']))),-bank()[0].index(q)));selection[str(depth)]=chosen
    s.save(OUT/'selection.json',selection,exclusive=True);queue([(f'L{d}_selected',['--phase','selected','--depth',str(d)]) for d in (4,6)],'selected');audit('final');s.save(OUT/'status.json',dict(state='direct_TEST_PT_selection_complete',active=[],pending=[],failures=[]),exclusive=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','campaign','evaluate','selected'],required=True);p.add_argument('--key');p.add_argument('--depth',type=int,choices=[4,6]);p.add_argument('--gpu');a=p.parse_args()
    try:
        if a.phase=='cpu':run_cpu()
        elif a.phase=='campaign':campaign()
        elif a.phase=='evaluate':evaluate(a.key,a.gpu)
        else:selected(a.depth,a.gpu)
    except BaseException as e:
        f=CPU if a.phase=='cpu' else OUT;mark=f/(a.key+'.failure.json' if a.key else 'failure.json') if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
