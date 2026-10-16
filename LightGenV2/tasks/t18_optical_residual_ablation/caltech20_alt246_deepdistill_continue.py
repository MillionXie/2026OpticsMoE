"""Six-layer residual student; a frozen two-layer teacher is TRAIN-only."""
import argparse,copy,hashlib,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
import caltech20_alt246_headcal as h
import caltech20_alt246_deepdistill as prior
from caltech20_alt246_genreg_model import RegularizedMoE

g,b,s,engine,HERE,ROOT=h.g,h.g.b,h.s,h.engine,h.HERE,h.ROOT
PROFILE=HERE/'caltech20_alt246_deepdistill_continue_profile.json'
FILES=('caltech20_alt246_deepdistill_continue.py','caltech20_alt246_deepdistill_continue_profile.json','check_caltech20_alt246_deepdistill_continue.py','audit_caltech20_alt246_deepdistill_continue.py','CALTECH20_ALT246_KD_CONTINUE_PROTOCOL_20261016.md')
RUN='caltech20_alt246_L6_KDcontinue160_s17_20261016'
PRIOR_RUN=prior.RUN
CPU=HERE/'runs/smoke/caltech20_alt246_deepdistill_continue_cpu_20261016'

def config():
    p=s.read(PROFILE);cfg=g.config(6);cfg.update(p['training']);return cfg

def sources():
    src=prior.sources()
    for n in FILES:src[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return src

def identity():
    src=sources();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(src,commit);return dict(sources=src,git_commit=commit)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)

def admit():
    p=s.read(PROFILE);prior.admit();root=HERE/'runs/simulation'/prior.RUN;assert s.read(root/'status.json')['state']=='L6_distillation_review_science_test_complete'
    meta=s.read(root/'metadata.json');assert meta['sources']==prior.sources();engine.verify_sources(meta['sources'],meta['git_commit']);pid=meta['pid'];proc=Path('/proc',str(pid),'cmdline');assert not proc.exists() or prior.RUN.encode() not in proc.read_bytes()
    for rec,folder in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        q=s.read(root/rec);assert q['actual_audit_returncode']==0 and q['audit_sha256']==p['prior_receipts'][folder]==s.sha(root/folder/'audit.json')
    for key in ('parent_last','teacher','prior_best','prior_train','prior_val','prior_test'):assert s.sha(p[key]['path'])==p[key]['sha256']
    assert s.sha(root/'fit/history.json')==p['prior_history_sha256'];return p


def cpu_ready():
    proof=s.read(CPU/'checks.json');assert proof['passed'] and proof['sources']==sources();engine.verify_sources(proof['sources'],proof['git_commit'])

teacher_model,kd,objective,gradients,gate=prior.teacher_model,prior.kd,prior.objective,prior.gradients,prior.gate

def fit(a):
    p=admit();cpu_ready();cfg=config();assert not a.out.exists() and a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();claim(a.out.with_name(a.out.name+'.claim.json'),'L6_frozen_L2_teacher_fit');a.out.mkdir()
    s.gpu_budget.ALLOWED.update(b.GPU.values());policy=s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4)
    last=torch.load(p['parent_last']['path'],map_location='cpu',weights_only=False);assert last['epoch']==140 and last['steps']==12040;teacher=teacher_model(p,'cuda');data,val=b.load('train'),b.load('val');proof=gate(last['model'],teacher,data,cfg,'cuda');s.save(a.out/'numeric_gate.json',proof,exclusive=True)
    model=RegularizedMoE(6).cuda();ema=copy.deepcopy(model).eval();model.load_state_dict(last['model']);ema.load_state_dict(last['ema']);phase=[];electronic=[]
    for n,v in model.named_parameters():(phase if n=='raw_router' or n.startswith('phase_sets.') else electronic).append(v)
    optimizer=torch.optim.AdamW([dict(params=phase,weight_decay=0),dict(params=electronic,weight_decay=cfg['electronic_weight_decay'])],lr=cfg['lr']);optimizer.load_state_dict(last['optimizer']);moment=g.optimizer_moment_digest(optimizer.state_dict());assert moment==g.optimizer_moment_digest(last['optimizer'])
    assert [q['weight_decay'] for q in optimizer.param_groups]==[0,cfg['electronic_weight_decay']]
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,threshold=.002,threshold_mode='abs',min_lr=2e-5);scheduler.load_state_dict(last['scheduler'])
    assert [q['lr'] for q in optimizer.param_groups]==last['scheduler']['_last_lr']
    meta=dict(identity(),pid=os.getpid(),config=cfg,profile=p,gpu_uuid=a.gpu,policy=policy,parent_raw_digest=s.digest(last['model']),parent_ema_digest=s.digest(last['ema']),initial_optimizer_moment_digest=moment,teacher_digest=s.digest(teacher.state_dict()));s.save(a.out/'metadata.json',meta,exclusive=True)
    best=p['prior_metrics']['val']['balanced_nll'];selected=dict(epoch=p['prior_selected_epoch'],head_step=p['prior_selected_head_step'],best_path=p['prior_best']['path'],train_path=p['prior_train']['path'],val_path=p['prior_val']['path'],metrics=p['prior_metrics']);history=[];steps=12040;started=time.monotonic();weights=(len(data[1])/(20*torch.bincount(data[1],minlength=20).float())).cuda()
    for epoch in range(141,161):
        assert time.monotonic()-started<cfg['maximum_seconds'],'Method incomplete: preserve and stop, no retry';begin=time.monotonic();order,theta=engine.epoch_plan(epoch,len(data[1]),cfg);record={k:[] for k in ('mix','flip','mask')};lr=[q['lr'] for q in optimizer.param_groups];coverage=0;samples=0;kl=0.
        for batch,off in enumerate(range(0,len(order),16)):
            idx=order[off:off+16];x,y,mask,hashes=g.mixed(data,idx,theta,epoch,batch,cfg,'cuda')
            for kind,digest in hashes.items():record[kind].append(digest)
            _,_,_,_,stats=gradients(model,teacher,x,y,mask,weights,cfg,4);coverage+=stats['eligible'];samples+=stats['samples'];kl+=stats['mean_temperature_scaled_kl']*len(idx)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);previous={n:v.detach().cpu().clone() for n,v in ema.named_parameters()} if epoch==160 and batch==85 else None;optimizer.step();steps+=1
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.mul_(cfg['ema_decay']).add_(mp,alpha=1-cfg['ema_decay'])
        assert samples==1372 and s.digest(teacher.state_dict())==meta['teacher_digest'] and all(v.grad is None for v in teacher.parameters())
        tm,tr=b.evaluate(ema,data);vm,vr=b.evaluate(ema,val);entry=dict(epoch=epoch,train=tm,val=vm,lr=lr,adam_steps=steps,order_sha256=s.digest(order),affine_sha256=s.digest(theta),**{kind+'_sha256':hashlib.sha256(''.join(v).encode()).hexdigest() for kind,v in record.items()},teacher_eligible=coverage,teacher_samples=samples,mean_temperature_scaled_kl=kl/samples,teacher_digest=s.digest(teacher.state_dict()),raw_digest=s.digest(model.state_dict()),ema_digest=s.digest(ema.state_dict()),epoch_seconds=time.monotonic()-begin);history.append(entry)
        if vm['balanced_nll']<best-1e-6:
            best=vm['balanced_nll'];engine.save_pt(a.out/'best.pt',dict(model=ema.state_dict(),epoch=epoch,head_step=0,contract=ema.contract(),sources=meta['sources'],config=cfg,metrics=dict(train=tm,val=vm)));b.save_np(a.out/'selected_train.npz',tr);b.save_np(a.out/'selected_val.npz',vr);selected=dict(epoch=epoch,head_step=0,best_path=str(a.out/'best.pt'),train_path=str(a.out/'selected_train.npz'),val_path=str(a.out/'selected_val.npz'),metrics=dict(train=tm,val=vm))
        scheduler.step(vm['balanced_nll']);engine.save_pt(a.out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),previous_final_ema=previous,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,steps=steps,sources=meta['sources'],config=cfg));s.save(a.out/'history.json',history);s.save(a.out/'status.json',dict(state='training',epoch=epoch,train=tm,val=vm,teacher_coverage=coverage/samples));print(dict(epoch=epoch,train=tm['accuracy'],val=vm['accuracy'],NLL=vm['balanced_nll'],coverage=coverage/samples),flush=True)
    assert steps==13760;identity();s.save(a.out/'fit_result.json',dict(passed=True,actual_epochs=160,additional_epochs=20,selected_epoch=selected['epoch'],selected_head_step=selected['head_step'],best_path=selected['best_path'],best_sha256=s.sha(selected['best_path']),selected_train_path=selected['train_path'],selected_val_path=selected['val_path'],metrics=selected['metrics'],steps=steps,fit_seconds=time.monotonic()-started,teacher_digest=meta['teacher_digest'],teacher_frozen=True,history_sha256=s.sha(a.out/'history.json'),last_sha256=s.sha(a.out/'last.pt'),stop_reason='registered20_epoch_method_review'),exclusive=True)

def test(a):
    p=admit();root=a.out.parent;f=s.read(root/'fit/fit_result.json');receipt=s.read(root/'audit_completed.json');assert receipt['actual_audit_returncode']==0 and receipt['audit_sha256']==s.sha(root/'science/audit.json') and not a.out.exists();claim(a.out.with_name(a.out.name+'.claim.json'),'L6_selected_once_test');a.out.mkdir()
    checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);digest=s.digest(checkpoint['model']);prior_checkpoint=torch.load(p['prior_best']['path'],map_location='cpu',weights_only=False)
    if digest==s.digest(prior_checkpoint['model']):
        prior=s.read(HERE/'runs/simulation'/PRIOR_RUN/'test/result.json');result=dict(prior,reused=True,new_test_evaluations=0)
    elif digest==s.digest(torch.load(s.read(HERE/'caltech20_alt246_deepdistill_profile.json')['prior_best']['path'],map_location='cpu',weights_only=False)['model']):
        original=s.read(HERE/'runs/simulation'/h.RUN/'test/result.json');result=dict(original,reused=True,new_test_evaluations=0)
    elif digest==h.config()['parent']['model_digest']:
        parent=h.config()['parent']
        with np.load(parent['test_path'],allow_pickle=False) as z:metrics=engine.metrics(z['labels'],z['probabilities'])
        result=dict(metrics=metrics,predictions_path=parent['test_path'],predictions_sha256=parent['test_sha256'],reused=True,new_test_evaluations=0)
    else:
        assert a.gpu in b.GPU.values() and os.environ.get('CUDA_VISIBLE_DEVICES')==a.gpu;b.slots();s.gpu_budget.ALLOWED.update(b.GPU.values());s.gpu_budget.configure(torch,a.gpu,shared=True,memory_gib=2);torch.set_num_threads(4);model=RegularizedMoE(6).cuda();checkpoint=torch.load(f['best_path'],map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['model'],strict=True);metrics,saved=b.evaluate(model,b.load('test'));b.save_np(a.out/'predictions.npz',saved);result=dict(metrics=metrics,predictions_path=str(a.out/'predictions.npz'),predictions_sha256=s.sha(a.out/'predictions.npz'),reused=False,new_test_evaluations=1)
    s.save(a.out/'result.json',dict(result,best_sha256=f['best_sha256'],model_digest=digest,selected_epoch=f['selected_epoch'],selected_head_step=f['selected_head_step'],teacher_in_inference=False,test_used_for_epoch_selection=False,test_informed_development=True),exclusive=True)

def run_child(root,phase,gpu):
    out=root/phase;log=(root/(phase+'.log')).open('x');child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(out),'--gpu',gpu],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT);rc=child.wait();log.close();s.save(root/(phase+'_exit.json'),dict(pid=child.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0,'Own producer failed; preserve/no retry'

def campaign(a):
    assert a.out==HERE/'runs/simulation'/RUN and not a.out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';p=admit();cpu_ready();assert b.slots()['total']==0;claim(a.out.with_name(a.out.name+'.campaign_claim.json'),'L6_KD_campaign');a.out.mkdir();s.save(a.out/'metadata.json',dict(identity(),pid=os.getpid(),config=config(),profile=p),exclusive=True)
    s.gpu_budget.ALLOWED.update(b.GPU.values());gpu=None
    for card in (4,2,5,0):
        try:s.gpu_budget.preflight(b.GPU[card],shared=True,memory_gib=2);gpu=b.GPU[card];break
        except AssertionError:continue
    assert gpu is not None,'No safe non-A100 slot';run_child(a.out,'fit',gpu)
    for phase,folder,rec in [('fit','science','audit_completed.json'),('test','saved_test_audit','test_audit_completed.json')]:
        if phase=='test':run_child(a.out,'test',gpu)
        claim(a.out/(phase+'_audit_claim.json'),'independent_KD_'+phase)
        with (a.out/(phase+'_audit.log')).open('x') as log:rc=subprocess.run([sys.executable,str(HERE/'audit_caltech20_alt246_deepdistill_continue.py'),'--base',str(a.out),'--out',str(a.out/folder),'--phase',phase],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT).returncode
        assert rc==0 and s.read(a.out/folder/'audit.json')['passed'],'Own saved audit failed; preserve/no retry';s.save(a.out/rec,dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(a.out/folder/'audit.json')),exclusive=True)
    result=s.read(a.out/'test/result.json');s.save(a.out/'status.json',dict(state='L6_KDcontinue160_science_test_complete',test=result['metrics'],performance_maintained=result['metrics']['accuracy']>=p['prior_test_accuracy'],target_six_above_four=result['metrics']['accuracy']>p['reference_four_test_accuracy'],teacher_in_inference=False),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','test','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        path=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not path.exists():s.save(path,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
