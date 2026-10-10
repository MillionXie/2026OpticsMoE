"""T18 training-only MixUp/focal profiles with unchanged optical inference.

The parent was TEST-selected by explicit user authorization. Every subsequent
score is consequently a DEVELOPMENT score; train gradients use train only.
"""
import argparse
import copy
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import run as t
from test_sweep import ALLOWED_GPUS, DATA_SHA, state_sha, check_predictions

PARENT_SHA = '058b758ce585dde9e9ba092d28feca89465c9444f48be5c087bab5a86ccf565f'
SCOPE = 'TEST-SELECTED DEVELOPMENT, NOT INDEPENDENT GENERALIZATION'


def classification_loss(prob, target, class_weights, gamma=0.):
    # Soft-label class weighting is applied to each target class, including MixUp.
    logp = prob.clamp_min(1e-12).log()
    return (-(target * class_weights[None] * (1-prob).pow(gamma) * logp).sum(1)).mean()


def smoke():
    p = t.torch.tensor([[.8,.2],[.4,.6]],requires_grad=True)
    y = t.torch.tensor([[1.,0.],[0.,1.]])
    w = t.torch.ones(2)
    assert t.torch.allclose(classification_loss(p,y,w), -t.torch.log(t.torch.tensor([.8,.6])).mean())
    lam=.9; mix=lam*y+(1-lam)*y.flip(0)
    assert t.torch.allclose(mix.sum(1),t.torch.ones(2))
    expected=lam*classification_loss(p,y,w)+(1-lam)*classification_loss(p,y.flip(0),w)
    assert t.torch.allclose(classification_loss(p,mix,w),expected)
    loss=classification_loss(p,mix,w,1.);loss.backward()
    assert t.torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert float(loss)<float(classification_loss(p,mix,w))
    soft=t.torch.tensor([[.7,.3]])
    same=t.b.F.kl_div(soft.log(),soft,reduction='batchmean')
    assert abs(float(same))<1e-7
    print('CPU loss/soft-target/gradient checks passed')


def train(a):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ALLOWED_GPUS
    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert gpu not in procs
    assert t.r.sha(a.data)==DATA_SHA and t.r.sha(a.checkpoint)==PARENT_SHA
    ck=t.torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert ck['depth']==6 and ck['seed']==17 and ck['arch']=='moe'
    assert ck['sources']['parent']==t.source_identity()
    approved=hashlib.sha256(subprocess.check_output(['git','show',
        'fad46834a:LightGenV2/tasks/t18_optical_residual_ablation/continue_residual.py'])).hexdigest()
    assert ck['sources']['continuation']==approved
    assert ck['config']['residual_rho']==.3 and ck['config']['dataset']=='MangoLeafVarietyBD_raw_v2'
    cfg=copy.deepcopy(ck['config']);cfg.update(epochs=30,minimum_epochs=10,patience=10,
        min_delta=.0002,lr=.00015,ema_decay=.99,training_profile='rho03_'+a.profile+'_development30',
        continuation_augmentation_epoch_offset=200,parent_checkpoint_sha256=PARENT_SHA,
        mixup_alpha=.1 if a.profile=='mixup' else 0.,focal_gamma=1. if a.profile=='focal' else 0.,
        selection='best EMA by validation balanced NLL during training; saved states compared on test as development')
    teacher=None
    if a.profile=='distill':
        from train_teacher import Teacher,source as teacher_source
        teacher_ck=t.torch.load(a.teacher,map_location='cpu',weights_only=False)
        assert teacher_ck['sources']==teacher_source() and teacher_ck['data_sha256']==DATA_SHA
        assert t.r.read(a.teacher.parent/'result.json')['eligible_for_distillation']
        teacher=Teacher().cuda();teacher.load_state_dict(teacher_ck['model']);teacher.eval()
        for q in teacher.parameters():q.requires_grad_(False)
        cfg.update(distillation_weight=.3,distillation_temperature=2.,
            teacher_checkpoint_sha256=t.r.sha(a.teacher),teacher_sources=teacher_ck['sources'],
            teacher_used_only_during_training=True)
    a.out.mkdir(parents=True,exist_ok=False);folder=a.out/'moe_L6_seed17';folder.mkdir()
    src=dict(parent=t.source_identity(),regularized=t.r.sha(Path(__file__)))
    t.r.save(a.out/'metadata.json',dict(config=cfg,sources=src,depth=6,seed=17,pid=os.getpid(),
        command=sys.argv,gpu_uuid=gpu,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        parent_checkpoint=str(a.checkpoint),parent_training_sources=ck['sources'],
        parent_state_sha256=state_sha(ck['model']),data_sha256=DATA_SHA,time=t.r.now(),
        scope=SCOPE,training_reads_test=False,optimizer_restart=True,ema_restart=True))
    t.torch.set_num_threads(4);t.r.setseed(17)
    model=t.build('moe',6,cfg);model.load_state_dict(ck['model'],strict=True)
    ema=copy.deepcopy(model).eval()
    for q in ema.parameters():q.requires_grad_(False)
    data=t.k.load_data(a.data,'train');val=t.k.load_data(a.data,'val')
    opt=t.torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0)
    scheduler=t.torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr']*.1)
    weights=len(data[1])/(8*t.torch.bincount(data[1],minlength=8).float())
    initial=t.b.evaluate(ema,val,'moe',16)[0];best=initial['balanced_nll'];wait=0
    t.r.save(folder/'initial_validation.json',initial)
    def checkpoint(state,epoch):
        return dict(model=state,epoch=epoch,arch='moe',depth=6,seed=17,config=cfg,sources=src)
    t.r.save_torch(folder/'best_checkpoint.pt',checkpoint(ema.state_dict(),0))
    history=[];gradients=[];orders=[];transforms=[];mixup_hashes=[];started=time.time()
    for epoch in range(1,31):
        model.train();actual_epoch=epoch+200
        order=t.r.epoch_order(17,actual_epoch,len(data[1]))
        theta=t.b.affine_parameters(len(data[1]),17,actual_epoch,cfg['augmentation'])
        orders.append(t.r.sha_tensor(order));transforms.append(t.r.sha_tensor(theta))
        rng=t.np.random.RandomState(170000+actual_epoch);mixhash=hashlib.sha256();total=0
        for batch,idx in enumerate(order.split(16)):
            x=t.b.encode(data[0][idx],theta[idx]);y=data[1][idx]
            target=t.b.F.one_hot(y,8).float()*(1-cfg['label_smoothing'])+cfg['label_smoothing']/8
            if cfg['mixup_alpha']:
                lam=float(rng.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam)
                permutation=rng.permutation(len(idx));j=t.torch.tensor(permutation,device=x.device)
                x=lam*x+(1-lam)*x[j];target=lam*target+(1-lam)*target[j]
                mixhash.update(t.np.asarray([lam],dtype='float64').tobytes());mixhash.update(permutation.tobytes())
            opt.zero_grad(set_to_none=True);prob,capture,_=t.b.forward(model,x,'moe')
            loss=classification_loss(prob,target,weights,cfg['focal_gamma'])
            if teacher is not None:
                temperature=cfg['distillation_temperature']
                with t.torch.no_grad():soft_target=(teacher(x)/temperature).softmax(1)
                log_student=(prob.clamp_min(1e-12).log()/temperature).log_softmax(1)
                loss=loss+cfg['distillation_weight']*temperature**2*t.b.F.kl_div(log_student,soft_target,reduction='batchmean')
            loss=loss-cfg['capture_weight']*capture.clamp_min(1e-12).log().mean()+cfg['phase_smooth_weight']*t.b.phase_smoothness(model)
            assert t.torch.isfinite(loss);loss.backward()
            if batch==0:
                norms={name:float(q.grad.norm()) for name,q in model.named_parameters()}
                assert all(t.np.isfinite(v) for v in norms.values()) and all(v>0 for v in norms.values())
                gradients.append(dict(epoch=epoch,norms=norms))
            t.torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            with t.torch.no_grad():
                for q,new in zip(ema.parameters(),model.parameters()):q.lerp_(new,1-cfg['ema_decay'])
            total+=float(loss.detach())*len(idx)
        vm,_=t.b.evaluate(ema,val,'moe',16)
        h=dict(epoch=epoch,val=vm,train_loss=total/len(data[1]),lr=opt.param_groups[0]['lr'])
        if epoch==1 or epoch%5==0:h['train']=t.b.evaluate(ema,data,'moe',16)[0]
        history.append(h);mixup_hashes.append(mixhash.hexdigest())
        if vm['balanced_nll']<best-cfg['min_delta']:
            best=vm['balanced_nll'];wait=0;t.r.save_torch(folder/'best_checkpoint.pt',checkpoint(ema.state_dict(),epoch))
        else:wait+=1
        scheduler.step()
        last=checkpoint(model.state_dict(),epoch);last.update(ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict())
        t.r.save_torch(folder/'last_checkpoint.pt',last)
        t.r.save(folder/'history.json',history);t.r.save(folder/'gradients.json',gradients)
        status=dict(state='training',profile=a.profile,epoch=epoch,val_accuracy=vm['accuracy'],
            val_balanced_nll=vm['balanced_nll'],seconds=time.time()-started,test_read=False)
        t.r.save(a.out/'status.json',status);print(t.json.dumps(status),flush=True)
        if epoch>=10 and wait>=10:break
    selected=t.torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    ema.load_state_dict(selected['model']);metrics={}
    for split,d in [('train',data),('val',val)]:
        metrics[split],rows=t.b.evaluate(ema,d,'moe',16);t.r.csvwrite(folder/(split+'_predictions.csv'),rows)
    t.r.save(a.out/'result.json',dict(profile=a.profile,selected_epoch=selected['epoch'],epochs_completed=epoch,
        metrics=metrics,parent_validation=initial,checkpoint_sha256=t.r.sha(folder/'best_checkpoint.pt'),
        orders=orders,transforms=transforms,mixup_hashes=mixup_hashes,scope=SCOPE,seconds=time.time()-started))
    t.r.save(a.out/'status.json',dict(state='validation_complete',test_read=False,time=t.r.now()))


def evaluate(a):
    # Only the six saved new states plus the immutable, already tested parent.
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES');assert gpu in ALLOWED_GPUS
    assert t.r.sha(a.data)==DATA_SHA
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=t.r.read(a.data.parent/'data_manifest.json');t.torch.set_num_threads(4)
    parent_record=t.r.read(a.parent_sweep/'results.json')['winner']
    assert parent_record['checkpoint_sha256']==PARENT_SHA
    check_predictions(Path(parent_record['predictions']),parent_record['test'],manifest)
    items=[]
    for profile in a.profiles:
        for name in ['best_checkpoint.pt','last_checkpoint.pt']:
            path=a.candidates/profile/'moe_L6_seed17'/name
            ck=t.torch.load(path,map_location='cpu',weights_only=False)
            assert ck['sources']['parent']==t.source_identity()
            archived=hashlib.sha256(subprocess.check_output(['git','show',
                'af8aedaef:LightGenV2/tasks/t18_optical_residual_ablation/regularized_continuation.py'])).hexdigest()
            assert ck['sources']['regularized'] in {t.r.sha(Path(__file__)),archived}
            assert ck['config']['parent_checkpoint_sha256']==PARENT_SHA
            assert ck['config']['training_profile']=='rho03_'+profile+'_development30'
            assert ck['depth']==6 and ck['config']['residual_rho']==.3
            for key in ['model'] if name.startswith('best') else ['model','ema']:
                items.append(dict(profile=profile,checkpoint=str(path),checkpoint_sha256=t.r.sha(path),
                    state_key=key,state_sha256=state_sha(ck[key]),epoch=ck['epoch'],
                    kind='best_ema' if name.startswith('best') else 'last_'+key))
    t.r.save(a.out/'manifest.json',dict(items=items,data_sha256=DATA_SHA,scope=SCOPE,
        parent=parent_record,time=t.r.now(),command=sys.argv,
        selection='test accuracy maximum, ties test balanced NLL minimum; parent remains eligible'))
    data=t.k.load_data(a.data,'test');known={parent_record['state_sha256']:dict(test=parent_record['test'],predictions=parent_record['predictions'])}
    results=[dict(parent_record,profile='parent',reused=True)]
    for i,item in enumerate(items):
        sid=item['state_sha256'];reused=sid in known
        if not reused:
            ck=t.torch.load(item['checkpoint'],map_location='cpu',weights_only=False)
            model=t.build('moe',6,ck['config']);model.load_state_dict(ck[item['state_key']],strict=True)
            tm,rows=t.b.evaluate(model,data,'moe',16);preds=a.out/f'state_{i}_test_predictions.csv'
            t.r.csvwrite(preds,rows);check_predictions(preds,tm,manifest)
            known[sid]=dict(test=tm,predictions=str(preds));del model,ck;gc.collect();t.torch.cuda.empty_cache()
        results.append(dict(item,**known[sid],reused=reused))
    winner=sorted(results,key=lambda q:(-q['test']['accuracy'],q['test']['balanced_nll'],q['checkpoint'],q['state_key']))[0]
    t.r.save(a.out/'results.json',dict(winner=winner,results=results,scope=SCOPE,time=t.r.now()))
    t.r.save(a.out/'status.json',dict(state='complete',time=t.r.now()))
    print(t.json.dumps(dict(winner_profile=winner['profile'],accuracy=winner['test']['accuracy'])),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','train','evaluate'],required=True)
    p.add_argument('--profile',choices=['mixup','focal','distill']);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--data',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--candidates',type=Path);p.add_argument('--parent-sweep',type=Path)
    p.add_argument('--profiles',nargs='+',choices=['mixup','focal','distill'],default=['mixup','focal'])
    a=p.parse_args()
    if a.phase=='smoke':smoke()
    elif a.phase=='train':train(a)
    else:evaluate(a)


if __name__=='__main__':main()
