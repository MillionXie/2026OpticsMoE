"""Validation-only paired optimization; test evaluation is a separate locked action."""
import argparse,copy,gc,json,math,os,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import train_adrenal_regularized as old
from train_adrenal_regularized import r,torch,np,F,PhaseLayer,affine_parameters,augment,phase_smoothness
from calibration import choose_thresholds,full_metrics


class CoarsePhase(PhaseLayer):
    """Bilinear latent phase grid; identical physical aperture and propagation."""
    def __init__(self,original,factor):
        super().__init__(original.grid_size,parameterization='sigmoid',init='zeros')
        self.raw_phase=torch.nn.Parameter(torch.zeros(tuple(math.ceil(s/factor) for s in original.grid_size)))

    def get_phase(self):
        raw=F.interpolate(self.raw_phase[None,None],size=self.grid_size,mode='bilinear',align_corners=False)[0,0]
        return 2*math.pi*raw.sigmoid()


def build(v,protocol):
    model=r.build(v['architecture'],r.CONFIGS[v['id']])
    if protocol['phase_factor']>1:
        for name,module in list(model.named_modules()):
            if isinstance(module,PhaseLayer):
                parent,_,attr=name.rpartition('.')
                setattr(model.get_submodule(parent),attr,CoarsePhase(module,protocol['phase_factor']))
    return model.cuda()


def loss_terms(model,pred,y,weights,cfg):
    p=r.probabilities(pred).clamp_min(1e-9)
    target=F.one_hot(y,2)*(1-cfg['label_smoothing'])+cfg['label_smoothing']/2
    nll=(-(target*p.log()).sum(1)*weights[y]).mean()
    mse=r.objective(pred,y,model.masks)
    smooth=phase_smoothness(model)
    return nll+cfg['mse_weight']*mse+cfg['phase_smooth_weight']*smooth,dict(nll=float(nll.detach()),mse=float(mse.detach()),smooth=float(smooth.detach()))


@torch.no_grad()
def evaluate(model,data):
    model.eval();allp=[];capture=[];routes=[];power=[]
    for x,y,_ in r.batches(data):
        out=model(x);allp.append(r.probabilities(out).cpu().numpy())
        capture.append(((out['intensity']*model.masks.sum(0)).sum((-1,-2))/out['intensity'].sum((-1,-2)).clamp_min(1e-12)).cpu().numpy())
        if out['route_probabilities'] is not None:
            routes.append(out['route_probabilities'].cpu().numpy());power.append(out['stage_input_power'].cpu().numpy())
    p=np.concatenate(allp);y=data[1].cpu().numpy();m=r.metrics(y,p)
    losses=-np.log(np.maximum(p[np.arange(len(y)),y],1e-9))
    m['nll']=float(losses.mean());m['balanced_nll']=float(np.mean([losses[y==c].mean() for c in [0,1]]))
    m['detector_capture']=float(np.concatenate(capture).mean())
    if routes:
        a=np.concatenate(routes);w=a*a/(a*a).sum(1,keepdims=True)
        m['routing']=dict(probability_std=a.std(0).tolist(),mean_entrance_power=w.mean(0).tolist(),entrance_power_std=w.std(0).tolist(),largest_counts=np.bincount(w.argmax(1),minlength=9).tolist(),stage_mean_power=np.concatenate(power).mean(0).tolist())
    rows=[dict(sample_id=str(i),label_true=int(t),label_pred=int(q[1]>.5),score0=float(q[0]),score1=float(q[1])) for i,t,q in zip(data[2],y,p)]
    return m,rows


def train(v,seed,data,val,cfg,out,sources):
    dest=out/v['id']/('seed'+str(seed));dest.mkdir(parents=True,exist_ok=False)
    r.setseed(seed);model=build(v,cfg);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    weights=len(data[1])/(2*torch.bincount(data[1],minlength=2).float())
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['lr']*.1)
    history=[];diagnostics=[];best=float('inf');wait=0;orders=[];transforms=[];started=time.time()
    init,_=evaluate(model,val);r.save(dest/'initial_validation.json',init)
    for epoch in range(1,cfg['epochs']+1):
        model.train();order=r.epoch_order(seed,epoch,len(data[1]));theta=affine_parameters(len(data[1]),seed,epoch,cfg['augmentation'])
        orders.append(r.sha_tensor(order));transforms.append(r.sha_tensor(theta));total=0.
        for batch,(x,y,indices) in enumerate(r.batches(data,order)):
            opt.zero_grad(set_to_none=True);pred=model(augment(x,theta[indices]));loss,parts=loss_terms(model,pred,y,weights,cfg)
            assert torch.isfinite(loss);loss.backward()
            if batch==0:
                grads={n:float(p.grad.norm()) for n,p in model.named_parameters()}
                assert all(np.isfinite(g) and g>0 for g in grads.values()),grads
                diagnostics.append(dict(epoch=epoch,gradient_norm=grads))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-cfg['ema_decay'])
            total+=float(loss.detach())*len(y)
        vm,_=evaluate(ema,val)
        row=dict(epoch=epoch,train_augmented_loss=total/len(data[1]),validation=vm,lr=opt.param_groups[0]['lr'])
        if epoch==1 or epoch%5==0:row['clean_train']=evaluate(ema,data)[0]
        history.append(row)
        if vm['balanced_nll']<best-cfg['min_delta']:
            best=vm['balanced_nll'];wait=0
            r.save_torch(dest/'best_checkpoint.pt',dict(model=ema.state_dict(),epoch=epoch,variant=v,seed=seed,config=cfg,validation=vm,sources=sources))
        else:wait+=1
        scheduler.step()
        r.save(dest/'history.json',history);r.save(dest/'gradients.json',diagnostics)
        r.save_torch(dest/'last_checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,config=cfg,variant=v,seed=seed,sources=sources))
        r.save(out/'status.json',dict(state='training',variant=v['id'],seed=seed,epoch=epoch,val_auroc=vm['auroc'],val_balanced_nll=vm['balanced_nll']))
        print(json.dumps(dict(variant=v['id'],seed=seed,epoch=epoch,auc=vm['auroc'],bnll=vm['balanced_nll'],seconds=time.time()-started)),flush=True)
        if epoch>=cfg['minimum_epochs'] and wait>=cfg['patience']:break
    changes={n:dict(rms=float((p-initial[n]).square().mean().sqrt()),max_abs=float((p-initial[n]).abs().max())) for n,p in model.named_parameters()}
    assert all(a['rms']>0 for a in changes.values())
    ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(ck['model'])
    metrics={}
    for split,d in [('train',data),('val',val)]:
        metrics[split],rows=evaluate(ema,d);r.csvwrite(dest/(split+'_predictions.csv'),rows)
        if split=='val':
            thresholds=choose_thresholds(np.array([a['label_true'] for a in rows]),np.array([[a['score0'],a['score1']] for a in rows]));r.save(dest/'thresholds.json',thresholds)
    result=dict(variant=v['id'],seed=seed,selected_epoch=ck['epoch'],epochs_completed=epoch,metrics=metrics,parameters=sum(p.numel() for p in model.parameters()),changes=changes,orders=orders,transforms=transforms,seconds=time.time()-started,checkpoint_sha256=r.sha(dest/'best_checkpoint.pt'))
    r.save(dest/'summary.json',result)
    del model,ema,opt,ck,initial;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','train','test'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[17]);p.add_argument('--depths',type=int,nargs='+',default=[2,4,6]);args=p.parse_args()
    cfg=r.read(args.profile);r.EXP.update(batch_size=cfg['batch_size'],data_npz=str(args.data.resolve()));r.setup()
    variants=[v for v in r.VARIANTS if v['activation']=='relu_softsign' and v['depth'] in args.depths]
    sources={str(f.relative_to(old.TASK)):r.sha(f) for f in sorted(old.ARCHIVE.rglob('*')) if f.suffix in ['.py','.yaml']}
    sources['runner']=r.sha(__file__);sources['augmentation']=r.sha(old.__file__);sources['profile']=r.sha(args.profile)
    if args.phase=='test':
        lock=r.read(args.out/'test_lock.json');assert lock['sources']==sources
        for rel,h in lock['files'].items():assert r.sha(args.out/rel)==h,rel
        with np.load(args.data,allow_pickle=False) as z:x=z['test_images'].copy();y=z['test_labels'].reshape(-1).copy();ids=z['test_ids'].copy()
        test=(F.interpolate(torch.from_numpy(x[:,None]),size=(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).cuda(),torch.from_numpy(y).long().cuda(),ids)
        results=[]
        for rel in lock['models']:
            dest=args.out/rel;ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model=build(ck['variant'],cfg);model.load_state_dict(ck['model']);m,rows=evaluate(model,test);r.csvwrite(dest/'test_predictions.csv',rows)
            threshold=r.read(dest/'thresholds.json')['policies']['val_balanced']['threshold'];prob=np.array([[a['score0'],a['score1']] for a in rows])
            item=dict(variant=ck['variant']['id'],seed=ck['seed'],test=m,val_threshold=full_metrics(y,prob,threshold,0.));results.append(item);del model,ck;torch.cuda.empty_cache()
        r.save(args.out/'test_results.json',results);r.save(args.out/'test_execution.json',dict(command=sys.argv,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),time=r.now()));return
    args.out.mkdir(parents=True,exist_ok=False)
    r.save(args.out/'metadata.json',dict(command=sys.argv,config=cfg,sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),data_sha256=r.sha(args.data),python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),test_read=False,seeds=args.seeds,variants=variants))
    data,val=r.getdata('train'),r.getdata('val')
    assert not set(data[2])&set(val[2])
    if args.phase=='smoke':
        reports=[];weights=len(data[1])/(2*torch.bincount(data[1],minlength=2).float())
        for v in variants:
            model=build(v,cfg);loss,_=loss_terms(model,model(data[0][:8]),data[1][:8],weights,cfg);loss.backward();g={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(np.isfinite(z) and z>0 for z in g.values());reports.append(dict(variant=v['id'],gradients=g,parameters=sum(p.numel() for p in model.parameters())));del model;torch.cuda.empty_cache()
        r.save(args.out/'smoke.json',reports);return
    results=[];files={};models=[]
    for seed in args.seeds:
        for v in variants:
            results.append(train(v,seed,data,val,cfg,args.out,sources));rel=Path(v['id'])/('seed'+str(seed));models.append(rel.as_posix())
            for name in ['best_checkpoint.pt','thresholds.json']:files[(rel/name).as_posix()]=r.sha(args.out/rel/name)
            r.save(args.out/'validation_results.json',results)
    r.save(args.out/'test_lock.json',dict(sources=sources,files=files,models=models,selection='minimum validation balanced NLL; EMA weights',test_status='previously inspected official test; retrospective development',time=r.now()))
    r.save(args.out/'status.json',dict(state='training_complete_test_not_read'))


if __name__=='__main__':main()
