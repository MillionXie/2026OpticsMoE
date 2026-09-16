"""One small shared frozen image encoder, followed by paired optical training."""
import argparse,copy,json,subprocess,sys,time
from pathlib import Path
import adrenal_generalization as g
from adrenal_generalization import r,torch,np,F,old


class TinyFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features=torch.nn.Sequential(torch.nn.Conv2d(1,8,5,padding=2,bias=False),torch.nn.GroupNorm(2,8),torch.nn.ReLU(),torch.nn.AvgPool2d(2),torch.nn.Conv2d(8,8,3,padding=1,bias=False),torch.nn.GroupNorm(2,8),torch.nn.ReLU(),torch.nn.AvgPool2d(2))
        self.head=torch.nn.Sequential(torch.nn.Dropout(.35),torch.nn.Flatten(),torch.nn.Linear(8*7*7,2))

    def maps(self,x):
        x=F.interpolate(x,size=(28,28),mode='bilinear',align_corners=False,antialias=True)
        return self.features(x)

    def forward(self,x):return self.head(self.maps(x))

    @torch.no_grad()
    def encode(self,x):
        z=self.maps(x);b=len(x)
        z=z.reshape(b,2,4,7,7).permute(0,1,3,2,4).reshape(b,1,14,28)
        z=F.interpolate(z,size=(100,100),mode='bilinear',align_corners=False)
        return z/z.amax((-1,-2),keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def cnn_eval(model,data):
    model.eval();ps=[]
    for x,y,_ in r.batches(data):ps.append(model(x).softmax(1).cpu().numpy())
    p=np.concatenate(ps);y=data[1].cpu().numpy();m=r.metrics(y,p);l=-np.log(np.maximum(p[np.arange(len(y)),y],1e-9));m['balanced_nll']=float(np.mean([l[y==c].mean() for c in [0,1]]));return m


def pretrain(args,cfg,data,val):
    r.setseed(args.seeds[0]);model=TinyFrontend().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,60,eta_min=.0003)
    weights=len(data[1])/(2*torch.bincount(data[1],minlength=2).float());best=float('inf');wait=0;hist=[]
    r.EXP['batch_size']=64
    for epoch in range(1,61):
        model.train();theta=g.affine_parameters(len(data[1]),args.seeds[0],epoch,cfg['augmentation']);total=0
        for x,y,idx in r.batches(data,r.epoch_order(args.seeds[0],epoch,len(data[1]))):
            opt.zero_grad(set_to_none=True);logp=model(g.augment(x,theta[idx])).log_softmax(1);target=F.one_hot(y,2)*.95+.025;loss=(-(target*logp).sum(1)*weights[y]).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(y)
        tm=cnn_eval(model,data);vm=cnn_eval(model,val);hist.append(dict(epoch=epoch,train=tm,val=vm,augmented_loss=total/len(data[1])));r.save(args.out/'frontend_history.json',hist)
        if vm['balanced_nll']<best-.0005:
            best=vm['balanced_nll'];wait=0;r.save_torch(args.out/'best_checkpoint.pt',dict(model=model.state_dict(),epoch=epoch,seed=args.seeds[0],train=tm,val=vm))
        else:wait+=1
        scheduler.step();r.save_torch(args.out/'last_checkpoint.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch))
        if epoch>=20 and wait>=12:break
    ck=torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    r.save(args.out/'summary.json',dict(selected_epoch=ck['epoch'],epochs_completed=epoch,train=ck['train'],val=ck['val'],feature_parameters=sum(p.numel() for p in model.features.parameters()),training_head_parameters=sum(p.numel() for p in model.head.parameters()),checkpoint_sha256=r.sha(args.out/'best_checkpoint.pt'),test_read=False))
    r.save(args.out/'status.json',dict(state='complete_frontend_pretraining_test_not_read'))


def install_frontend(frontend):
    base_augment=g.augment;base_evaluate=g.evaluate
    @torch.no_grad()
    def encoded_augment(x,theta):return frontend.encode(base_augment(x,theta))
    class InputWrapper(torch.nn.Module):
        def __init__(self,optical):super().__init__();self.optical=optical
        @property
        def masks(self):return self.optical.masks
        def forward(self,x):return self.optical(frontend.encode(x))
    g.augment=encoded_augment
    g.evaluate=lambda model,data:base_evaluate(InputWrapper(model),data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['frontend','smoke','train','test'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--frontend',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[17]);p.add_argument('--depths',type=int,nargs='+',default=[2,4,6]);args=p.parse_args()
    cfg=r.read(args.profile);r.EXP.update(batch_size=cfg['batch_size'],data_npz=str(args.data.resolve()));r.setup()
    sources={str(f.relative_to(old.TASK)):r.sha(f) for f in sorted(old.ARCHIVE.rglob('*')) if f.suffix in ['.py','.yaml']};sources.update(runner=r.sha(__file__),training=r.sha(g.__file__),augmentation=r.sha(old.__file__),profile=r.sha(args.profile))
    if args.frontend:sources['frontend_checkpoint']=r.sha(args.frontend)
    if args.phase!='test':
        args.out.mkdir(parents=True,exist_ok=False);r.save(args.out/'metadata.json',dict(command=sys.argv,config=cfg,sources=sources,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),data_sha256=r.sha(args.data),python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),test_read=False,seeds=args.seeds,frontend_file=str(args.frontend)))
    if args.phase=='frontend':return pretrain(args,cfg,r.getdata('train'),r.getdata('val'))
    assert args.frontend is not None
    frontend=TinyFrontend().cuda();frontend.load_state_dict(torch.load(args.frontend,map_location='cpu',weights_only=False)['model']);frontend.eval()
    for p in frontend.parameters():p.requires_grad_(False)
    before={n:r.sha_tensor(p) for n,p in frontend.state_dict().items()};install_frontend(frontend)
    variants=[v for v in r.VARIANTS if v['activation']=='relu_softsign' and v['depth'] in args.depths]
    if args.phase=='test':
        lock=r.read(args.out/'test_lock.json');assert lock['sources']==sources
        for rel,h in lock['files'].items():assert r.sha(args.out/rel)==h
        with np.load(args.data,allow_pickle=False) as z:x=z['test_images'].copy();y=z['test_labels'].reshape(-1).copy();ids=z['test_ids'].copy()
        data=(F.interpolate(torch.from_numpy(x[:,None]),size=(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1).cuda(),torch.from_numpy(y).long().cuda(),ids);results=[]
        for rel in lock['models']:
            dest=args.out/rel;ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model=g.build(ck['variant'],cfg);model.load_state_dict(ck['model']);m,rows=g.evaluate(model,data);r.csvwrite(dest/'test_predictions.csv',rows)
            t=r.read(dest/'thresholds.json')['policies']['val_balanced']['threshold'];prob=np.array([[a['score0'],a['score1']] for a in rows]);cal=g.full_metrics(y,prob,t,None);cal.pop('detector_plane_mse')
            results.append(dict(variant=ck['variant']['id'],seed=ck['seed'],test=m,val_threshold=cal));del model,ck;torch.cuda.empty_cache()
        r.save(args.out/'test_results.json',results);r.save(args.out/'test_execution.json',dict(command=sys.argv,sources=sources,time=r.now()));return
    data,val=r.getdata('train'),r.getdata('val')
    r.save(args.out/'frontend_identity.json',dict(parameter_sha256=before,train_encoded_sha256=r.sha_tensor(frontend.encode(data[0])),val_encoded_sha256=r.sha_tensor(frontend.encode(val[0])),feature_parameters=sum(p.numel() for p in frontend.features.parameters()),classifier_head_used_by_optics=False))
    if args.phase=='smoke':
        weights=len(data[1])/(2*torch.bincount(data[1],minlength=2).float());rows=[]
        for v in variants:
            model=g.build(v,cfg);encoded=frontend.encode(data[0][:8]);assert encoded.shape==(8,1,100,100) and 0<=encoded.min() and encoded.max()<=1
            loss,_=g.loss_terms(model,model(encoded),data[1][:8],weights,cfg);loss.backward();grads={n:float(p.grad.norm()) for n,p in model.named_parameters()};assert all(np.isfinite(x) and x>0 for x in grads.values());assert all(p.grad is None for p in frontend.parameters());rows.append(dict(variant=v['id'],gradients=grads));del model;torch.cuda.empty_cache()
        r.save(args.out/'smoke.json',rows);return
    results=[];files={};models=[]
    for seed in args.seeds:
        for v in variants:
            results.append(g.train(v,seed,data,val,cfg,args.out,sources));rel=Path(v['id'])/('seed'+str(seed));models.append(rel.as_posix())
            for name in ['best_checkpoint.pt','thresholds.json']:files[(rel/name).as_posix()]=r.sha(args.out/rel/name)
            assert before=={n:r.sha_tensor(p) for n,p in frontend.state_dict().items()}
            r.save(args.out/'validation_results.json',results)
    r.save(args.out/'test_lock.json',dict(sources=sources,files=files,models=models,selection='minimum validation balanced NLL; EMA weights',time=r.now()));r.save(args.out/'status.json',dict(state='training_complete_test_not_read',frontend_unchanged=True))


if __name__=='__main__':main()
