"""Full-split eight-class pilot; archived optical propagation is unchanged.

Train reads train/val only. Test requires source/data/weight and validation locks.
RGB tiling is fixed for both optical architectures, never a trainable classifier.
"""
import argparse,copy,gc,hashlib,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from train_adrenal_regularized import r,torch,np,F,affine_parameters,augment,phase_smoothness,ARCHIVE,TASK
from models import OpticalMoE,RelayFanoutMoE,D2NN
from optical_reference.optics import DetectorArray
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,roc_auc_score


class BloodMoE(OpticalMoE):
    def __init__(self,cfg):
        torch.nn.Module.__init__(self);self.net=RelayFanoutMoE(cfg,num_classes=8)


class TinyCNN(torch.nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.features=torch.nn.Sequential(torch.nn.Conv2d(3,16,3,padding=1),torch.nn.ReLU(),torch.nn.AvgPool2d(2),torch.nn.Conv2d(16,32,3,padding=1),torch.nn.ReLU(),torch.nn.AvgPool2d(2),torch.nn.AvgPool2d(2))
        self.head=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(dropout),torch.nn.Linear(288,8))
    def forward(self,x):return self.head(self.features(x))


def build(arch,depth,cfg):
    if arch=='cnn':return TinyCNN(cfg['cnn']['dropout']).cuda()
    c=copy.deepcopy(r.CONFIGS[f'{arch}_L{depth}_oeo_relu_softsign']);d=cfg['detector']
    detector=DetectorArray(8,500,d['size'],'fixed_2x2',True,start_pos_x=d['x'],start_pos_y=d['y'],n_det_sets=d['rows'],det_steps_x=d['gap_x'],det_steps_y=d['gap_y'])
    if arch=='moe':
        c['detector'].update(start_pos_x=d['x'],start_pos_y=d['y'],N_det_sets=d['rows'],det_steps_x=d['gap_x'],det_steps_y=d['gap_y'],detector_size=d['size'])
        m=BloodMoE(c);assert torch.equal(m.masks,detector.masks)
    else:m=D2NN(c);m.detector=detector
    assert m.masks.shape==(8,500,500) and float(m.masks.sum(0).max())==1
    with torch.no_grad():
        for name,p in m.named_parameters():
            assert name.endswith('raw_phase');p.uniform_(-cfg['phase_init_raw_uniform'],cfg['phase_init_raw_uniform'])
    return m.cuda()


def load_data(path,split):
    with np.load(path,allow_pickle=False) as z:x=z[split+'_images'].copy();y=z[split+'_labels'].reshape(-1).copy()
    assert x.dtype==np.uint8 and x.shape==(len(y),28,28,3)
    assert len(y)=={'train':11959,'val':1712,'test':3421}[split] and set(y)==set(range(8))
    return torch.from_numpy(x.transpose(0,3,1,2).copy()).float().cuda()/255,torch.from_numpy(y).long().cuda(),np.array([f'{split}_{i}' for i in range(len(y))])


def encode(x,theta=None,cnn=False):
    if theta is not None or not cnn:
        x=F.interpolate(x,size=(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    if theta is not None:x=augment(x,theta)
    if cnn:return F.interpolate(x,size=(28,28),mode='bilinear',align_corners=False,antialias=True) if theta is not None else x
    x=F.interpolate(x,size=(50,50),mode='bilinear',align_corners=False,antialias=True)
    return torch.cat([torch.cat([x[:,0:1],x[:,1:2]],3),torch.cat([x[:,2:3],x.mean(1,keepdim=True)],3)],2)


def forward(model,x,arch):
    if arch=='cnn':return model(x).softmax(1),None,None
    out=model(x);e=out['energies'];p=(e+1e-12)/(e.sum(1,keepdim=True)+8e-12)
    intensity=out['intensity'];capture=(intensity*model.masks.sum(0)).sum((-1,-2))/intensity.sum((-1,-2)).clamp_min(1e-12)
    return p,capture,out


def metrics(y,p):
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=list(range(8)));rec=np.diag(cm)/cm.sum(1);nll=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12))
    return dict(n=len(y),support=np.bincount(y,minlength=8).tolist(),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(rec.mean()),macro_f1=float(f1_score(y,pred,average='macro')),macro_ovr_auroc=float(roc_auc_score(y,p,multi_class='ovr',average='macro')),recall=rec.tolist(),confusion_matrix=cm.tolist(),nll=float(nll.mean()),balanced_nll=float(np.mean([nll[y==k].mean() for k in range(8)])))


@torch.no_grad()
def evaluate(model,data,arch,batch):
    model.eval();probs=[];captures=[]
    for idx in torch.arange(len(data[1])).split(batch):
        p,c,_=forward(model,encode(data[0][idx],cnn=arch=='cnn'),arch);probs.append(p.cpu().numpy())
        if c is not None:captures.append(c.cpu().numpy())
    p=np.concatenate(probs);y=data[1].cpu().numpy();m=metrics(y,p)
    if captures:m['detector_capture']=float(np.concatenate(captures).mean())
    rows=[dict(sample_id=str(sid),label_true=int(label),label_pred=int(q.argmax()),**{f'score{k}':float(q[k]) for k in range(8)}) for sid,label,q in zip(data[2],y,p)]
    return m,rows


def sources(profile):
    files=[Path(__file__),Path(__file__).with_name('train_adrenal_regularized.py'),profile]+[p for p in ARCHIVE.rglob('*') if p.suffix in ['.py','.yaml']]
    return {str(p.relative_to(TASK)).replace('\\','/'):r.sha(p) for p in files}


def train(arch,depth,seed,data,val,cfg,out,src):
    name=f'{arch}_L{depth}_seed{seed}';dest=out/name;dest.mkdir();r.setseed(seed);model=build(arch,depth,cfg);ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    initial={n:p.detach().clone() for n,p in model.named_parameters()};batch=cfg['cnn']['batch_size'] if arch=='cnn' else cfg['batch_size'];lr=cfg['cnn']['lr'] if arch=='cnn' else cfg['lr'];epochs=cfg['cnn']['epochs'] if arch=='cnn' else cfg['epochs']
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg['cnn']['weight_decay'] if arch=='cnn' else 0)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=lr*.1);weights=len(data[1])/(8*torch.bincount(data[1],minlength=8).float())
    best=float('inf');wait=0;history=[];grads=[];orders=[];transforms=[];started=time.time();r.save(dest/'initial_validation.json',evaluate(model,val,arch,batch)[0])
    for epoch in range(1,epochs+1):
        model.train();order=r.epoch_order(seed,epoch,len(data[1]));theta=affine_parameters(len(data[1]),seed,epoch,cfg['augmentation']);orders.append(r.sha_tensor(order));transforms.append(r.sha_tensor(theta));total=0
        for b,idx in enumerate(order.split(batch)):
            y=data[1][idx];x=encode(data[0][idx],theta[idx],cnn=arch=='cnn');opt.zero_grad(set_to_none=True);p,c,_=forward(model,x,arch)
            target=F.one_hot(y,8)*(1-cfg['label_smoothing'])+cfg['label_smoothing']/8
            loss=(-(target*p.clamp_min(1e-12).log()).sum(1)*weights[y]).mean()
            if arch!='cnn':loss=loss-cfg['capture_weight']*c.clamp_min(1e-12).log().mean()+cfg['phase_smooth_weight']*phase_smoothness(model)
            assert torch.isfinite(loss);loss.backward()
            if b==0:
                norms={n:float(q.grad.norm()) for n,q in model.named_parameters()};assert all(np.isfinite(v) for v in norms.values()) and any(v>0 for v in norms.values()),(name,epoch,norms);grads.append(dict(epoch=epoch,norms=norms))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            with torch.no_grad():
                for a,bp in zip(ema.parameters(),model.parameters()):a.lerp_(bp,1-cfg['ema_decay'])
            total+=float(loss.detach())*len(idx)
        vm,_=evaluate(ema,val,arch,batch);h=dict(epoch=epoch,train_loss=total/len(data[1]),val=vm,lr=opt.param_groups[0]['lr'])
        if epoch==1 or epoch%5==0:h['train']=evaluate(ema,data,arch,batch)[0]
        history.append(h)
        if vm['balanced_nll']<best-cfg['min_delta']:
            best=vm['balanced_nll'];wait=0;r.save_torch(dest/'best_checkpoint.pt',dict(model=ema.state_dict(),epoch=epoch,arch=arch,depth=depth,seed=seed,sources=src,config=cfg))
        else:wait+=1
        scheduler.step();r.save(dest/'history.json',history);r.save(dest/'gradients.json',grads)
        r.save_torch(dest/'last_checkpoint.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,arch=arch,depth=depth,seed=seed,sources=src,config=cfg))
        status=dict(state='training',model=name,epoch=epoch,val_accuracy=vm['accuracy'],val_balanced_accuracy=vm['balanced_accuracy'],val_balanced_nll=vm['balanced_nll'],seconds=time.time()-started);r.save(out/'status.json',status);print(json.dumps(status),flush=True)
        if epoch>=cfg['minimum_epochs'] and wait>=cfg['patience']:break
    updates={n:float((q-initial[n]).square().mean().sqrt()) for n,q in model.named_parameters()};assert all(v>0 for v in updates.values())
    ck=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);ema.load_state_dict(ck['model']);measure={}
    for split,d in [('train',data),('val',val)]:measure[split],rows=evaluate(ema,d,arch,batch);r.csvwrite(dest/(split+'_predictions.csv'),rows)
    result=dict(name=name,arch=arch,depth=depth,seed=seed,selected_epoch=ck['epoch'],epochs_completed=epoch,parameters=sum(p.numel() for p in model.parameters()),metrics=measure,orders=orders,transforms=transforms,updates=updates,seconds=time.time()-started,checkpoint_sha256=r.sha(dest/'best_checkpoint.pt'))
    r.save(dest/'summary.json',result);del model,ema,opt,ck,initial;gc.collect();torch.cuda.empty_cache();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','smoke','train','test'],required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--profile',type=Path,default=Path(__file__).with_name('bloodmnist_profile.json'));p.add_argument('--out',type=Path,required=True);p.add_argument('--depths',type=int,nargs='+',default=[2,4,6]);p.add_argument('--seed',type=int,default=17);a=p.parse_args();cfg=r.read(a.profile)
    if a.phase=='prepare':
        a.out.mkdir(parents=True,exist_ok=False);a.data.parent.mkdir(parents=True,exist_ok=True)
        if not a.data.exists():
            tmp=a.data.with_suffix('.download');urllib.request.urlretrieve(cfg['url'],tmp);assert hashlib.md5(tmp.read_bytes()).hexdigest()==cfg['md5'];tmp.rename(a.data)
        assert hashlib.md5(a.data.read_bytes()).hexdigest()==cfg['md5'];r.save(a.out/'download.json',dict(url=cfg['url'],license=cfg['license'],md5=cfg['md5'],sha256=r.sha(a.data),bytes=a.data.stat().st_size,time=r.now(),original_source='https://data.mendeley.com/datasets/snkd93bnjr/1',test_arrays_read=False));return
    assert hashlib.md5(a.data.read_bytes()).hexdigest()==cfg['md5'];torch.set_num_threads(4);r.setseed(a.seed);src=sources(a.profile)
    if a.phase=='test':
        lock=r.read(a.out/'test_lock.json');assert lock['sources']==src and lock['data_sha256']==r.sha(a.data);assert r.sha(a.out/'validation_results.json')==lock['validation_sha256']
        for x in lock['models']:assert r.sha(a.out/x['name']/'best_checkpoint.pt')==x['checkpoint_sha256']
        # Replay every selected validation checkpoint before opening test arrays.
        val=load_data(a.data,'val');replay=[]
        for x in lock['models']:
            model=build(x['arch'],x['depth'],cfg);ck=torch.load(a.out/x['name']/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);vm,_=evaluate(model,val,x['arch'],cfg['cnn']['batch_size'] if x['arch']=='cnn' else cfg['batch_size']);assert vm==r.read(a.out/x['name']/'summary.json')['metrics']['val'];replay.append(x['name']);del model,ck;torch.cuda.empty_cache()
        test=load_data(a.data,'test');results=[]
        for x in lock['models']:
            model=build(x['arch'],x['depth'],cfg);ck=torch.load(a.out/x['name']/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);m,rows=evaluate(model,test,x['arch'],cfg['cnn']['batch_size'] if x['arch']=='cnn' else cfg['batch_size']);r.csvwrite(a.out/x['name']/'test_predictions.csv',rows);results.append(dict(name=x['name'],metrics=m));del model,ck;torch.cuda.empty_cache()
        with np.load(a.data,allow_pickle=False) as z:
            hashes={split:{hashlib.sha256(img.tobytes()).hexdigest() for img in z[split+'_images']} for split in ['train','val','test']}
        r.save(a.out/'image_overlap_audit.json',dict(exact_rgb_overlap={s+'_'+t:len(hashes[s]&hashes[t]) for s,t in [('train','val'),('train','test'),('val','test')]},scope='Exact images only; no patient identity inference'))
        r.save(a.out/'test_results.json',results);r.save(a.out/'test_execution.json',dict(command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),lock_sha256=r.sha(a.out/'test_lock.json'),validation_replayed=replay,time=r.now()));r.save(a.out/'status.json',dict(state='test_complete',time=r.now()));return
    a.out.mkdir(parents=True,exist_ok=False);data=load_data(a.data,'train');val=load_data(a.data,'val');majority=int(data[1].bincount().argmax())
    r.save(a.out/'metadata.json',dict(command=sys.argv,config=cfg,sources=src,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),data_sha256=r.sha(a.data),python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),time=r.now(),train_support=data[1].bincount().tolist(),val_support=val[1].bincount().tolist(),majority_class=majority,majority_validation_accuracy=float((val[1]==majority).float().mean()),test_read=False))
    variants=[('cnn',0)]+[(arch,d) for d in a.depths for arch in ['moe','d2nn']]
    if a.phase=='smoke':
        rows=[]
        for arch,d in variants:
            model=build(arch,d,cfg);x=encode(data[0][:8],cnn=arch=='cnn');prob,c,_=forward(model,x,arch);loss=-prob[torch.arange(8),data[1][:8]].log().mean()
            if c is not None:loss=loss-.2*c.log().mean()
            loss.backward();g={n:float(q.grad.norm()) for n,q in model.named_parameters()};assert all(np.isfinite(z) and z>0 for z in g.values()),(arch,d,g);rows.append(dict(arch=arch,depth=d,input_shape=list(x.shape),output_shape=list(prob.shape),parameters=sum(q.numel() for q in model.parameters()),gradients=g));r.save(a.out/'smoke.json',rows);del model;torch.cuda.empty_cache()
        r.save(a.out/'smoke.json',rows);return
    results=[]
    for arch,d in variants:
        results.append(train(arch,d,a.seed,data,val,cfg,a.out,src));r.save(a.out/'validation_results.json',results)
    r.save(a.out/'test_lock.json',dict(sources=src,data_sha256=r.sha(a.data),validation_sha256=r.sha(a.out/'validation_results.json'),models=[{k:x[k] for k in ['name','arch','depth','seed','checkpoint_sha256']} for x in results],time=r.now(),selection=cfg['selection']))
    r.save(a.out/'status.json',dict(state='training_complete_test_not_read',time=r.now()))


if __name__=='__main__':main()
