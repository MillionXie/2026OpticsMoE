"""Tiny final-CCD residual readout. TRAIN gradients, TEST-selected development."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    # Immutable reports/history snapshots avoid Windows reader replacement locks.
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def metrics(labels,probability):
    predicted=probability.argmax(1);cm=np.zeros((10,10),dtype=np.int64)
    np.add.at(cm,(labels,predicted),1);support=cm.sum(1)
    recall=np.divide(cm.diagonal(),support,out=np.zeros(10),where=support>0)
    return dict(samples=len(labels),correct=int((predicted==labels).sum()),accuracy=float((predicted==labels).mean()),macro_recall=float(recall[support>0].mean()),support=support.tolist(),confusion_matrix=cm.tolist())


def build_head():
    model=nn.Sequential(nn.Linear(256,32),nn.ReLU(),nn.Linear(32,10))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    assert sum(p.numel() for p in model.parameters())==8554
    return model


def numpy_probability(state,features,base):
    x=np.maximum(features@state['0.weight'].numpy().T+state['0.bias'].numpy(),0)
    logits=x@state['2.weight'].numpy().T+state['2.bias'].numpy()+np.log(np.maximum(base,1e-8))
    logits=logits-logits.max(1,keepdims=True);p=np.exp(logits)
    return p/p.sum(1,keepdims=True)


def load_features(session,arm,role):
    contract=json.loads((session/'contract.json').read_text())
    report=json.loads((session/'report.json').read_text())
    assert contract['role']==role and report['state']=='complete' and report['sdk_released']
    n={'train':761,'test':1833}[role]
    assert contract['samples']==n and report['samples_per_arm']==n and report['physical_ccd']==n*8
    folder=session/arm;rows=np.load(folder/'predictions.npz',allow_pickle=False)
    ids=rows['ids'].copy();labels=rows['labels'].copy();base=rows['probabilities'].copy()
    assert len(ids)==n and len(set(ids.tolist()))==n and base.shape==(n,10)
    ccd=np.load(folder/'2/ccd.npy',mmap_mode='r',allow_pickle=False)
    assert ccd.shape==(n,478,478) and ccd.dtype==np.uint8
    features=[]
    with torch.inference_mode():
        for start in range(0,n,32):
            x=torch.from_numpy(np.asarray(ccd[start:start+32]).copy()).float()
            x=torch.log1p(x/x.mean((-2,-1),keepdim=True).clamp_min(1e-6))
            features.append(F.adaptive_avg_pool2d(x[:,None],(16,16)).flatten(1).numpy())
    receipts=sorted((folder/'2').glob('[0-9]*.json'))
    assert len(receipts)==n
    for j,path in enumerate(receipts):
        r=json.loads(path.read_text());assert r['index']==j and r['sample_id']==str(ids[j]) and r['arm']==arm and r['stage']=='2'
    return dict(features=np.concatenate(features),ids=ids,labels=labels,base=base,contract=contract,session_report_sha256=sha(session/'report.json'),predictions_sha256=sha(folder/'predictions.npz'))


def fit(a):
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4)
    assert 1<=a.epochs<=300
    assets=a.train.parent/'assets';data=assets/'caltech10.npz'
    for arm in ('residual','control'):
        folder=a.out/arm;folder.mkdir();started=time.perf_counter()
        train=load_features(a.train,arm,'train');test=load_features(a.test,arm,'test')
        assert sha(data)==train['contract']['data_sha256']
        with np.load(data,allow_pickle=False) as original:
            for role,value in (('train',train),('test',test)):
                assert np.array_equal(value['ids'],original[role+'_ids']) and np.array_equal(value['labels'],original[role+'_labels'])
        assert sha(assets/(arm+'.pt'))==train['contract']['checkpoint_sha256'][arm]
        assert not set(train['ids'].tolist())&set(test['ids'].tolist())
        for key in ('coordinates_sha256','checkpoint_sha256','exposure_us','gain','wait_ms','phase_transform','camera_orientation','response_lut_sha256','data_sha256'):
            assert train['contract'][key]==test['contract'][key],key
        mean=train['features'].mean(0,dtype=np.float64).astype(np.float32)
        std=train['features'].std(0,dtype=np.float64).astype(np.float32).clip(.001)
        x=torch.from_numpy((train['features']-mean)/std);xt=torch.from_numpy((test['features']-mean)/std)
        y=torch.from_numpy(train['labels'].astype(np.int64));base=torch.from_numpy(train['base']).float().clamp_min(1e-8).log()
        bt=torch.from_numpy(test['base']).float().clamp_min(1e-8).log()
        np.savez_compressed(folder/'features.npz',train_features=train['features'],test_features=test['features'],train_ids=train['ids'],test_ids=test['ids'],train_labels=train['labels'],test_labels=test['labels'],train_base=train['base'],test_base=test['base'],mean=mean,std=std)
        torch.manual_seed(17);rng=torch.Generator().manual_seed(17);model=build_head()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=1e-5)
        history=[];best=-1;steps=0
        for epoch in range(a.epochs+1):
            loss_sum=0.
            if epoch:
                model.train();order=torch.randperm(len(y),generator=rng)
                for ix in order.split(64):
                    optimizer.zero_grad(set_to_none=True);logits=base[ix]+model(x[ix]);loss=F.cross_entropy(logits,y[ix])
                    assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.)
                    optimizer.step();steps+=1;loss_sum+=float(loss.detach())*len(ix)
                schedule.step()
            model.eval()
            with torch.inference_mode():
                p=(base+model(x)).softmax(1).numpy();pt=(bt+model(xt)).softmax(1).numpy()
            tm=metrics(train['labels'],p);te=metrics(test['labels'],pt)
            row=dict(epoch=epoch,train_loss=loss_sum/len(y) if epoch else None,train_accuracy=tm['accuracy'],test_accuracy=te['accuracy'],test_correct=te['correct'],optimizer_steps=steps,elapsed_s=time.perf_counter()-started)
            history.append(row)
            checkpoint=dict(epoch=epoch,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},mean=mean,std=std,train_metrics=tm,test_metrics=te,optimizer_steps=steps,selection='first highest TEST accuracy; development, not independent generalization',optical_weights_frozen=True,electronic_parameters=8554)
            if te['correct']>best:
                best=te['correct'];torch.save(checkpoint,folder/'best_checkpoint.pt')
            if epoch==a.epochs:torch.save(checkpoint,folder/'last_checkpoint.pt')
            if epoch%10==0 or epoch==a.epochs:
                save(folder/f'history_e{epoch:03d}.json',history)
                print(json.dumps(dict(arm=arm,**row)),flush=True)
        save(folder/'history.json',history)
        audits={}
        for kind in ('best','last'):
            ck=torch.load(folder/(kind+'_checkpoint.pt'),map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
            with torch.inference_mode():tp=(base+model(x)).softmax(1).numpy();vp=(bt+model(xt)).softmax(1).numpy()
            for split,features,baseline,labels,prob in (('train',x.numpy(),train['base'],train['labels'],tp),('test',xt.numpy(),test['base'],test['labels'],vp)):
                independent=numpy_probability(ck['model'],features,baseline)
                error=float(np.max(np.abs(prob-independent)));assert error<2e-5
                assert metrics(labels,independent)['correct']==ck[split+'_metrics']['correct']
                audits[kind+'_'+split]=dict(numpy_probability_max_error=error,metrics=metrics(labels,prob))
            np.savez_compressed(folder/(kind+'_predictions.npz'),train_ids=train['ids'],test_ids=test['ids'],train_labels=train['labels'],test_labels=test['labels'],train_probability=tp,test_probability=vp)
        best_ck=torch.load(folder/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        save(folder/'report.json',dict(state='complete',arm=arm,selected_epoch=best_ck['epoch'],epochs_completed=a.epochs,optimizer_steps=steps,train_samples=761,test_samples=1833,val_samples_used=0,selection=best_ck['selection'],kind='hybrid_fixed_optics_plus_final_CCD_residual_readout',electronic_parameters=8554,optical_weights_frozen=True,raw_RGB_input_used=False,feature_normalization_fit_on='TRAIN only',direct_train=metrics(train['labels'],train['base']),direct_test=metrics(test['labels'],test['base']),audits=audits,source_sha256=sha(__file__),train_session_report_sha256=train['session_report_sha256'],test_session_report_sha256=test['session_report_sha256'],files={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.pt','.npz')},elapsed_s=time.perf_counter()-started))
    save(a.out/'report.json',dict(state='complete',sdk_opened=False,validation_used=False,results={arm:json.loads((a.out/arm/'report.json').read_text()) for arm in ('residual','control')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=300)
    fit(p.parse_args())
