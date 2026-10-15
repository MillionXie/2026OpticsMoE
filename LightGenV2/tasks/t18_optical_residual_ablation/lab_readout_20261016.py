"""Tiny final-CCD residual readout. TRAIN gradients, TEST-selected development."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from caltech10_sim_head_model import CCDHead,ccd_features


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
    model=CCDHead().net
    assert sum(p.numel() for p in model.parameters())==8554
    return model


def numpy_probability(state,features,base):
    x=np.maximum(features@state['0.weight'].numpy().T+state['0.bias'].numpy(),0)
    logits=x@state['2.weight'].numpy().T+state['2.bias'].numpy()+np.log(np.maximum(base,1e-8))
    logits=logits-logits.max(1,keepdims=True);p=np.exp(logits)
    return p/p.sum(1,keepdims=True)


def require_simulation_handoff(a):
    """No random-initialization fallback after the corrected human instruction."""
    path=a.train.parent/'simulation_head_handoff_20261016.json'
    deadline=time.monotonic()+3000
    if not path.exists():
        status=a.out.parent/'awaiting_simulation_head.json'
        if not status.exists():save(status,dict(state='waiting_for_same_simulation_trained_head',random_head_training_allowed=False,optical_capture_unchanged=True,expected_manifest=str(path)))
        print(json.dumps({'state':'waiting_for_simulation_head','manifest':str(path)}),flush=True)
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError('Simulation head not delivered; no random head was trained')
        time.sleep(10)
    value=json.loads(path.read_text())
    assert value['schema']=='t18_simulation_head_handoff_v1' and value['architecture']==[256,32,10] and value['electronic_parameters']==8554
    assert value['normalization']=='frozen_simulation_TRAIN_mean_std' and value['feature_contract']=='rawCCD_relative_mean_eps1e-6_log1p_pool16'
    assert 1<=value['finetune_epochs']<=20 and 0<float(value['learning_rate'])<=.001
    assert set(value['arms'])=={'residual','control'}
    return path,value


def initialized_head(entry,optical_sha):
    path=Path(entry['checkpoint']);assert sha(path)==entry['sha256']
    assert entry['optical_checkpoint_sha256']==optical_sha
    ck=torch.load(path,map_location='cpu',weights_only=False)
    state=ck[entry['model_key']];prefix=entry.get('state_prefix','')
    model=build_head();mapped={k:state[prefix+k].detach().cpu().clone() for k in model.state_dict()}
    model.load_state_dict(mapped,strict=True)
    normalization=ck[entry['normalization_key']] if entry.get('normalization_key') else ck
    mean=np.asarray(normalization[entry['mean_key']],dtype=np.float32).reshape(256)
    std=np.asarray(normalization[entry['std_key']],dtype=np.float32).reshape(256)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>=.001-1e-7).all()
    return model,mean,std


def load_features(session,arm,role):
    contract=json.loads((session/'contract.json').read_text())
    report=json.loads((session/'report.json').read_text())
    assert contract['role']==role and report['state']=='complete' and report['sdk_released']
    depth=contract.get('main_depth',len(contract['main_stages']));assert depth in (2,3,4)
    n={'train':761,'test':1833}[role]
    assert contract['samples']==n and report['samples_per_arm']==n and report['physical_ccd']==n*2*(depth+1)
    folder=session/arm;rows=np.load(folder/'predictions.npz',allow_pickle=False)
    ids=rows['ids'].copy();labels=rows['labels'].copy();base=rows['probabilities'].copy()
    assert len(ids)==n and len(set(ids.tolist()))==n and base.shape==(n,10)
    final_stage=str(depth-1);ccd=np.load(folder/final_stage/'ccd.npy',mmap_mode='r',allow_pickle=False)
    assert ccd.shape==(n,478,478) and ccd.dtype==np.uint8
    features=[]
    with torch.inference_mode():
        for start in range(0,n,32):
            x=torch.from_numpy(np.asarray(ccd[start:start+32]).copy()).float()
            features.append(ccd_features(x).numpy())
    receipts=sorted((folder/final_stage).glob('[0-9]*.json'))
    assert len(receipts)==n
    for j,path in enumerate(receipts):
        r=json.loads(path.read_text());assert r['index']==j and r['sample_id']==str(ids[j]) and r['arm']==arm and r['stage']==final_stage
    return dict(features=np.concatenate(features),ids=ids,labels=labels,base=base,contract=contract,session_report_sha256=sha(session/'report.json'),predictions_sha256=sha(folder/'predictions.npz'))


def fit(a):
    handoff_path,handoff=require_simulation_handoff(a)
    a.epochs=handoff['finetune_epochs']
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4)
    assert 1<=a.epochs<=300
    assets=getattr(a,'assets',None) or a.train.parent/'assets';data=assets/'caltech10.npz'
    for arm in ('residual','control'):
        folder=a.out/arm;folder.mkdir();started=time.perf_counter()
        train=load_features(a.train,arm,'train');test=load_features(a.test,arm,'test')
        assert sha(data)==train['contract']['data_sha256']
        with np.load(data,allow_pickle=False) as original:
            for role,value in (('train',train),('test',test)):
                assert np.array_equal(value['ids'],original[role+'_ids']) and np.array_equal(value['labels'],original[role+'_labels'])
        assert sha(assets/train['contract'].get('checkpoint_files',{}).get(arm,arm+'.pt'))==train['contract']['checkpoint_sha256'][arm]
        assert not set(train['ids'].tolist())&set(test['ids'].tolist())
        for key in ('coordinates_sha256','checkpoint_sha256','exposure_us','gain','wait_ms','phase_transform','camera_orientation','response_lut_sha256','data_sha256'):
            assert train['contract'][key]==test['contract'][key],key
        model,mean,std=initialized_head(handoff['arms'][arm],train['contract']['checkpoint_sha256'][arm])
        x=torch.from_numpy((train['features']-mean)/std);xt=torch.from_numpy((test['features']-mean)/std)
        y=torch.from_numpy(train['labels'].astype(np.int64));base=torch.from_numpy(train['base']).float().clamp_min(1e-8).log()
        bt=torch.from_numpy(test['base']).float().clamp_min(1e-8).log()
        np.savez_compressed(folder/'features.npz',train_features=train['features'],test_features=test['features'],train_ids=train['ids'],test_ids=test['ids'],train_labels=train['labels'],test_labels=test['labels'],train_base=train['base'],test_base=test['base'],mean=mean,std=std)
        torch.manual_seed(17);rng=torch.Generator().manual_seed(17)
        optimizer=torch.optim.AdamW(model.parameters(),lr=float(handoff['learning_rate']),weight_decay=.001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs,eta_min=min(1e-5,float(handoff['learning_rate'])))
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
            checkpoint=dict(epoch=epoch,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},mean=mean,std=std,train_metrics=tm,test_metrics=te,optimizer_steps=steps,selection='first highest TEST accuracy; development, not independent generalization',optical_weights_frozen=True,electronic_parameters=8554,simulation_head_sha256=handoff['arms'][arm]['sha256'],simulation_normalization_frozen=True,handoff_sha256=sha(handoff_path))
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
        save(folder/'report.json',dict(state='complete',arm=arm,selected_epoch=best_ck['epoch'],epochs_completed=a.epochs,optimizer_steps=steps,train_samples=761,test_samples=1833,val_samples_used=0,selection=best_ck['selection'],kind='same_simulation_head_physical_finetuning',electronic_parameters=8554,optical_weights_frozen=True,raw_RGB_input_used=False,feature_normalization_fit_on='simulation TRAIN; frozen during physical adaptation',simulation_head_sha256=handoff['arms'][arm]['sha256'],handoff_sha256=sha(handoff_path),direct_train=metrics(train['labels'],train['base']),direct_test=metrics(test['labels'],test['base']),audits=audits,source_sha256=sha(__file__),train_session_report_sha256=train['session_report_sha256'],test_session_report_sha256=test['session_report_sha256'],files={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.pt','.npz')},elapsed_s=time.perf_counter()-started))
    save(a.out/'report.json',dict(state='complete',sdk_opened=False,validation_used=False,results={arm:json.loads((a.out/arm/'report.json').read_text()) for arm in ('residual','control')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=300);p.add_argument('--assets',type=Path)
    fit(p.parse_args())
