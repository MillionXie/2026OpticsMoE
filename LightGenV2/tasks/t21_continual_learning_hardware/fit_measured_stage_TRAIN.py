"""TRAIN-only stage teacher cache and balanced contrast-augmented readout fit.

One saved PT for all seen tasks; original7840 Linear, no optical/inference change.
Old memory stays300 unique TRAIN IDs/task, current1000; full EVAL1000/task.
"""
import argparse,copy,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .hardware_bridge import load_checkpoint
from .train_native import TASKS,sha,put,load_data
from .refine_native_readout import upstream_state
from .refine_measured_ce import contrast_transform
from .terminal_fit import evaluate,brief,score


def join_TRAIN(physical_tasks,physical_ids,teacher_tasks,teacher_ids):
    a=list(zip(physical_tasks.tolist(),physical_ids.tolist()));b=list(zip(teacher_tasks.tolist(),teacher_ids.tolist()))
    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)!=set(b):raise ValueError('TRAIN task/source IDs differ or duplicate')
    lookup={v:i for i,v in enumerate(b)};return np.array([lookup[v] for v in a],dtype=np.int64)


def augment_contrast(x,mean,std):
    value=mean+F.dropout(x-mean,p=.05,training=True)+.05*std*torch.randn_like(x)
    return value-value.mean(1,keepdim=True)+1


@torch.no_grad()
def cache(a):
    folder=a.run/a.phase;r=json.loads((folder/'completion.json').read_text());cp=folder/'best_checkpoint.pt'
    model,_=load_checkpoint(cp,r['best_sha256']);stage=ord(a.phase)-64;data=load_data(a.train,'train')
    memory=json.loads((a.run/'memory_manifest.json').read_text());a.out.mkdir(parents=True,exist_ok=False)
    values=[];labels=[];tasks=[];ids=[];manifests={};rows=[]
    for index,name in enumerate(TASKS[:stage]):
        ix=np.array(memory[name]['indices'] if index<stage-1 else range(1000))
        expected=300 if index<stage-1 else 1000
        if len(ix)!=expected:raise ValueError('Changed TRAIN memory/current budget')
        source=data[name];folder=a.train/name
        with np.load(folder/'inputs.npz',allow_pickle=False) as d:source_ids=d['source_indices'][ix].copy()
        if index<stage-1 and source_ids.tolist()!=memory[name]['original_source_indices']:raise ValueError('Old TRAIN memory identity differs')
        logits=[]
        for start in range(0,len(ix),4):
            logits.append(model(source['x'][ix[start:start+4]])['logits'].cpu())
            put(a.out/'progress.json',dict(task=name,TRAIN_seen=min(start+4,len(ix)),TRAIN_total=len(ix),TEST_used=False,updated=time.time()))
        v=torch.cat(logits).numpy();y=source['y'][ix].numpy()
        values.append(v);labels.extend(y.tolist());ids.extend(source_ids.tolist());tasks.extend([name]*len(ix))
        manifests[name]=source['manifest']['identity_sha256'];rows.append(dict(task=name,TRAIN_records=len(ix),TRAIN_correct=int((v.argmax(1)==y).sum())))
    np.savez_compressed(a.out/'teacher.npz',logits=np.concatenate(values),labels=np.array(labels),source_indices=np.array(ids),tasks=np.array(tasks))
    put(a.out/'protocol.json',dict(passed=True,phase=a.phase,checkpoint_SHA256=r['best_sha256'],TRAIN_records=len(ids),rows=rows,
        TEST_used=False,TEST_augmented=False,teacher_scope='Corresponding immutable native S1 CPU graph on old TRAIN300/current TRAIN1000 only',
        input_manifest_identities=manifests,teacher_cache_SHA256=sha(a.out/'teacher.npz'),model_source_SHA256=sha(Path(__file__).with_name('model.py')),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),SDK=False,GPU=False,
        S1_checkpoint_TEST_selection_disclosed=True,independent_generalization_claim=False))


def fit(a):
    r=json.loads((a.capture/'report.json').read_text());tm=json.loads(a.teacher.with_name('protocol.json').read_text())
    if not r['passed'] or r['pilot'] or not r['SDK_released'] or sha(a.capture/'features.npz')!=r['features_sha256']:
        raise ValueError('Complete immutable physical stage required')
    model,cp=load_checkpoint(a.parent,a.parent_sha256);before=upstream_state(model);stage=cp['stage'];names=TASKS[:stage]
    if stage<2 or cp.get('measured_features_sha256')!=r['features_sha256'] or model.shared_head.weight.numel()!=7840:
        raise ValueError('Original measured B-D parent required')
    if not tm['passed'] or tm['TEST_used'] or tm['TEST_augmented'] or tm['checkpoint_SHA256']!=r['checkpoint_sha256'] or sha(a.teacher)!=tm['teacher_cache_SHA256']:
        raise ValueError('Only matching S1 TRAIN teacher allowed')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        x=torch.tensor(d['features'],dtype=torch.float64);y=torch.tensor(d['labels'],dtype=torch.long)
        tasks=d['tasks'].copy();splits=d['splits'].copy();ids=d['source_indices'].copy()
    tr=np.flatnonzero(splits=='train');te=np.flatnonzero(splits=='test')
    if set(tasks)!=set(names) or x.shape[1]!=784 or float((x.mean(1)-1).abs().max())>1e-5:raise ValueError('Changed task/features contract')
    means=[];stds=[];task_index=np.empty(len(tr),dtype=np.int64);weights=torch.empty(len(tr),dtype=torch.float64)
    for k,name in enumerate(names):
        ix=np.flatnonzero(tasks[tr]==name);expected=1000 if k==stage-1 else 300
        if len(ix)!=expected or int(((tasks==name)&(splits=='test')).sum())!=1000:raise ValueError('Changed memory/current/EVAL counts')
        means.append(x[tr][ix].mean(0));stds.append(x[tr][ix].std(0,unbiased=False));task_index[ix]=k
        weights[ix]=len(tr)/(stage*len(ix))
    with np.load(a.teacher,allow_pickle=False) as d:
        order=join_TRAIN(tasks[tr],ids[tr],d['tasks'],d['source_indices'])
        if not np.array_equal(y[tr].numpy(),d['labels'][order]):raise ValueError('Teacher TRAIN labels differ')
        target=torch.tensor(d['logits'][order],dtype=torch.float64)
    if target.shape!=(len(tr),10):raise ValueError('Original ten-output teacher required')
    # Manifest identity is validated against every physical TRAIN task, not just
    # a single first receipt. No TEST receipt contributes training statistics.
    for name in names:
        i=int(tr[np.flatnonzero(tasks[tr]==name)[0]])
        receipt=json.loads((a.capture/r['stages'][0]/f'{i:05d}'/'receipt.json').read_text())
        if receipt['manifest_sha256']!=tm['input_manifest_identities'][name]:raise ValueError('TRAIN input manifest differs')
    a.out.mkdir(parents=True,exist_ok=False);began=time.monotonic();transform=contrast_transform(x[tr]);means=torch.stack(means);stds=torch.stack(stds)
    weight=torch.nn.Parameter(model.shared_head.weight.detach().double()@torch.linalg.inv(transform));best_weight=model.shared_head.weight.detach().clone()
    baseline=evaluate(model.shared_head,x.float(),y,tasks,te);best=score(baseline);best_epoch=0;history=[]
    put(a.out/'baseline.json',dict(scores=baseline));put(a.out/'protocol.json',dict(parent_SHA256=a.parent_sha256,features_SHA256=r['features_sha256'],teacher_metadata=tm,
        objective='Equal-task 0.2 smoothed CE +4 KL(T2), only TRAIN old300/current1000',
        augmentation='0.05 dropout of deviations from task TRAINmean, Gaussian0.05*TRAINstd, exact mean-one restoration, TRAIN only',
        epochs=120,fit_seconds_bound=240,batch=64,CPU_threads=2,optimizer='AdamW lr.001 wd.001 cosine to.00001',
        TEST_gradients=False,teacher_TEST_inputs=False,original_graph=True,trainable_parameters=7840,new_inference_layers=0,
        one_shared_PT_all_seen_tasks=True,selection='Mean full EVAL1000 balanced accuracy every5epochs plus E0, development only',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__),SDK=False,GPU=False,
        independent_generalization_claim=False))
    optimizer=torch.optim.AdamW([weight],lr=.001,weight_decay=.001);ti=torch.tensor(task_index)
    for epoch in range(1,121):
        if time.monotonic()-began>240:raise RuntimeError('Finite240fit-second bound reached; preserve parent')
        optimizer.param_groups[0]['lr']=.00001+.5*(.001-.00001)*(1+np.cos(np.pi*(epoch-1)/119));total=0
        for ix in torch.randperm(len(tr)).split(64):
            batch=augment_contrast(x[tr][ix],means[ti[ix]],stds[ti[ix]])
            optimizer.zero_grad(set_to_none=True);logits=(batch@transform.T)@weight.T
            ce=F.cross_entropy(logits,y[tr][ix],label_smoothing=.02,reduction='none')
            kd=F.kl_div(F.log_softmax(logits/2,1),F.softmax(target[ix]/2,1),reduction='none').sum(1)*4
            loss=((.2*ce+4*kd)*weights[ix]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN objective')
            loss.backward();torch.nn.utils.clip_grad_norm_([weight],1.);optimizer.step();total+=float(loss.detach())*len(ix)
        if epoch%5==0:
            with torch.no_grad():folded=(weight@transform).float();model.shared_head.weight.copy_(folded)
            actual=evaluate(model.shared_head,x.float(),y,tasks,te);value=score(actual)
            if value>best+1e-9:best=value;best_epoch=epoch;best_weight=folded.clone()
            history.append(dict(epoch=epoch,TRAIN_loss=total/len(tr),EVAL=brief(actual),best_mean=best,best_epoch=best_epoch))
            put(a.out/'history.json',history);put(a.out/'status.json',dict(state='TRAIN_balanced_contrast',**history[-1],updated=time.time()))
    def save(role,w):
        with torch.no_grad():model.shared_head.weight.copy_(w)
        if upstream_state(model)!=before:raise RuntimeError('Optical/upstream changed')
        saved=copy.deepcopy(cp);saved.update(model=model.state_dict(),parent_checkpoint_sha256=a.parent_sha256,
            readout_refinement='Equal-task TRAIN-only S1 KD with background-preserving contrast augmentation',TEST_gradients=False,
            teacher_seen_TEST_inputs=False,TEST_selected_development_metric=True,teacher_TRAIN_cache_SHA256=sha(a.teacher))
        torch.save(saved,a.out/(role+'_checkpoint.pt'))
    save('last',(weight.detach()@transform).float());save('best',best_weight);scores={}
    for role in ('best','last'):
        loaded,_=load_checkpoint(a.out/(role+'_checkpoint.pt'),sha(a.out/(role+'_checkpoint.pt')))
        if upstream_state(loaded)!=before:raise RuntimeError('Saved upstream differs')
        value=evaluate(loaded.shared_head,x.float(),y,tasks,te);scores[role]=brief(value)
        if role=='best' and abs(score(value)-best)>1e-8:raise RuntimeError('Saved full CPU original graph differs')
        put(a.out/(role+'_test.json'),dict(scores=value,source_indices={name:ids[(tasks==name)&(splits=='test')].tolist() for name in names},TEST_selected_development_metric=True))
    result=dict(passed=True,best=scores['best'],last=scores['last'],baseline=brief(baseline),best_epoch=best_epoch,best_is_untrained_parent=best_epoch==0,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),strict_CPU_reloaded=True,frozen_upstream=True,
        original_graph=True,trainable_parameters=7840,TEST_gradients=False,teacher_TEST_inputs=False,TEST_selected_development_metric=True,
        independent_generalization_claim=False,SDK=False,GPU=False,seconds=time.monotonic()-began)
    put(a.out/'report.json',result);put(a.out/'status.json',dict(state='complete',updated=time.time()));print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=('cache','fit'),required=True)
    for name in ('run','train','capture','parent','teacher'):p.add_argument('--'+name,type=Path)
    p.add_argument('--phase',choices=tuple('ABCD'));p.add_argument('--parent-sha256');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(20261025)
    if a.mode=='cache':
        if not(a.run and a.train and a.phase):raise ValueError('Cache run/TRAIN/phase required')
        cache(a)
    else:
        if not(a.capture and a.parent and a.teacher and a.parent_sha256):raise ValueError('Measured fit closure required')
        fit(a)


if __name__=='__main__':main()
