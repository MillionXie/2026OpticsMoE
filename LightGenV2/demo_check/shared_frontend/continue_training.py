"""Low-LR paired continuation with frozen features and validation protection."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('continuation_base',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def optimizer_digest(state):
    values={f'{i}.{k}':v for i,s in state['state'].items() for k,v in s.items() if torch.is_tensor(v)}
    return r.tensors_sha(values)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','train','evaluate'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run',type=Path)
    p.add_argument('--profile',type=Path,default=HERE/'continuation.json');a=p.parse_args()
    profile=json.loads(a.profile.read_text());source=json.loads((a.source/'metadata.json').read_text());cfg=source['config'];ocfg=source['optical_config']
    assert profile['resume_epoch']==cfg['epochs']==20
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.manual_seed(cfg['seed'])
    assert r.sha(a.data)==source['data_sha256']
    with np.load(a.data,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    assert not any(k.startswith('test') for k in arrays)
    train=tuple(torch.from_numpy(arrays['train_'+k]) for k in ['images','labels','domains'])
    val=tuple(torch.from_numpy(arrays['validation_'+k]) for k in ['images','labels','domains'])
    frontend_path=a.source/'frontend/best_checkpoint.pt'
    frontend=r.SharedFrontend(torch.load(frontend_path,map_location='cpu',weights_only=False)['model']).cuda()
    frozen=r.tensors_sha(frontend.state_dict())
    metadata=dict(config=cfg,optical_config=ocfg,profile=profile,source_run=str(a.source),source_training_commit=source['git_commit'],
        command=sys.argv,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        data_sha256=r.sha(a.data),data_manifest_sha256=source['data_manifest_sha256'],split_sha256=source['split_sha256'],
        frontend_checkpoint_sha256=r.sha(frontend_path),python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),
        source_sha256={str(f.relative_to(r.TASK)):r.sha(f) for f in [HERE/'continue_training.py',a.profile,HERE/'run.py',HERE/'model.py',r.TASK/'pure_optical/models.py',r.TASK/'pure_optical/run.py',r.TASK/'frozen_electronic/model.py',r.base.ARCHIVE/'optical_reference/optics.py']})
    r.save(a.out/'metadata.json',metadata);summaries=[];paired=[]
    if a.phase=='train':
        (a.out/'frontend').mkdir();shutil.copyfile(frontend_path,a.out/'frontend/best_checkpoint.pt')
    for architecture in cfg['architectures']:
        model=r.FrontendOptics(frontend,architecture,ocfg).cuda()
        if a.phase=='evaluate':
            summary=json.loads((a.run/architecture/'summary.json').read_text());path=a.run/architecture/'best_checkpoint.pt'
            assert r.sha(path)==summary['checkpoint_sha256']
            ck=torch.load(path,map_location='cpu',weights_only=False);model.optical.load_state_dict(ck['model'])
            metrics,prob=r.base.evaluate(model,val,cfg);train_metrics,_=r.base.evaluate(model,train,cfg)
            assert metrics==summary['validation'] and train_metrics==summary['train_unaugmented']
            with (a.run/architecture/'validation_predictions.csv').open() as f:rows=list(csv.DictReader(f))
            assert [x['sample_id'] for x in rows]==arrays['validation_ids'].tolist()
            assert np.array_equal(np.array([[float(x[f'p{i}']) for i in range(10)] for x in rows],dtype=np.float32),prob)
            assert frozen==r.tensors_sha(frontend.state_dict())
            summaries.append(dict(architecture=architecture,predictions_bitwise_equal=True,train_and_validation_metrics_equal=True));del model;continue
        path=a.source/architecture/'last_checkpoint.pt';source_hash=r.sha(path)
        last=torch.load(path,map_location='cpu',weights_only=False);assert last['epoch']==20
        model.optical.load_state_dict(last['model'])
        assert last['frontend_tensors_sha256']==frozen
        optimizer=torch.optim.Adam(model.optical.parameters(),lr=profile['learning_rate'],weight_decay=0.)
        optimizer.load_state_dict(last['optimizer'])
        assert optimizer_digest(optimizer.state_dict())==optimizer_digest(last['optimizer'])
        steps={int(x['step']) for x in optimizer.state.values()};assert steps=={20*math.ceil(len(train[1])/cfg['batch_size'])}
        resumed_validation,_=r.base.evaluate(model,val,cfg);assert resumed_validation==last['validation']
        resumed_train,_=r.base.evaluate(model,train,cfg)
        resume=dict(epoch=20,source_checkpoint_sha256=source_hash,optimizer_tensor_sha256=optimizer_digest(last['optimizer']),
                    optimizer_steps=list(steps),validation=resumed_validation,train_unaugmented=resumed_train,
                    phase_tensors_sha256=r.tensors_sha(model.optical.state_dict()))
        if a.phase=='smoke':
            images,labels,_,_=next(r.batches(*train,torch.arange(32),32))
            model.train();optimizer.zero_grad(set_to_none=True);loss=r.objective(model(images),labels);loss.backward()
            gradients={n:float(v.grad.norm()) for n,v in model.optical.named_parameters()}
            assert all(np.isfinite(v) and v>0 for v in gradients.values())
            optimizer.step();assert {int(x['step']) for x in optimizer.state.values()}=={next(iter(steps))+1}
            assert all(p.grad is None for p in frontend.parameters()) and frozen==r.tensors_sha(frontend.state_dict())
            assert r.sha(path)==source_hash
            summaries.append(dict(architecture=architecture,resume=resume,gradients=gradients,passed=True));del model,optimizer;continue
        dest=a.out/architecture;dest.mkdir();r.save(dest/'resume.json',resume)
        previous=torch.load(a.source/architecture/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        shutil.copyfile(a.source/architecture/'best_checkpoint.pt',dest/'best_checkpoint.pt')
        best=(previous['validation']['accuracy'],-previous['validation']['loss']);nll_ceiling=previous['validation']['loss']
        old_history=json.loads((a.source/architecture/'history.json').read_text())
        monitor_best=min(x['validation']['loss'] for x in old_history);stale=0;history=[];started=time.perf_counter();reason='maximum_epoch'
        for epoch in range(21,profile['maximum_epoch']+1):
            progress=(epoch-21)/max(1,profile['maximum_epoch']-21)
            lr=profile['minimum_learning_rate']+(profile['learning_rate']-profile['minimum_learning_rate'])*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:group['lr']=lr
            model.train();total=0.;correct=0;n=0;features=hashlib.sha256()
            order=torch.randperm(len(train[1]),generator=torch.Generator().manual_seed(cfg['seed']*1000003+epoch));aug_seed=cfg['seed']*99991+epoch
            for images,labels,_,_ in r.batches(*train,order,cfg['batch_size'],augment_seed=aug_seed):
                optimizer.zero_grad(set_to_none=True);output=model(images);loss=r.objective(output,labels)
                assert bool(torch.isfinite(loss));loss.backward();assert all(p.grad is None for p in frontend.parameters())
                torch.nn.utils.clip_grad_norm_(model.optical.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=float(loss)*len(labels);correct+=int((output['probabilities'].argmax(1)==labels).sum());n+=len(labels)
                features.update(output['features'].cpu().numpy().tobytes())
            validation,_=r.base.evaluate(model,val,cfg);train_metrics,_=r.base.evaluate(model,train,cfg)
            assert frozen==r.tensors_sha(frontend.state_dict())
            key=(validation['accuracy'],-validation['loss']);eligible=validation['loss']<=nll_ceiling
            improved=eligible and key>best
            if validation['loss']<monitor_best-profile['early_stopping_min_nll_improvement']:monitor_best=validation['loss'];stale=0
            else:stale+=1
            row=dict(epoch=epoch,train_online_loss=total/n,train_online_accuracy=correct/n,train_unaugmented=train_metrics,
                     validation=validation,accuracy_gap=train_metrics['accuracy']-validation['accuracy'],
                     nll_gap=validation['loss']-train_metrics['loss'],learning_rate=lr,
                     order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),augmentation_seed=aug_seed,
                     input_feature_sha256=features.hexdigest(),frontend_frozen_verified=True,
                     eligible_by_nll=eligible,selected_improvement=improved,nll_stale_epochs=stale,seconds=time.perf_counter()-started)
            history.append(row)
            payload=dict(model=model.optical.state_dict(),epoch=epoch,architecture=architecture,config=cfg,optical_config=ocfg,profile=profile,
                         validation=validation,train_unaugmented=train_metrics,frontend_tensors_sha256=frozen)
            if improved:best=key;torch.save(payload,dest/'best_checkpoint.pt')
            torch.save(dict(**payload,optimizer=optimizer.state_dict(),monitor_best=monitor_best,nll_stale_epochs=stale),dest/'last_checkpoint.pt')
            r.save(dest/'history.json',history);r.save(a.out/'status.json',dict(state='training',architecture=architecture,epoch=epoch))
            print(json.dumps(dict(architecture=architecture,epoch=epoch,train=train_metrics['accuracy'],validation=validation['accuracy'],
                                 train_nll=train_metrics['loss'],validation_nll=validation['loss'],stale=stale,seconds=row['seconds'])),flush=True)
            if stale>=profile['early_stopping_patience']:reason='validation_nll_early_stopping';break
        selected=torch.load(dest/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.optical.load_state_dict(selected['model'])
        metrics,prob=r.base.evaluate(model,val,cfg);train_metrics,_=r.base.evaluate(model,train,cfg)
        r.write_predictions(dest/'validation_predictions.csv',arrays,prob)
        summary=dict(architecture=architecture,selected_epoch=selected['epoch'],stop_epoch=epoch,stop_reason=reason,
                     validation=metrics,train_unaugmented=train_metrics,accuracy_gap=train_metrics['accuracy']-metrics['accuracy'],
                     previous_best_validation=previous['validation'],nll_selection_ceiling=nll_ceiling,
                     checkpoint_sha256=r.sha(dest/'best_checkpoint.pt'),resume_checkpoint_sha256=source_hash,
                     frontend_tensors_sha256=frozen,epochs_completed=len(history),seconds=time.perf_counter()-started)
        assert (metrics['accuracy'],-metrics['loss'])>=(previous['validation']['accuracy'],-previous['validation']['loss'])
        assert metrics['loss']<=nll_ceiling and frozen==r.tensors_sha(frontend.state_dict())
        r.save(dest/'summary.json',summary);summaries.append(summary);r.save(a.out/'results.json',summaries)
        paired.append({x['epoch']:(x['order_sha256'],x['augmentation_seed'],x['input_feature_sha256']) for x in history})
        del model,optimizer,last,previous,selected,output,loss;torch.cuda.empty_cache()
    if a.phase=='train':
        overlap=set(paired[0])&set(paired[1]);assert all(paired[0][e]==paired[1][e] for e in overlap)
        r.save(a.out/'fairness.json',dict(same_frontend=True,matching_features_on_common_epochs=True,common_epochs=sorted(overlap),
                                        same_maximum_budget_and_early_stopping_rule=True,actual_epochs=[len(x) for x in paired]))
    else:r.save(a.out/(a.phase+'.json'),summaries)
    r.save(a.out/'status.json',dict(state='complete',phase=a.phase))


if __name__=='__main__':main()
