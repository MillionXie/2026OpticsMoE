"""Adapt only the EXISTING 5140-parameter Linear256->20 to frozen CCD features.

Full selected TRAIN1000; no VAL. TEST411 selects the checkpoint and is explicitly
a development metric, never an untouched generalization estimate. No new head,
normalizer, optical forward, early fusion update, SDK or GPU is introduced.
"""
import argparse,json,platform,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
from lab_caltech20_rich_model_20261018 import sha
from lab_caltech20_rich_capture_20261018 import metrics
from lab_caltech10_20261015 import save


def existing_head(state):
    assert state['linear.weight'].shape==(20,256) and state['linear.bias'].shape==(20,)
    head=nn.Linear(256,20);head.load_state_dict({k:state['linear.'+k] for k in ('weight','bias')},strict=True)
    assert sum(p.numel() for p in head.parameters())==5140
    return head


def load_features(a):
    release=json.loads(a.release.read_text());report=json.loads((a.capture/'report.json').read_text())
    assert release['capture_released'] and report['state']=='complete' and report['sdk_released']
    assert not report['pilot'] and report['physical_ccd']==4233 and report['arm']==a.arm
    assert report['release_sha256']==sha(a.release)
    spec=next(s for s in release['models'] if s['arm']==a.arm)
    assert sha(a.assets/spec['file'])==spec['sha256']
    saved=torch.load(a.assets/spec['file'],map_location='cpu',weights_only=False);state=saved['model'];head=existing_head(state)
    manifest=json.loads((a.assets/'data_manifest.json').read_text());assert manifest['sha256']==release['data_sha256']
    assert sha(a.assets/manifest['file'])==manifest['sha256']
    tensors={};files={}
    with np.load(a.assets/manifest['file'],allow_pickle=False) as original:
        for role,n in (('train',1000),('test',411)):
            path=a.capture/a.arm/role/'predictions.npz';files[role]=sha(path)
            with np.load(path,allow_pickle=False) as z:
                x,y,ids,p=[z[k].copy() for k in ('features','labels','ids','probabilities')]
            assert x.shape==(n,256) and p.shape==(n,20) and np.isfinite(x).all() and len(ids)==n
            assert np.array_equal(y,original[role+'_labels']) and np.array_equal(ids,original[role+'_ids'])
            if role=='train':assert np.bincount(y,minlength=20).tolist()==[50]*20
            with torch.inference_mode():actual=torch.softmax(head(torch.from_numpy(x)),1).numpy()
            assert np.max(np.abs(actual-p))<=3e-6,'Features must be exact existing final-head inputs'
            tensors[role]=dict(x=torch.from_numpy(x),y=torch.from_numpy(y.astype(np.int64)),ids=ids)
    assert not set(tensors['train']['ids'].tolist())&set(tensors['test']['ids'].tolist())
    return release,spec,state,head,tensors,files


@torch.inference_mode()
def evaluate(head,role):
    probabilities=torch.softmax(head(role['x']),1).numpy()
    return metrics(role['y'].numpy(),probabilities),probabilities


def fit(a):
    assert a.epochs==120 and a.lr==.0005 and a.weight_decay==.0001
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(17)
    release,spec,original,head,data,files=load_features(a);started=time.perf_counter()
    baseline,baseline_p=evaluate(head,data['test']);train0,_=evaluate(head,data['train'])
    bundled=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    git=str(bundled) if bundled.exists() else shutil.which('git')
    actual_git=subprocess.check_output([git,'rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip() if git else None
    protocol=dict(task='existing_final_5140_linear_CCD_adaptation',arm=a.arm,epochs=120,batch_size=64,seed=17,optimizer='AdamW',lr=a.lr,weight_decay=a.weight_decay,loss='cross_entropy; no label smoothing or feature transformation',train_samples=1000,TEST_samples=411,VALID_samples=0,checkpoint_selection='highest TEST accuracy, first tie; development/test-informed, not independent generalization',TEST_evaluation_epochs=[0,1,*range(10,121,10)],TEST_gradients=False,trainable_parameters=5140,new_network_layers=0,upstream_electronic_and_optical_parameters_frozen=True,original_PT_SHA256=spec['sha256'],data_sha256=release['data_sha256'],feature_files_sha256=files,release_sha256=sha(a.release),source_sha256=sha(__file__),actual_checkout_git=actual_git,python=sys.version,torch=torch.__version__,platform=platform.platform(),SDK_opened=False,GPU_used=False)
    save(a.out/'protocol.json',protocol)
    optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=a.weight_decay);history=[];steps=0
    best_score=baseline['accuracy'];best_epoch=0
    def checkpoint(epoch,kind):
        state={k:v.detach().clone() for k,v in original.items()}
        for k,v in head.state_dict().items():state['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        target=a.out/(kind+'.pt');torch.save(dict(model=state,epoch=epoch,protocol=protocol,original_checkpoint_sha256=spec['sha256'],only_existing_linear_adapted=True),target)
    checkpoint(0,'best');save(a.out/'progress.json',dict(state='fitting',epoch=0,baseline_TEST=baseline,baseline_TRAIN=train0,TEST_is_development=True))
    for epoch in range(1,121):
        head.train();order=torch.randperm(1000);loss_total=0.
        for begin in range(0,1000,64):
            ix=order[begin:begin+64];optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(data['train']['x'][ix]),data['train']['y'][ix]);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
            optimizer.step();steps+=1;loss_total+=float(loss.detach())*len(ix)
        row=dict(epoch=epoch,TRAIN_loss=loss_total/1000,optimizer_steps=steps)
        if epoch==1 or epoch%10==0:
            head.eval();train,_=evaluate(head,data['train']);test,_=evaluate(head,data['test'])
            row.update(TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'])
            if test['accuracy']>best_score:best_score=test['accuracy'];best_epoch=epoch;checkpoint(epoch,'best')
            save(a.out/'progress.json',dict(state='fitting',latest=row,best_epoch=best_epoch,best_TEST_development_accuracy=best_score,elapsed_s=time.perf_counter()-started))
        history.append(row)
    checkpoint(120,'last');save(a.out/'history.json',history)
    results={}
    for kind in ('best','last'):
        path=a.out/(kind+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
        assert all(torch.equal(v,saved['model'][k]) for k,v in original.items() if not k.startswith('linear.'))
        fresh=existing_head(saved['model']).cpu().eval();train,_=evaluate(fresh,data['train']);test,probabilities=evaluate(fresh,data['test'])
        prediction=a.out/(kind+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=data['test']['ids'],labels=data['test']['y'].numpy(),probabilities=probabilities)
        results[kind]=dict(epoch=saved['epoch'],TRAIN=train,TEST_development=test,PT_SHA256=sha(path),predictions_SHA256=sha(prediction),fresh_CPU_reload_verified=True,all_nonterminal_parameters_exactly_unchanged=True)
    assert results['best']['epoch']==best_epoch and results['best']['TEST_development']['accuracy']==best_score
    report=dict(state='complete',arm=a.arm,baseline_TRAIN=train0,baseline_TEST_development=baseline,results=results,optimizer_steps=steps,epochs=120,elapsed_s=time.perf_counter()-started,source_sha256=sha(__file__),protocol_sha256=sha(a.out/'protocol.json'),history_sha256=sha(a.out/'history.json'),SDK_opened=False,GPU_used=False,TEST_is_development_not_independent=True,all_original_input_files_SHA256_unchanged=all(sha(a.capture/a.arm/role/'predictions.npz')==digest for role,digest in files.items()))
    assert report['all_original_input_files_SHA256_unchanged'] and sha(a.assets/spec['file'])==spec['sha256']
    save(a.out/'report.json',report);print(json.dumps(dict(state='complete',arm=a.arm,baseline_TEST=baseline['accuracy'],best_TEST_development=best_score,best_epoch=best_epoch,last_TEST_development=results['last']['TEST_development']['accuracy'],elapsed_s=report['elapsed_s'])),flush=True)


def selftest():
    torch.manual_seed(17);torch.set_num_threads(2)
    original=nn.Linear(256,20);state={'linear.'+k:v.detach().clone() for k,v in original.state_dict().items()};head=existing_head(state)
    x=torch.randn(8,256);labels=torch.arange(8);before=head.weight.detach().clone()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=.0001)
    optimizer.zero_grad();nn.functional.cross_entropy(head(x),labels).backward();optimizer.step()
    assert not torch.equal(before,head.weight.detach()) and sum(p.numel() for p in head.parameters())==5140
    print('Synthetic existing5140 linear initialization/gradient check passed; no real data/SDK/GPU')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--arm',choices=('residual','control'))
    for name in ('capture','assets','release','out'):p.add_argument('--'+name,type=Path)
    p.add_argument('--epochs',type=int,default=120);p.add_argument('--lr',type=float,default=.0005);p.add_argument('--weight-decay',type=float,default=.0001);a=p.parse_args()
    if a.selftest:selftest()
    else:
        try:fit(a)
        except BaseException as error:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),automatic_retry=False,SDK_opened=False))
            raise
