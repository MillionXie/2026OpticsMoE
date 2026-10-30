"""Fixed-budget CRC9 D2NN-only run; original data and preprocessing are reused."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model_linear_ccd import LinearD2NNCCD
from .train import encode, write
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import load_domain, sha256


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval()
    scores = torch.cat([model(x[i:i+batch])['probabilities'].cpu() for i in range(0, len(y), batch)])
    labels=y.cpu(); pred=scores.argmax(1)
    classes=scores.shape[1]
    cm=torch.bincount(labels*classes+pred, minlength=classes*classes).reshape(classes,classes)
    metrics=dict(accuracy=float((pred==labels).float().mean()),
                 balanced_accuracy=float((cm.diag()/cm.sum(1).clamp_min(1)).mean()),
                 cross_entropy=float(-scores[torch.arange(len(y)),labels].log().mean()),
                 confusion_matrix=cm.tolist(), n=len(y))
    return metrics, scores.numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); uuid=require_non_a100_gpu()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=json.loads(a.profile.read_text());torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    metadata=dict(profile=cfg,profile_sha256=sha256(a.profile),command=sys.argv,pid=os.getpid(),
                  python=platform.python_version(),torch=torch.__version__,gpu_uuid=uuid,
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  data_sha256=sha256(a.data),manifest_sha256=sha256(a.manifest),
                  source_commit=os.environ.get('T17_SOURCE_COMMIT'),test_read=False)
    sources=[Path(__file__),Path(__file__).with_name('model_linear_ccd.py'),
             Path(__file__).with_name('train.py'),Path(__file__).with_name('model_four.py'),
             Path('LightGenV2/demo_check/pure_optical/models.py'),
             Path('LightGenV2/demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py'),
             Path('LightGenV2/tasks/t11_lifelong_optics/crc9_data.py')]
    metadata['source_sha256']={str(p):sha256(p) for p in sources}
    write(a.out/'metadata.json',metadata)
    try:
        # Deliberately load no TEST images or labels until the selected PT is locked.
        manifest=json.loads(a.manifest.read_text())
        if manifest['license']!='CC BY 4.0' or manifest['domain']!='A_original':raise ValueError('wrong dataset')
        if manifest['cache_sha256']!=metadata['data_sha256']:raise ValueError('data hash mismatch')
        from LightGenV2.tasks.t11_lifelong_optics.crc9_data import CLASS_NAMES
        if tuple(manifest['class_names'])!=CLASS_NAMES:raise ValueError('class order mismatch')
        with np.load(a.data,allow_pickle=False) as z:
            ids={s:z[s+'_ids'].astype(str) for s in ('train','val','test')}
            if any(len(ids[s])!=n for s,n in [('train',5026),('val',718),('test',1436)]):raise ValueError('split size')
            if any(len(set(v))!=len(v) for v in ids.values()):raise ValueError('duplicate IDs')
            if any(set(ids[s])&set(ids[t]) for s,t in [('train','val'),('train','test'),('val','test')]):raise ValueError('split overlap')
            x={s:encode(z[s+'_images'],'four_top2').cuda() for s in ('train','val')}
            y={s:torch.as_tensor(z[s+'_labels'],device='cuda',dtype=torch.long) for s in ('train','val')}
        model=LinearD2NNCCD(cfg['layers']).cuda()
        assert sum(p.numel() for p in model.parameters())==cfg['phase_parameters']
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
        rng=np.random.default_rng(cfg['seed']);history=[];best=(-1.,float('-inf'));started=time.time()
        for epoch in range(1,cfg['epochs']+1):
            begin=time.time();model.train();loss_sum=0.;order=rng.permutation(len(y['train']))
            for i in range(0,len(order),cfg['batch']):
                ix=torch.as_tensor(order[i:i+cfg['batch']],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x['train'][ix])['logits'],y['train'][ix])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.phases):raise RuntimeError('phase gradient failure')
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();loss_sum+=float(loss.detach())*len(ix)
            val,_=evaluate(model,x['val'],y['val'],cfg['batch'])
            row=dict(epoch=epoch,train_objective=loss_sum/len(order),validation=val)
            if epoch==1 or epoch%cfg['training_evaluate_every']==0:
                row['training'],_=evaluate(model,x['train'],y['train'],cfg['batch'])
            score=(val['balanced_accuracy'],-val['cross_entropy']);improved=score>best
            if improved:best=score
            row.update(seconds=time.time()-begin,selected=improved)
            history.append(row);write(a.out/'history.json',history)
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    config=metadata,validation=val,shuffle_rng=rng.bit_generator.state)
            torch.save(ck,a.out/'last_checkpoint.pt')
            if improved:torch.save(ck,a.out/'best_checkpoint.pt')
            write(a.out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],best_validation=best[0]))
            print(json.dumps(row),flush=True)
        ck=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'])
        lock=dict(selected_epoch=ck['epoch'],checkpoint_sha256=sha256(a.out/'best_checkpoint.pt'),
                  selection=cfg['selection'],test_evaluations_before_lock=0)
        write(a.out/'test_selection_lock.json',lock)
        train,_=evaluate(model,x['train'],y['train'],cfg['batch'])
        val,vp=evaluate(model,x['val'],y['val'],cfg['batch'])
        assert val==ck['validation']
        np.savez_compressed(a.out/'val_predictions.npz',ids=ids['val'],labels=y['val'].cpu().numpy(),scores=vp,predictions=vp.argmax(1))
        with np.load(a.data,allow_pickle=False) as z:
            xt=encode(z['test_images'],'four_top2').cuda();yt=torch.as_tensor(z['test_labels'],device='cuda',dtype=torch.long)
        test,tp=evaluate(model,xt,yt,cfg['batch'])
        np.savez_compressed(a.out/'test_predictions.npz',ids=ids['test'],labels=yt.cpu().numpy(),scores=tp,predictions=tp.argmax(1))
        result=dict(**lock,train=train,validation=val,test=test,test_evaluations=1,
                    completed_epochs=cfg['epochs'],elapsed_seconds=time.time()-started,
                    role='development_followup_after_observed_original_TEST')
        write(a.out/'result.json',result);write(a.out/'status.json',dict(state='complete',**result))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        write(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(error)));raise
    finally:
        torch.cuda.empty_cache()
        write(a.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time(),note='Process exit releases CUDA; verify PID absence externally'))


if __name__=='__main__':main()
