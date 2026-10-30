"""Unified panel-A optical LG: four experts, Top-2, fixed CCD class regions."""
import argparse
import hashlib
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
from .model_four import FourRouterClassification, CONTRACT
from .train import encode, write
from .train_linear_ccd_8class import prepare_index, load_encoded
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import CLASS_NAMES, sha256


def route_statistics(power):
    pairs=np.sort(np.argsort(-power,axis=1)[:,:2],axis=1)
    counts={f'{i}-{j}':float(((pairs[:,0]==i)&(pairs[:,1]==j)).mean()) for i in range(4) for j in range(i+1,4)}
    mean=power.mean(0)
    return dict(mean_power=mean.tolist(),std_power=power.std(0).tolist(),
                selected_fraction=(power>0).mean(0).tolist(),top2_pair_fraction=counts,
                fraction_outside_most_common_pair=1.-max(counts.values()),
                effective_experts_from_mean_power=float(1./np.square(mean).sum()),
                mean_per_sample_effective_experts=float((1./np.square(power).sum(1)).mean()),
                expert_numbering='zero-based 0..3',samples=len(power))


@torch.no_grad()
def evaluate(model,x,y,batch):
    scores=[];routes=[];dense=[];capture=[]
    model.eval()
    for start in range(0,len(y),batch):
        out=model(x[start:start+batch],return_dense_route=True)
        scores.append(out['probabilities'].cpu());routes.append(out['route_power'].cpu());dense.append(out['dense_route_power'].cpu())
        capture.append(out['router_capture'].cpu())
    p=torch.cat(scores);q=torch.cat(routes).numpy();soft=torch.cat(dense).numpy()
    labels=y.cpu();pred=p.argmax(1);classes=p.shape[1]
    cm=torch.bincount(labels*classes+pred,minlength=classes*classes).reshape(classes,classes)
    metrics=dict(accuracy=float((pred==labels).float().mean()),
                 balanced_accuracy=float((cm.diag()/cm.sum(1).clamp_min(1)).mean()),
                 cross_entropy=float(-p[torch.arange(len(y)),labels].log().mean()),
                 confusion_matrix=cm.tolist(),n=len(y),router=route_statistics(q),
                 dense_router_mean_power=soft.mean(0).tolist())
    capture=torch.cat(capture).numpy()
    metrics['router_capture']=dict(mean=float(capture.mean()),minimum=float(capture.min()),
                                   p05=float(np.quantile(capture,.05)))
    return metrics,p.numpy(),q,soft


def prepare_dataset(data_root,cfg):
    if cfg['dataset']!='crc9':
        rows,ids,identity=prepare_index(data_root,cfg)
        hashes={s:[r['pixel_sha256'] for r in rr] for s,rr in rows.items()}
        def load(split):return load_encoded(data_root,rows[split])
        return load,ids,identity,hashes
    data=data_root/'A_original.npz';manifest_path=data_root/'A_original_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    actual=sha256(data)
    if actual!=cfg['source_archive_sha256'] or manifest['cache_sha256']!=actual:raise ValueError('CRC9 archive mismatch')
    if manifest['license']!='CC BY 4.0' or manifest['domain']!='A_original' or tuple(manifest['class_names'])!=CLASS_NAMES:raise ValueError('CRC9 identity')
    with np.load(data,allow_pickle=False) as z:ids={s:z[s+'_ids'].astype(str) for s in ('train','val','test')}
    for s,v in ids.items():
        if len(v)!=cfg['expected_counts'][s] or len(set(v))!=len(v):raise ValueError('CRC9 split count/IDs')
    if any(set(ids[s])&set(ids[t]) for s,t in [('train','val'),('train','test'),('val','test')]):raise ValueError('CRC9 ID overlap')
    hashes={}
    def load(split):
        # TEST images and labels are opened only after validation selection is locked.
        with np.load(data,allow_pickle=False) as z:
            images=z[split+'_images'];labels=z[split+'_labels']
        if images.dtype!=np.uint8 or images.shape!=(len(ids[split]),224,224,3) or set(labels.tolist())!=set(range(9)):raise ValueError('CRC9 arrays')
        hashes[split]=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        return encode(images,'four_top2').cuda(),torch.tensor(labels,device='cuda',dtype=torch.long)
    identity=dict(archive_sha256=actual,manifest_sha256=sha256(manifest_path),classes=list(CLASS_NAMES),
                  counts={s:len(v) for s,v in ids.items()},patient_independence_verified=False)
    return load,ids,identity,hashes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,required=True);parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    gpu=require_non_a100_gpu();cfg=json.loads(args.profile.read_text())
    if (cfg['architecture']!='optical_four_top2_ccd' or cfg['experts']!=4 or cfg['main_layers']!=2
        or cfg['OEO_stages']!=2 or cfg['route_balance_loss_weight']!=0):raise ValueError('LG contract')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    try:
        load,ids,identity,hashes=prepare_dataset(args.data_root,cfg)
        parent=Path(__file__).resolve().parent;lg=parent.parents[1]
        sources=[Path(__file__),parent/'model_four.py',parent/'train.py',parent/'train_linear_ccd_8class.py',
                 parent/'train_linear_ccd.py',parent/'gpu_policy.py',lg/'demo_check/pure_optical/models.py',
                 lg/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py',
                 lg/'tasks/t11_lifelong_optics/crc9_data.py']
        metadata=dict(profile=cfg,profile_sha256=sha256(args.profile),command=sys.argv,pid=os.getpid(),
                      git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256={str(p):sha256(p) for p in sources},
                      data=identity,python=platform.python_version(),torch=torch.__version__,gpu_uuid=gpu,
                      optical_contract=dict(CONTRACT),test_pixels_read=False)
        write(args.out/'metadata.json',metadata)
        x={};y={}
        for split in ('train','val'):x[split],y[split]=load(split)
        if set(hashes['train'])&set(hashes['val']):raise ValueError('train/validation pixel overlap')
        model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda()
        assert sum(p.numel() for p in model.parameters())==cfg['trainable_parameters']
        assert set(dict(model.named_parameters()))=={'first_phase','global_phase','router_phase'}
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
        rng=np.random.default_rng(cfg['seed']);best=(-1.,float('-inf'));history=[];started=time.time()
        for epoch in range(1,cfg['epochs']+1):
            begin=time.time();model.train();order=rng.permutation(len(y['train']));loss_sum=0.
            for offset in range(0,len(order),cfg['batch']):
                idx=torch.as_tensor(order[offset:offset+cfg['batch']],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x['train'][idx])['logits'],y['train'][idx])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('phase gradient failure')
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();loss_sum+=float(loss.detach())*len(idx)
            val,*_=evaluate(model,x['val'],y['val'],cfg['batch'])
            row=dict(epoch=epoch,train_objective=loss_sum/len(order),validation=val)
            if epoch==1 or epoch%cfg['training_evaluate_every']==0:row['training'],*_=evaluate(model,x['train'],y['train'],cfg['batch'])
            score=(val['balanced_accuracy'],-val['cross_entropy']);improved=score>best
            if improved:best=score
            row.update(seconds=time.time()-begin,selected=improved);history.append(row)
            write(args.out/'history.json',history)
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                    config=metadata,validation=val,shuffle_rng=rng.bit_generator.state)
            torch.save(ck,args.out/'last_checkpoint.pt')
            if improved:torch.save(ck,args.out/'best_checkpoint.pt')
            write(args.out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],best_validation=best[0]))
            print(json.dumps(row),flush=True)
        ck=torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        lock=dict(selected_epoch=ck['epoch'],checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),
                  selection=cfg['selection'],test_evaluations_before_lock=0)
        write(args.out/'test_selection_lock.json',lock);metrics={}
        for split,key in [('train','train'),('val','validation'),('test','test')]:
            if split=='test':x[split],y[split]=load(split)
            metrics[key],p,q,soft=evaluate(model,x[split],y[split],cfg['batch'])
            np.savez_compressed(args.out/f'{split}_predictions.npz',ids=ids[split],labels=y[split].cpu().numpy(),
                                scores=p,predictions=p.argmax(1),route_power=q,dense_route_power=soft)
            if split=='test':
                seen=set(hashes['train'])|set(hashes['val']);clean=np.array([h not in seen for h in hashes['test']])
                labels=y[split].cpu().numpy();pred=p.argmax(1);classes=cfg['num_classes']
                cm=np.bincount(labels[clean]*classes+pred[clean],minlength=classes*classes).reshape(classes,classes)
                metrics['test_clean']=dict(accuracy=float((pred[clean]==labels[clean]).mean()),balanced_accuracy=float((cm.diagonal()/cm.sum(1)).mean()),
                                           cross_entropy=float(-np.log(p[clean,labels[clean]]).mean()),confusion_matrix=cm.tolist(),n=int(clean.sum()))
                write(args.out/'test_overlap_audit.json',dict(removed_test_ids=ids[split][~clean].tolist(),clean_test_ids=ids[split][clean].tolist()))
        assert metrics['validation']==ck['validation']
        result=dict(**lock,**metrics,test_evaluations=1,completed_epochs=cfg['epochs'],elapsed_seconds=time.time()-started,
                    role='development_followup_fixed_four_expert_optical_LG')
        write(args.out/'result.json',result);write(args.out/'status.json',dict(state='complete',**result));print(json.dumps(result),flush=True)
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error)));raise
    finally:
        torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time(),note='Verify PID absence after exit'))


if __name__=='__main__':main()
