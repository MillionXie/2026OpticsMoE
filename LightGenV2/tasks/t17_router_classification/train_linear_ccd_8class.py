"""Eight-class image-index datasets; reuse the three-phase CCD-only D2NN."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .model_linear_ccd import LinearD2NNCCD
from .train_linear_ccd import evaluate
from .train import encode, write
from .gpu_policy import require_non_a100_gpu
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def build_model(cfg):
    if cfg.get('architecture') == 'panel_b_depth_oeo':
        from .model_depth_ccd import DepthD2NN
        if cfg.get('intermediate_OEO') is not True:
            raise ValueError('Panel B requires its agreed per-layer OEO')
        return DepthD2NN(layers=cfg['layers'], num_classes=8)
    return LinearD2NNCCD(layers=cfg['layers'], num_classes=8)


def prepare_index(data_root, cfg):
    """Use provenance/CSV only; never read test pixels during training."""
    manifest_path=data_root/'experiment_data_manifest.json'
    index_path=data_root/'images_index.csv'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['license']!='CC BY 4.0':raise ValueError('dataset license')
    rows=list(csv.DictReader(index_path.open(encoding='utf-8',newline='')))
    if cfg['dataset']=='kather2016':
        archive=data_root/'kather2016_hf_4fd718b.parquet'
        if manifest['source_archive_sha256']!=cfg['source_archive_sha256']:raise ValueError('Kather archive identity')
        classes=manifest['classes'];expected={s:set(v) for s,v in manifest['split_ids'].items()}
        if any({r['source_id'] for r in rows if r['split']==s}!=v for s,v in expected.items()):raise ValueError('Kather split IDs')
    elif cfg['dataset']=='bloodmnist128':
        archive=data_root/'bloodmnist_128.npz'
        if manifest['sha256']!=cfg['source_archive_sha256']:raise ValueError('Blood archive identity')
        keep=set(manifest['train_indices'])
        excluded=set(range(11959))-keep
        if excluded!=set(manifest['removed_train_val_matches']+manifest['removed_train_duplicates']):raise ValueError('Blood exclusions')
        rows=[r for r in rows if r['split']!='train' or int(r['source_index']) in keep]
        classes=[None]*8
        for r in rows:
            label=int(r['class_id'])
            if classes[label] is not None and classes[label]!=r['class_name']:raise ValueError('class mapping')
            classes[label]=r['class_name']
    else:raise ValueError(cfg['dataset'])
    if sha256(archive)!=cfg['source_archive_sha256']:raise ValueError('archive SHA mismatch')
    grouped={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    ids={s:np.array([r.get('source_id',s+'_'+r.get('source_index','')) for r in rr]) for s,rr in grouped.items()}
    for split,rr in grouped.items():
        if len(rr)!=cfg['expected_counts'][split] or len(set(ids[split]))!=len(rr):raise ValueError('count/duplicate IDs')
        if set(int(r['class_id']) for r in rr)!=set(range(8)):raise ValueError('missing class')
        for r in rr:
            path=(data_root/r['relative_path']).resolve()
            if not path.is_relative_to(data_root.resolve()) or not path.is_file():raise ValueError('image path')
    hashes={s:[r['pixel_sha256'] for r in rr] for s,rr in grouped.items()}
    overlap={s+'_'+t:len(set(hashes[s])&set(hashes[t])) for s,t in [('train','val'),('train','test'),('val','test')]}
    # Do not silently alter historical TEST or use its outcomes to curate training.
    if overlap['train_val']:raise ValueError('train/validation pixel overlap')
    identity=dict(manifest_sha256=sha256(manifest_path),index_sha256=sha256(index_path),
                  archive_sha256=cfg['source_archive_sha256'],classes=classes,
                  counts={s:len(rr) for s,rr in grouped.items()},pixel_hash_overlap_counts=overlap,
                  duplicate_hashes_within_split={s:len(h)-len(set(h)) for s,h in hashes.items()},
                  patient_independence_verified=False)
    return grouped,ids,identity


def load_encoded(data_root, rows):
    arrays=[]
    for row in rows:
        with Image.open(data_root/row['relative_path']) as im:array=np.array(im.convert('RGB'))
        if hashlib.sha256(array.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('image pixels changed')
        arrays.append(array)
    x=encode(np.stack(arrays),'four_top2').cuda()
    y=torch.tensor([int(row['class_id']) for row in rows],device='cuda',dtype=torch.long)
    return x,y


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();uuid=require_non_a100_gpu()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    cfg=json.loads(args.profile.read_text());torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    try:
        rows,ids,identity=prepare_index(args.data_root,cfg)
        root=Path(__file__).resolve().parents[2]
        paths=[Path(__file__),Path(__file__).with_name('model_linear_ccd.py'),Path(__file__).with_name('train_linear_ccd.py'),
               Path(__file__).with_name('train.py'),Path(__file__).with_name('model_four.py'),
               root/'demo_check/pure_optical/models.py',
               root/'demo_check/adrenal_softsign_code_export_20260915_145336/code/optical_reference/optics.py']
        if cfg.get('architecture') == 'panel_b_depth_oeo':
            paths.append(Path(__file__).with_name('model_depth_ccd.py'))
        metadata=dict(profile=cfg,profile_sha256=sha256(args.profile),command=sys.argv,pid=os.getpid(),
                      git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_commit=os.environ.get('T17_SOURCE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      source_sha256={str(p):sha256(p) for p in paths},data=identity,
                      python=platform.python_version(),torch=torch.__version__,gpu_uuid=uuid,test_pixels_read=False)
        write(args.out/'metadata.json',metadata)
        x={};y={}
        for split in ('train','val'):x[split],y[split]=load_encoded(args.data_root,rows[split])
        model=build_model(cfg).cuda()
        assert sum(p.numel() for p in model.parameters())==cfg['phase_parameters']
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=0)
        rng=np.random.default_rng(cfg['seed']);history=[];best=(-1.,float('-inf'));started=time.time()
        for epoch in range(1,cfg['epochs']+1):
            begin=time.time();model.train();order=rng.permutation(len(y['train']));loss_sum=0.
            for i in range(0,len(order),cfg['batch']):
                index=torch.as_tensor(order[i:i+cfg['batch']],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x['train'][index])['logits'],y['train'][index])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.phases):raise RuntimeError('phase gradient failure')
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();loss_sum+=float(loss.detach())*len(index)
            val,_=evaluate(model,x['val'],y['val'],cfg['batch'])
            row=dict(epoch=epoch,train_objective=loss_sum/len(order),validation=val,
                     order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
            if epoch==1 or epoch%cfg['training_evaluate_every']==0:
                row['training'],_=evaluate(model,x['train'],y['train'],cfg['batch'])
            score=(val['balanced_accuracy'],-val['cross_entropy']);improved=score>best
            if improved:best=score
            row.update(seconds=time.time()-begin,selected=improved);history.append(row)
            write(args.out/'history.json',history)
            checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                            config=metadata,validation=val,shuffle_rng=rng.bit_generator.state)
            torch.save(checkpoint,args.out/'last_checkpoint.pt')
            if improved:torch.save(checkpoint,args.out/'best_checkpoint.pt')
            write(args.out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],best_validation=best[0]))
            print(json.dumps(row),flush=True)
        ck=torch.load(args.out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        lock=dict(selected_epoch=ck['epoch'],checkpoint_sha256=sha256(args.out/'best_checkpoint.pt'),
                  selection=cfg['selection'],test_evaluations_before_lock=0)
        write(args.out/'test_selection_lock.json',lock)
        metrics={}
        for split,key in [('train','train'),('val','validation'),('test','test')]:
            if split=='test':x[split],y[split]=load_encoded(args.data_root,rows[split])
            metrics[key],scores=evaluate(model,x[split],y[split],cfg['batch'])
            np.savez_compressed(args.out/f'{split}_predictions.npz',ids=ids[split],labels=y[split].cpu().numpy(),
                                scores=scores,predictions=scores.argmax(1))
            if split=='test':
                seen={r['pixel_sha256'] for s in ('train','val') for r in rows[s]}
                clean=np.array([r['pixel_sha256'] not in seen for r in rows[split]],dtype=bool)
                labels=y[split].cpu().numpy();pred=scores.argmax(1)
                cm=np.bincount(labels[clean]*8+pred[clean],minlength=64).reshape(8,8)
                metrics['test_clean']=dict(accuracy=float((pred[clean]==labels[clean]).mean()),
                    balanced_accuracy=float((cm.diagonal()/cm.sum(1)).mean()),
                    cross_entropy=float(-np.log(scores[clean,labels[clean]]).mean()),
                    confusion_matrix=cm.tolist(),n=int(clean.sum()))
                write(args.out/'test_overlap_audit.json',dict(policy='Report full TEST plus pixel-disjoint TEST subset; no extra inference/selection',
                    removed_test_ids=ids[split][~clean].tolist(),clean_test_ids=ids[split][clean].tolist()))
        assert metrics['validation']==ck['validation']
        result=dict(**lock,**metrics,test_evaluations=1,completed_epochs=cfg['epochs'],elapsed_seconds=time.time()-started,
                    role=cfg.get('result_role','development_followup_historical_panel_A_baseline_source_unconfirmed'))
        write(args.out/'result.json',result);write(args.out/'status.json',dict(state='complete',**result))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error)));raise
    finally:
        torch.cuda.empty_cache()
        write(args.out/'exit_receipt.json',dict(pid=os.getpid(),timestamp=time.time(),note='Verify PID absence externally after process exit'))


if __name__=='__main__':main()
