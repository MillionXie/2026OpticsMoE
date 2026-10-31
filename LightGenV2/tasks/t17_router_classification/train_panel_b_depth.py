"""Matched-budget panel-B workers. Original TEST is explicitly development data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .model_depth_ccd import DepthLG, DepthD2NN, depth_contract
from .train import encode, write
from .train_boundary_development import boundary_objective
from .test_exposed_data import load_raw_splits
from .development_teacher_data import development_training_membership
from .phase_optimizer import WrappedPhaseAdam
from .gpu_policy import require_non_a100_gpu
from .train_optical_four_ccd import route_statistics
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


JOBS = {'lg4':('lg',4), 'lg6':('lg',6),
        'd2nn3':('d2nn',2), 'd2nn5':('d2nn',4), 'd2nn7':('d2nn',6)}


def build_job(name):
    if name not in JOBS: raise ValueError('Expected one of '+','.join(JOBS))
    arch, main = JOBS[name]
    return DepthLG(main,8) if arch == 'lg' else DepthD2NN(main+1,8)


def schedule(epoch, cfg):
    if not 1 <= epoch <= cfg['epochs']: raise ValueError(epoch)
    stage = 0 if epoch <= cfg['stage_epochs'] else 1
    local_epoch = epoch - stage*cfg['stage_epochs']
    base = cfg['stage_learning_rates'][stage]
    factor = .2 + .8*.5*(1+np.cos(np.pi*(local_epoch-1)/(cfg['stage_epochs']-1)))
    return stage, float(base*factor), cfg['stage_focal_gamma'][stage]


@torch.no_grad()
def evaluate(model, x, y, batch):
    model.eval(); scores=[]; routes=[]; dense=[]
    for start in range(0,len(y),batch):
        out = model(x[start:start+batch],return_dense_route=True) if isinstance(model,DepthLG) else model(x[start:start+batch])
        scores.append(out['probabilities'].cpu().numpy())
        if isinstance(model,DepthLG):
            routes.append(out['route_power'].cpu().numpy());dense.append(out['dense_route_power'].cpu().numpy())
    p=np.concatenate(scores);labels=y.cpu().numpy();pred=p.argmax(1)
    matrix=np.bincount(labels*8+pred,minlength=64).reshape(8,8)
    q=np.concatenate(routes) if routes else np.empty((len(y),0),dtype=np.float32)
    soft=np.concatenate(dense) if dense else q.copy()
    metrics=dict(n=len(y),correct=int(np.trace(matrix)),accuracy=float((pred==labels).mean()),
                 balanced_accuracy=float((matrix.diagonal()/matrix.sum(1)).mean()),
                 cross_entropy=float(-np.log(p[np.arange(len(y)),labels]).mean()),confusion_matrix=matrix.tolist())
    if routes:metrics['router']=route_statistics(q)
    return metrics,p,q,soft


def metric_key(metrics):
    return metrics['balanced_accuracy'], -metrics['cross_entropy']


def train_one(name, out, cfg, metadata, x, y, ids, union, labels, teacher, membership):
    out.mkdir(exist_ok=False)
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    model=build_job(name).cuda();arch,main=JOBS[name]
    contract=depth_contract(main)
    expected=contract['LG_phase_parameters' if arch=='lg' else 'D2NN_phase_parameters']
    assert sum(p.numel() for p in model.parameters())==expected
    assert not any(isinstance(m,(torch.nn.Linear,torch.nn.Conv2d)) for m in model.modules())
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    job_metadata=dict(metadata,job=name,architecture=arch,contract=contract,
                      actual_parameter_count=expected,initialization='from_scratch_all_raw_phase_zero',
                      validation_selection='macro recall then original CCD probability NLL',
                      original_test_ground_truth_in_student_loss=False,
                      TEST_evaluations_before_weight_lock=0)
    write(out/'metadata.json',job_metadata)
    best=(-1.,float('-inf'));selected_epoch=0;history=[];optimizer=None;started=time.time()
    for epoch in range(1,cfg['epochs']+1):
        stage,lr,gamma=schedule(epoch,cfg)
        if epoch in (1,cfg['stage_epochs']+1):
            if stage:
                chosen=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(chosen['model'],strict=True)
            optimizer=WrappedPhaseAdam(list(model.parameters()),lr,cfg['gradient_clip'])
        for group in optimizer.param_groups:group['lr']=lr
        model.train();order=rng.permutation(membership);loss_sum=0.;begin=time.time()
        for start in range(0,len(order),cfg['batch']):
            idx=torch.as_tensor(order[start:start+cfg['batch']],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            result=model(union[idx])
            loss=boundary_objective(result['logits'],labels[idx],idx<len(y['train']),teacher[idx],
                                    cfg['training_temperature'],cfg['teacher_confidence'],
                                    cfg['energy_nll_weight'],focal_gamma=gamma)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite panel B training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Missing or nonfinite phase gradient')
            optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        row=dict(epoch=epoch,stage=stage,lr=lr,focal_gamma=gamma,training_objective=loss_sum/len(order),
                 training_inputs_seen=len(order),order_sha256=hashlib.sha256(order.tobytes()).hexdigest())
        # Exactly the same evaluation frequency and checkpoint rule at all depths.
        if epoch==1 or epoch%cfg['evaluate_every']==0:
            val,*_=evaluate(model,x['val'],y['val'],cfg['batch']);row['validation']=val
            if metric_key(val)>best:
                best=metric_key(val);selected_epoch=epoch
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                epoch=epoch,validation=val,metadata=job_metadata),out/'best_checkpoint.pt')
        if epoch==1 or epoch%cfg['train_evaluate_every']==0:
            row['original_TRAIN'],*_=evaluate(model,x['train'],y['train'],cfg['batch'])
        row.update(seconds=time.time()-begin,selected_epoch=selected_epoch);history.append(row)
        write(out/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                        shuffle_rng=rng.bit_generator.state,metadata=job_metadata),out/'last_checkpoint.pt')
        write(out/'status.json',dict(state='training',epoch=epoch,total_epochs=cfg['epochs'],selected_epoch=selected_epoch,pid=os.getpid()))
        print(json.dumps(dict(job=name,**row)),flush=True)
    chosen=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'],strict=True)
    write(out/'selection_lock.json',dict(checkpoint_sha256=sha256(out/'best_checkpoint.pt'),selected_epoch=selected_epoch,
                                      selection='validation macro then CCD NLL',TEST_evaluations_this_job_before_lock=0))
    result=dict(role=metadata['role'],independent_test_claim=False,job=name,completed_epochs=cfg['epochs'],
                selected_epoch=selected_epoch,checkpoint_sha256=sha256(out/'best_checkpoint.pt'),
                elapsed_seconds=time.time()-started,initialization=job_metadata['initialization'],contract=contract)
    for split,key in [('train','train'),('val','val'),('test','original_test_dev')]:
        metrics,p,q,soft=evaluate(model,x[split],y[split],cfg['batch']);result[key]=metrics
        np.savez_compressed(out/f'{split}_predictions.npz',ids=ids[split],labels=y[split].cpu().numpy(),
                            scores=p,predictions=p.argmax(1),route_power=q,dense_route_power=soft)
    result['phase_update_L2']={k:float((v.detach().cpu()-initial[k]).norm()) for k,v in model.state_dict().items()}
    write(out/'result.json',result);write(out/'status.json',dict(state='complete',pid=os.getpid(),**result))
    del model,optimizer,initial,chosen;torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    for arg in ('profile','data-root','teacher','out'):parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--jobs',nargs='+',choices=list(JOBS),required=True)
    args=parser.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(args.profile.read_text(encoding='utf-8'))
    if cfg['epochs']!=2*cfg['stage_epochs'] or cfg['epochs']%cfg['evaluate_every']:raise ValueError('Complete two-stage budget required')
    if len(set(args.jobs))!=len(args.jobs):raise ValueError('Duplicate worker jobs')
    torch.set_num_threads(4);args.out.mkdir(exist_ok=False,parents=True)
    source_files=[Path(__file__),Path(__file__).with_name('model_depth_ccd.py'),Path(__file__).with_name('model_four.py'),
                  Path(__file__).with_name('model_linear_ccd.py'),Path(__file__).with_name('train_boundary_development.py'),
                  Path(__file__).with_name('phase_optimizer.py')]
    metadata=dict(role='ORIGINAL_TEST_EXPOSED_PANEL_B_DEVELOPMENT',independent_test_claim=False,
                  profile=cfg,profile_sha256=sha256(args.profile),source_commit=os.environ.get('T17_SOURCE_COMMIT'),
                  source_sha256={p.name:sha256(p) for p in source_files},pid=os.getpid(),gpu_uuid=gpu,
                  command=sys.argv,torch=torch.__version__,original_TEST_in_teacher_training=True,
                  original_TEST_teacher_pseudo_labels_in_student_training=True,teacher_absent_from_inference=True,
                  matched_budget_jobs=list(JOBS),LG2_reference_is_reused_separate_optimization_budget=True)
    write(args.out/'metadata.json',metadata);write(args.out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
    try:
        raw,identity=load_raw_splits(args.data_root,cfg);membership,removed=development_training_membership(raw)
        x={};y={};ids={};probabilities=[];cache_manifest=json.loads((args.teacher/'cache_manifest.json').read_text())
        for split in ('train','val','test'):
            part=raw[split]
            x[split]=encode(part['images'],'four_top2').cuda();y[split]=torch.as_tensor(part['labels'],device='cuda',dtype=torch.long);ids[split]=part['ids']
            if split!='val':
                path=args.teacher/f'teacher_logits_{split}.npz';assert sha256(path)==cache_manifest[path.name]
                with np.load(path,allow_pickle=False) as z:
                    assert np.array_equal(z['ids'],ids[split]) and np.array_equal(z['labels'],part['labels'])
                    probabilities.append(torch.from_numpy(z['logits'][:,0].copy()).cuda().softmax(1))
            del part['images']
        teacher=torch.cat(probabilities);union=torch.cat((x['train'],x['test']));labels=torch.cat((y['train'],y['test']))
        confidence=teacher.max(1).values.cpu().numpy();valid=(membership<len(y['train']))|(confidence[membership]>=cfg['teacher_confidence'])
        membership=membership[valid]
        metadata.update(data=identity,training_membership_count=len(membership),teacher_cache_manifest_sha256=sha256(args.teacher/'cache_manifest.json'))
        write(args.out/'metadata.json',metadata);write(args.out/'training_membership.json',dict(indices=membership.tolist(),removed_pixel_duplicates=removed,low_confidence_original_TEST_inputs_excluded_from_training=int((~valid).sum()),evaluation_sets_unchanged=True))
        for name in args.jobs:
            write(args.out/'status.json',dict(state='training',job=name,pid=os.getpid()))
            train_one(name,args.out/name,cfg,metadata,x,y,ids,union,labels,teacher,membership)
        write(args.out/'status.json',dict(state='complete',jobs=args.jobs,pid=os.getpid()))
    except BaseException as error:
        write(args.out/'status.json',dict(state='failed_or_interrupted',error=repr(error),pid=os.getpid()));raise
    finally:
        torch.cuda.empty_cache();write(args.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
