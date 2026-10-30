"""Optical boundary training with explicit TEST-trained teacher pseudo labels."""
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model_four import FourRouterClassification
from .train import encode,write
from .train_optical_four_ccd import evaluate
from .test_exposed_data import load_raw_splits
from .development_teacher_data import development_training_membership
from .gpu_policy import require_non_a100_gpu
from .phase_optimizer import WrappedPhaseAdam
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def boundary_objective(log_probabilities,labels,is_original_train,teacher_probabilities,temperature,confidence,nll_weight,class_weights=None):
    if temperature<=0:raise ValueError('positive training temperature required')
    teacher_confidence,pseudo=teacher_probabilities.max(1)
    targets=torch.where(is_original_train,labels,pseudo)
    mask=is_original_train|(teacher_confidence>=confidence)
    if not mask.any():return log_probabilities.sum()*0.
    return (F.cross_entropy(log_probabilities[mask]/temperature,targets[mask],weight=class_weights)
            +nll_weight*F.nll_loss(log_probabilities[mask],targets[mask],weight=class_weights))


def training_error_class_weights(confusion_matrix):
    matrix=np.asarray(confusion_matrix,dtype=np.float64)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or np.any(matrix<0) or np.any(matrix.sum(1)<=0):
        raise ValueError('Full square TRAIN confusion matrix with every class required')
    error=.1+1-np.diag(matrix)/matrix.sum(1)
    return np.clip(error/error.mean(),.5,2.)


def class_pair_route_targets(original_train_class_counts):
    """Deterministic prior-balanced training targets; no per-image oracle at inference."""
    counts=np.asarray(original_train_class_counts,dtype=np.float64)
    if counts.ndim!=1 or len(counts)<2 or np.any(counts<=0):raise ValueError('Positive original TRAIN class counts required')
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    load=np.zeros(4);targets=np.full((len(counts),4),.01,dtype=np.float32)
    for c in sorted(range(len(counts)),key=lambda c:(-counts[c],c)):
        def cost(pair):
            candidate=load.copy();candidate[list(pair)]+=counts[c]/2
            return float(np.square(candidate).sum()),pair
        pair=min(pairs,key=cost);load[list(pair)]+=counts[c]/2;targets[c,list(pair)]+=.48
    return targets


def dense_route_supervision(dense_power,labels,is_original_train,teacher_probabilities,class_targets,confidence):
    teacher_confidence,pseudo=teacher_probabilities.max(1)
    targets=torch.where(is_original_train,labels,pseudo)
    mask=is_original_train|(teacher_confidence>=confidence)
    if not mask.any():return dense_power.sum()*0.
    return F.kl_div(dense_power[mask].clamp_min(1e-12).log(),class_targets[targets[mask]],reduction='batchmean')


def initialize_main_phases(model,kind,seed):
    if kind=='parent':return
    if kind=='common_phase_gauge':
        if model.variant!='optical' or model.cfg.get('oeo_activation')!='intensity_softsign':
            raise ValueError('Gauge initialization requires the admitted optical/OEO geometry')
        records=[]
        with torch.no_grad():
            # One common offset across all four experts preserves their interference.
            for name in ('first_phase','global_phase'):
                value=getattr(model,name);p=value.detach().double().sigmoid();phase=2*torch.pi*p
                original=float((2*torch.pi*p*(1-p)).clamp_min(1e-20).log().mean());best=(original,0.)
                for index in range(1,64):
                    offset=2*torch.pi*index/64;q=torch.remainder(phase+offset,2*torch.pi)/(2*torch.pi)
                    score=float((2*torch.pi*q*(1-q)).clamp_min(1e-20).log().mean())
                    if score>best[0]:best=(score,offset)
                if best[1]!=0:
                    q=torch.remainder(phase+best[1],2*torch.pi)/(2*torch.pi)
                    value.copy_(torch.logit(q.clamp(1e-9,1-1e-9)).to(value.dtype))
                records.append(dict(parameter=name,common_offset_rad=best[1],mean_log_phase_Jacobian_before=original,
                                    mean_log_phase_Jacobian_after=best[0],geometric_Jacobian_gain=float(np.exp(best[0]-original)),
                                    offset_selection_uses_images_or_labels=False))
        return records
    if kind!='diffuse_uniform_phase':raise ValueError(kind)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if model.first_phase.is_cuda else []):
        torch.manual_seed(seed)
        with torch.no_grad():
            for value in (model.first_phase,model.global_phase):
                value.copy_(torch.logit(.01+.98*torch.rand_like(value)))


def configure_phase_optimizer(model,phase_lr,variant):
    train_router=variant.get('train_router',False)
    model.router_phase.requires_grad_(train_router)
    parameters=[model.first_phase,model.global_phase]
    if variant.get('phase_optimizer')=='wrapped_phase_adam':
        if train_router:raise ValueError('Wrapped phase admission currently requires the router frozen')
        return WrappedPhaseAdam(parameters,phase_lr,variant.get('phase_gradient_clip',5.)),parameters
    if variant.get('phase_optimizer','raw_adam')!='raw_adam':raise ValueError('Unknown phase optimizer')
    groups=[dict(params=parameters.copy(),lr=phase_lr)]
    if train_router:
        router_lr=variant['router_lr']
        if router_lr<=0:raise ValueError('positive router LR required for joint training')
        groups.append(dict(params=[model.router_phase],lr=router_lr));parameters.append(model.router_phase)
    return torch.optim.Adam(groups),parameters


def main():
    ap=argparse.ArgumentParser()
    for name in ('profile','data-root','parent','teacher','selection-root','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();gpu=require_non_a100_gpu();cfg=json.loads(a.profile.read_text());torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'status.json',dict(state='loading_verified_inputs',pid=os.getpid()))
    try:
        raw,identity=load_raw_splits(a.data_root,cfg);membership,removed=development_training_membership(raw)
        x={};y={};ids={};teacher=[]
        for split,part in raw.items():
            x[split]=encode(part['images'],'four_top2').cuda();y[split]=torch.as_tensor(part['labels'],device='cuda',dtype=torch.long);ids[split]=part['ids']
            if split!='val':
                p=a.teacher/f'teacher_logits_{split}.npz'
                assert sha256(p)==json.loads((a.teacher/'cache_manifest.json').read_text())[p.name]
                z=np.load(p,allow_pickle=False);assert np.array_equal(z['ids'],ids[split]) and np.array_equal(z['labels'],part['labels'])
                teacher.append(torch.from_numpy(z['logits'][:,0]).cuda().softmax(1))
            del part['images']
        n_train=len(y['train']);union=torch.cat((x['train'],x['test']));labels=torch.cat((y['train'],y['test']));teacher=torch.cat(teacher)
        route_class_targets=None
        if cfg.get('routing_pair_class_counts') is not None:
            counts=torch.bincount(y['train'],minlength=cfg['num_classes']).cpu().tolist()
            if counts!=cfg['routing_pair_class_counts']:raise ValueError('Router supervision must use exact original TRAIN counts')
            route_class_targets=torch.as_tensor(class_pair_route_targets(counts),device='cuda')
        write(a.out/'training_membership.json',dict(indices=membership.tolist(),removed=removed,full_permutation_each_epoch=True,evaluation_set_unchanged=True))
        parent=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        if cfg.get('expected_parent_checkpoint_sha256') and cfg['expected_parent_checkpoint_sha256']!=sha256(a.parent/'best_checkpoint.pt'):
            raise ValueError('Unexpected fixed parent checkpoint')
        loss_weights=None
        if cfg.get('training_class_weights') is not None:
            if cfg['class_weight_parent_sha256']!=sha256(a.parent/'best_checkpoint.pt'):raise ValueError('Class weights use another parent')
            if cfg['class_weight_class_names']!=identity['classes']:raise ValueError('Class-weight label order')
            p=a.parent/'train_predictions.npz'
            if cfg['class_weight_TRAIN_predictions_sha256']!=sha256(p):raise ValueError('Parent TRAIN prediction identity')
            with np.load(p,allow_pickle=False) as z:
                assert np.array_equal(z['ids'],ids['train']) and np.array_equal(z['labels'],raw['train']['labels'])
                c=cfg['num_classes'];matrix=np.bincount(z['labels']*c+z['scores'].argmax(1),minlength=c*c).reshape(c,c)
            expected=training_error_class_weights(matrix)
            if not np.array_equal(matrix,np.asarray(cfg['class_weight_TRAIN_confusion_matrix'])) or not np.allclose(expected,cfg['training_class_weights'],rtol=1e-7):raise ValueError('TRAIN-derived weight mismatch')
            loss_weights=torch.as_tensor(expected,device='cuda',dtype=torch.float32)
        baseline=json.loads((a.selection_root/'result.json').read_text())['selected_D2NN_development_reference']
        parent_model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda().eval();parent_model.load_state_dict(parent['model'])
        parent_val,parent_val_scores,parent_val_routes,_=evaluate(parent_model,x['val'],y['val'],cfg['batch']);parent_dev,*_=evaluate(parent_model,x['test'],y['test'],cfg['batch'])
        parent_state={k:v.detach().cpu().clone() for k,v in parent_model.state_dict().items()};del parent_model
        metadata=dict(role='ORIGINAL_TEST_EXPOSED_BOUNDARY_DEVELOPMENT',independent_test_claim=False,profile=cfg,data=identity,
                      source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256=sha256(Path(__file__)),command=os.sys.argv,
                      torch=torch.__version__,gpu_uuid=gpu,pid=os.getpid(),parent_checkpoint_sha256=sha256(a.parent/'best_checkpoint.pt'),
                      original_test_in_teacher_training=True,original_test_inputs_in_student_training=True,
                      original_test_ground_truth_used_in_student_loss=False,teacher_hard_pseudo_labels_on_original_test=True,
                      teacher_confidence=cfg['teacher_confidence'],router_frozen_by_variant={v['name']:not v.get('train_router',False) for v in cfg['variants']},parameter_count=479364,
                      training_class_weights=cfg.get('training_class_weights'),class_weight_source='parent_original_TRAIN_confusion_matrix' if loss_weights is not None else None,
                      routing_supervision_class_targets=route_class_targets.cpu().tolist() if route_class_targets is not None else None,
                      routing_supervision_absent_from_inference=True,
                      training_temperature_absent_from_inference=True,teacher_absent_from_inference=True,
                      parent_validation=parent_val,parent_original_test_development=parent_dev,D2NN_reference=baseline)
        write(a.out/'metadata.json',metadata);results=[]
        key=lambda m:(m['accuracy'],m['balanced_accuracy'],-m['cross_entropy'])
        for variant in cfg['variants']:
            if time.time()>=cfg['stop_training_unix']:break
            out=a.out/variant['name'];out.mkdir();model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda()
            model.load_state_dict(parent_state)
            initialization_report=initialize_main_phases(model,variant['initialization'],cfg['seed'])
            routing_weight=variant.get('routing_supervision_weight',0.)
            if routing_weight and (route_class_targets is None or not variant.get('train_router',False)):
                raise ValueError('Dense router supervision requires training targets and trainable optical router')
            gauge_error=None
            if variant['initialization']=='common_phase_gauge':
                _,scores,routes,_=evaluate(model,x['val'],y['val'],cfg['batch'])
                gauge_error=float(np.max(np.abs(scores-parent_val_scores)))
                if not np.allclose(scores,parent_val_scores,rtol=1e-5,atol=1e-6) or not np.array_equal(scores.argmax(1),parent_val_scores.argmax(1)) or not np.array_equal(routes,parent_val_routes):
                    raise ValueError('Common-phase initialization failed full VAL numerical invariance')
            optimizer,trainable=configure_phase_optimizer(model,cfg['phase_lr'],variant);rng=np.random.default_rng(cfg['seed'])
            base_lrs=[group['lr'] for group in optimizer.param_groups]
            trial_metadata=dict(metadata,router_frozen=not variant.get('train_router',False),trainable_parameter_count=sum(p.numel() for p in trainable),
                                initialization_diagnostics=initialization_report,gauge_full_VAL_probability_max_difference=gauge_error)
            write(out/'metadata.json',trial_metadata)
            best=key(parent_dev);best_epoch=0;last_improvement=0;history=[];started=time.time();completed=0;stop_reason='epoch_limit';evaluations=1
            def save_best(state,epoch,kind,val,dev):
                torch.save(dict(model=state,epoch=epoch,weights_kind=kind,validation_diagnostic=val,original_test_dev=dev,metadata=trial_metadata,variant=variant),out/'best_checkpoint.pt')
            save_best(parent_state,0,'parent',parent_val,parent_dev)
            for epoch in range(1,cfg['epochs']+1):
                if time.time()>=cfg['stop_training_unix']:stop_reason='deadline';break
                begin=time.time();model.train();order=rng.permutation(membership);loss_sum=0.;route_loss_sum=0.;count=0;partial=False
                factor=.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['epochs']-1)))
                lr=cfg['phase_lr']*factor
                for group,base_lr in zip(optimizer.param_groups,base_lrs):group['lr']=base_lr*factor
                for start in range(0,len(order),cfg['batch']):
                    if time.time()>=cfg['stop_training_unix']:partial=True;stop_reason='deadline';break
                    idx=torch.as_tensor(order[start:start+cfg['batch']],device='cuda');optimizer.zero_grad(set_to_none=True)
                    result=model(union[idx],return_dense_route=bool(routing_weight));loss=boundary_objective(result['logits'],labels[idx],idx<n_train,teacher[idx],cfg['training_temperature'],cfg['teacher_confidence'],cfg['energy_nll_weight'],loss_weights)
                    if routing_weight:
                        route_loss=dense_route_supervision(result['dense_route_power'],labels[idx],idx<n_train,teacher[idx],route_class_targets,cfg['teacher_confidence'])
                        loss=loss+routing_weight*route_loss;route_loss_sum+=float(route_loss.detach())*len(idx)
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite boundary loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):raise RuntimeError('trainable phase gradient failure')
                    if not variant.get('train_router',False):assert model.router_phase.grad is None
                    if not isinstance(optimizer,WrappedPhaseAdam):
                        torch.nn.utils.clip_grad_norm_(trainable,cfg['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step()
                    loss_sum+=float(loss.detach())*len(idx);count+=len(idx)
                row=dict(epoch=epoch,training_objective=loss_sum/max(count,1),seen_training_inputs=count,full_epoch=not partial,lr=lr)
                if routing_weight:row['unweighted_dense_routing_KL']=route_loss_sum/max(count,1)
                if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==cfg['epochs'] or partial:
                    val,*_=evaluate(model,x['val'],y['val'],cfg['batch']);dev,*_=evaluate(model,x['test'],y['test'],cfg['batch']);evaluations+=1
                    row.update(validation_diagnostic=val,original_test_dev=dev)
                    admissible=val['accuracy']>=parent_val['accuracy']-.02 and val['balanced_accuracy']>=parent_val['balanced_accuracy']-.03
                    if admissible and key(dev)>best:
                        best=key(dev);best_epoch=epoch;last_improvement=epoch
                        save_best({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch,'online',val,dev)
                row.update(seconds=time.time()-begin,best_epoch=best_epoch);history.append(row);write(out/'history.json',history)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metadata=trial_metadata,variant=variant),out/'last_checkpoint.pt')
                completed=epoch if not partial else epoch-1
                write(out/'status.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,best_original_test_development_accuracy=best[0]));write(a.out/'status.json',dict(state='training',variant=variant['name'],epoch=epoch,pid=os.getpid()));print(json.dumps(row),flush=True)
                if partial:break
                if epoch>=variant.get('minimum_epochs',cfg['minimum_epochs']) and epoch-last_improvement>=cfg['patience']:stop_reason='plateau';break
            chosen=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
            if not variant.get('train_router',False):assert torch.equal(model.router_phase.detach().cpu(),parent_state['router_phase'])
            result=dict(role=metadata['role'],independent_test_claim=False,variant=variant,selected_epoch=best_epoch,selected_kind=chosen['weights_kind'],completed_epochs=completed,
                        router_frozen=trial_metadata['router_frozen'],trainable_parameter_count=trial_metadata['trainable_parameter_count'],
                        router_phase_L2_change=float((model.router_phase.detach().cpu()-parent_state['router_phase']).norm()),
                        stop_reason=stop_reason,elapsed_seconds=time.time()-started,selection_evaluations=evaluations,checkpoint_sha256=sha256(out/'best_checkpoint.pt'))
            for split in ('train','val','test'):
                metrics,p,q,dense=evaluate(model,x[split],y[split],cfg['batch']);result[split if split!='test' else 'original_test_dev']=metrics
                np.savez_compressed(out/f'{split}_predictions.npz',ids=ids[split],labels=y[split].cpu().numpy(),scores=p,predictions=p.argmax(1),route_power=q,dense_route_power=dense)
            result['gain_over_D2NN_pp']=100*(result['original_test_dev']['accuracy']-baseline['original_test_dev']['accuracy'])
            write(out/'result.json',result);write(out/'status.json',dict(state='complete',**result));results.append(result)
            del model,optimizer;torch.cuda.empty_cache()
        write(a.out/'result.json',dict(role=metadata['role'],independent_test_claim=False,variants=results));write(a.out/'status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as exc:
        write(a.out/'status.json',dict(state='failed_or_interrupted',error=repr(exc),pid=os.getpid()));raise
    finally:torch.cuda.empty_cache();write(a.out/'exit_receipt.json',dict(pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
