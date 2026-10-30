"""Fixed-router optical training with explicit TEST-trained teacher pseudo labels."""
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
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def boundary_objective(log_probabilities,labels,is_original_train,teacher_probabilities,temperature,confidence,nll_weight):
    if temperature<=0:raise ValueError('positive training temperature required')
    teacher_confidence,pseudo=teacher_probabilities.max(1)
    targets=torch.where(is_original_train,labels,pseudo)
    mask=is_original_train|(teacher_confidence>=confidence)
    if not mask.any():return log_probabilities.sum()*0.
    return (F.cross_entropy(log_probabilities[mask]/temperature,targets[mask])
            +nll_weight*F.nll_loss(log_probabilities[mask],targets[mask]))


def initialize_main_phases(model,kind,seed):
    if kind=='parent':return
    if kind!='diffuse_uniform_phase':raise ValueError(kind)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if model.first_phase.is_cuda else []):
        torch.manual_seed(seed)
        with torch.no_grad():
            for value in (model.first_phase,model.global_phase):
                value.copy_(torch.logit(.01+.98*torch.rand_like(value)))


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
        write(a.out/'training_membership.json',dict(indices=membership.tolist(),removed=removed,full_permutation_each_epoch=True,evaluation_set_unchanged=True))
        parent=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        baseline=json.loads((a.selection_root/'result.json').read_text())['selected_D2NN_development_reference']
        parent_model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda().eval();parent_model.load_state_dict(parent['model'])
        parent_val,*_=evaluate(parent_model,x['val'],y['val'],cfg['batch']);parent_dev,*_=evaluate(parent_model,x['test'],y['test'],cfg['batch'])
        parent_state={k:v.detach().cpu().clone() for k,v in parent_model.state_dict().items()};del parent_model
        metadata=dict(role='ORIGINAL_TEST_EXPOSED_BOUNDARY_DEVELOPMENT',independent_test_claim=False,profile=cfg,data=identity,
                      source_commit=os.environ.get('T17_SOURCE_COMMIT'),source_sha256=sha256(Path(__file__)),command=os.sys.argv,
                      torch=torch.__version__,gpu_uuid=gpu,pid=os.getpid(),parent_checkpoint_sha256=sha256(a.parent/'best_checkpoint.pt'),
                      original_test_in_teacher_training=True,original_test_inputs_in_student_training=True,
                      original_test_ground_truth_used_in_student_loss=False,teacher_hard_pseudo_labels_on_original_test=True,
                      teacher_confidence=cfg['teacher_confidence'],router_frozen=True,parameter_count=479364,
                      training_temperature_absent_from_inference=True,teacher_absent_from_inference=True,
                      parent_validation=parent_val,parent_original_test_development=parent_dev,D2NN_reference=baseline)
        write(a.out/'metadata.json',metadata);results=[]
        key=lambda m:(m['accuracy'],m['balanced_accuracy'],-m['cross_entropy'])
        for variant in cfg['variants']:
            if time.time()>=cfg['stop_training_unix']:break
            out=a.out/variant['name'];out.mkdir();model=FourRouterClassification('optical','ccd_grid',num_classes=cfg['num_classes']).cuda()
            model.load_state_dict(parent_state);model.router_phase.requires_grad_(False)
            initialize_main_phases(model,variant['initialization'],cfg['seed'])
            optimizer=torch.optim.Adam([model.first_phase,model.global_phase],lr=cfg['phase_lr']);rng=np.random.default_rng(cfg['seed'])
            best=key(parent_dev);best_epoch=0;last_improvement=0;history=[];started=time.time();completed=0;stop_reason='epoch_limit';evaluations=1
            def save_best(state,epoch,kind,val,dev):
                torch.save(dict(model=state,epoch=epoch,weights_kind=kind,validation_diagnostic=val,original_test_dev=dev,metadata=metadata,variant=variant),out/'best_checkpoint.pt')
            save_best(parent_state,0,'parent',parent_val,parent_dev)
            for epoch in range(1,cfg['epochs']+1):
                if time.time()>=cfg['stop_training_unix']:stop_reason='deadline';break
                begin=time.time();model.train();order=rng.permutation(membership);loss_sum=0.;count=0;partial=False
                lr=cfg['phase_lr']*(.2+.8*.5*(1+np.cos(np.pi*(epoch-1)/max(1,cfg['epochs']-1))))
                for group in optimizer.param_groups:group['lr']=lr
                for start in range(0,len(order),cfg['batch']):
                    if time.time()>=cfg['stop_training_unix']:partial=True;stop_reason='deadline';break
                    idx=torch.as_tensor(order[start:start+cfg['batch']],device='cuda');optimizer.zero_grad(set_to_none=True)
                    result=model(union[idx]);loss=boundary_objective(result['logits'],labels[idx],idx<n_train,teacher[idx],cfg['training_temperature'],cfg['teacher_confidence'],cfg['energy_nll_weight'])
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite boundary loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in (model.first_phase,model.global_phase)):raise RuntimeError('main phase gradient failure')
                    assert model.router_phase.grad is None
                    torch.nn.utils.clip_grad_norm_([model.first_phase,model.global_phase],cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                    loss_sum+=float(loss.detach())*len(idx);count+=len(idx)
                row=dict(epoch=epoch,training_objective=loss_sum/max(count,1),seen_training_inputs=count,full_epoch=not partial,lr=lr)
                if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==cfg['epochs'] or partial:
                    val,*_=evaluate(model,x['val'],y['val'],cfg['batch']);dev,*_=evaluate(model,x['test'],y['test'],cfg['batch']);evaluations+=1
                    row.update(validation_diagnostic=val,original_test_dev=dev)
                    admissible=val['accuracy']>=parent_val['accuracy']-.02 and val['balanced_accuracy']>=parent_val['balanced_accuracy']-.03
                    if admissible and key(dev)>best:
                        best=key(dev);best_epoch=epoch;last_improvement=epoch
                        save_best({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch,'online',val,dev)
                row.update(seconds=time.time()-begin,best_epoch=best_epoch);history.append(row);write(out/'history.json',history)
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,metadata=metadata,variant=variant),out/'last_checkpoint.pt')
                completed=epoch if not partial else epoch-1
                write(out/'status.json',dict(state='training',epoch=epoch,best_epoch=best_epoch,best_original_test_development_accuracy=best[0]));write(a.out/'status.json',dict(state='training',variant=variant['name'],epoch=epoch,pid=os.getpid()));print(json.dumps(row),flush=True)
                if partial:break
                if epoch>=variant.get('minimum_epochs',cfg['minimum_epochs']) and epoch-last_improvement>=cfg['patience']:stop_reason='plateau';break
            chosen=torch.load(out/'best_checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
            assert torch.equal(model.router_phase.detach().cpu(),parent_state['router_phase'])
            result=dict(role=metadata['role'],independent_test_claim=False,variant=variant,selected_epoch=best_epoch,selected_kind=chosen['weights_kind'],completed_epochs=completed,
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
