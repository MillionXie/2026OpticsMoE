"""Bounded TRAIN-only adaptation of the approved full70 existing condition head.

No camera, optical-mask, alpha, Qwen, diffusion, sampling or model-structure
change. A derived checkpoint is a new candidate and requires its own SIM/EXP.
"""
import argparse,json,math,os,random,sys,time
from pathlib import Path
from .scene_lab_protocol import put,sha
from .scene_lab_middle8 import F6,PAIRS,checked_binding,require_training,hex_sha

PARENT='aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8'
NOMINAL='bf70b5ca54c5d3d0455eae812b1ed8fc03452fe2666db48c90176a58b9acfcb8'
SCHEMA='scene_lab_full70_existing_head_adaptation_v1'


def full_binding(binding):
    checked_binding(binding)
    if binding['schema']!='scene_lab_full_unet8_binding_v1'or binding['generator_sha256']!=PARENT:
        raise ValueError('Only this explicitly approved full70 parent is eligible')
    return binding


def fit_protocol(steps,seconds,lr,batch,anchor,paired_consistency=0.):
    if (type(steps)is not int or not 4<=steps<=1200 or not math.isfinite(seconds)or not 0<seconds<=900
        or not math.isfinite(lr)or not 0<lr<=1e-4 or type(batch)is not int or not 1<=batch<=4
        or anchor not in (0.,.1,1.) or paired_consistency not in (0.,1.)
        or (paired_consistency and anchor!=.1)):
        raise ValueError('Finite existing-head recipe required; registered anchor only0,0.1 or1.0')


def paired_calibration_loss(actual,nominal,target,anchor,consistency):
    """TRAIN-only agreement with a stop-gradient nominal branch; no inference change."""
    from .scene_lab_adapt import domain_alignment_loss
    value=domain_alignment_loss(actual,nominal,target,anchor)
    if not consistency:return value
    if consistency!=1. or anchor!=.1 or nominal is None:raise ValueError('Only the registered paired calibration recipe')
    from .qwen3_scene_conditioning import alignment_loss
    agreement=alignment_loss(actual,{k:v.detach()for k,v in nominal.items()})[0]/3
    return(value+agreement)/2


def equal_value(left,right):
    import torch
    if torch.is_tensor(left)or torch.is_tensor(right):
        return torch.is_tensor(left)and torch.is_tensor(right)and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict)and isinstance(right,dict):
        return set(left)==set(right)and all(equal_value(left[k],right[k])for k in left)
    if isinstance(left,(list,tuple))or isinstance(right,(list,tuple)):
        return type(left)is type(right)and len(left)==len(right)and all(equal_value(a,b)for a,b in zip(left,right))
    return type(left)is type(right)and left==right


def checked_child(child,parent,report,binding):
    """Prove that ONLY the original post-Qwen head changed, not CCD inputs."""
    full_binding(binding);meta=child.get('lab_adaptation',{})
    if (meta.get('schema')!=SCHEMA or meta.get('parent_generator_SHA256')!=PARENT
        or meta.get('frontend_F6_SHA256')!=F6 or meta.get('TRAIN')!=624
        or meta.get('TEST_gradients')is not False or meta.get('added_deployment_parameters')!=0
        or meta.get('frozen_body_exact')is not True or meta.get('trainable_parameters')!=7791616
        or not all(hex_sha(meta.get(k))for k in ('measured_ledger_SHA256','measured_features_SHA256','nominal_conditions_SHA256'))
        or report.get('lab_adaptation_complete')is not True or report.get('parent_generator_SHA256')!=PARENT
        or report.get('complete')is not True or report.get('compression_training_performed')is not True
        or report.get('depth_compression')!=parent['depth_compression']
        or not hex_sha(child.get('source_commit'),40)or report.get('source_commit')!=child.get('source_commit')):
        raise ValueError('Explicit complete TRAIN-only same-body adaptation lineage required')
    allowed={'head','source_commit','lab_adaptation'}
    if set(child)!=set(parent)|{'lab_adaptation'}:
        raise ValueError('Unexpected derived checkpoint fields')
    if not all(equal_value(v,child[k])for k,v in parent.items()if k not in allowed):
        raise ValueError('Frozen optical/diffusion/sampling/metadata body changed; old CCDs cannot be reused')
    if (set(child['head'])!=set(parent['head'])
        or any(v.shape!=child['head'][k].shape or v.dtype!=child['head'][k].dtype for k,v in parent['head'].items())):
        raise ValueError('Original electronic head structure/dtype changed')
    return meta


def nominal(a,binding):
    import torch
    from .scene_lab_middle8 import check_data
    from .scene_lab_middle8_runtime import load_head
    from .scene_lab_adapt import match_train
    from .scene_tail_batches import pack_features
    report=json.loads((a.nominal_features/'report.json').read_text());file=a.nominal_features/'paired_cache.pt'
    if not report['complete']or sha(file)!=NOMINAL or report['cache_sha256']!=NOMINAL:raise ValueError('Current nominal cache changed')
    raw=torch.load(file,map_location='cpu',weights_only=False,mmap=True)
    if raw['pair_sha256']!=PAIRS or raw['optical_frontend']['checkpoint_sha256']!=F6:raise ValueError('Wrong nominal upstream')
    requests=check_data(a.data)['train'];require_training(requests);pairs=match_train(requests,raw['records'])
    head,parent=load_head(a.parent/'best_checkpoint.pt',json.loads((a.parent/'report.json').read_text()),binding)
    torch.set_num_threads(2);targets={};began=time.monotonic()
    with torch.inference_mode():
        for index in range(0,len(pairs),4):
            group=pairs[index:index+4];h,v,p=pack_features([raw['features'][key]for _,key in group],'cpu')
            values=head(h,v,p)
            for i,(_,key)in enumerate(group):targets[key]={k:t[i].cpu().clone()for k,t in values.items()}
            if (index+4)%64==0 or index+4>=len(pairs):
                (a.out/'progress.json').write_text(json.dumps(dict(stage='frozen_parent_TRAIN_conditions_CPU',completed=min(index+4,len(pairs)),total=624)))
    payload=dict(schema='scene_lab_middle8_nominal_conditions_v1',binding=binding,pair_sha256=PAIRS,TRAIN_only=True,
        records=raw['records'],targets=targets,source_nominal_cache_SHA256=NOMINAL,source_nominal_cache=str(file.resolve()),
        parent_generator_SHA256=PARENT,TEST_gradients=False)
    torch.save(payload,a.out/'paired_cache.pt')
    put(a.out/'report.json',dict(complete=True,TRAIN=624,cache_sha256=sha(a.out/'paired_cache.pt'),
        source_nominal_cache_SHA256=NOMINAL,parent_generator_SHA256=PARENT,CPU_only=True,
        electronic_teacher_target_bank_used=False,SDK=False,TEST_gradients=False,seconds=time.monotonic()-began))


def fit(a,binding):
    import torch
    from .scene_lab_middle8_runtime import load_head,matched_training_inputs
    from .scene_lab_adapt import domain_alignment_loss
    from .scene_tail_batches import pack_features
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d':raise ValueError('Only authorized deployment GPU1')
    fit_protocol(a.steps,a.fit_seconds,a.lr,a.batch_size,a.nominal_anchor,a.paired_consistency)
    if a.fixed_sim_reference not in (None,PARENT)or bool(a.fixed_sim_reference)!=bool(a.paired_consistency):
        raise ValueError('Explicit approved fixed original SIM required only for the new paired calibration')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25,0);torch.manual_seed(20261101)
    physical_report=json.loads((a.measured/'report.json').read_text());physical_file=a.measured/'TRAIN_measured_features.pt'
    nominal_report=json.loads((a.nominal/'report.json').read_text());nominal_file=a.nominal/'paired_cache.pt'
    if(not physical_report['complete']or physical_report['contexts']!=624 or sha(physical_file)!=physical_report['feature_SHA256']
       or not nominal_report['complete']or sha(nominal_file)!=nominal_report['cache_sha256']):raise ValueError('Changed or incomplete TRAIN caches')
    physical=torch.load(physical_file,map_location='cpu',weights_only=False,mmap=True)
    normal=torch.load(nominal_file,map_location='cpu',weights_only=False,mmap=True)
    pairs=matched_training_inputs(physical,normal)
    if physical['binding']!=binding or normal['parent_generator_SHA256']!=PARENT or normal['source_nominal_cache_SHA256']!=NOMINAL:
        raise ValueError('Another candidate/cache is not the approved nominal teacher')
    if set(physical['features'])!={sid for sid,_ in pairs}or set(normal['targets'])!={key for _,key in pairs}:raise ValueError('All624 exact TRAIN contexts required')
    original,parent=load_head(a.parent/'best_checkpoint.pt',json.loads((a.parent/'report.json').read_text()),binding)
    head=original.cuda().train().requires_grad_(True);params=list(head.parameters());count=sum(p.numel()for p in params)
    if count!=7791616:raise ValueError('Original-head parameter budget changed')
    raw=None
    if a.nominal_anchor:
        path=Path(normal['source_nominal_cache'])
        if sha(path)!=NOMINAL:raise ValueError('Anchoring features changed')
        raw=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    def objective(indices,device='cuda'):
        group=[pairs[i]for i in indices];h,v,p=pack_features([physical['features'][sid]for sid,_ in group],device)
        target={k:torch.stack([normal['targets'][key][k]for _,key in group]).to(device).float()for k in ('text','image','pooled')}
        anchored=None
        if raw is not None:
            nh,nv,npad=pack_features([raw['features'][key]for _,key in group],device);anchored=head(nh,nv,npad)
        return paired_calibration_loss(head(h,v,p),anchored,target,a.nominal_anchor,a.paired_consistency)
    def evaluate(device='cuda'):
        head.eval()
        with torch.no_grad():score=sum(float(objective(list(range(i,min(i+a.batch_size,624))),device))*(min(a.batch_size,624-i))for i in range(0,624,a.batch_size))/624
        head.train();return score
    baseline=evaluate();best=baseline;best_step=0;step=0;rng=random.Random(20261101);order=[];seen=set();began=time.monotonic()
    history=[dict(step=0,TRAIN_condition_loss=baseline)]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
    def save(name,score):
        child=dict(parent,head={k:v.detach().cpu().clone()for k,v in head.state_dict().items()},source_commit=os.environ['T12_SOURCE_COMMIT'],
            lab_adaptation=dict(schema=SCHEMA,parent_generator_SHA256=PARENT,frontend_F6_SHA256=F6,TRAIN=624,
                measured_ledger_SHA256=physical['ledger_SHA256'],measured_features_SHA256=physical_report['feature_SHA256'],
                nominal_conditions_SHA256=nominal_report['cache_sha256'],TRAIN_condition_loss=score,step=step,
                TEST_gradients=False,added_deployment_parameters=0,frozen_body_exact=True,trainable_parameters=count,
                nominal_anchor_weight=a.nominal_anchor,paired_consistency_weight=a.paired_consistency,
                fixed_SIM_reference_SHA256=a.fixed_sim_reference,
                requires_fresh_same_weight_SIM=not bool(a.fixed_sim_reference)))
        torch.save(child,a.out/name)
    put(a.out/'protocol.json',dict(parent_generator_SHA256=PARENT,TRAIN=624,TEST_gradients=False,steps_max=a.steps,
        fit_seconds_max=a.fit_seconds,lr=a.lr,batch_size=a.batch_size,weight_decay=.01,nominal_anchor_weight=a.nominal_anchor,
        loss='mean(normalized_MSE + 0.1 cosine distance), original-head nominal TRAIN conditions',
        paired_consistency_weight=a.paired_consistency,nominal_agreement_branch_stop_gradient=True,
        fixed_SIM_reference_SHA256=a.fixed_sim_reference,
        comparison='Fixed pre-adaptation SIM, not same-PT SIM'if a.fixed_sim_reference else'Fresh same-PT SIM required',
        head_parameters=count,new_deployment_parameters=0,best_selection='TRAIN loss only; subsequent complete TEST500 is development selection'))
    save('best_checkpoint.pt',baseline);save('last_checkpoint.pt',baseline)
    for step in range(1,a.steps+1):
        if not order:order=list(range(624));rng.shuffle(order)
        group=order[:a.batch_size];del order[:a.batch_size];seen.update(group)
        lr=a.lr*min(1.,step/30)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for g in optimizer.param_groups:g['lr']=lr
        loss=objective(group)
        if not torch.isfinite(loss):raise ValueError('Nonfinite TRAIN-only fit')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
        if step%20==0:(a.out/'progress.json').write_text(json.dumps(dict(stage='TRAIN_existing_head_fit',step=step,loss=float(loss),distinct_TRAIN_seen=len(seen))))
        if step%100==0 or step==a.steps or time.monotonic()-began>=a.fit_seconds:
            score=evaluate();history.append(dict(step=step,TRAIN_condition_loss=score,seconds=time.monotonic()-began));(a.out/'history.json').write_text(json.dumps(history,indent=2))
            save('last_checkpoint.pt',score)
            if score<best:best=score;best_step=step;save('best_checkpoint.pt',score)
            if time.monotonic()-began>=a.fit_seconds:break
    selected=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    report=dict(complete=True,lab_adaptation_complete=True,parent_generator_SHA256=PARENT,
        source_commit=os.environ['T12_SOURCE_COMMIT'],source_SHA256=sha(__file__),TRAIN=624,distinct_TRAIN_seen=len(seen),
        depth_compression=parent['depth_compression'],compression_training_performed=True,
        compression_training_inherited_not_repeated=True,best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),
        steps=step,best_step=best_step,baseline_TRAIN_condition_loss=baseline,best_TRAIN_condition_loss=best,
        nominal_anchor_weight=a.nominal_anchor,TEST_gradients=False,SDK=False,added_deployment_parameters=0,
        paired_consistency_weight=a.paired_consistency,fixed_SIM_reference_SHA256=a.fixed_sim_reference,
        quality_accepted=False,original_parent_head_retained=best_step==0,requires_fresh_same_weight_SIM=not bool(a.fixed_sim_reference))
    checked_child(selected,parent,report,binding)
    head.cpu();torch.cuda.empty_cache();head.load_state_dict(selected['head'],strict=True)
    verified=evaluate('cpu')
    if abs(verified-best)>max(2e-6,abs(best)*1e-3):raise ValueError('Strict CPU selected-head replay differs from GPU TRAIN loss')
    report.update(strict_CPU_TRAIN_condition_loss=verified,frozen_body_exact=True)
    put(a.out/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['nominal','fit'],required=True)
    for name in ('binding','parent','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('nominal-features','data','measured','nominal'):p.add_argument('--'+name,type=Path)
    p.add_argument('--steps',type=int,default=1200);p.add_argument('--fit-seconds',type=float,default=900)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--nominal-anchor',type=float,default=0.)
    p.add_argument('--paired-consistency',type=float,default=0.);p.add_argument('--fixed-sim-reference')
    a=p.parse_args();binding=full_binding(json.loads(a.binding.read_text()))
    if not hex_sha(os.environ.get('T12_SOURCE_COMMIT'),40):raise ValueError('Published execution-source commit required')
    if a.mode=='nominal'and(a.nominal_features is None or a.data is None):p.error('Nominal mode requires explicit original features/data')
    if a.mode=='fit'and(a.measured is None or a.nominal is None):p.error('Fit requires complete measured and nominal TRAIN caches')
    if a.mode=='nominal'and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Nominal target creation is CPU only')
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'execution.json',dict(argv=sys.argv,source_commit=os.environ['T12_SOURCE_COMMIT'],source_SHA256=sha(__file__),SDK=False))
    try:(nominal if a.mode=='nominal'else fit)(a,binding)
    except BaseException as e:
        import traceback
        put(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
