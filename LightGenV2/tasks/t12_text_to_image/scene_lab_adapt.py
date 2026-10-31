"""Measured TRAIN replay and existing-head adaptation; never opens an SDK/TEST.

Teacher = frozen F9 on matching original F6 SIM TRAIN features. Student reads
the six actual CCDs through the original complete Qwen. Only its existing
7,791,616-parameter condition head is optimized. No new deployment parameters,
optical/alpha changes, diffusion changes or claim of passed generation quality.
"""
import argparse,json,os,random,math,time,sys,traceback
from pathlib import Path
from types import SimpleNamespace
from .scene_lab_protocol import Ledger,STAGES,identity,sha,put
from .scene_lab_native import F9,F6,NativeRunner,load_head


def validate_nominal_anchor(weight):
    if not math.isfinite(weight) or not 0<=weight<=1:
        raise ValueError('Nominal TRAIN anchoring weight must be finite in [0,1]')
    return weight


def domain_alignment_loss(actual,nominal,target,weight):
    """Optional dual-domain TRAIN objective, with no new deployment module."""
    from .qwen3_scene_conditioning import alignment_loss
    validate_nominal_anchor(weight)
    physical=alignment_loss(actual,target)[0]/3
    if weight==0:return physical
    if nominal is None:raise ValueError('Nominal TRAIN features required for anchoring')
    return (physical+weight*alignment_loss(nominal,target)[0]/3)/(1+weight)


def match_train(requests,records):
    """Join physical and SIM inputs by exact source pixels AND public prompt."""
    if not requests or any(r.get('split')!='train' for r in requests+records):raise ValueError('Only TRAIN may be used for adaptation')
    def key(row):return row['reference_sha256'],row['prompt']
    by_key={key(row):row for row in records}
    if len(by_key)!=len(records) or len({key(r) for r in requests})!=len(requests) or set(by_key)!={key(r) for r in requests}:
        raise ValueError('Physical and original SIM TRAIN cohorts differ')
    if len({r['sample_id'] for r in requests})!=len(requests):raise ValueError('Duplicate physical request')
    if len({row['input_key'] for row in records})!=len(records):raise ValueError('Duplicate SIM feature key')
    return [(row['sample_id'],by_key[key(row)]['input_key']) for row in requests]


def gpu_setup(a):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d':raise ValueError('Use only user-authorized shared GPU1')
    if not 0<a.gpu_memory_fraction<=.25:raise ValueError('Keep the shared memory cap at or below one quarter')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction,0)


def streaming_admission(ledger):
    """Read-only final-output replay can overlap capture, never field creation.

    Every preceding cohort must already be complete. The frozen final-stage
    input producer must be finished/GPU-released. Each sample still passes the
    full Ledger.infer SHA/lineage checks after its actual final CCD arrives.
    """
    base=ledger.root.parent
    for stage in STAGES[:-1]:
        report=json.loads((base/(stage+'_outputs_data_stream_complete.json')).read_text())
        if not report['complete']or report['count']!=len(ledger.requests)or report['ledger_SHA256']!=ledger.key:raise ValueError('Incomplete/wrong preceding actual cohort')
    release=json.loads((base/('native_'+STAGES[-1])/'release.json').read_text())
    if release['returncode']!=0 or not release['own_GPU_process_absent']:raise ValueError('Final input producer must finish and release its GPU first')
    if any(not(ledger.folder(STAGES[-1],r)/'input.ready').exists()for r in ledger.requests):raise ValueError('All final native inputs required before streaming replay')


def wait_final_capture(ledger,request,seconds=600):
    deadline=time.monotonic()+seconds;marker=ledger.folder(STAGES[-1],request)/'capture.ready'
    while not marker.exists():
        pointer=ledger.root.parent/(STAGES[-1]+'_active_capture.json');job='capture_'+STAGES[-1]
        if pointer.exists():job=json.loads(pointer.read_text())['job']
        returned=ledger.root.parent/(job+'_return.json')
        if returned.exists()and json.loads(returned.read_text())['returncode']!=0:raise RuntimeError('Final actual capture stopped; never substitute ideal data')
        if time.monotonic()>=deadline:raise TimeoutError('Bounded wait for an actual final CCD expired')
        time.sleep(.2)


def complete_cohort_admission(ledger):
    """Closed DATA cohorts admit replay, not unverified model inputs.

    Ledger.infer still hashes all six original frames and tickets for EACH
    request before its forward. Avoid a redundant whole-TRAIN disk sweep.
    """
    for stage in STAGES:
        report=json.loads((ledger.root.parent/(stage+'_outputs_data_stream_complete.json')).read_text())
        if not report['complete']or report['count']!=len(ledger.requests)or report['ledger_SHA256']!=ledger.key:raise ValueError('Incomplete/wrong actual DATA cohort')


def features(a):
    import torch
    from .scene_paired_render import install_optical_frontend
    from .qwen3_scene_paired_train import new_qwen
    c=json.loads((a.ledger/'contract.json').read_text());ledger=Ledger(a.ledger,c['session'],c['models'],c['requests'])
    if c['models']['F6']!=F6 or c['models']['F9']!=F9 or c['session']['scope']!='reviewed_physical':raise ValueError('Wrong physical model/session')
    if len(ledger.requests)!=608 or any(r['split']!='train'for r in ledger.requests):raise ValueError('Only the complete TRAIN608 adaptation cohort is eligible')
    if a.stream_last_stage:streaming_admission(ledger)
    else:complete_cohort_admission(ledger)
    task=Path(__file__).resolve().parent;base=task/'runs/simulation'
    qwen,processor=new_qwen(SimpleNamespace(qwen=a.qwen,qwen_side=256))
    optical=install_optical_frontend(qwen,base/'qwen3_scene_LG_frontend_20261029',base/'qwen3_scene_LG_teacher_features_20261029',a.sim_cache)
    if optical['checkpoint_sha256']!=F6:raise ValueError('Frontend changed')
    head,_=load_head(base/'scene_natural_single_EG_joint12_training_20261030/best_checkpoint.pt',expected_sha256=F9)
    runner=NativeRunner(qwen,processor,head.cuda(),base/'scene_two_sentence_pairs_20261029',frontend_sha256=F6,head_sha256=F9,export_features=True)
    values={};chains={};began=time.monotonic()
    for index,request in enumerate(ledger.requests):
        if a.stream_last_stage:wait_final_capture(ledger,request)
        chain=[];result=ledger.infer(request,runner,receipt_chain=chain);values[request['sample_id']]=result['features']
        chains[request['sample_id']]=identity(chain)
        if (index+1)%16==0 or index+1==len(ledger.requests):
            progress=dict(phase='complete_measured_Qwen_feature_replay',completed=index+1,total=len(ledger.requests),seconds=time.monotonic()-began)
            (a.out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        time.sleep(.03)
    payload=dict(schema='scene_lab_measured_feature_cache_v1',records=ledger.requests,features=values,ledger_SHA256=ledger.key,
        physical_receipt_chain_SHA256=chains,F9=F9,F6=F6,TRAIN_only=True,all_six_stages_measured=True,source_commit=a.source_commit)
    torch.save(payload,a.out/'paired_cache.pt')
    put(a.out/'report.json',dict(complete=True,cache_sha256=sha(a.out/'paired_cache.pt'),TRAIN=len(values),TRAIN_only=True,TEST_opened=False,
        source_commit=a.source_commit,source_SHA256=sha(__file__),native_adapter_SHA256=sha(Path(__file__).with_name('scene_lab_native.py')),
        ledger_SHA256=ledger.key,F9=F9,F6=F6,all_six_stages_measured=True,streamed_while_final_CCD_capture=a.stream_last_stage,native_forward_attempts=runner.native_forward_attempts,
        complete_Qwen_forwards=runner.completed_forwards,SDK_used=False,SDXL_images_generated=0,quality_accepted=False))


def fit(a):
    import torch
    from .qwen3_scene_conditioning import QwenSDXLConditions,alignment_loss
    from .scene_tail_batches import pack_features
    physical_report=json.loads((a.features/'report.json').read_text());sim_report=json.loads((a.sim_cache/'report.json').read_text())
    if not physical_report['complete'] or physical_report['cache_sha256']!=sha(a.features/'paired_cache.pt'):raise ValueError('Measured cache checksum changed')
    if not sim_report['complete'] or sim_report['cache_sha256']!=sha(a.sim_cache/'paired_cache.pt'):raise ValueError('Original SIM cache checksum changed')
    physical=torch.load(a.features/'paired_cache.pt',map_location='cpu',weights_only=False)
    sim=torch.load(a.sim_cache/'paired_cache.pt',map_location='cpu',weights_only=False,mmap=True)
    if physical['schema']!='scene_lab_measured_feature_cache_v1' or physical['F9']!=F9 or physical['F6']!=F6 or not physical['all_six_stages_measured']:
        raise ValueError('Only actual complete frozen F6/F9 features may adapt this head')
    if sim['optical_frontend']['checkpoint_sha256']!=F6:raise ValueError('Wrong nominal optical teacher')
    pairs=match_train(physical['records'],sim['records'])
    nominal_anchor=validate_nominal_anchor(a.nominal_anchor_weight)
    sim_keys=dict(pairs)
    if len(pairs)!=608 or set(physical['features'])!={r[0] for r in pairs}:raise ValueError('Complete TRAIN608 required')
    parent_path=Path(__file__).resolve().parent/'runs/simulation/scene_natural_single_EG_joint12_training_20261030/best_checkpoint.pt'
    teacher,parent=load_head(parent_path,expected_sha256=F9);teacher=teacher.cuda()
    head=QwenSDXLConditions(joint_image_context=True).cuda();head.load_state_dict(parent['head'],strict=True)
    targets={};initial={n:v.detach().clone() for n,v in head.state_dict().items()}
    with torch.inference_mode():
        for sid,key in pairs:
            hidden,visual,padding=pack_features([sim['features'][key]],'cuda')
            targets[sid]={n:t[0].cpu() for n,t in teacher(hidden,visual,padding).items()}
    # Normal tensors are made outside inference mode when assembling a batch.
    del teacher;torch.cuda.empty_cache();head.train().requires_grad_(True)
    parameters=list(head.parameters());count=sum(p.numel() for p in parameters)
    if count!=7791616:raise ValueError('Existing F9 condition-head parameter count changed')
    ids=[sid for sid,_ in pairs];rng=random.Random(20261031);opt=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=.01)
    def objective(indices):
        names=[ids[i] for i in indices];hidden,visual,padding=pack_features([physical['features'][sid] for sid in names],'cuda')
        wanted={n:torch.stack([targets[sid][n] for sid in names]).cuda().float() for n in ('text','image','pooled')}
        nominal=None
        if nominal_anchor:
            h,v,p=pack_features([sim['features'][sim_keys[sid]] for sid in names],'cuda')
            nominal=head(h,v,p)
        return domain_alignment_loss(head(hidden,visual,padding),nominal,wanted,nominal_anchor)
    def evaluate():
        head.eval()
        with torch.no_grad():value=sum(float(objective(list(range(i,min(i+a.batch_size,len(ids))))))*(min(i+a.batch_size,len(ids))-i) for i in range(0,len(ids),a.batch_size))/len(ids)
        head.train();return value
    baseline=evaluate();best=baseline;best_step=0;history=[dict(step=0,TRAIN_condition_loss=baseline)];began=time.monotonic();step=0
    def save(name,score):
        adapted=dict(parent,head={n:v.detach().cpu() for n,v in head.state_dict().items()},source_commit=a.source_commit,
            lab_adaptation=dict(schema='scene_lab_existing_head_adaptation_v1',parent_F9_SHA256=F9,frontend_F6_SHA256=F6,
                measured_ledger_SHA256=physical['ledger_SHA256'],physical_feature_cache_SHA256=physical_report['cache_sha256'],
                SIM_feature_cache_SHA256=sim_report['cache_sha256'],steps=step,TRAIN_condition_loss=score,TEST_used=False,
                nominal_TRAIN_anchor_weight=nominal_anchor,
                unchanged_frontend_alpha_optics_UNet_LoRA=True,added_deployment_parameters=0))
        torch.save(adapted,a.out/name)
    save('best_checkpoint.pt',baseline);save('last_checkpoint.pt',baseline)
    put(a.out/'protocol.json',dict(parent_F9_SHA256=F9,frontend_F6_SHA256=F6,steps_max=a.steps,fit_seconds_max=a.fit_seconds,
        batch_size=a.batch_size,lr=a.lr,weight_decay=.01,trainable_existing_head_parameters=count,TRAIN=608,
        loss='mean(normalized_MSE + 0.1*(1-cosine)) for text/image/pooled, matching frozen F9 SIM conditions',
        nominal_TRAIN_anchor_weight=nominal_anchor,domain_objective='(actual_CCD_TRAIN_loss + w*nominal_TRAIN_loss)/(1+w)',
        selected_by='TRAIN condition loss only; not an image-quality or independent-generalization metric',
        TEST_used=False,DEV_used=False,SDK_used=False,new_parameters=0,physical_ledger_SHA256=physical['ledger_SHA256']))
    for step in range(1,a.steps+1):
        lr=a.lr*min(1.,step/30)*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
        for group in opt.param_groups:group['lr']=lr
        loss=objective(rng.sample(range(len(ids)),min(a.batch_size,len(ids))))
        if not torch.isfinite(loss):raise ValueError('Nonfinite measured-head loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);opt.step()
        if step%20==0:
            progress=dict(phase='TRAIN_existing_electronic_condition_head',step=step,max_steps=a.steps,loss=float(loss),seconds=time.monotonic()-began)
            (a.out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        if step%100==0 or step==a.steps or time.monotonic()-began>=a.fit_seconds:
            score=evaluate();history.append(dict(step=step,TRAIN_condition_loss=score,seconds=time.monotonic()-began));(a.out/'history.json').write_text(json.dumps(history,indent=2))
            save('last_checkpoint.pt',score)
            if score<best:best=score;best_step=step;save('best_checkpoint.pt',score)
            if time.monotonic()-began>=a.fit_seconds:break
    # Strictly rebuild the selected original head; parent optical/UNet states
    # are carried through unchanged, and no earlier-CCD electronic is touched.
    selected=torch.load(a.out/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    head.load_state_dict(selected['head'],strict=True);head.eval()
    verified=evaluate()
    if abs(verified-best)>max(1e-7,abs(best)*1e-5):raise ValueError('Selected-head reload mismatch')
    if set(selected['head'])!=set(parent['head']) or any(selected['head'][n].shape!=parent['head'][n].shape for n in parent['head']):raise ValueError('Head structure changed')
    if any(not torch.equal(selected['unet_lora'][n],parent['unet_lora'][n]) for n in parent['unet_lora']):raise ValueError('Frozen UNet LoRA changed')
    put(a.out/'report.json',dict(complete=True,source_commit=a.source_commit,source_SHA256=sha(__file__),TRAIN=608,steps=step,best_step=best_step,
        baseline_TRAIN_condition_loss=baseline,best_TRAIN_condition_loss=best,strict_reload_TRAIN_condition_loss=verified,
        best_sha256=sha(a.out/'best_checkpoint.pt'),last_sha256=sha(a.out/'last_checkpoint.pt'),parent_F9_SHA256=F9,frontend_F6_SHA256=F6,
        trainable_existing_head_parameters=count,added_deployment_parameters=0,original_best_retained=best_step==0,
        nominal_TRAIN_anchor_weight=nominal_anchor,
        original_optics_alpha_Qwen_and_UNet_LoRA_unchanged=True,TEST_used=False,DEV_used=False,SDK_used=False,
        best_selected_by='TRAIN-only condition reconstruction loss',generation_quality_accepted=False,physical_5percent_gate_passed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('features','fit'),required=True)
    for name in ('ledger','features','sim-cache','qwen'):p.add_argument('--'+name,type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--gpu-memory-fraction',type=float,default=.25);p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--fit-seconds',type=float,default=900)
    p.add_argument('--nominal-anchor-weight',type=float,default=0.,help='Optional TRAIN-only nominal-condition preservation; no inference change')
    p.add_argument('--stream-last-stage',action='store_true',help='Read-only complete-Qwen replay after each actual last CCD arrives; fit still waits for all608')
    a=p.parse_args()
    validate_nominal_anchor(a.nominal_anchor_weight)
    if len(a.source_commit)!=40 or not 4<=a.steps<=1200 or not 1<=a.batch_size<=4 or not 0<a.lr<=1e-4 or not 0<a.fit_seconds<=900:raise ValueError('Bounded original-head training required')
    if a.mode=='features' and (not a.ledger or not a.qwen or not a.sim_cache):raise ValueError('Complete physical ledger/Qwen/original cache required')
    if a.mode=='fit' and (not a.features or not a.sim_cache):raise ValueError('Measured and original SIM caches required')
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=a.source_commit,source_SHA256=sha(__file__),SDK=False))
    try:
        gpu_setup(a)
        (features if a.mode=='features' else fit)(a)
    except BaseException as error:
        put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
