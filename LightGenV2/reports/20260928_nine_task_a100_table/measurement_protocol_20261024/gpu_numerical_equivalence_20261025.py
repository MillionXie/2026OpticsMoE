"""Native-A100 equivalence outside timing; replaces the slow auxiliary CPU gate.

Original arithmetic, precision, actual non-timing TEST inputs and full outputs
are compared on the SAME GPU. Each timed repeat still loads a fresh process.
No validation call is charged as timing, used as warmup, or called a CPU proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
KIND='same_A100_native_GPU_equivalence_20261025'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    from measure_abo_text_qwen import write as save
    save(path,value)


def source_inventory():
    paths={Path(__file__)}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value and Path(value).is_absolute():
            p=Path(value).resolve()
            if p.suffix=='.py' and p.is_relative_to(REPO):paths.add(p)
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}


def platform_proof(path,value,check_sources):
    if (value.get('status')!='passed' or value.get('proof_kind')!=KIND
            or value.get('no_gpu') is not False or value.get('cuda_initialized') is not True
            or value.get('gpu_uuid')!=GPU or value.get('mps_percent')!=100
            or value.get('visible_multiprocessors')!=108 or value.get('power_limit_w')!=250
            or value.get('separate_validation_process_not_a_timed_repeat') is not True):
        raise ValueError('Complete native A100 numerical validation required, not a CPU label')
    audit=Path(path).parent/'process_audit.json'
    if sha(audit)!=value['process_audit_sha256']:raise ValueError('Validation GPU guard evidence changed')
    events=json.loads(audit.read_text());bound=False
    for event in events:
        p=event['platform']
        if (p['uuid']!=GPU or p['name']!='NVIDIA A100-PCIE-40GB' or abs(p['power_limit_w']-250)>.1):
            raise ValueError('Validation physical platform differs')
        clients=set(event['clients'])
        if clients-{value['pid']}:raise ValueError('Foreign validation MPS client')
        if clients:bound=True
        elif bound:raise ValueError('Validation MPS client disappeared')
        if any(int(row[0]) not in {value['pid'],*event['servers']} or row[1].strip()!=GPU for row in event['processes']):
            raise ValueError('Foreign validation GPU process')
    if (not bound or events[0]['unix']>value['first_forward_unix']
            or events[-1]['unix']<value['last_forward_unix']):raise ValueError('Validation guard does not cover both full forwards')
    if check_sources:
        for name,digest in value['sources'].items():
            if sha(REPO/name)!=digest:raise ValueError('Validated original source changed: '+name)
    return value


def t12_proof(path,system,*,check_sources=True):
    if system!='qwen':raise ValueError('This GPU proof is only the missing Qwen system; reuse the qualified Ours CPU proof')
    from t12_live_frontend import BASELINE_SHA
    v=json.loads(Path(path).read_text());platform_proof(path,v,check_sources)
    q=v.get('qwen_report',{})
    if (v.get('baseline_sha256')!=BASELINE_SHA
            or [c.get('test_index') for c in v.get('checks',[])]!=[200,2303]
            or not all(c.get('bitwise_equal') is True and c.get('original_reference_bitwise_equal') is True
                       and c.get('target_passed_to_model') is False for c in v['checks'])
            or q.get('language_layers_retained')!=28 or q.get('vision_tower_used') is not False
            or q.get('language_model_head_used') is not False or not v.get('actual_NN_shapes')):
        raise ValueError('Actual full Qwen28 original/native-GPU two-input comparison missing')
    return v


def lgvq_proof(path,*,check_sources=True):
    import lgvq_live_frontend as live
    v=json.loads(Path(path).read_text());platform_proof(path,v,check_sources)
    if (v.get('manifest_sha256')!=live.MANIFEST_SHA or v.get('original_front_parameter_binding_sha256')!=live.BINDING_SHA
            or [(c.get('target'),c.get('field_index')) for c in v.get('checks',[])]
            !=[('temporal',13),('temporal',14),('spatial',200),('spatial',557)]):
        raise ValueError('Original four-field native GPU proof required')
    for c in v['checks']:
        if (c.get('body_sha256')!=live.PINS[c['target']][1] or c.get('bitwise_equal') is not True
                or c.get('full_original_output_bitwise_equal') is not True
                or c.get('selected_NN_stage_instrumentation_tested') is not True
                or c.get('frontend_transformer_layers_executed')!=0 or len(c.get('six_passes',[]))!=6):
            raise ValueError('Full native frontend/model/merger proof differs')
    return v


def validate_t12(root,guard):
    import torch
    import t12_baseline_live as baseline
    from t12_live_frontend import DATA,TEST_SHA,PINS,ASSETS,reference_input
    from LightGenV2.tasks.t12_text_to_image.product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    from LightGenV2.tasks.t12_text_to_image.product_repair_model import one_step_edit
    from LightGenV2.tasks.t12_text_to_image.export_matched_summary import noise_batch
    baseline.original_statements()
    for path,pin in [(DATA/'test.jsonl',TEST_SHA),*[(ASSETS/'datasets'/name,pin) for name,pin in PINS.items()]]:
        if sha(path)!=pin:raise ValueError('Original real input identity differs')
    write(root/'status.json',dict(status='loading_original_Qwen28_GPU',pid=os.getpid()))
    runtime=baseline.load_runtime(torch.device('cuda:0'));guard.bind()
    dataset=ExpandedUnifiedProductEditDataset(DATA,'test',256,ASSETS/'datasets/abo_unified_expanded_instructions_qwen2_v2.pt')
    frozen=source_inventory();checks=[];shapes=[];first=None;last=None
    with torch.inference_mode():
        for index in (200,2303):
            guard.fast();raw=dataset[index];live=reference_input(dataset,index)
            if (not torch.equal(raw['reference'],live['reference']) or raw['prompt']!=live['prompt']
                    or raw['sample_id']!=live['sample_id']):raise ValueError('Original reference-only input differs')
            tokens=baseline.inputs(runtime,live['prompt']);ref=raw['reference'][None].to(runtime.device)
            write(root/'status.json',dict(status='original_AST_vs_live_GPU',index=index,pid=os.getpid()))
            if first is None:first=time.time()
            expected=baseline.gold_forward(runtime,ref,tokens,live['prompt'],index)
            handles=[]
            for group in ('qwen','adapter','unet','vae'):
                for name,module in getattr(runtime,group).named_modules():
                    if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                        def observe(m,a,b,name=group+'.'+name):
                            shapes.append(dict(test_index=index,module=name,kind=type(m).__name__,input_shape=list(a[0].shape),
                                               output_shape=list(b.shape),weight_shape=list(m.weight.shape),groups=getattr(m,'groups',None)))
                        handles.append(module.register_forward_hook(observe))
            try:actual=baseline.forward(runtime,ref,tokens,index);torch.cuda.synchronize()
            finally:
                for handle in handles:handle.remove()
            if not torch.equal(expected,actual):raise ValueError('Original AST/live full GPU RGB differs')
            if not torch.isfinite(actual).all():raise ValueError('Nonfinite original GPU RGB')
            last=time.time();guard.check()
            checks.append(dict(test_index=index,sample_id=live['sample_id'],bitwise_equal=True,
                original_reference_bitwise_equal=True,target_passed_to_model=False,input_ids=tokens['input_ids'].tolist(),
                attention_mask=tokens['attention_mask'].tolist(),output_shape=list(actual.shape)))
    if frozen!=source_inventory():raise ValueError('Original GPU validation source changed')
    return dict(checks=checks,actual_NN_shapes=shapes,sources=frozen,baseline_sha256=baseline.BASELINE_SHA,
        qwen_report=runtime.qwen_report,frozen_dependency_binding=runtime.frozen_dependency_binding,
        original_export_source_commit=baseline.EXPORT_COMMIT,original_export_source_sha256=baseline.EXPORT_SHA,
        first_forward_unix=first,last_forward_unix=last,native_math='original native GPU FP16 autocast, FP16 VAE; no CPU precision substitution')


def validate_lgvq(root,guard):
    import torch
    import lgvq_live_frontend as live
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import load_model,forward
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    from stage_meter import StageMeter
    if sha(live.MANIFEST)!=live.MANIFEST_SHA:raise ValueError('Original TEST manifest differs')
    tests=[r for r in read_manifest(live.MANIFEST) if r.split=='test'];assert len(tests)==558
    write(root/'status.json',dict(status='loading_native_LGVQ_GPU',pid=os.getpid()))
    front=live.LiveFront('cuda:0');front.load_spatial_stem();guard.bind()
    models={target:load_model(target,live.TASK/'runs/simulation'/live.PINS[target][0]/'best_checkpoint.pt','cuda:0')
            for target in ('temporal','spatial')}
    frozen=source_inventory();checks=[];first=None;last=None
    with torch.inference_mode():
        for target in ('temporal','spatial'):
            run,pin,grid,count=live.PINS[target]
            model,settings=models[target]
            if target=='spatial' and (settings.quality_branch_enabled or not settings.electronic_quality_residual_enabled
                    or not settings.custom_conv_electronic_enabled or model.frame_stem is not None):raise ValueError('Original spatial architecture differs')
            for field_index in ((13,14) if target=='temporal' else (200,557)):
                guard.fast();indices,valid=live.field_indices(558,count,field_index);rows=[tests[i] for i in indices]
                write(root/'status.json',dict(status='original_vs_observed_GPU',target=target,field_index=field_index,pid=os.getpid()))
                if first is None:first=time.time()
                gold,meta=front.build(rows,target);original=forward(model,gold)
                meter=StageMeter(torch.cuda);meter.sample_index=field_index;meter.valid_samples=sum(valid)
                observed,observed_meta=front.build(rows,target,meter)
                live.equal_tree(gold,observed,'inputs');live.equal_tree(meta,observed_meta,'input_metadata')
                covered=live.instrument_body(model,meter);shapes=[];handles=[]
                for name,module in model.named_modules():
                    if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):
                        def hook(m,a,b,name=name):
                            shapes.append(dict(name=name,type=type(m).__name__,input_shape=list(a[0].shape),output_shape=list(b.shape),
                                               weight_shape=list(m.weight.shape),charged_serial=not live.residual_name(name)))
                        handles.append(module.register_forward_hook(hook))
                try:actual=forward(model,observed);live.equal_tree(original,actual,'full_original_output');torch.cuda.synchronize()
                finally:
                    for handle in handles:handle.remove()
                    meter.close()
                passes=[r for r in meter.rows if r['family']=='optical_simulation_FFT']
                if len(passes)!=6 or not any(r['stage']=='frame_merger' for r in meter.rows):raise ValueError('Six full passes/real merger absent')
                checks.append(dict(target=target,body_sha256=pin,field_index=field_index,test_indices=indices,valid=valid,
                    source_ids=meta['source_ids'],bitwise_equal=True,inputs={k:live.tensor_record(v) for k,v in gold.items()},
                    prediction=live.tensor_record(original['prediction']),six_passes=[r['stage'] for r in passes],
                    full_original_output_bitwise_equal=True,selected_NN_stage_instrumentation_tested=True,
                    stage_records=meter.rows,source_ledger=meter.ledger,nn_module_shapes=shapes,
                    covered_neural_modules=sorted(covered),frontend_transformer_layers_executed=0))
                last=time.time();guard.check()
            del model
    if frozen!=source_inventory():raise ValueError('Original LGVQ validation source changed')
    return dict(checks=checks,sources=frozen,manifest_sha256=live.MANIFEST_SHA,original_front_parameter_binding_sha256=live.BINDING_SHA,
                first_forward_unix=first,last_forward_unix=last,native_math='original GPU FP32 body/stem, native BF16 patch/position/embedding and original FP16 CPU feature return')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('t12','lgvq'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    root=args.output.resolve()
    task='t12_text_to_image' if args.task=='t12' else 't06_video_quality_assessment'
    if (root.exists() or not root.is_relative_to(REPO/'LightGenV2/tasks'/task/'runs/simulation')
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100'):
        raise ValueError('Fresh actual A100/MPS100 validation only')
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from measure_t12_cost import check_platform
    engine.GPU=GPU;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY'];root.mkdir(parents=True)
    guard=engine.MPSGuard(SimpleNamespace(deadline=args.deadline));base=guard.check;mutex=threading.Lock()
    def check():
        with mutex:
            base();raw=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,power.limit,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5)
            guard.audit[-1]['platform']=check_platform(raw)
    guard.check=check;guard.check();stop=threading.Event();errors=[]
    def loading_guard():
        while not stop.wait(1):
            try:guard.check()
            except BaseException as e:errors.append(e);return
    monitor=threading.Thread(target=loading_guard,daemon=True);monitor.start();proof=None
    try:
        import torch
        torch.set_num_threads(8)
        proof=validate_t12(root,guard) if args.task=='t12' else validate_lgvq(root,guard)
        if errors:raise RuntimeError('Full loading guard failed') from errors[0]
        guard.fast();guard.check();props=torch.cuda.get_device_properties(0)
        if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108:raise ValueError('Wrong physical A100/SM count')
        proof.update(status='passed',proof_kind=KIND,no_gpu=False,cuda_initialized=True,pid=os.getpid(),gpu_uuid=GPU,
            power_limit_w=250,mps_percent=100,visible_multiprocessors=108,separate_validation_process_not_a_timed_repeat=True,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    except BaseException as e:write(root/'status.json',dict(status='failed',error=repr(e)));raise
    finally:stop.set();monitor.join(6);guard.close(root/'process_audit.json')
    proof['process_audit_sha256']=sha(root/'process_audit.json');write(root/'contract.json',proof)
    (t12_proof(root/'contract.json','qwen') if args.task=='t12' else lgvq_proof(root/'contract.json'))
    write(root/'status.json',dict(status='complete',proof_kind=KIND,contract_sha256=sha(root/'contract.json')))


if __name__=='__main__':main()
