"""Original LGVQ video-to-CPU-MOS costs; full front, selected NN, real fields.

No training, cached online features, unused Transformers or dummy warmup.
Temporal200 uses thirteen coherent16x4 fields, including eight padding slots.
The complete frontend wall is counted ONCE; its component spans are diagnostics.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
import lgvq_live_frontend as live
from measure_t12_cost import GPU,PROTOCOL,MAX_QUEUE_SECONDS,check_platform
from measure_abo_text_qwen import sha,write
TASK=live.TASK
QWEN_REFERENCE=TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024/qwen100_repaired01/qwen_cap100_repeat01/report.json'


def field_plan(target):
    count=live.PINS[target][3]
    return [live.field_indices(200,count,i) for i in range(math.ceil(200/count))]


def proof_contract(path,*,check_sources=True):
    proof=json.loads(Path(path).read_text(encoding='utf8'))
    expected=[('temporal',13),('temporal',14),('spatial',200),('spatial',557)]
    if (proof.get('status')!='passed' or proof.get('no_gpu') is not True
            or proof.get('cuda_initialized') is not False
            or proof.get('manifest_sha256')!=live.MANIFEST_SHA
            or proof.get('original_front_parameter_binding_sha256')!=live.BINDING_SHA
            or [(c.get('target'),c.get('field_index')) for c in proof.get('checks',[])]!=expected):
        raise ValueError('Complete actual original four-field CPU proof required')
    for check in proof['checks']:
        if (check.get('body_sha256')!=live.PINS[check['target']][1]
                or check.get('bitwise_equal') is not True or check.get('full_original_output_bitwise_equal') is not True
                or check.get('selected_NN_stage_instrumentation_tested') is not True
                or check.get('frontend_transformer_layers_executed')!=0 or len(check.get('six_passes',[]))!=6):
            raise ValueError('Original live frontend/full model/selected NN proof differs')
    if check_sources and any(sha(REPO/p)!=h for p,h in proof['sources'].items()):
        raise ValueError('Actual CPU-proven original source changed')
    return proof


def source_files():
    paths={Path(__file__),*(REPO/p for p in live.sources())}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value and Path(value).is_absolute():
            path=Path(value).resolve()
            if path.suffix=='.py' and path.is_relative_to(REPO):paths.add(path)
    return {p.relative_to(REPO).as_posix():sha(p) for p in sorted(paths)}


def costs(target,fields,stages):
    plan=field_plan(target)
    if len(fields)!=len(plan) or sum(sum(f['valid']) for f in fields)!=200:
        raise ValueError('Complete first200 original fields required')
    front=sum(f['frontend_wall_ms'] for f in fields)
    gaps=sum(f['internal_gap_ms'] for f in fields)
    selected=sum(r['synchronized_wall_ms'] for r in stages if r['region']!='frontend' and r['charged'])
    charged=front+selected+gaps;optical=len(plan)*6*1.0447
    seconds=(charged+optical)/1000;joules=(104.230*charged+121.746*optical)/1000
    return dict(seconds200=seconds,joules200=joules,sample_ms=seconds*5,sample_j=joules/200,
        equivalent_frame_ms=seconds*1.25,equivalent_frame_j=joules/800,
        frontend_wall_ms_total=front,charged_electronic_ms_total=charged,optical_ms_total=optical,
        field_count=len(plan),valid_videos=200,executed_video_slots=sum(len(f['valid']) for f in fields))


def file_records(tests,hashed=False):
    result={}
    for row in tests:
        path=Path(row.video_path);s=path.stat()
        result[row.sample_id]=dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)
        if hashed:result[row.sample_id]['sha256']=sha(path)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=('temporal','spatial'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-proof',type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),required=True);p.add_argument('--deadline',type=float,required=True)
    args=p.parse_args();root=args.output.resolve();plan=field_plan(args.target)
    if (root.exists() or not root.is_relative_to(TASK/'runs/simulation')
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100'
            or not 0<args.deadline-time.time()<=MAX_QUEUE_SECONDS):raise ValueError('Fresh finite original A100/MPS100 point only')
    proof_contract(args.cpu_proof)
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as original
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    from measure_salicon import shared
    from stage_meter import StageMeter
    engine.GPU=GPU;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY']
    guard=engine.MPSGuard(SimpleNamespace(deadline=args.deadline));base_check=guard.check;mutex=threading.Lock()
    def check():
        with mutex:
            base_check()
            raw=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,power.limit,memory.total',
                '--format=csv,noheader,nounits'],text=True,timeout=5)
            guard.audit[-1]['platform']=check_platform(raw)
    guard.check=check;guard.check()
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(8);root.mkdir(parents=True);meter=StageMeter(torch.cuda)
    fields=[];predictions=[];retained=[];shapes=[];handles=[];sampler=None;telemetry=[];phase={'name':'loading'}
    def status(name,**extra):write(root/'status.json',dict(status=name,pid=os.getpid(),fields=len(fields),
        valid_samples=sum(sum(f['valid']) for f in fields),**extra))
    status('loading')
    try:
        if sha(live.MANIFEST)!=live.MANIFEST_SHA:raise ValueError('Original TEST manifest differs')
        tests=[r for r in read_manifest(live.MANIFEST) if r.split=='test']
        if len(tests)!=558:raise ValueError('Original full558 TEST identity required, not re-evaluation')
        tests=tests[:200];before=file_records(tests)
        qref=json.loads(QWEN_REFERENCE.read_text());qref_sha=sha(QWEN_REFERENCE)
        if set(qref['input_files'])!={r.sample_id for r in tests}:raise ValueError('Shared Qwen input set differs')
        with torch.inference_mode():
            device=torch.device('cuda:0');front=live.LiveFront(device)
            if args.target=='spatial':front.load_spatial_stem()
            run,pin,grid,slots=live.PINS[args.target]
            checkpoint=TASK/'runs/simulation'/run/'best_checkpoint.pt'
            model,settings=original.load_model(args.target,checkpoint,device);model.requires_grad_(False)
            if args.target=='spatial' and (settings.quality_branch_enabled or not settings.electronic_quality_residual_enabled
                    or not settings.custom_conv_electronic_enabled or model.frame_stem is not None):
                raise ValueError('Original spatial192/custom-conv/residual architecture differs')
            live.instrument_body(model,meter)
            networks={'body':model,'native_patch_position':front.visual}
            if front.stem is not None:networks['original_FrameStem192']=front.stem
            for group,network in networks.items():
                for name,module in network.named_modules():
                    if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):
                        def observe(owner,inputs,result,name=group+'.'+name):
                            shapes.append(dict(field=meter.sample_index,region=phase['name'],module=name,kind=type(owner).__name__,
                                input_shape=list(inputs[0].shape),output_shape=list(result.shape),weight_shape=list(owner.weight.shape),
                                groups=getattr(owner,'groups',None),parallel_residual=name.startswith('body.') and live.residual_name(name[5:])))
                        handles.append(module.register_forward_hook(observe))
            props=torch.cuda.get_device_properties(0)
            if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108:raise ValueError('Wrong A100/SM admission')
            guard.bind();frozen=source_files()
            identity=dict(protocol_id=PROTOCOL,target=args.target,system='ours',repeat=args.repeat,pid=os.getpid(),batch=1,
                gpu_uuid=GPU,power_limit_w=250,visible_multiprocessors=108,mps_percent=100,
                body_sha256=pin,manifest_sha256=live.MANIFEST_SHA,front_binding_sha256=live.BINDING_SHA,
                cpu_contract=str(args.cpu_proof.resolve()),cpu_contract_sha256=sha(args.cpu_proof),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),sources=frozen,
                video_metadata_before=before,qualified_Qwen_input_reference=str(QWEN_REFERENCE),qualified_Qwen_report_sha256=qref_sha,
                original_front_tensors=front.tensor_checks,original_stem_sha256=live.STEM_SHA if args.target=='spatial' else None,
                native_prompt=next(c for c in front.binding['matched_prompt_checks'] if c['target']==args.target),
                parameter_dtypes={name:sorted({str(v.dtype) for v in net.parameters() if v.is_floating_point()}) for name,net in networks.items()},
                explicit_warmup_forwards=0,first_field_retained=True,frontend_transformer_layers_executed=0,
                accounting='entire frontend wall once + original selected body NN/boundary spans + all field gaps + six parameter passes per field',
                cost_scope='mixed hardware projection, not device measured E2E',formal_accuracy='confirmed original full558, not re-evaluated here',
                clock_anchor=dict(unix=time.time(),monotonic=time.perf_counter()),command=sys.argv)
            write(root/'identity.json',identity);write(root/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen))
            raw=root/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('first200_live_original_fields_zero_warmup');time.sleep(.15)
            for index,(indices,valid) in enumerate(plan):
                guard.fast();sampler.check();rows=[tests[i] for i in indices];meter.sample_index=index;meter.valid_samples=sum(valid)
                phase['name']='frontend';row_start=len(meter.rows);start=time.perf_counter()
                batch,meta=front.build(rows,args.target,meter);torch.cuda.synchronize();front_end=time.perf_counter()
                for item in meter.rows[row_start:]:item['region']='frontend'
                phase['name']='body';row_start=len(meter.rows)
                with meter.scope('input.original_CPU_features_H2D','input_transfer'):
                    moved={k:v.to(device) for k,v in batch.items()}
                result=original.forward(model,moved)
                with meter.scope('output.full_field_CPU_MOS','output',gpu=False):
                    host=result['prediction'].detach().float().cpu().reshape(-1)
                returned=time.perf_counter()
                for item in meter.rows[row_start:]:item['region']='body'
                if host.numel()!=slots or not torch.isfinite(host).all():raise ValueError('Original complete field prediction invalid')
                fields.append(dict(field=index,test_indices=indices,valid=valid,source_ids=meta['source_ids'],
                    input_start=start,frontend_end=front_end,cpu_result_return=returned,frontend_wall_ms=(front_end-start)*1000,
                    internal_gap_ms=0. if not fields else (start-fields[-1]['cpu_result_return'])*1000,
                    original_native_prompt=meta,predictions=host.tolist(),output_shape=list(result['prediction'].shape)))
                predictions.extend(host.tolist());retained.append(batch)
                if index==0 or (index+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            board=interval_energy([(fields[0]['input_start'],fields[-1]['cpu_result_return'])],telemetry,0.)
            guard.check()
            # Original inputs/content hashes are verified only after T1.
            after=file_records(tests,True)
            for sid,item in after.items():
                if {k:v for k,v in item.items() if k!='sha256'}!=before[sid] or item['sha256']!=qref['input_files'][sid]['sha256']:
                    raise ValueError('Original video bytes changed or differ from qualified Qwen')
            views={}
            from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front import _sample_positions
            from experiments.lgvq_four_stage_optical_electronic_109_no_attention_vqa.data import FRAME_FRACTIONS
            for row in tests:
                capture=cv2.VideoCapture(str(row.video_path))
                try:
                    total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                finally:capture.release()
                if min(total,width,height)<=0:raise ValueError('Original video dimensions invalid')
                views[row.sample_id]=dict(width=width,height=height,reported_frame_count=total,
                    Qwen_front_source_selected_positions=list(_sample_positions(total,4)),Qwen_front_crop=.65,Qwen_front_size=448,
                    raw224_source_selected_positions=[min(total-1,max(0,round((total-1)*x))) for x in FRAME_FRACTIONS]
                        if args.target=='spatial' else None,raw224_crop=1.0 if args.target=='spatial' else None,
                    metadata_read_after_T1=True,positions_derived_from_original_producer_not_observed_decoder_seek=True)
            prepared=[{k:live.tensor_record(v) for k,v in batch.items()} for batch in retained]
            if frozen!=source_files() or sha(live.MANIFEST)!=live.MANIFEST_SHA or sha(QWEN_REFERENCE)!=qref_sha:
                raise ValueError('Frozen source/manifest/reference changed')
            counts=Counter((r['sample_index'],r['family']) for r in meter.rows)
            if any(counts[i,'optical_simulation_FFT']!=6 for i in range(len(plan))):raise ValueError('Six real propagations per field required')
            write(root/'input_files.json',after);write(root/'video_views.json',views);write(root/'prepared_inputs.json',prepared)
            np.savez_compressed(root/'field_predictions.npz',predictions=np.asarray(predictions,np.float32),
                valid=np.asarray([v for _,flags in plan for v in flags],bool),test_indices=np.asarray([i for ids,_ in plan for i in ids],np.int64))
            cost=costs(args.target,fields,meter.rows);elapsed=fields[-1]['cpu_result_return']-fields[0]['input_start']
            write(root/'summary.json',dict(status='complete',target=args.target,repeat=args.repeat,**cost,
                actual_software_wall_seconds=elapsed,diagnostic_software_board_j200=board['energy_j'],
                weighted_software_board_w=board['energy_j']/elapsed,common_input_tokens200=None,
                common_tokens_require_qualified_Qwen_input_pairing=True,not_a_new_full_TEST_accuracy_evaluation=True))
            status('complete')
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(root/'telemetry.csv',telemetry)
        guard.close(root/'process_audit.json');meter.close()
        for handle in handles:handle.remove()
        shared.write_csv(root/'fields.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
            for k,v in item.items()} for item in fields]);shared.write_csv(root/'stages.csv',meter.rows)
        write(root/'stage_ledger.json',meter.ledger);write(root/'nn_module_shapes.json',shapes)
        write(root/'files_manifest.json',[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(root.iterdir()) if p.is_file() and p.name!='files_manifest.json'])


if __name__=='__main__':main()
