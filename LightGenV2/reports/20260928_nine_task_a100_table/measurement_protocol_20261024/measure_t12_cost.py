"""Original T12 RGB/UTF8-to-CPU-RGB B1 first200 cost worker; never trains.

Requires an actual original two-input CPU numerical proof before loading a
model. Ours preserves full software execution, charges selected NN/frontend
and real gaps plus six parameter-optical passes. Qwen charges continuous E2E.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
import t12_live_frontend as original
from t12_reference_files import load_bundle,read_input,verify_original_bytes
from measure_abo_text_qwen import GPU,PROTOCOL,sha,write
TASK=original.TASK
FULL_TEST_PSNR={'ours':34.27775097739636,'qwen':27.26045327136914}
MAX_QUEUE_SECONDS=7200


def check_platform(raw):
    values=[v.strip() for v in raw.strip().split(',')]
    if (len(values)!=4 or values[:2]!=[GPU,'NVIDIA A100-PCIE-40GB']
            or abs(float(values[2])-250)>.1 or abs(float(values[3])-40960)>512):
        raise ValueError('Wrong physical lab A100 UUID/type/cap/VRAM')
    return dict(uuid=values[0],name=values[1],power_limit_w=float(values[2]),memory_total_mib=float(values[3]))


def proof_contract(path,system,*,check_sources=True):
    value=json.loads(Path(path).read_text(encoding='utf8'))
    key='body_sha256' if system=='ours' else 'baseline_sha256'
    pin=original.BODY_SHA if system=='ours' else original.BASELINE_SHA
    if (value.get('status')!='passed' or value.get('no_gpu') is not True or value.get(key)!=pin
            or [r.get('test_index') for r in value.get('checks',[])]!=[200,2303]
            or not all(r.get('bitwise_equal') and r.get('target_passed_to_model') is False for r in value['checks'])):
        raise ValueError('Actual original T12 numerical CPU proof required before model loading')
    if system=='qwen':
        report=value.get('qwen_report',{})
        if (report.get('language_layers_retained')!=28 or report.get('vision_tower_used') is not False
                or report.get('language_model_head_used') is not False):
            raise ValueError('Original full text-only Qwen28 CPU proof required')
    if check_sources:
        for name,digest in value['sources'].items():
            if sha(REPO/name)!=digest:raise ValueError('Verified original CPU source changed: '+name)
    return value


def original_cost(system,calls,stages,board_j):
    if len(calls)!=200:raise ValueError('Complete first200 required')
    elapsed=calls[-1]['cpu_result_return']-calls[0]['input_start']
    gaps=sum(c['internal_gap_ms'] for c in calls)
    if system=='qwen':
        return dict(seconds200=elapsed,joules200=338.2*elapsed+board_j,
            charged_electronic_ms_total=None,optical_ms_total=0.)
    charged=sum(r['synchronized_wall_ms'] for r in stages if r['charged'])+gaps
    optical=200*6*1.0447
    return dict(seconds200=(charged+optical)/1000,joules200=(121.746*optical+104.230*charged)/1000,
        charged_electronic_ms_total=charged,optical_ms_total=optical)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--cpu-proof',type=Path,required=True);parser.add_argument('--system',choices=('ours','qwen'),required=True)
    parser.add_argument('--repeat',type=int,choices=(1,2,3),required=True);parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args();folder=args.output.resolve()
    if folder.exists() or not folder.is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New immutable canonical cost point required')
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100'
            or not 0<args.deadline-time.time()<=MAX_QUEUE_SECONDS):raise ValueError('Physical lab A100/private MPS100/finite deadline required')
    proof=proof_contract(args.cpu_proof,args.system);bundle=load_bundle(args.inputs)
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    from measure_salicon import shared
    from stage_meter import StageMeter
    engine.GPU=GPU;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY']
    guard=engine.MPSGuard(SimpleNamespace(deadline=args.deadline));original_check=guard.check;platform_lock=threading.Lock()
    def checked_platform():
        with platform_lock:
            original_check()
            raw=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,power.limit,memory.total',
                '--format=csv,noheader,nounits'],text=True,timeout=5)
            guard.audit[-1]['platform']=check_platform(raw)
    guard.check=checked_platform;guard.check()
    import numpy as np
    import torch
    torch.set_num_threads(8);folder.mkdir(parents=True)
    meter=StageMeter(torch.cuda);calls=[];predictions=[];shapes=[];handles=[];sampler=None;telemetry=[]
    def status(name,**extra):write(folder/'status.json',dict(status=name,pid=os.getpid(),samples=len(calls),**extra))
    status('loading')
    try:
        with torch.inference_mode():
            device=torch.device('cuda:0')
            if args.system=='ours':
                from safetensors import safe_open
                from transformers import AutoProcessor
                from LightGenV2.tasks.t12_text_to_image.sealed_editor import build_sealed
                if sha(original.BODY)!=original.BODY_SHA or sha(original.MODEL/'model.safetensors')!=original.WEIGHT_SHA:
                    raise ValueError('Adopted original body/native embedding file differs')
                model=build_sealed(torch.load(original.BODY,map_location='cpu',weights_only=False)).to(device).eval().requires_grad_(False)
                if sum(p.numel() for p in model.parameters())!=17026642:raise ValueError('Original 17M model required')
                processor=AutoProcessor.from_pretrained(original.MODEL,local_files_only=True)
                with safe_open(original.MODEL/'model.safetensors',framework='pt',device='cpu') as stream:
                    weight=stream.get_tensor('model.language_model.embed_tokens.weight')
                embedding=torch.nn.Embedding.from_pretrained(weight,freeze=True).to(device)
                original.patch(model,meter);groups={'model':model}
            else:
                import t12_baseline_live as baseline
                # Import original noise/oracle definitions before source freeze.
                from LightGenV2.tasks.t12_text_to_image.export_matched_summary import noise_batch
                runtime=baseline.load_runtime(device)
                groups={name:getattr(runtime,name) for name in ('qwen','adapter','unet','vae')}
            props=torch.cuda.get_device_properties(0)
            cap=float(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
            if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
                raise ValueError('Wrong physical A100/cap')
            guard.bind()
            for group,network in groups.items():
                for name,module in network.named_modules():
                    if isinstance(module,(torch.nn.Linear,torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                        def observe(owner,inputs,result,name=group+'.'+name):
                            shapes.append(dict(index=meter.sample_index,module=name,kind=type(owner).__name__,
                                input_shape=list(inputs[0].shape),output_shape=list(result.shape),
                                weight_shape=list(owner.weight.shape),groups=getattr(owner,'groups',None)))
                        handles.append(module.register_forward_hook(observe))
            frozen=original.sources();manifest=args.inputs/'input_manifest.json'
            identity=dict(protocol_id=PROTOCOL,system=args.system,repeat=args.repeat,batch=1,pid=os.getpid(),
                body_sha256=original.BODY_SHA if args.system=='ours' else original.BASELINE_SHA,
                gpu_uuid=GPU,visible_multiprocessors=108,power_limit_w=cap,mps_percent=100,
                cpu_contract=str(args.cpu_proof.resolve()),cpu_contract_sha256=sha(args.cpu_proof),
                input_root=str(args.inputs.resolve()),input_manifest_sha256=sha(manifest),
                sources=frozen,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                explicit_warmup_forwards=0,first_sample_retained=True,formal_full_TEST_reference_psnr=FULL_TEST_PSNR[args.system],
                input_contract='same original reference RGB PNG + UTF8; target never loaded online',
                output_contract='float32 CPU RGB[3,256,256], original clamp before [0,1] mapping',
                noise='Ours original CPU RGB generator seed1042+index then H2D; Qwen original native-device latent noise_batch seed1042+index',
                original_small_autocast=False if args.system=='ours' else None,
                original_qwen_autocast='float16' if args.system=='qwen' else None,
                stage_ledger=meter.ledger,command=sys.argv,
                floating_parameter_dtypes={name:sorted({str(p.dtype) for p in network.parameters() if p.is_floating_point()})
                    for name,network in groups.items()},
                clock_anchor=dict(unix=time.time(),monotonic=time.perf_counter()),
                cost_scope='continuous file-to-CPU-RGB E2E' if args.system=='qwen' else 'mixed selected-NN/parameter-optics projection, not device measured E2E')
            if args.system=='ours':identity['native_embedding_dtype']=str(embedding.weight.dtype)
            if args.system=='qwen':identity.update(qwen_report=runtime.qwen_report,frozen_dependency_binding=runtime.frozen_dependency_binding)
            write(folder/'identity.json',identity);write(folder/'admission_bound.json',dict(pid=os.getpid(),unix=time.time(),sources=frozen))
            raw=folder/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('first200_original_reference_zero_warmup');time.sleep(.15)
            for index,record in enumerate(bundle['records']):
                guard.fast();sampler.check();meter.sample_index=index;start=time.perf_counter()
                with meter.scope('input.reference_RGB_UTF8_decode_H2D','input',gpu=False):
                    reference,prompt=read_input(args.inputs,record);reference=reference[None].to(device);torch.cuda.synchronize()
                if args.system=='ours':
                    with meter.scope('input.original_tokenizer_embedding','input_neural'):
                        text,mask,tokens=original.online_embeddings(processor,embedding,prompt,device)
                    with meter.scope('input.original_seeded_RGB_noise','input',gpu=False):
                        noise=torch.randn((1,3,256,256),generator=torch.Generator().manual_seed(1042+index)).to(device)
                        torch.cuda.synchronize()
                    result=model(reference,text,mask,noise)
                else:
                    with meter.scope('input.original_tokenizer_H2D','input',gpu=False):
                        tokens=baseline.inputs(runtime,prompt);torch.cuda.synchronize()
                    with meter.scope('baseline.full_original_Qwen28_UNet_VAE','baseline_full'):
                        result=baseline.forward(runtime,reference,tokens,index)
                with meter.scope('output.CPU_RGB_clamp','output',gpu=False):
                    host=result[0].detach().float().cpu().clamp(-1,1).add(1).mul(.5)
                returned=time.perf_counter()
                if tuple(host.shape)!=(3,256,256) or not torch.isfinite(host).all():raise ValueError('Original RGB output invalid')
                ids=tokens['input_ids'].detach().cpu().tolist();mask=tokens['attention_mask'].detach().cpu().tolist()
                gap=0. if not calls else (start-calls[-1]['cpu_result_return'])*1000
                calls.append(dict(index=index,sample_id=record['sample_id'],input_start=start,cpu_result_return=returned,
                    internal_gap_ms=gap,input_ids=ids,attention_mask=mask,valid_input_tokens=sum(mask[0]),output_shape=list(host.shape)))
                predictions.append(host.numpy().copy())
                if index==0 or (index+1)%25==0:status('evaluating')
            sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
            elapsed=calls[-1]['cpu_result_return']-calls[0]['input_start']
            board=interval_energy([(calls[0]['input_start'],calls[-1]['cpu_result_return'])],telemetry,0)
            costs=original_cost(args.system,calls,meter.rows,board['energy_j'])
            if args.system=='ours':
                counts=Counter(r['stage'] for r in meter.rows)
                for mode in ('language','vision'):
                    if counts[mode+'.router_simulation']!=200 or counts[mode+'.expert_global_simulation']!=400:
                        raise ValueError('Six original optical passes missing')
            guard.check();verify_original_bytes(args.inputs,bundle)
            if frozen!=original.sources() or sha(manifest)!=identity['input_manifest_sha256']:
                raise ValueError('Frozen original source/input metadata changed')
            np.savez_compressed(folder/'RGB_outputs.npz',predictions=np.asarray(predictions,dtype=np.float32))
            write(folder/'summary.json',dict(status='complete',system=args.system,repeat=args.repeat,valid_samples=200,
                formal_full2304_reference_PSNR=FULL_TEST_PSNR[args.system],actual_software_wall_seconds=elapsed,
                diagnostic_software_board_j200=board['energy_j'],weighted_software_board_w=board['energy_j']/elapsed,
                **costs,sample_ms=costs['seconds200']*5,sample_j=costs['joules200']/200,
                common_input_tokens200=sum(c['valid_input_tokens'] for c in calls) if args.system=='qwen' else None,
                common_tokens_require_actual_Qwen_ID_mask_pairing=True,not_a_new_full_TEST_accuracy_evaluation=True))
            status('complete')
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(folder/'telemetry.csv',telemetry)
        guard.close(folder/'process_audit.json');meter.close()
        for handle in handles:handle.remove()
        shared.write_csv(folder/'calls.csv',calls);shared.write_csv(folder/'stages.csv',meter.rows)
        write(folder/'stage_ledger.json',meter.ledger);write(folder/'nn_module_shapes.json',shapes)
        if predictions and not (folder/'RGB_outputs.npz').exists():
            np.savez_compressed(folder/'RGB_outputs_partial.npz',predictions=np.asarray(predictions,dtype=np.float32))
        write(folder/'files_manifest.json',[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(folder.iterdir()) if p.is_file() and p.name!='files_manifest.json'])


if __name__=='__main__':main()
