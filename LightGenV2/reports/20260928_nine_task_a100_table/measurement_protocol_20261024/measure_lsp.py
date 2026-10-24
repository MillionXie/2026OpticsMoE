"""Original .7348 LSP and matched frozen-Qwen pose baseline, live image input.

Keep original annotation-defined person crop, native vision/decoder and metric.
No cached stem as online input, no fabricated CCD or replaced residual. The
separate CPU contract verifies real-image logits before formal CUDA admission.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location('lsp_shared_salicon_scope',HERE/'measure_salicon.py')
salicon=importlib.util.module_from_spec(spec);spec.loader.exec_module(salicon)
shared=salicon.shared;sha=salicon.sha;write=salicon.write
TASK=REPO/'LightGenV2/tasks/t02_keypoint_detection'
PINS={'ours':'495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518',
      'qwen':'0a4569f288f6de424b1b412452fa804a96e58d8f7e8655efaa23f461ab8cc735'}


def patch(model,meter):salicon.patch(model,meter,head_label='head.pose_heatmap_decoder')


def load(args,device):
    import torch
    from LightGenV2.tasks.t02_keypoint_detection import modeling as ours
    from LightGenV2.tasks.t02_keypoint_detection.settings import load_settings
    from LightGenV2.tasks.t02_keypoint_detection.lab_runtime import PORTABLE_PICKLE
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 import modeling as baseline
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings as baseline_settings
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import _read_source,split_standard_protocol,_find_joints_mat
    if sha(args.checkpoint)!=PINS[args.system]:raise ValueError('Wrong requested LSP model')
    settings=(load_settings if args.system=='ours' else baseline_settings)(args.config)
    settings.model_id=str(args.model);settings.cache_dir=None;settings.local_files_only=True;settings.download=False
    settings.data_root=args.data_root
    loaded=ours.load_vision_backbone(settings,device);settings.resolve_architecture(loaded.model)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,pickle_module=PORTABLE_PICKLE)
    if args.system=='ours':
        if payload['checkpoint_architecture']!=ours.architecture_label(settings) or payload['router_contract_sha256']!=settings.router_contract_sha256:
            raise ValueError('Requested legacy optical architecture or router differs')
        model=ours.build_student(loaded,settings)
        model.core.load_state_dict(payload['core'],strict=True);model.head.load_state_dict(payload['head'],strict=True)
        model.core.set_phase_dropout_active(False)
        if sum(p.numel() for p in model.head.parameters())!=133425:raise ValueError('Wrong original pose head')
    else:
        model=baseline.build_teacher(loaded,settings);model.head.load_state_dict(payload['head'],strict=True)
        if sum(p.numel() for p in model.head.parameters())!=1102990:raise ValueError('Wrong original frozen teacher head')
    model.eval().requires_grad_(False)
    original=_read_source(args.data_root/'lsp_dataset','lsp',settings.visibility_policy)
    _,records=split_standard_protocol(original,[])
    if len(records)!=1000 or len({r.sample_id for r in records})!=1000:raise ValueError('Canonical LSP TEST1000 required')
    # Bind official labels and original200 image paths before CUDA measurement;
    # bytes are hashed after timing so no file-content prewarming is introduced.
    selected=records[:200]
    info=dict(system=args.system,checkpoint=str(args.checkpoint),checkpoint_sha256=PINS[args.system],
              config=str(args.config),config_sha256=sha(args.config),resolved_settings=vars(settings),
              annotation_path=str(_find_joints_mat(args.data_root/'lsp_dataset')),
              annotation_sha256=sha(_find_joints_mat(args.data_root/'lsp_dataset')),
              test_ids=[r.sample_id for r in records],selected_epoch=payload.get('epoch'),weight_kind=payload.get('weight_variant'),
              original_full_test_reference=.7347857142857143 if args.system=='ours' else .7226,
              original_full_test_samples=1000,crop_protocol='original GT-joint person box, original margin and bilinear224; not full-frame person detection',
              native_model_assets={p.name:sha(p) for p in sorted(args.model.iterdir()) if p.is_file() and p.suffix in ('.json','.safetensors')})
    return model,loaded.processor,selected,settings,info


def prepare(processor,record,settings,device):
    from PIL import Image
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import transform_person
    with Image.open(record.image_path) as source:
        original_size=list(source.size);image=source.convert('RGB')
    image,keypoints,visible,crop,flipped=transform_person(image,record.keypoints,image_size=settings.image_size,
        crop_margin=settings.crop_margin,training=False,scale_jitter=settings.crop_scale_jitter,
        center_jitter=settings.crop_center_jitter,flip_probability=settings.horizontal_flip_probability,
        brightness_jitter=settings.brightness_jitter,contrast_jitter=settings.contrast_jitter)
    if flipped:raise ValueError('TEST input unexpectedly augmented')
    cpu=processor.image_processor(images=[image],return_tensors='pt')
    metadata=dict(original_size=original_size,processed_size=list(image.size),crop_box=crop,
                  pixel_values_shape=list(cpu['pixel_values'].shape),image_grid_thw=cpu['image_grid_thw'].tolist())
    return {name:cpu[name].to(device,non_blocking=True) for name in ('pixel_values','image_grid_thw')},metadata,keypoints,visible


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--system',choices=('ours','qwen'),required=True)
    for name in ('config','checkpoint','model','data-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--inspect-only',action='store_true');parser.add_argument('--repeat',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--gpu-uuid',default='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55');args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    if not args.inspect_only:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!='100':raise ValueError('Explicit UUID/MPS100 required')
        admission=shared.guard(args.gpu_uuid);props=torch.cuda.get_device_properties(0)
        if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108:raise ValueError('Wrong A100 or cap')
    else:admission=None
    device=torch.device('cpu' if args.inspect_only else 'cuda:0');torch.set_num_threads(8);torch.manual_seed(42)
    model,processor,rows,settings,info=load(args,device)
    meter=shared.meter_lib.StageMeter(None if args.inspect_only else torch.cuda)
    if args.system=='ours':patch(model,meter)
    info.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        script_sha256=sha(__file__),salicon_scope_sha256=sha(HERE/'measure_salicon.py'),stage_ledger=meter.ledger,
        command=sys.argv,repeat=args.repeat,explicit_warmup_forwards=0,cpu_threads=8,gpu_admission=admission,
        transformer_precision='original BF16 visual; FP32 original body/head; no new global autocast',torch=torch.__version__)
    args.output.mkdir(parents=True);write(args.output/'identity.json',info)
    records=[];predictions=[];targets=[];visibility=[];guard=None;sampler=None
    try:
        if args.inspect_only:
            meter.close();batch,metadata,_,_=prepare(processor,rows[0],settings,device)
            with torch.inference_mode():
                original=model(**batch)[0]
                if args.system=='ours':patch(model,meter)
                observed=model(**batch)[0]
            if not torch.equal(original,observed):raise RuntimeError('Real-image original model changed by timing')
            write(args.output/'cpu_contract.json',dict(status='passed',bitwise_equal=True,max_abs_difference=0.,sample_id=rows[0].sample_id,**metadata))
            print('Actual-image CPU contract passed; no CUDA forward');return
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
        from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
        from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
        from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import hardargmax_coordinates
        engine.GPU=args.gpu_uuid;engine.PIPE=os.environ['CUDA_MPS_PIPE_DIRECTORY']
        guard=engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800));guard.bind()
        raw=args.output/'telemetry_stdout.jsonl';sampler=ProcessTelemetrySampler(args.gpu_uuid,raw,command=observer_command(args.gpu_uuid,raw))
        sampler.start();sampler.set_phase('continuous200_no_warmup');time.sleep(.15)
        with torch.inference_mode():
            for i,row in enumerate(rows):
                guard.fast();sampler.check();meter.sample_index=i;start=time.perf_counter()
                with meter.scope('input.file_person_crop_processor_h2d','input',gpu=False):
                    batch,metadata,target,visible=prepare(processor,row,settings,device);torch.cuda.synchronize()
                if args.system=='ours':output=model(**batch)[0]
                else:
                    with meter.scope('qwen.full_native_vision_pose_head','baseline'):output=model(**batch)[0]
                with meter.scope('output.pose_hardargmax','output'):predicted=hardargmax_coordinates(output,settings.image_size)
                with meter.scope('output.cpu_coordinates','output',gpu=False):result=predicted.cpu().numpy()
                returned=time.perf_counter()
                predictions.append(result[0]);targets.append(target);visibility.append(visible)
                gap=0. if not records else (start-records[-1]['cpu_result_return'])*1000
                records.append(dict(index=i,sample_id=row.sample_id,image_path=str(row.image_path),input_start=start,
                    cpu_result_return=returned,internal_gap_ms=gap,**metadata))
        sampler.request('barrier');time.sleep(.12);telemetry=sampler.stop();sampler=None
        import numpy as np
        from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import pose_scales
        hits=0;joints=0
        for row,record,prediction,target,visible in zip(rows,records,predictions,targets,visibility):
            record['image_sha256']=sha(row.image_path);torso,head=pose_scales(target,visible)
            distance=np.linalg.norm(prediction-target,axis=-1);valid=visible & np.isfinite(distance)
            if not np.isfinite(torso) or torso<=0:valid[:]=False
            hit=int(((distance<=.2*torso)&valid).sum());total=int(valid.sum());hits+=hit;joints+=total
            record.update(predicted_coordinates=prediction.tolist(),target_coordinates=target.tolist(),visible=visible.tolist(),
                          torso_scale=float(torso),head_scale=float(head),pck_hits=hit,pck_joints=total)
        np.savez_compressed(args.output/'raw_pose_outputs.npz',predicted=np.stack(predictions),targets=np.stack(targets),visible=np.stack(visibility))
        elapsed=records[-1]['cpu_result_return']-records[0]['input_start']
        charged=sum(r['synchronized_wall_ms'] for r in meter.rows if r['charged'])+sum(r['internal_gap_ms'] for r in records)
        if args.system=='ours':
            optical=200*3*.0010447;duration=optical+charged/1000;energy=shared.meter_lib.optical_energy_j(optical,charged/1000)
            extra=dict(optical_ms_total=1000*optical,charged_electronic_ms_total=charged,power_basis='fixed121.746/104.230W phase estimate; FFT power diagnostic only')
        else:
            duration=elapsed;power=interval_energy([(records[0]['input_start'],records[-1]['cpu_result_return'])],telemetry,338.2)
            energy=power['energy_j'];extra=dict(power_integration=power,mean_board_power_w=energy/elapsed-338.2)
        summary=dict(status='complete',system=args.system,repeat=args.repeat,valid_samples=200,measured200_pck=hits/max(1,joints),
            evaluated_joints=joints,pck_hits=hits,full1000_reference=info['original_full_test_reference'],actual_software_wall_seconds=elapsed,
            cost=shared.meter_lib.equivalent_cost(duration,energy,200),**extra)
        if sha(__file__)!=info['script_sha256'] or sha(HERE/'measure_salicon.py')!=info['salicon_scope_sha256']:raise ValueError('Measurement source changed')
        for entry in meter.ledger:
            if sha(entry['source_file'])!=entry['source_sha256']:raise ValueError('Native source changed')
        guard.check();write(args.output/'summary.json',summary);save_samples(args.output/'telemetry.csv',telemetry);print(json.dumps(summary))
    except BaseException as error:
        write(args.output/'failure.json',dict(error=repr(error),completed_samples=len(records)));raise
    finally:
        if sampler is not None:
            from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
            save_samples(args.output/'telemetry.csv',sampler.stop(raise_on_error=False))
        if guard is not None:guard.close(args.output/'process_audit.json')
        meter.close()
        if hasattr(model,'restore_native'):model.restore_native()
        else:model.close()
        shared.write_csv(args.output/'samples.csv',records);shared.write_csv(args.output/'stages.csv',meter.rows)
        write(args.output/'stage_ledger.json',meter.ledger)
        write(args.output/'files_manifest.json',[dict(file=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(args.output.iterdir()) if f.is_file() and f.name!='files_manifest.json'])


if __name__=='__main__':main()
