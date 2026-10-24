"""Bind original SALICON .8625/.8748 graphs to the approved input-inclusive scope.

TEST order and RGB224 bicubic transform are from the original task manifest;
GT maps are evaluated after the timing window, never fed to inference. The
optical source is not reimplemented: AST spans wrap unchanged statements.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
spec=importlib.util.spec_from_file_location("salicon_shared_meter",HERE/"measure_ours.py")
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
sha=shared.meter_lib.file_sha256
write=shared.write_json
TASK=REPO/"LightGenV2/tasks/t03_saliency"
PINS={"ours":"036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe",
      "qwen":"689d086b5a390c399df109d07b73185575cd2245998b7bd427f6ccea5b0fe019"}


def one_index(body,predicate):
    values=[i for i,value in enumerate(body) if predicate(value)]
    if len(values)!=1:raise ValueError("Ambiguous original neural boundary: "+str(values))
    return values[0]


def patch(model,meter):
    visual=model.visual
    meter.wrap(visual.patch_embed,"forward","frontend.patch_embedding",family="frontend")
    for name in ("fast_pos_embed_interpolate","rot_pos_emb"):
        if not hasattr(visual,name):raise ValueError("Native Qwen frontend interface changed")
        meter.wrap(visual,name,"frontend."+name,family="frontend")
    meter.wrap(visual.merger,"forward","bridge.native_merger",family="bridge")
    for i,merger in enumerate(visual.deepstack_merger_list):
        meter.wrap(merger,"forward",f"bridge.deepstack_merger{i}",family="bridge")
    h=model.core.hybrid;b=h.optical_branch;c=b.core
    body=shared.meter_lib.statements(h,"forward_groups")
    ix=shared.meter_lib.assign_index(body,"input_latent")
    meter.patch_statements(h,"forward_groups",[("vision.input_adapter_norm","adapter",True,ix,ix)])
    for name in ("output_norm","output_adapter"):
        meter.wrap(getattr(h,name),"forward","vision."+name,family="adapter")
    meter.wrap(h,"_fuse","vision.fusion_audit",family="fusion",charged=False)
    for i,block in enumerate(h.blocks):
        meter.wrap(block,"forward",f"vision.parallel_residual{i+1}",family="parallel_residual",charged=False)
    body=shared.meter_lib.statements(c,"encode_groups")
    ix=shared.meter_lib.assign_index(body,"projected")
    meter.patch_statements(c,"encode_groups",[("vision.amplitude_linear_norm_softplus","amplitude_encoding_nn",True,ix,ix)])
    body=shared.meter_lib.statements(b,"_encode_input_fields")
    ix=shared.meter_lib.assign_index(body,"rms")
    meter.patch_statements(b,"_encode_input_fields",[("vision.amplitude_rms","amplitude_encoding_nn",True,ix,len(body)-1)])
    meter.wrap(c,"_direct_amplitude_load","vision.field_layout",family="layout",charged=False)
    meter.wrap(c.router,"_simulate","vision.router_simulation",family="simulation_only",charged=False)
    body=shared.meter_lib.statements(c.router,"forward")
    first=shared.meter_lib.assign_index(body,"energy")
    last=one_index(body,lambda s:s.startswith("weights, selected, indices ="))
    meter.patch_statements(c.router,"forward",[("vision.router_energy_top2","router",True,first,last)])
    meter.wrap(b,"_simulate_detector_roi","vision.expert_global_simulation",family="simulation_only",charged=False)
    normalizer=b.ccd_normalizer
    body=shared.meter_lib.statements(normalizer,"forward")
    first=shared.meter_lib.assign_index(body,"value")
    # Keep the original clamp/normalization math; validation is not counted as
    # a neural layer. Safety checks still execute unchanged outside this span.
    first_group=("vision.ccd_value","ccd_nonlinearity",True,first,first)
    next_neural=one_index(body,lambda s:s.startswith("frame_mean =")) if any(s.startswith("frame_mean =") for s in body) else len(body)-1
    groups=[first_group,("vision.ccd_normalization","ccd_nonlinearity",True,next_neural,len(body)-1)]
    if first==next_neural:groups=[("vision.ccd_normalization","ccd_nonlinearity",True,first,len(body)-1)]
    meter.patch_statements(normalizer,"forward",groups)
    for name,readout,output in (("expert",b.expert_readout,b.expert_output_adapter),("global",c.readout,c.output_adapter)):
        body=shared.meter_lib.statements(readout,"forward_intensity")
        first=one_index(body,lambda s:s.startswith("detector_intensity =") and ".clamp_min(" in s)
        meter.patch_statements(readout,"forward_intensity",[(f"vision.{name}_ccd_pool_norm_nonlinearity","ccd_readout",True,first,len(body)-1)])
        meter.wrap(output,"forward",f"vision.{name}_ccd_output_linear",family="ccd_readout")
    meter.wrap(model.head,"forward","head.saliency_decoder",family="head")


def load(args,device):
    import torch
    from LightGenV2.tasks.t03_saliency.settings import load_settings
    from LightGenV2.tasks.t03_saliency.modeling import build_student,load_vision_backbone,architecture_label
    from LightGenV2.tasks.t03_saliency.aligned_baseline import AlignedReadout
    from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import FrozenQwenVisionTeacher
    if sha(args.checkpoint)!=PINS[args.system]:raise ValueError("Wrong requested SALICON checkpoint")
    settings=load_settings(args.config)
    settings.model_id=str(args.model);settings.cache_dir=None
    settings.local_files_only=True;settings.download=False
    loaded=load_vision_backbone(settings,device)
    settings.resolve_architecture(loaded.model)
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    if args.system=="ours":
        if payload["architecture"]!=architecture_label(settings):raise ValueError("Optical architecture mismatch")
        model=build_student(loaded,settings)
        model.core.load_state_dict(payload["core"],strict=True)
        model.head.load_state_dict(payload["saliency_head"],strict=True)
        model.core.set_phase_dropout_active(False)
    else:
        head=AlignedReadout(settings.vision_hidden_size).to(device)
        head.load_state_dict(payload["head"],strict=True)
        if sum(p.numel() for p in head.parameters())!=282596:raise ValueError("Wrong aligned head")
        model=FrozenQwenVisionTeacher(loaded,head)
    model.eval().requires_grad_(False)
    import csv
    with args.manifest.open(encoding="utf-8-sig",newline="") as f:
        rows=[r for r in csv.DictReader(f) if r["split"]=="validation"]
    if len(rows)!=5000 or len({r["sample_id"] for r in rows})!=5000:raise ValueError("Original validation5000 required")
    selected=rows[:200]
    for r in selected:
        if not Path(r["image_path"]).is_file() or not Path(r["density_path"]).is_file():raise FileNotFoundError(r)
    info=dict(system=args.system,checkpoint=str(args.checkpoint),checkpoint_sha256=PINS[args.system],
              config=str(args.config),config_sha256=sha(args.config),manifest_sha256=sha(args.manifest),
              original_full_test_reference=.8624925081777596 if args.system=="ours" else .87483828,
              original_full_test_samples=5000,selected_epoch=payload["epoch"],weight_kind=payload.get("weight_kind"),
              resolved_settings=vars(settings),
              processor="original image_processor, RGB224 bicubic, no language prompt",
              preprocessing="same original _aligned_transform(training=False) image branch",
              native_model_assets={p.name:sha(p) for p in sorted(args.model.iterdir()) if p.is_file() and p.suffix in (".json",".safetensors")})
    return model,loaded.processor,selected,info


def prepare(processor,row,device):
    from PIL import Image
    with Image.open(row["image_path"]) as src:
        source_size=list(src.size)
        image=src.convert("RGB").resize((224,224),Image.Resampling.BICUBIC)
    cpu=processor.image_processor(images=[image],return_tensors="pt")
    metadata=dict(original_size=source_size,processed_size=[224,224],
                  pixel_values_shape=list(cpu["pixel_values"].shape),image_grid_thw=cpu["image_grid_thw"].tolist())
    return {name:cpu[name].to(device,non_blocking=True) for name in ("pixel_values","image_grid_thw")},metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--system",choices=("ours","qwen"),required=True)
    for name in ("config","checkpoint","manifest","model","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--inspect-only",action="store_true")
    p.add_argument("--repeat",type=int,choices=(1,2,3),default=1)
    p.add_argument("--gpu-uuid",default="GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    if not args.inspect_only:
        if (os.environ.get("CUDA_VISIBLE_DEVICES")!=args.gpu_uuid or os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")!="100"):
            raise ValueError("Explicit UUID and private MPS100 required")
        admission=shared.guard(args.gpu_uuid)
        props=torch.cuda.get_device_properties(0)
        if props.name!="NVIDIA A100-PCIE-40GB" or props.multi_processor_count!=108:raise ValueError("Wrong card/cap")
    else:admission=None
    device=torch.device("cpu" if args.inspect_only else "cuda:0")
    torch.set_num_threads(8);torch.manual_seed(42)
    model,processor,rows,info=load(args,device)
    meter=shared.meter_lib.StageMeter(None if args.inspect_only else torch.cuda)
    if args.system=="ours":patch(model,meter)
    info.update(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
                script_sha256=sha(__file__),stage_ledger=meter.ledger,command=sys.argv,repeat=args.repeat,
                explicit_warmup_forwards=0,cpu_threads=8,gpu_admission=admission,
                torch=torch.__version__,transformer_precision="original BF16 vision; original FP32 core/head, no new global autocast")
    args.output.mkdir(parents=True);write(args.output/"identity.json",info)
    records=[];densities=[];guard=None;sampler=None;telemetry=[]
    try:
        if args.inspect_only:
            meter.close();batch,metadata=prepare(processor,rows[0],device)
            with torch.inference_mode():
                original=model(**batch)[0]
                if args.system=="ours":patch(model,meter)
                instrumented=model(**batch)[0]
            if not torch.equal(original,instrumented):raise RuntimeError("Instrumentation changes original real-image logits")
            write(args.output/"cpu_contract.json",dict(status="passed",bitwise_equal=True,max_abs_difference=0.,sample_id=rows[0]["sample_id"],**metadata))
            print("Real-image CPU contract passed; no GPU inference performed")
            return
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
        from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
        from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
        from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
        engine.GPU=args.gpu_uuid;engine.PIPE=os.environ["CUDA_MPS_PIPE_DIRECTORY"]
        guard=engine.MPSGuard(SimpleNamespace(deadline=time.time()+1800));guard.bind()
        raw=args.output/"telemetry_stdout.jsonl"
        sampler=ProcessTelemetrySampler(args.gpu_uuid,raw,command=observer_command(args.gpu_uuid,raw))
        sampler.start();sampler.set_phase("continuous200_no_warmup");time.sleep(.15)
        with torch.inference_mode():
            for i,row in enumerate(rows):
                guard.fast();sampler.check();meter.sample_index=i
                start=time.perf_counter()
                with meter.scope("input.file_resize_processor_h2d","input",gpu=False):
                    batch,metadata=prepare(processor,row,device);torch.cuda.synchronize()
                # The unmodified task model also executes untimed numerical
                # propagation and layout for Ours. Never charge its FFT as light.
                if args.system=="ours":logits=model(**batch)[0]
                else:
                    with meter.scope("qwen.full_native_vision_aligned_head","baseline"):
                        logits=model(**batch)[0]
                with meter.scope("output.saliency_softmax","output"):
                    density=logits.flatten(1).softmax(-1).reshape(-1,224,224)
                with meter.scope("output.cpu_map","output",gpu=False):
                    result=density.cpu().numpy()
                returned=time.perf_counter()
                gap=0. if not records else (start-records[-1]["cpu_result_return"])*1000
                records.append(dict(index=i,sample_id=row["sample_id"],image_path=row["image_path"],
                                    input_start=start,cpu_result_return=returned,internal_gap_ms=gap,**metadata))
                densities.append(result[0])
        sampler.request("barrier");time.sleep(.12)
        telemetry=sampler.stop();sampler=None
        elapsed=records[-1]["cpu_result_return"]-records[0]["input_start"]
        import numpy as np
        from PIL import Image
        from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
        targets=[]
        for row,record in zip(rows,records):
            record["image_sha256"]=sha(row["image_path"])
            with Image.open(row["density_path"]) as src:target=np.asarray(src.convert("L").resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255.
            if target.sum()<=0:raise ValueError("Empty density target")
            targets.append(target/target.sum())
        cc=independent_cc(np.stack(densities),np.stack(targets))
        for record,value in zip(records,cc):record["cc_float64"]=float(value)
        np.savez_compressed(args.output/"predicted_density_maps.npz",maps=np.stack(densities))
        charged=sum(r["synchronized_wall_ms"] for r in meter.rows if r["charged"])+sum(r["internal_gap_ms"] for r in records)
        if args.system=="ours":
            optical=200*3*.0010447;duration=optical+charged/1000
            energy=shared.meter_lib.optical_energy_j(optical,charged/1000)
            extra=dict(optical_ms_total=optical*1000,charged_electronic_ms_total=charged,
                       power_basis="fixed121.746/104.230W phase estimate; telemetry of FFT simulation is diagnostic only")
        else:
            duration=elapsed;power=interval_energy([(records[0]["input_start"],records[-1]["cpu_result_return"])],telemetry,338.2)
            energy=power["energy_j"];extra=dict(power_integration=power,mean_board_power_w=energy/elapsed-338.2)
        summary=dict(status="complete",system=args.system,repeat=args.repeat,valid_samples=200,
                     measured200_cc_float64=float(np.mean(cc)),full5000_reference=info["original_full_test_reference"],
                     actual_software_wall_seconds=elapsed,cost=shared.meter_lib.equivalent_cost(duration,energy,200),**extra)
        if sha(__file__)!=info["script_sha256"]:raise ValueError("Bound measurement source changed")
        for entry in meter.ledger:
            if sha(entry["source_file"])!=entry["source_sha256"]:raise ValueError("Measured source changed")
        guard.check();write(args.output/"summary.json",summary);print(json.dumps(summary))
        save_samples(args.output/"telemetry.csv",telemetry)
    except BaseException as error:
        write(args.output/"failure.json",dict(error=repr(error),completed_samples=len(records)));raise
    finally:
        if sampler is not None:
            from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
            save_samples(args.output/"telemetry.csv",sampler.stop(raise_on_error=False))
        if guard is not None:guard.close(args.output/"process_audit.json")
        meter.close()
        if hasattr(model,"restore_native"):model.restore_native()
        else:model.close()
        shared.write_csv(args.output/"samples.csv",records);shared.write_csv(args.output/"stages.csv",meter.rows)
        write(args.output/"stage_ledger.json",meter.ledger)
        write(args.output/"files_manifest.json",[{"file":f.name,"bytes":f.stat().st_size,"sha256":sha(f)}
            for f in sorted(args.output.iterdir()) if f.is_file() and f.name!="files_manifest.json"])


if __name__=="__main__":main()
