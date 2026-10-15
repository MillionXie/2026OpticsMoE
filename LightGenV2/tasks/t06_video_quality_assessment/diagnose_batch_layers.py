"""Bounded CPU input/RoPE checks and frozen A100 layer/route diagnostics.

Diagnostic outputs never replace full-TEST metrics or performance costs.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
from pathlib import Path
import subprocess

from . import diagnose_batch_numerics as previous
from . import evaluate_temporal_baseline_batches as frozen
from .audit_batch_semantics import source_record

BATCHES = (1, 2, 4)
REFERENCE_BYTE_LIMIT = 8 * 1024**3


def prepare_cpu(args, rows):
    import torch
    if args.baseline == "qwen":
        from transformers import AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration
        from accelerate import init_empty_weights
        from . import quality_token_common as core
        from . import quality_token_batch_benchmark_200 as original
        processor = AutoProcessor.from_pretrained(str(args.model), min_pixels=448**2,
            max_pixels=448**2, local_files_only=True, trust_remote_code=True)
        prompt = core.render_prompt(processor, "temporal")
        with init_empty_weights():
            meta = Qwen3VLForConditionalGeneration(AutoConfig.from_pretrained(str(args.model), local_files_only=True))
        def prepare(selected):
            return original.prepare_batch(selected, processor, prompt, torch.device("cpu"), 448)
        return prepare, meta.model
    from deepseek_vl2.models import DeepseekVLV2Processor
    from .visual_backbone_quality import decode_ordered_frames
    from . import quality_token_common as core
    processor = DeepseekVLV2Processor.from_pretrained(str(args.model))
    def prepare(selected):
        prepared, positions = [], []
        for row in selected:
            frames, where = decode_ordered_frames(Path(row["video_path"]), core.FRAME_FRACTIONS[4], 384)
            content = "\n".join([*[f"Frame {i + 1}: <image>" for i in range(4)], core.PROMPTS["temporal"]])
            conversation = [{"role": "<|User|>", "content": content,
                "images": [f"frame_{i+1:02d}.png" for i in range(4)]}, {"role": "<|Assistant|>", "content": ""}]
            prepared.append(processor.process_one(conversations=conversation, images=frames, system_prompt=""))
            positions.append(where)
        return processor.batchify(prepared), positions
    return prepare, None


def cpu_checks(args, rows):
    import torch
    prepare, meta = prepare_cpu(args, rows)
    all_records, sources = {}, []
    if meta is not None:
        sources = [source_record(meta.get_rope_index), source_record(meta.forward)]
    else:
        from deepseek_vl2.models.modeling_deepseek import DeepseekV2Model
        sources = [source_record(DeepseekV2Model.forward)]
    for batch in BATCHES:
        records = []
        for start in range(0, 8, batch):
            inputs, frames = prepare(rows[start:start+batch])
            basic = previous.input_records(inputs, batch)
            mask = inputs["attention_mask"].bool()
            ids = inputs["input_ids"]
            if meta is not None:
                pos, delta = meta.get_rope_index(ids, inputs.get("image_grid_thw"),
                    inputs.get("video_grid_thw"), attention_mask=inputs["attention_mask"])
                pos_kind = "actual_native_get_rope_index_CPU"
            else:
                # The deployed DeepseekV2Model source creates arange for a fresh no-cache call.
                # This is a source-derived CPU check, not a live model hook.
                pos = torch.arange(ids.shape[1]).reshape(1,1,-1).expand(1,batch,-1)
                delta, pos_kind = None, "source_derived_fresh_DeepseekV2_position_arange"
            for i in range(batch):
                valid = mask[i]
                last = int(torch.arange(ids.shape[1])[valid].max())
                records.append(dict(sample_id=rows[start+i]["sample_id"], input=basic[i], frames=frames[i],
                    sequence_length=ids.shape[1], valid_tokens=int(valid.sum()), padding_tokens=int((~valid).sum()),
                    last_valid_index=last, last_valid_token_id=int(ids[i,last]),
                    valid_position_sha256=previous.tensor_hash(pos[:,i,valid]),
                    position_scope=pos_kind, rope_delta=int(delta[i]) if delta is not None else None))
        all_records[str(batch)] = records
    old = json.loads((args.previous_diagnostic/"diagnosis.json").read_text())
    for batch in BATCHES:
        actual = all_records[str(batch)]
        if [r["input"] for r in actual] != old["inputs"][str(batch)]["input_fingerprints"]:
            raise ValueError("CPU preparation no longer matches the preserved formal diagnostic")
        if [r["frames"] for r in actual] != old["inputs"][str(batch)]["selected_frame_positions"]:
            raise ValueError("Frame identity changed")
    def key(record):
        return tuple(record[k] for k in ("sample_id","valid_tokens","last_valid_token_id","valid_position_sha256"))
    stable = {str(b): [key(r) for r in all_records[str(b)]] == [key(r) for r in all_records["1"]] for b in BATCHES}
    frozen.write_json(args.output/"cpu_checks.json",dict(baseline=args.baseline, records=all_records,
        valid_tokens_positions_readout_equal_to_B1=stable, native_sources=sources,
        previous_diagnostic_sha256=frozen.sha256(args.previous_diagnostic/"diagnosis.json"),
        scope="No backbone forward; Qwen actual RoPE helper, DeepSeek source-derived positions; internal GPU masks remain to be traced"))
    print(json.dumps(dict(baseline=args.baseline, positions_equal=stable)),flush=True)


def first_tensor(value):
    import torch
    if torch.is_tensor(value):
        return value
    if isinstance(value,(tuple,list)):
        for item in value:
            result = first_tensor(item)
            if result is not None:
                return result
    return None


class Trace:
    """Full boundary tensors in bounded CPU memory; only statistics reach disk."""
    def __init__(self):
        self.references, self.statistics, self.events = {}, {}, []
        self.byte_count, self.mode, self.start, self.mask, self.grid, self.tiles = 0, "reference", 0, None, None, None

    def setup(self, inputs, start):
        self.start = start
        self.mask = inputs["attention_mask"].bool().cpu()
        keys = set(inputs.keys())
        self.grid = inputs["video_grid_thw"].cpu().prod(-1).tolist() if "video_grid_thw" in keys else None
        self.tiles = None
        if "images_spatial_crop" in keys:
            crops = inputs["images_spatial_crop"].cpu()
            self.tiles = [sum(1+int(w*h) for w,h in row if int(w)>0 and int(h)>0) for row in crops]

    def split(self, tensor, kind):
        import torch
        value = tensor.detach().cpu()
        count = len(self.mask)
        if kind == "route":
            value = value.reshape(count,self.mask.shape[1],-1)
            return [value[i][self.mask[i]] for i in range(count)]
        if kind == "language":
            if value.ndim < 3 or value.shape[:2] != self.mask.shape:
                raise ValueError(f"Language alignment mismatch: {tuple(value.shape)}")
            return [value[i][self.mask[i]] for i in range(count)]
        if self.grid is not None:
            sizes = [int(value.shape[0]*g/sum(self.grid)) for g in self.grid]
        else:
            sizes = self.tiles
        if sizes is None or sum(sizes) != value.shape[0]:
            raise ValueError("Vision sample packing does not align exactly")
        return list(torch.split(value,sizes,dim=0))

    def capture(self, stage, tensor, kind):
        import torch
        if tensor is None:
            raise ValueError(f"No tensor at {stage}")
        if stage not in self.events:
            self.events.append(stage)
        for i,value in enumerate(self.split(tensor,kind)):
            key = (stage,self.start+i)
            if self.mode == "reference":
                if key in self.references:
                    raise ValueError("Repeated boundary for the same video")
                value = value.clone()
                self.byte_count += value.numel()*value.element_size()
                if self.byte_count > REFERENCE_BYTE_LIMIT:
                    raise MemoryError("Bounded CPU reference limit exceeded")
                self.references[key] = value
                continue
            reference = self.references[key]
            if reference.shape != value.shape:
                raise ValueError(f"Stage shape differs at {stage}")
            if kind == "route":
                left,right = reference.sort(-1).values,value.sort(-1).values
                changed = int((left!=right).any(-1).sum())
                stat = self.statistics.setdefault(stage,dict(kind=kind,changed_token_routes=0,tokens=0))
                stat["changed_token_routes"] += changed
                stat["tokens"] += len(value)
            else:
                delta = value.float()-reference.float()
                if not torch.isfinite(delta).all():
                    raise ValueError(f"Nonfinite stage delta at {stage}")
                stat = self.statistics.setdefault(stage,dict(kind=kind,max_abs=0.,square_sum=0.,reference_square_sum=0.,entries=0,changed_entries=0))
                stat["max_abs"] = max(stat["max_abs"],float(delta.abs().max()))
                stat["square_sum"] += float(delta.double().square().sum())
                stat["reference_square_sum"] += float(reference.double().square().sum())
                stat["entries"] += value.numel()
                stat["changed_entries"] += int(torch.count_nonzero(delta))

    def report(self):
        result = {}
        for stage in self.events:
            if stage not in self.statistics:
                continue
            row = dict(self.statistics[stage])
            if row["kind"] != "route":
                row["rms"] = (row["square_sum"]/row["entries"])**.5
                row["relative_l2"] = (row["square_sum"]/max(row["reference_square_sum"],1e-30))**.5
            result[stage] = row
        return result


def install_hooks(model, trace):
    handles = []
    for name,module in model.named_modules():
        cls = module.__class__.__name__
        language = cls.endswith("DecoderLayer")
        vision = cls in ("Qwen3VLVisionBlock","Block","PatchEmbed","Qwen3VLVisionPatchEmbed","Qwen3VLVisionPatchMerger","MlpProjector")
        first_component = (".blocks.0." in name or ".layers.0." in name) and name.endswith((".self_attn",".attn",".mlp"))
        if language or vision or first_component:
            kind = "language" if language or ".layers." in name else "vision"
            def before(_module,args,stage=name,kind=kind):
                trace.capture(stage+":input",first_tensor(args),kind)
            def after(_module,args,out,stage=name,kind=kind):
                trace.capture(stage+":output",first_tensor(out),kind)
            handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        if cls == "MoEGate":
            def route(_module,args,out,stage=name):
                trace.capture(stage+":expert_ids",out[0],"route")
            handles.append(module.register_forward_hook(route))
    if not handles:
        raise ValueError("No actual architecture boundaries discovered")
    return handles


def detach_cost_instrumentation(timer, handles):
    """Direct diagnostic forwards do not use/reset the formal cost timer."""
    timer.close()
    for handle in handles:
        handle.remove()
    handles.clear()


def gpu_checks(args):
    import torch
    from . import reproduce_temporal_baseline as reproduction
    reproduction.cuda_guard()
    old = json.loads((args.previous_diagnostic/"diagnosis.json").read_text())
    model,head,rows,prepare,transfer,forward,timer,original_handles = frozen.load_runtime(args)
    rows = rows[:8]
    handles, gate_contexts, saved_flags = [], [], None
    result = {}
    try:
        detach_cost_instrumentation(timer, original_handles)
        for mode in ("native","bf16_accumulation_control" if args.baseline=="qwen" else "gate_autocast_disabled"):
            trace = Trace()
            handles = install_hooks(model,trace)
            if mode == "bf16_accumulation_control":
                saved_flags = (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                               torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            if mode == "gate_autocast_disabled":
                for gate in (m for m in model.modules() if m.__class__.__name__=="MoEGate"):
                    def enter(_module,_args):
                        ctx = torch.autocast("cuda",enabled=False); ctx.__enter__(); gate_contexts.append(ctx)
                    def leave(_module,_args,_out):
                        gate_contexts.pop().__exit__(None,None,None)
                    handles.extend([gate.register_forward_pre_hook(enter),gate.register_forward_hook(leave)])
            mode_result = {}
            for batch in BATCHES:
                trace.mode = "reference" if batch==1 else "compare"
                trace.statistics = {}
                predictions, features, logits = [], [], []
                for start in range(0,8,batch):
                    frozen.process_audit()
                    cpu,where = prepare(rows[start:start+batch]); inputs = transfer(cpu)
                    if previous.input_records(cpu,batch) != old["inputs"][str(batch)]["input_fingerprints"][start:start+batch]:
                        raise ValueError("Traced inputs differ from original probe")
                    trace.setup(inputs,start)
                    feature,logit = previous.diagnostic_forward(args,model,head,inputs,False)
                    scores = torch.load(args.checkpoint,map_location="cpu",weights_only=False)["level_scores"].float()
                    values = (logit.softmax(-1) @ scores).tolist()
                    # CPU reduction may add tiny differences; native trace must remain faithful.
                    if mode=="native" and max(abs(a-b) for a,b in zip(values,old["probes"]["native"][str(batch)]["predictions"][start:start+batch])) > .001:
                        raise ValueError("Trace changed native predictions; no causal conclusion allowed")
                    predictions.extend(values); features.append(feature); logits.append(logit)
                    del inputs
                stats = trace.report()
                first = next((s for s,v in stats.items() if v.get("changed_entries",v.get("changed_token_routes",0))),None)
                mode_result[str(batch)] = dict(predictions=predictions,stages=stats,first_observed_divergent_boundary=first,
                    reference_bytes=trace.byte_count)
                torch.save(dict(sample_ids=[r["sample_id"] for r in rows],features=torch.cat(features),logits=torch.cat(logits)),args.output/f"{mode}_b{batch}_readout.pt")
                frozen.write_json(args.output/"gpu_layers.json",dict(baseline=args.baseline,mode_in_progress=mode,results={**result,mode:mode_result}))
                print(json.dumps(dict(mode=mode,batch=batch,first_observed=first)),flush=True)
            result[mode] = mode_result
            for handle in handles: handle.remove()
            handles = []
            if saved_flags is not None:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = saved_flags
                saved_flags = None
            del trace
        frozen.process_audit()
        frozen.write_json(args.output/"gpu_layers.json",dict(baseline=args.baseline,results=result,
            scope="same eight TEST videos B1/2/4, no training, no paper metric/cost replacement",
            limit="first observed instrumented boundary, not proof of the earliest primitive operation"))
    finally:
        while gate_contexts: gate_contexts.pop().__exit__(None,None,None)
        for handle in handles+original_handles: handle.remove()
        if saved_flags is not None:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = saved_flags
        timer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("cpu","gpu"),required=True)
    parser.add_argument("--baseline",choices=("qwen","deepseek"),required=True)
    for name in ("manifest","checkpoint","model","output","previous-diagnostic","reproduction-receipt"):
        parser.add_argument("--"+name,type=Path,required=name!="reproduction-receipt")
    args = parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path): setattr(args,name,value.resolve())
    frozen.preflight_files(args)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ("-1" if args.phase=="cpu" else frozen.A100_UUID):
        raise ValueError("Wrong device visibility for selected phase")
    args.output.mkdir(parents=True,exist_ok=False)
    identity = dict(baseline=args.baseline,phase=args.phase,pid=os.getpid(),status="running",
        source_sha256=frozen.sha256(Path(__file__)),checkpoint_sha256=frozen.sha256(args.checkpoint),
        manifest_sha256=frozen.sha256(args.manifest),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    frozen.write_json(args.output/"status.json",identity)
    try:
        import torch
        torch.set_num_threads(2)
        with torch.inference_mode():
            if args.phase=="cpu":
                from . import quality_token_common as core
                rows = [r for r in core.read_manifest(args.manifest) if r["split"]=="test"][:8]
                cpu_checks(args,rows)
            else: gpu_checks(args)
        frozen.write_json(args.output/"status.json",{**identity,"status":"complete"})
    except BaseException as error:
        frozen.write_json(args.output/"status.json",{**identity,"status":"failed","error":repr(error)})
        raise


if __name__=="__main__":
    main()
