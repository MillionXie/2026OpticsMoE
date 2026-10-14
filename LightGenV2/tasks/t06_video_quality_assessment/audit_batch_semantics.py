"""CPU/meta architecture and source audit; no backbone inference or CUDA."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import subprocess

from . import evaluate_temporal_baseline_batches as frozen


def inventory(model):
    import torch
    model.eval().requires_grad_(False)
    norms, dropout = [], []
    for name, module in model.named_modules():
        kind = module.__class__.__name__
        if "Norm" in kind or isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            record = dict(name=name,kind=kind,training=module.training)
            for key in ("normalized_shape","track_running_stats","momentum","eps"):
                if hasattr(module,key):
                    value=getattr(module,key)
                    record[key]=list(value) if isinstance(value,tuple) else value
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                record.update(is_batchnorm=True,running_mean_present=module.running_mean is not None,
                              running_var_present=module.running_var is not None)
            norms.append(record)
        if isinstance(module,torch.nn.modules.dropout._DropoutNd):
            dropout.append(dict(name=name,kind=kind,p=module.p,training=module.training))
    return dict(module_counts=dict(Counter(m.__class__.__name__ for m in model.modules())),
        normalization=norms,dropout=dropout,
        batchnorm_count=sum(r.get("is_batchnorm",False) for r in norms),
        training_modules=[name for name,m in model.named_modules() if m.training])


def source_record(symbol):
    path=Path(inspect.getfile(symbol))
    source,line=inspect.getsourcelines(symbol)
    return dict(symbol=symbol.__name__,path=str(path),file_sha256=frozen.sha256(path),
                first_line=line,source="".join(source))


def load_cpu_structure(args):
    from accelerate import init_empty_weights
    if args.baseline=="qwen":
        from transformers import AutoConfig,Qwen3VLForConditionalGeneration
        config=AutoConfig.from_pretrained(str(args.model),local_files_only=True)
        with init_empty_weights():
            model=Qwen3VLForConditionalGeneration(config)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextRMSNorm,Qwen3VLVisionAttention,Qwen3VLTextAttention,Qwen3VLModel)
        sources=[source_record(c) for c in (Qwen3VLTextRMSNorm,Qwen3VLVisionAttention,Qwen3VLTextAttention)]
        sources.append(source_record(Qwen3VLModel.forward))
        return model,"meta architecture only, no pretrained weight load",sources
    if args.baseline=="deepseek":
        from deepseek_vl2.models import DeepseekVLV2Processor
        from transformers import AutoConfig,AutoModelForCausalLM
        config=AutoConfig.from_pretrained(str(args.model),local_files_only=True,trust_remote_code=True)
        with init_empty_weights():
            model=AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        from deepseek_vl2.models.modeling_deepseek import MoEGate,DeepseekV2MoE,DeepseekV2RMSNorm,DeepseekV2Attention
        sources=[source_record(c) for c in (MoEGate,DeepseekV2MoE,DeepseekV2RMSNorm,DeepseekV2Attention)]
        sources.append(source_record(model.prepare_inputs_embeds))
        return model,"meta architecture only, no pretrained weight load",sources
    if args.baseline=="clip":
        import clip
        model,_=clip.load(str(args.model),device="cpu",jit=False)
        return model,"original frozen backbone loaded on CPU, no forward",[source_record(model.__class__)]
    from ultralytics import YOLO
    model=YOLO(str(args.model)).model.cpu()
    return model,"original frozen backbone loaded on CPU, no forward",[source_record(model.__class__)]


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise RuntimeError("Architecture audit requires all CUDA devices hidden")
    if args.output.exists():
        raise FileExistsError("Preserve existing source audit")
    owner=Path(__file__).resolve().parent/"runs/simulation"
    if not args.output.is_relative_to(owner.resolve()) or args.output==owner.resolve():
        raise ValueError("Canonical run child required")
    import torch
    torch.set_num_threads(2)
    model,load_scope,sources=load_cpu_structure(args)
    result=dict(baseline=args.baseline,scope=load_scope,model_identity=str(args.model),
        config_sha256=frozen.sha256(args.model/"config.json") if args.model.is_dir() else None,
        model_file_sha256=frozen.sha256(args.model) if args.model.is_file() else None,
        architecture=inventory(model),sources=sources,
        pytorch_batchnorm_forward=source_record(torch.nn.modules.batchnorm._BatchNorm.forward),
        deployed_loader_sha256=frozen.sha256(Path(frozen.__file__)),
        audit_source_sha256=frozen.sha256(Path(__file__)),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        limits=["Meta model inventories modules/configuration, not trained activation values",
                "eval inventory matches deployed loader declarations; does not retroactively record every runtime flag",
                "No new GPU inference, performance score, weight change or timing result"])
    if args.baseline=="deepseek":
        gates=[m for m in model.modules() if m.__class__.__name__=="MoEGate"]
        result["moe_config"]=[dict(top_k=g.top_k,experts=g.n_routed_experts,
            scoring_func=g.scoring_func,topk_method=g.topk_method,seq_aux=g.seq_aux,
            auxiliary_loss_alpha=g.alpha,training=g.training) for g in gates]
    args.output.mkdir(parents=True,exist_ok=False)
    frozen.write_json(args.output/"audit.json",result)
    summary={k:result["architecture"][k] for k in ("module_counts","batchnorm_count","training_modules")}
    summary.update(baseline=args.baseline,moe_config=result.get("moe_config",[])[:1])
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",choices=("qwen","deepseek","clip","yolo"),required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.model,args.output=args.model.resolve(),args.output.resolve()
    run(args)


if __name__=="__main__":
    main()
