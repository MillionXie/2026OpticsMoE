"""Exact student graph with only its used frozen Qwen weights materialized.

Native Transformer layers are constructed on meta, replaced by the original
student replacements, and NEVER loaded/executed. CPU electronics is supported.
The native tokenizer, patch embedding, positions, merger and token table stay.
"""
import gc
from pathlib import Path
import numpy as np
from common import ROOT, CHECKPOINT_SHA, setup_imports, model_config, sha, write

def create(device='auto',export_native=False,force_fp32=False):
    setup_imports()
    import torch
    import transformers
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from LightGenV2.tasks.t08_abo_image_text_retrieval import optical_moe as m
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import LoadedBackbone
    from abo_dual.backend import Backend
    checkpoint=ROOT/'assets/best_checkpoint.pt'
    if sha(checkpoint)!=CHECKPOINT_SHA: raise ValueError('Wrong epoch-25 EMA checkpoint')
    from memory import inference_policy,place_inference_model,cast_fp32
    dev,low_vram,fp32=inference_policy(device)
    if force_fp32: fp32=True
    print('Loading fixed student; execution device:',dev,'CPU token table:',low_vram,flush=True)
    if dev.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    settings=m.load_settings(model_config())
    settings.model_id=str(ROOT/'models/Qwen3-VL-Embedding-2B')
    settings.local_files_only=True; settings.cache_dir=None
    if fp32: settings.amp_enabled=False; settings.dtype='float32'
    m.seed_everything(42)
    processor=transformers.AutoProcessor.from_pretrained(settings.model_id,local_files_only=True,
        min_pixels=settings.processor_min_pixels,max_pixels=settings.processor_max_pixels)
    cfg=transformers.AutoConfig.from_pretrained(settings.model_id,local_files_only=True)
    with init_empty_weights(include_buffers=False):
        model=transformers.Qwen3VLForConditionalGeneration(cfg)
    model.requires_grad_(False).eval()
    # Build/load on CPU first: avoid a CUDA peak during checkpoint construction.
    loaded=LoadedBackbone(model,processor,torch.device('cpu'),0.0)
    replacement,readout=m.build_student(loaded,settings)
    payload=m.load_checkpoint(checkpoint,replacement,readout)
    metadata=payload.get('metadata',{}); del payload
    replacement.use_student()
    # LM vocabulary logits and disabled DeepStack heads are not on retrieval path.
    model.lm_head=torch.nn.Identity()
    native_keys={n for n,p in model.named_parameters() if p.is_meta}
    if any('.layers.' in n or '.blocks.' in n for n in native_keys):
        raise RuntimeError('Unexpected native Transformer parameters remain in student')
    snapshot=Path(settings.model_id)
    compact=snapshot/'native_student.safetensors'
    files=[compact] if compact.exists() else sorted(snapshot.glob('model*.safetensors'))
    native={}
    for f in files:
        with safe_open(f,framework='pt',device='cpu') as weights:
            for key in native_keys.intersection(weights.keys()): native[key]=weights.get_tensor(key)
    missing=native_keys-native.keys()
    if missing: raise RuntimeError('Missing native Qwen weights: '+str(sorted(missing)))
    if export_native:
        from safetensors.torch import save_file
        save_file({k:v.contiguous() for k,v in native.items()},str(compact))
        write(ROOT/'results/compact_frontend.json',{'native_keys':sorted(native),'sha256':sha(compact),
              'bytes':compact.stat().st_size,'transformer_weights_loaded':False,'checkpoint_sha256':CHECKPOINT_SHA})
    # assign=True avoids allocating the omitted 2B Transformer weights.
    model.load_state_dict(native,strict=False,assign=True)
    del native
    meta=[n for n,p in model.named_parameters() if p.is_meta]
    if meta: raise RuntimeError('Unmaterialized active parameters: '+str(meta))
    # CPU uses FP32 for compatibility with this older lab CPU; optics stays FP32.
    if fp32: cast_fp32(model,replacement.language_model,preserve_cpu_table=low_vram)
    place_inference_model(model,replacement.language_model,dev,cpu_token_table=low_vram,
                          embedding_output_dtype=torch.float32 if fp32 else None)
    model.eval().requires_grad_(False); readout.to(dev)
    loaded=LoadedBackbone(model,processor,dev,0.0)
    replacement.vision_surrogate.eval(); replacement.language_surrogate.eval()
    replacement.set_phase_dropout_active(False); readout.eval()
    b=Backend.__new__(Backend)
    b.m=m; b.settings=settings; b.device=dev; b.loaded=loaded
    b.replacement=replacement; b.readout=readout; b.metadata=metadata; b.reference_phases=None
    b.cpu_token_table=low_vram
    b.branches={k:getattr(replacement,k+'_surrogate').core.optical_branch for k in ('vision','language')}
    b.original={}; b.guarded=False
    # The original lists are meta-only; teacher mode is deliberately unavailable.
    b.replacement.original_vision=[]; b.replacement.original_language=[]
    gc.collect()
    print('Student ready:',dev,'FP32' if fp32 else settings.dtype,flush=True)
    return b

def forward(b,sample,measured=None,*,release=False):
    import torch
    b.install(measured or {})
    try:
        with torch.inference_mode(),torch.autocast(device_type=b.device.type,
            dtype=torch.bfloat16 if b.settings.dtype=='bfloat16' else torch.float16,
            enabled=b.device.type=='cuda' and b.settings.amp_enabled):
            out,_=b.m.student_embeddings(b.loaded.model,b.replacement,b.readout,b.inputs(sample))
        return out.detach().float().cpu().numpy()[0]
    finally:
        b.install({})
        if release:
            from memory import release_transients
            release_transients(b)

def phases(b):
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optical_artifacts import phase_tensors
    result={}
    for name,branch in b.branches.items():
        physical=phase_tensors(branch.core)
        for stage,value in [('router',branch.core.router.active_phase()),('expert',physical['physical_expert_mosaic_rad']),('global',physical['physical_global_phase_rad'])]:
            result[name+'_'+stage]=value.detach().float().cpu().numpy()
    return result

def optical_contract(b):
    core=b.branches['vision'].core; r=core.router
    return {'checkpoint_sha256':CHECKPOINT_SHA,'active_size':478,'input_size':224,'canvas_size':518,
        'input_normalization':core.amplitude_slm_input_normalization,
        'expert_apertures':[[a.x0,a.y0,a.x1,a.y1] for a in core.geometry.expert_apertures],
        'router':{'bounds':[list(x) for x in r.detector_bounds],'temperature':r.temperature,
            'top_k':r.top_k,'eps':r.eps,'normalization':r.weight_normalization,
            'score_normalization':r.score_normalization},
        'router_quality':{n:float(getattr(b.settings,'optical_router_'+n)) for n in (
            'maximum_saturated_pixel_fraction','minimum_p99_uint8','minimum_dynamic_range_uint8','minimum_topk_probability_margin')}}
