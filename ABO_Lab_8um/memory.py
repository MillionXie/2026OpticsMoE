"""Inference placement only: no learned parameters or optical equations changed."""
import types
import torch

def inference_policy(requested='auto'):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested=='auto' else torch.device(requested)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use the GPU environment; no silent CPU fallback.')
    low_vram=device.type=='cuda' and torch.cuda.get_device_properties(device).total_memory<=4*1024**3
    # Pascal cannot run the original BF16 autocast. FP32 also preserves FFT support.
    fp32=device.type=='cpu' or (device.type=='cuda' and torch.cuda.get_device_capability(device)[0]<8)
    return device,low_vram,fp32

def place_inference_model(model,language_model,device,*,cpu_token_table=False):
    if not cpu_token_table:
        model.to(device)
        return
    table=language_model.embed_tokens
    table.cpu()
    language_model.embed_tokens=torch.nn.Identity()
    try: model.to(device)
    finally: language_model.embed_tokens=table
    original=table.forward
    def lookup(this,input_ids):
        return original(input_ids.to('cpu')).to(device)
    table.forward=types.MethodType(lookup,table)

def release_transients(b):
    """Call only AFTER the whole sample returns, or stops at a CCD boundary.

    Intra-sample residuals and Language stage-1 state MUST remain until consumed.
    No weights, recorded CCDs, BMPs or checkpoint files are deleted.
    """
    def has_tensor(value):
        if torch.is_tensor(value): return True
        if isinstance(value,dict): return any(has_tensor(x) for x in value.values())
        if isinstance(value,(list,tuple)): return any(has_tensor(x) for x in value)
        return False
    for module in b.loaded.model.modules():
        for name,value in list(vars(module).items()):
            if name.startswith(('last_','_stage1_')) and has_tensor(value) and not isinstance(value,torch.nn.Parameter):
                setattr(module,name,{} if isinstance(value,dict) else [] if isinstance(value,list) else None)
    b.replacement.last_language_hidden=None
    if hasattr(b.loaded.model,'_grocery_retrieval_optical_last_hidden'):
        b.loaded.model._grocery_retrieval_optical_last_hidden=None
    if b.device.type=='cuda': torch.cuda.empty_cache()

def memory_report(b):
    report={'device':str(b.device),'precision':'fp32' if not b.settings.amp_enabled else str(b.settings.dtype),
            'cpu_token_embedding_lookup':b.cpu_token_table,'checkpoint_files_deleted':False,
            'gpu_name':None,'peak_allocated_mib':None,'peak_reserved_mib':None}
    if b.device.type=='cuda':
        torch.cuda.synchronize(b.device)
        report.update(gpu_name=torch.cuda.get_device_name(b.device),
                      peak_allocated_mib=torch.cuda.max_memory_allocated(b.device)/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(b.device)/1024**2)
    return report
