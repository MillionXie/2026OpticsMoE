"""One exact real CPU boundary check, no model forward or CUDA initialization."""
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as measurement

def main():
    import torch
    import lgvq_live_frontend as live
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    model='/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203'
    measurement.prepare_cpu_worker(model)
    row=next(r for r in read_manifest(live.MANIFEST) if r.split=='test')
    prepared=measurement.prepare_video(dict(sample_id=row.sample_id,video_path=str(row.video_path)))
    _,patch,_=scope.processor_adapter(type(scope.CPU_PROCESSOR.image_processor)._preprocess,lambda *a:None)
    pixels,record=measurement.patch_video(prepared,0,0,patch)
    if prepared['normalized'] or torch.cuda.is_initialized():raise ValueError('Normalized inputs retained or CUDA initialized')
    phases=['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual','merger','adapter_language','decoder']+[f'oeo{i}' for i in range(1,7)]+[f'optics{i}' for i in range(1,7)]
    kernels=[dict(field=0,phase=p,start_us=i*10,end_us=i*10+5) for i,p in enumerate(phases)]
    patch_rows=[dict(record,slot=i,worker_thread=i) for i in range(16)]
    scheduled=measurement.summarize_compute(kernels,patch_rows,[dict(thread_cpu_s=.001)],1)
    if abs(sum(scheduled['equivalent_s_per_group'].values())-scheduled['time16_s'])>1e-9:
        raise ValueError('CPU-stage partitioned equivalent schedule did not close')
    print(json.dumps(dict(status='CPU_NATIVE_RESIZE_NORMALIZE_PATCH_BYTES_EQUAL',shape=list(pixels.shape),record=record,
        CUDA_initialized=False,no_model_forward=True)))

if __name__=='__main__':main()
