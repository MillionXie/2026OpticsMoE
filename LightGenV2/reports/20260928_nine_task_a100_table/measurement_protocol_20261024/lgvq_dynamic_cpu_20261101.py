"""Continuous FIFO CPU producer and separately labelled fixed-optics estimate.

Real ThreadPoolExecutor workers take the next video, independent of the GPU.
The ideal compute-only estimate independently list-schedules FIFO jobs on 16
anonymous workers; it never reuses historical OS thread IDs or charges FFT.
"""
import heapq
import math
import threading
import time
from collections import defaultdict

import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as prep
import lgvq_patch_layout_20261029 as layout
import recalculate_lg_front1886_20261030 as optical

PENDING=[]
FUTURES=[]


def prepare_video(row):
    import torch
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
    result=prep.prepare_video(row)
    _,native,_=scope.processor_adapter(Qwen2VLImageProcessorFast._preprocess,lambda *a:None)
    chunks=[native(torch.from_numpy(a),h,w,t,p,m).reshape(-1,3*t*p*p)
            for a,h,w,t,p,m in scope.NORMALIZED]
    reference=torch.cat(chunks)
    if scope.digest(reference.numpy().tobytes())!=result['original_flat_sha256']:
        raise ValueError('Independent native reference patch differs')
    result['native_BF16_patch_sha256']=layout.digest_tensor(reference.to(torch.bfloat16))
    return result


def patch_video(prepared,field,slot,patch_function):
    import cv2
    from PIL import Image
    proc=prep.processor(prepared['processor_model_path'])
    prep.LOCAL.processor_resize_cpu_s=0.
    begin=time.perf_counter();ca=time.thread_time()
    resized=[cv2.resize(x,(448,448),interpolation=cv2.INTER_AREA) for x in prepared['raw_rgb_crops']]
    images=[Image.fromarray(x) for x in resized]
    cb=time.thread_time();b=time.perf_counter()
    normalized=proc(images=images,return_tensors='pt')
    cc=time.thread_time();c=time.perf_counter()
    if len(normalized)!=1:raise ValueError('Original four-frame geometry required')
    value,h,w,t,p,m=normalized[0]
    pixels=layout.strided_patch(value,h,w,t,p,m)
    cd=time.thread_time();end=time.perf_counter()
    extra=prep.LOCAL.processor_resize_cpu_s
    parts=(cb-ca+extra,cc-cb-extra,cd-cc)
    if any(not math.isfinite(x) or x<0 for x in parts):raise ValueError('Invalid CPU partition')
    record=dict(field=field,slot=slot,sample_id=prepared['sample_id'],worker_thread=threading.get_ident(),
        worker_native_id=threading.get_native_id(),host_start=begin,host_end=end,
        resize_end_wall=b,normalization_end_wall=c,thread_cpu_s=cd-ca,
        resize_cpu_s=parts[0],normalization_cpu_s=parts[1],patch_cpu_s=parts[2],
        processor_resize_cpu_s=extra,raw_patch_bytes_equal=False,
        validation_deferred_until_after_formal_window=True,
        CPU_dispatch_waits_retained_in_wall_not_charged_in_projection=True)
    # O(1) bookkeeping only; hashing/native recomputation must not block this queue.
    PENDING.append((pixels,resized,value,prepared,record))
    return pixels,record


def start(pool,prepared,patch_fn):
    if FUTURES:raise ValueError('One formal queue per fresh worker process')
    for i,p in enumerate(prepared):
        FUTURES.append(pool.submit(patch_video,p,i//16,i%16,patch_fn))


def futures(field):
    if len(FUTURES)!=512:raise ValueError('All 512 FIFO jobs must be submitted')
    return FUTURES[field*16:(field+1)*16]


def validate(output):
    import json
    for future in FUTURES:future.result(timeout=30)
    for pixels,resized,value,p,r in PENDING:
        if [scope.digest(x.tobytes()) for x in resized]!=p['expected_resized_SHA']:
            raise ValueError('Native resize bytes differ')
        expected=p['expected_normalized'][0]
        if list(value.shape)!=expected['shape'] or layout.digest_tensor(value)!=expected['sha256']:
            raise ValueError('Native normalization bytes differ')
        if layout.digest_tensor(pixels)!=p['native_BF16_patch_sha256']:
            raise ValueError('Native BF16 patch bytes differ')
        r.update(raw_patch_bytes_equal=True,resize_pixels_byte_equal=True,
                 normalized_pixels_byte_equal=True,original_post_cast_BF16_bytes_equal=True)
    formal=[r for *_,r in PENDING if r['field']>=0]
    if len(formal)!=512 or len({(r['field'],r['slot']) for r in formal})!=512:
        raise ValueError('Exactly 512 unique fresh CPU measurements required')
    receipt=dict(status='all_512_native_CPU_bytes_verified',formal_jobs=512,warm_jobs=48,
        actual_workers=len({r['worker_thread'] for r in formal}),FIFO_queue_no_GPU_backpressure=True,
        hash_validation_after_formal_window=True)
    with (output/'dynamic_CPU_validation.json').open('x') as f:
        json.dump(receipt,f,ensure_ascii=True,indent=2,allow_nan=False)
    PENDING.clear()


def projected_nodes(report,patch_rows,token_rows,fields):
    raw=dict(patch=patch_rows,tokenizer=token_rows)
    nodes=optical.nodes_for(report,raw,1886)
    cpu={n['id']:n for n in nodes if n['resource'].startswith('cpu_patch_')}
    workers=[(0.,i,None) for i in range(16)];heapq.heapify(workers)
    for r in sorted(patch_rows,key=lambda r:(r['field'],r['slot'])):
        at,worker,previous=heapq.heappop(workers)
        key=f'f{r["field"]}.patch{r["slot"]}'
        for suffix in ('.resize','.normalization',''):
            node=cpu[key+suffix]
            node['resource']='cpu_dynamic_'+str(worker)
            node['parents']=[previous] if previous else []
            previous=node['id'];at+=node['duration_s']
        heapq.heappush(workers,(at,worker,previous))
    return nodes


def summarize_compute(kernels,patch_rows,token_rows,fields):
    report=prep.NATIVE_SUMMARY(kernels,patch_rows,token_rows,fields)
    schedule=scope.compute_schedule(projected_nodes(report,patch_rows,token_rows,fields))
    by={n['id']:n for n in schedule['nodes']};parts=defaultdict(float)
    for key in schedule['additive_path']:
        n=by[key];parts[n['phase']]+=n['duration_s']/fields
    report.update(profile='LG_DYNAMIC_FIFO16_CONTINUOUS_CPU_1886_20261101',schedule=schedule,
        time16_s=schedule['total_s']/fields,equivalent_s_per_group=dict(parts),
        optical_layer_s=optical.optical_s(1886),SLM_hz=1886,
        CPU_preprocessing_raw_parts=patch_rows,CPU_resize_and_normalization_included=True,
        CPU_file_read_decode_RGB_crop_excluded=True,CPU_quality_pixel_input_preparation_excluded=True,
        actual_FIFO_worker_queue=True,projected_CPU_workers=16,
        projected_assignment='FIFO video jobs, next earliest free anonymous worker; no historical thread binding',
        CPU_to_backend_backpressure=False,backend_two_field_cross_group_pipeline=True,
        real_CPU_wall_s=max(r['host_end'] for r in patch_rows)-min(r['host_start'] for r in patch_rows),
        real_CPU_wall_is_not_compute_only_projection=True)
    for r in report['raw_phases']:
        if r['phase'].startswith('optics'):r['charged_compute_s']=optical.optical_s(1886)
    return report
