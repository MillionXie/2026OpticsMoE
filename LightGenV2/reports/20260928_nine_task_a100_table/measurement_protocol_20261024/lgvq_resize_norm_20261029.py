"""Native resize + normalization + exact optimized patch, all recomputed.

Only file reading/decoding, RGB conversion/65% crop and quality-pixel input
preparation remain outside. Four original raw RGB crops, not resized or
normalized pixels, feed every measured per-video preprocessing operation.
"""
import ast
from contextlib import contextmanager
import inspect
import json
import textwrap
import threading
import time
from types import MethodType

import lgvq_compute_scope_20261029 as scope
import lgvq_patch_layout_20261029 as layout

NATIVE_INIT=scope.prepare_cpu_worker;NATIVE_PREPARE=scope.prepare_video
NATIVE_SUMMARY=scope.summarize_compute
LOCAL=threading.local()

def observe_decode(original,capture):
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    class Hook(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if ast.unparse(node.func)=='cv2.resize':
                self.count+=1;node.func=ast.Name(id='_record_native_resize',ctx=ast.Load())
            return node
    hook=Hook();tree=ast.fix_missing_locations(hook.visit(tree))
    if hook.count!=1:raise ValueError('Exactly one original decode resize call required')
    namespace=dict(original.__globals__,_record_native_resize=capture)
    exec(compile(tree,'<observed_native_decode_resize>','exec'),namespace)
    return namespace[original.__name__],scope.digest(source.encode())

def prepare_cpu_worker(model_path):
    NATIVE_INIT(model_path)
    import cv2
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
    global CROPS,RESIZED_SHA,DECODE_SHA,MODEL_PATH
    CROPS=[];RESIZED_SHA=[];MODEL_PATH=model_path
    def capture(value,size,interpolation):
        CROPS.append(value.copy())
        resized=cv2.resize(value,size,interpolation=interpolation)
        RESIZED_SHA.append(scope.digest(resized.tobytes()))
        return resized
    observed,DECODE_SHA=observe_decode(native.decode_frames,capture)
    native.decode_frames=observed

def prepare_video(row):
    CROPS.clear();RESIZED_SHA.clear();prepared=NATIVE_PREPARE(row)
    prepared['raw_rgb_crops']=list(CROPS);prepared['expected_resized_SHA']=list(RESIZED_SHA)
    prepared['expected_normalized']=[dict(sha256=scope.digest(a.tobytes()),shape=list(a.shape),
        height=h,width=w,temporal=t,patch=p,merge=m) for a,h,w,t,p,m in prepared['normalized']]
    # Do not retain resized/normalized arrays as formal online inputs.
    prepared['normalized']=[]
    prepared['native_decode_source_sha256']=DECODE_SHA;prepared['processor_model_path']=MODEL_PATH
    if len(CROPS)!=4 or len(prepared['expected_normalized'])!=1:raise ValueError('Original four-frame geometry required')
    return prepared

def normalization_prefix(original):
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source);fn=tree.body[0]
    loops=[(i,n) for i,n in enumerate(fn.body) if isinstance(n,ast.For) and any(
        isinstance(x,ast.Assign) and isinstance(x.value,ast.Call)
        and ast.unparse(x.value.func)=='self.rescale_and_normalize' for x in n.body)]
    if len(loops)!=1:raise ValueError('Pinned native normalization boundary missing')
    index,loop=loops[0]
    end=next(i+1 for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
        and ast.unparse(n.value.func)=='self.rescale_and_normalize')
    loop.body=loop.body[:end]+ast.parse('normalized_inputs.append((patches,resized_height,resized_width,temporal_patch_size,patch_size,merge_size))').body
    fn.body=ast.parse('normalized_inputs=[]').body+fn.body[:index+1]+ast.parse('return normalized_inputs').body
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<native_resize_normalize_prefix>','exec'),namespace)
    return namespace[fn.name]

def processor(model_path):
    if not hasattr(LOCAL,'processor'):
        from transformers import AutoProcessor
        proc=AutoProcessor.from_pretrained(model_path,local_files_only=True).image_processor
        fn=normalization_prefix(type(proc)._preprocess)
        proc._preprocess=MethodType(fn,proc);native=proc.resize
        def timed_resize(*a,**kw):
            start=time.thread_time();value=native(*a,**kw)
            LOCAL.processor_resize_cpu_s+=time.thread_time()-start
            return value
        proc.resize=timed_resize;LOCAL.processor=proc
    return LOCAL.processor

def patch_video(prepared,field,slot,patch_function):
    import cv2
    import torch
    from PIL import Image
    proc=processor(prepared['processor_model_path'])
    LOCAL.processor_resize_cpu_s=0.
    start=time.perf_counter();cpu_start=time.thread_time()
    resized=[cv2.resize(frame,(448,448),interpolation=cv2.INTER_AREA) for frame in prepared['raw_rgb_crops']]
    images=[Image.fromarray(frame) for frame in resized]
    resize_end=time.thread_time();resize_wall_end=time.perf_counter()
    normalized=proc(images=images,return_tensors='pt')
    norm_end=time.thread_time();norm_wall_end=time.perf_counter()
    if len(normalized)!=1:raise ValueError('One original normalized four-frame batch required')
    value,h,w,t,p,m=normalized[0]
    pixels=layout.strided_patch(value,h,w,t,p,m)
    cpu_end=time.thread_time();end=time.perf_counter()
    resize_s=resize_end-cpu_start+LOCAL.processor_resize_cpu_s
    norm_s=norm_end-resize_end-LOCAL.processor_resize_cpu_s
    patch_s=cpu_end-norm_end
    if min(resize_s,norm_s,patch_s)<0:raise ValueError('CPU preprocessing partition inconsistent')
    # All verification below is bookkeeping, never an online neural feature.
    if [scope.digest(x.tobytes()) for x in resized]!=prepared['expected_resized_SHA']:
        raise ValueError('Native resize pixels differ')
    expected=prepared['expected_normalized'][0]
    if list(value.shape)!=expected['shape'] or layout.digest_tensor(value)!=expected['sha256']:
        raise ValueError('Native normalized pixels differ')
    native=patch_function(value,h,w,t,p,m).reshape(-1,3*t*p*p)
    raw_equal=layout.digest_tensor(native)==prepared['original_flat_sha256']
    exact=torch.equal(pixels.view(torch.int16),native.to(torch.bfloat16).view(torch.int16))
    if not raw_equal or not exact:raise ValueError('Native patch/cast bytes differ')
    return pixels,dict(field=field,slot=slot,worker_thread=threading.get_ident(),host_start=start,host_end=end,
        thread_cpu_s=cpu_end-cpu_start,raw_patch_bytes_equal=True,sample_id=prepared['sample_id'],
        resize_cpu_s=resize_s,normalization_cpu_s=norm_s,patch_cpu_s=patch_s,
        resize_end_wall=resize_wall_end,normalization_end_wall=norm_wall_end,
        processor_resize_cpu_s=LOCAL.processor_resize_cpu_s,
        resize_pixels_byte_equal=True,normalized_pixels_byte_equal=True,
        original_FP32_patch_SHA_equal=raw_equal,original_post_cast_BF16_bytes_equal=exact,
        original_decode_source_sha256=prepared['native_decode_source_sha256'],
        normalization_scope='native tensor input formatting, grouping, fused rescale/normalize; processor resize separately counted',
        resize_scope='original OpenCV INTER_AREA 448 resize and PIL image packing plus official processor resize',
        file_read_decode_RGB_crop_excluded=True,quality_pixel_input_preparation_excluded=True)

def summarize_compute(kernels,patch_rows,token_rows,fields):
    base=NATIVE_SUMMARY(kernels,patch_rows,token_rows,fields)
    nodes=[];index={(r['field'],r['slot']):r for r in patch_rows}
    for node in base['schedule']['nodes']:
        if node['phase']!='patch':nodes.append(node);continue
        slot=int(node['id'].split('.patch')[1]);r=index[(node['field'],slot)]
        if abs(r['resize_cpu_s']+r['normalization_cpu_s']+r['patch_cpu_s']-r['thread_cpu_s'])>1e-9:
            raise ValueError('Measured CPU stage sums do not close')
        resize=dict(node,id=node['id']+'.resize',phase='resize',duration_s=r['resize_cpu_s'])
        norm=dict(node,id=node['id']+'.normalize',phase='normalization',duration_s=r['normalization_cpu_s'],parents=[resize['id']])
        patch=dict(node,duration_s=r['patch_cpu_s'],parents=[norm['id']])
        nodes.extend([resize,norm,patch])
    schedule=scope.compute_schedule(nodes)
    if abs(schedule['total_s']-base['schedule']['total_s'])>1e-8:raise ValueError('CPU partition changed resource schedule')
    by={n['id']:n for n in schedule['nodes']};equivalent={}
    for key in schedule['additive_path']:
        n=by[key];equivalent[n['phase']]=equivalent.get(n['phase'],0)+n['duration_s']/fields
    base.update(profile='LG_RESIZE_NORMALIZATION_PATCH_COMPUTE_SCOPE_20261029',schedule=schedule,
        equivalent_s_per_group=equivalent,CPU_resize_and_normalization_included=True,
        CPU_preprocessing_raw_parts=patch_rows,CPU_file_read_decode_RGB_crop_excluded=True,
        CPU_quality_pixel_input_preparation_excluded=True)
    return base
