"""Observe seven authorized original Qwen workers; do not alter their inference.

Fresh original worker, no warmup. Native first Transformer pre-hook and the last
actual CUDA-to-CPU result transfer are observed with events. Primary wall/energy
uses first-block entry to the original CPU-result timestamp, per request (no gaps).
"""
from __future__ import annotations
import argparse, bisect, csv, hashlib, json, runpy, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def readrows(p):
    csv.field_size_limit(10000000)
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def block_candidate(model,mode):
    paths=(('model.visual.blocks','visual.blocks','blocks') if mode=='vision' else
           ('model.language_model.layers','language_model.layers','layers'))
    for path in paths:
        value=model
        try:
            for part in path.split('.'):value=getattr(value,part)
            if len(value):
                block=value[0];name=type(block).__name__
                if 'Qwen' in name and (('Vision' in name)==(mode=='vision')):return block,path+'.0'
        except (AttributeError,TypeError,IndexError):pass
    return None
def original_command(identity,output,repeat,deadline):
    cmd=list(identity['command']);script=Path(cmd[0])
    if script.name=='measure_t12_cost.py':cmd=[str(HERE/'gpu_checked_cost_entry_20261025.py'),'--task','t12']+cmd[1:]
    for flag,value in (('--output',str(output)),('--repeat',str(repeat))):
        i=cmd.index(flag);cmd[i+1]=value
    if '--deadline' in cmd:cmd[cmd.index('--deadline')+1]=str(deadline)
    return cmd
def integrate(intervals,power):
    ts=[float(x['host_monotonic_s']) for x in power];energy=0.;age=0.
    assert all(b>a for a,b in zip(ts,ts[1:]))
    for start,end in intervals:
        assert ts[0]<=start<end<=ts[-1]
        cursor=start
        while cursor<end:
            i=bisect.bisect_right(ts,cursor)-1;right=min(end,ts[i+1]);assert right>cursor
            energy+=float(power[i]['watts'])*(right-cursor);age=max(age,right-ts[i]);cursor=right
    assert age<=.25000001,age
    wall=sum(b-a for a,b in intervals)
    return dict(wall_s=wall,board_j=energy,host_proxy_j=338.2*wall,system_j=energy+338.2*wall,max_support_age_s=age)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['vision','language'],required=True);p.add_argument('--repeat',type=int,required=True)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(REPO/'LightGenV2/tasks'):raise ValueError('New task run required')
    template=json.loads(args.template.read_text());cmd=original_command(template,args.output,args.repeat,args.deadline)
    import torch
    from transformers import PreTrainedModel
    starts=[];copies=[];handles=[];attached=set();installed=[]
    post_init=PreTrainedModel.post_init;cpu=torch.Tensor.cpu
    def observed_post_init(model,*a,**kw):
        result=post_init(model,*a,**kw);candidate=block_candidate(model,args.mode)
        if candidate is not None:
            block,path=candidate
            if id(block) not in attached:
                attached.add(id(block));installed.append(dict(owner=type(model).__name__,path=path,block=type(block).__name__))
                def before(owner,inputs):
                    stamp=time.perf_counter();event=torch.cuda.Event(enable_timing=True);event.record()
                    starts.append((stamp,event,id(owner),type(owner).__name__))
                handles.append(block.register_forward_pre_hook(before))
        return result
    def observed_cpu(tensor,*a,**kw):
        if tensor.is_cuda:
            stamp=time.perf_counter();event=torch.cuda.Event(enable_timing=True);event.record()
            copies.append((stamp,event,tuple(tensor.shape),str(tensor.dtype)))
        return cpu(tensor,*a,**kw)
    PreTrainedModel.post_init=observed_post_init;torch.Tensor.cpu=observed_cpu
    success=False;error=None;original_argv=sys.argv
    try:
        sys.argv=cmd;runpy.run_path(cmd[0],run_name='__main__');success=True
    except BaseException as e:error=repr(e);raise
    finally:
        sys.argv=original_argv;PreTrainedModel.post_init=post_init;torch.Tensor.cpu=cpu
        for h in handles:h.remove()
        if args.output.exists():
            write(args.output/'first_tf_observer.json',dict(status='original_worker_complete' if success else 'failed',error=error,
                mode=args.mode,installed=installed,first_block_entries=len(starts),cuda_cpu_copies=len(copies),
                sources={str(Path(__file__).relative_to(REPO)):sha(__file__)},template=str(args.template),template_sha256=sha(args.template),
                native_model_math_unchanged=True,extra_model_forwards=0,explicit_warmup_forwards=0,
                wall_endpoint='original per-request CPU task result',cuda_endpoint='event immediately before last CUDA result transfer preceding CPU task result'))
            write(args.output/'first_tf_starts_raw.json',[dict(first_block_start=x[0],module_id=x[2],module_class=x[3]) for x in starts])
    if not success:return
    calls=readrows(args.output/('calls.csv' if (args.output/'calls.csv').exists() else 'samples.csv'))
    assert len(calls)==len(starts)==200,(len(calls),len(starts))
    assert len({x[2] for x in starts})==1,'More than one executed first block'
    torch.cuda.synchronize();rows=[];intervals=[]
    for i,(call,start) in enumerate(zip(calls,starts)):
        end=float(call['cpu_result_return']);a=start[0]
        assert float(call['input_start'])<=a<end
        possible=[x for x in copies if a<x[0]<end];assert possible,(i,'no result transfer')
        last=possible[-1];event_ms=start[1].elapsed_time(last[1]);assert event_ms>0
        intervals.append((a,end));rows.append(dict(index=i,sample_id=call.get('sample_id'),first_block_start=a,
            cpu_result_return=end,core_wall_ms=(end-a)*1000,diagnostic_core_cuda_ms=event_ms,
            cuda_result_transfer_start=last[0],cuda_result_shape=json.dumps(last[2]),module_class=start[3]))
    with (args.output/'first_tf_calls.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    energy=integrate(intervals,readrows(args.output/'telemetry.csv'))
    # Timing-only supplement keeps the original qualified output identity.
    original=args.template.parent;comparisons=[]
    import numpy as np
    for name in ('retrieval_outputs.npz','raw_pose_outputs.npz','predicted_density_maps.npz','grid_outputs.npz','RGB_outputs.npz'):
        if (args.output/name).exists() and (original/name).exists():
            with np.load(args.output/name) as new,np.load(original/name) as old:
                assert set(new.files)==set(old.files)
                for key in new.files:assert np.array_equal(new[key],old[key]),(name,key,'original output changed')
            comparisons.append(name)
    if 'ranking' in calls[0]:
        oldcalls=readrows(original/'calls.csv');assert [r['ranking'] for r in calls]==[r['ranking'] for r in oldcalls];comparisons.append('calls.csv:ranking')
    assert comparisons,'No original output comparison'
    write(args.output/'first_tf_summary.json',dict(status='complete',calls=200,repeat=args.repeat,mode=args.mode,
        **energy,mean_wall_ms=energy['wall_s']*5,mean_system_j=energy['system_j']/200,
        diagnostic_cuda_total_ms=sum(x['diagnostic_core_cuda_ms'] for x in rows),original_outputs_bitwise_equal=comparisons,
        no_pre_input_or_request_gaps_in_core=True,formal_accuracy_from_original_full_TEST=True))
    write(args.output/'first_tf_files_manifest.json',[dict(file=f.name,bytes=f.stat().st_size,sha256=sha(f))
        for f in sorted(args.output.iterdir()) if f.is_file() and f.name!='first_tf_files_manifest.json'])

if __name__=='__main__':main()
