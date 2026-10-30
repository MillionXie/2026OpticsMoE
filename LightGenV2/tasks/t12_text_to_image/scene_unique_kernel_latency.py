"""Attribute each raw CUPTI GPU activity exactly once via its CPU launch.

FunctionEvent parent links can repeat a kernel. Raw Chrome GPU records and
CUPTI correlation IDs avoid that ambiguity. Asynchronous GPU execution time is
never classified by overlap with a CPU optical scope; the launch time is used.
Missing/conflicting origins remain charged, rather than dropped.
"""
import collections,gzip,hashlib,json,math,sys

EXCLUSIONS=('T12/FFT_simulation','T12/electronic_residual','T12/pure_fusion')


def raw_activity_summary(events):
    launches=collections.defaultdict(list);scopes=collections.defaultdict(list)
    for event in events:
        if event.get('ph')!='X':continue
        args=event.get('args',{});thread=(event.get('pid'),event.get('tid'))
        if event.get('cat')in('cuda_runtime','cuda_driver')and 'correlation'in args:
            launches[args['correlation']].append(event)
        if event.get('cat')=='user_annotation'and event.get('name')in EXCLUSIONS:
            scopes[thread].append(event)
    groups=collections.defaultdict(float);nodes=[];seen=set();duplicates=0
    for index,event in enumerate(events):
        if event.get('ph')!='X'or event.get('cat')not in('kernel','gpu_memcpy','gpu_memset'):continue
        args=event.get('args',{});start=float(event['ts']);duration=float(event['dur'])
        if not math.isfinite(start+duration)or duration<0:raise ValueError('Invalid raw GPU activity')
        identity=(event.get('cat'),event.get('name'),event.get('pid'),event.get('tid'),start,duration,args.get('device'),args.get('context'),args.get('stream'),args.get('correlation'))
        if identity in seen:duplicates+=1;continue
        seen.add(identity);origins=launches.get(args.get('correlation'),[]);labels=[]
        for launch in origins:
            thread=(launch.get('pid'),launch.get('tid'));ts=float(launch['ts'])
            covering={q['name']for q in scopes[thread]if float(q['ts'])<=ts<=float(q['ts'])+float(q['dur'])}
            labels.append(next(('excluded/'+tag.rsplit('/',1)[-1]for tag in EXCLUSIONS if tag in covering),'retained_GPU'))
        group=labels[0]if labels and len(set(labels))==1 else'retained_unattributed_GPU'
        value=duration/1000;groups[group]+=value
        name=event.get('name','');nodes.append(dict(raw_trace_event_index=index,category=event['cat'],GPU_start_us=start,duration_us=duration,
            group=group,correlation=args.get('correlation'),stream=args.get('stream'),CPU_launch_records=len(origins),
            kernel_name_sha256=hashlib.sha256(name.encode()).hexdigest(),kernel_name_prefix=name[:180]))
    total=sum(n['duration_us']for n in nodes)/1000
    if not nodes or abs(sum(groups.values())-total)>max(1e-6,total*1e-9):raise ValueError('GPU duration partition did not close')
    retained=groups['retained_GPU']+groups['retained_unattributed_GPU']
    return dict(groups_ms=dict(groups),device_activity_ms=total,linked_activity_ms=total-groups['retained_unattributed_GPU'],
        retained_GPU_ms=retained,strict_GPU_no_FFT_with_residual_and_fusion_ms=total-groups['excluded/FFT_simulation'],
        raw_unique_GPU_activities=len(nodes),duplicate_raw_records_removed=duplicates,
        every_unique_GPU_activity_billed_once=True,unknown_origins_retained=True,nodes=nodes)


def main():
    if '--profile-known-work'not in sys.argv:raise ValueError('Use this entry for a fresh profile, not formal unprofiled wall')
    import torch
    from pathlib import Path
    from . import scene_compact_latency as latency
    from .scene_scoped_latency import PERIOD_MS
    from .white_reference_scene_probe import digest,write
    old=latency.Observer
    class UniqueObserver(old):
        def end(self,row):
            if not self.profile:return
            torch.cuda.synchronize();self.profiler.__exit__(None,None,None)
            trace=self.output/(row['sample_id']+'.trace.json');self.profiler.export_chrome_trace(str(trace))
            data=json.loads(trace.read_text());summary=raw_activity_summary(data['traceEvents'])
            counts=[dict(modality=name,router_calls=m.router_calls,pairs=m.pairs)for name,m in self.macros.items()]
            if counts:assert len(counts)==2 and all(v['router_calls']==v['pairs']==1 for v in counts)
            assert {q['scope']for q in self.cpu}=={'T12/CPU_preprocessing','T12/CPU_output'}
            result=latency.known_work(summary['retained_GPU_ms'],sum(q['process_CPU_ms']for q in self.cpu),6 if counts else 0,PERIOD_MS)
            result.update(sample_id=row['sample_id'],groups_ms=summary['groups_ms'],CPU_scopes=list(self.cpu),actual_modal_cycles=counts,
                strict_GPU_no_FFT_with_residual_and_fusion_ms=summary['strict_GPU_no_FFT_with_residual_and_fusion_ms'],
                FFT_cost_removed=True,electronic_residual_and_pure_fusion_excluded=True,CPU_submission_wait_wall_not_billed=True,
                profiler_CPU_measurement_may_include_instrumentation=True,attribution='Unique raw GPU activity -> correlated CPU launch -> annotated scope',
                raw_unique_GPU_activities=summary['raw_unique_GPU_activities'],duplicate_raw_records_removed=summary['duplicate_raw_records_removed'],
                every_unique_GPU_activity_billed_once=True,unknown_origins_retained=True)
            zipped=Path(str(trace)+'.gz')
            with gzip.open(zipped,'wb')as stream:stream.write(trace.read_bytes())
            with gzip.open(zipped,'rb')as stream:assert hashlib.sha256(stream.read()).hexdigest()==digest(trace)
            trace.unlink();nodes=self.output/(row['sample_id']+'.kernel_nodes.json');write(nodes,summary['nodes'])
            result.update(trace_gzip=zipped.name,trace_sha256=digest(zipped),kernel_nodes=nodes.name,kernel_nodes_sha256=digest(nodes))
            self.rows.append(result);write(self.output/'known_work_progress.json',dict(done=len(self.rows),latest=result));del self.profiler
    latency.Observer=UniqueObserver
    try:latency.main()
    finally:latency.Observer=old


if __name__=='__main__':main()
