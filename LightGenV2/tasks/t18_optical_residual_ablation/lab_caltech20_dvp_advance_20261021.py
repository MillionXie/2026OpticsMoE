"""User-authorized layerwise continuation, NOT a claim that old gates passed.

Capture L4 G2, then L6 G1/E2/G2/E3/G3, each TRAIN1600 -> TEST411.
One bounded suffix fit per newly measured layer, no repeated current-mask fit,
no scheduler, no inference graph/device changes. All source is Git-published.
"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

CORE_SHA='977eb00b88d85f1deb31eda557aa9e753fc6b15dfb8cf30d0aa529268c9c10d5'
LAYERWISE_SHA='decb4f13fefee737d7bf8b91e458a2b41bee4e351da472b52c0c676dacd57f76'
HARDWARE='0c6b75372fb5d2fc51f0cc2ae14eee3b5babac2cb8b62fc8709e9ac3a8417982'
DATA='e6fae9878ddb8fb7b622aeb7d53d7eb2e224db4d95a69cdf20c017d25b2f7d1c'


def schedule(depth):
    assert depth in (4,6)
    return ['3'] if depth==4 else ['1','2','3','4','5']


def admission(a):
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    torch.set_num_threads(2);torch.manual_seed(73)
    assert m.sha(loop.__file__)==CORE_SHA and m.sha(m.__file__)==LAYERWISE_SHA
    assert m.session_hardware_digest(a.sequence)==HARDWARE and not m.LOCK.exists()
    cfg=m.read(a.profile)
    assert cfg['schema']=='DVP_user_carry_layerwise_v1'
    assert cfg['user_current_stage_waiver'] and cfg['repeat_current_stage_fit'] is False
    assert cfg['TEST_gradient'] is False and cfg['VALID']==0
    data,dm=m.data_for(a.sequence);assert dm['sha256']==DATA
    hw=m.read(a.sequence/'calibration/admission/admitted_hardware.json')
    assert (hw['exposure_us'],hw['gain'],hw['wait_ms'],hw['orientation'])==(60006.,1.,240,'flip_v')
    for p,k in (('coordinates','coordinates_SHA256'),('response_lut','response_lut_SHA256'),('phase_lut','phase_LUT_SHA256')):
        assert m.sha(hw[p])==hw[k]
    roots={};proof={}
    for depth in (4,6):
        c=cfg['parents'][str(depth)];parent=a.sequence/c['relative']
        assert m.sha(parent)==c['SHA256']
        session=a.sequence/f'L{depth}_response_20261020'
        prefix=c['prefix'];stage=str(prefix-2)
        fields,identities=m.fields_for(session,prefix-1,data,dm,HARDWARE)
        cap=a.sequence/f'TRAIN_restored_same_mask_pair_20261021/L{depth}/capture'
        report=m.read(cap/'report.json');audit=m.read(cap/'independent_audit.json')
        assert report['state']=='complete' and report['physical_ccd']==2011 and report['SDK_released']
        assert audit['passed'] and audit['report_SHA256']==m.sha(cap/'report.json')
        assert audit['protocol_SHA256']==m.sha(cap/'protocol.json')
        assert m.read(cap/'same_stimulus_audit.json')['matched_samples']==2011
        fields[stage]={};paths={k:session/'capture'/k for k in fields if k!=stage};paths[stage]=cap
        for role,n in (('train',1600),('test',411)):
            p=cap/role/'ccd.npy';digest=m.sha(p);assert digest==report['arrays'][role]['SHA256']
            array=np.load(p,mmap_mode='r');assert array.shape==(n,478,478) and array.dtype==np.uint8
            fields[stage][role]=array;identities[stage+'/'+role]=digest
        pp=m.read(parent.parent/'protocol.json')
        assert pp['prefix_SHA256']==identities
        model,_=m.model_for(parent,c['SHA256'],depth,session/'source_reference.ts')
        score,_=loop.original_replay(model,data,fields,'test');assert score['correct']==c['correct']
        roots[depth]=dict(model=model,fields=fields,identities=identities,paths=paths,parent=parent,reference=session/'source_reference.ts')
        proof[str(depth)]=dict(PT=str(parent),PT_SHA256=c['SHA256'],TEST=score,
            measured_stages=list(fields),prefix_SHA256=identities,
            previous_threshold_correct=c['previous_threshold_correct'],previous_threshold_met=score['correct']>=c['previous_threshold_correct'],
            user_authorized_carry=True,next_stage=schedule(depth)[0])
    return m,loop,cfg,data,dm,hw,roots,proof


def acquire(m,loop,a,data,dm,hw,row,depth,stage,out):
    """New descendant capture with explicit immutable measured-prefix lineage."""
    import cv2
    import numpy as np
    from PIL import Image
    from lab_caltech20_dvp_TRAIN_response_20261020 import protected_paths
    from lab_caltech20_dvp_persistence_20261019 import FrameWriter,write_images
    model=row['model'];previous=row['fields'];digest=m.sha(row['parent'])
    assert list(previous)==['router']+[str(k) for k in range(int(stage))]
    assert not out.exists() and not m.LOCK.exists();out.mkdir()
    protected={str(p):m.sha(p) for p in protected_paths(hw,a.sequence)}
    drive=np.asarray(m.read(hw['response_lut'])['drive'],np.uint8)
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    # On release leave the newly displayed, audited nominal stage phase in place.
    phasepath=out/'phase.bmp';model.cpu().eval();model.measured={}
    phase_pixels=None;count=0;white=0;last_frame=-1;arrays={};pending=None;start=time.monotonic()
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),PT_SHA256=digest,
        parent_PT=str(row['parent']),stage=stage,depth=depth,data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,
        protected=protected,prefix_SHA256=row['identities'],prefix_reports={k:dict(path=str(v),report_SHA256=m.sha(v/'report.json')) for k,v in row['paths'].items()},
        TRAIN=1600,TEST=411,VALID=0,automatic_retry=False,original_layer_forward=True,
        no_old_failed_stage_CCD=True,phase_LUT_scope='linear voltage; NOT verified linear phase',user_stage_carry=True)
    m.save(out/'protocol.json',protocol)
    with m.LOCK.open('x') as f:json.dump(dict(pid=os.getpid(),camera='DVP',kind='user_carry_layerwise',out=str(out),stage=stage),f)
    writer=FrameWriter()
    def finish():
        nonlocal pending,count
        if pending is None:return
        stem,record=pending;result=writer.finish();record['raw_SHA256']=result['raw_SHA256']
        m.save(stem.with_suffix('.json'),record);count+=1;pending=None
        m.save(out/'progress.json',dict(state='capturing',depth=depth,stage=stage,frames=count,expected=2011,latest=record,elapsed_s=time.monotonic()-start))
    try:
        with m.Bench(hw,out) as bench:
            for role in ('train','test'):
                folder=out/role;folder.mkdir()
                values=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(len(data[role][1]),478,478))
                for begin in range(0,len(data[role][1]),8):
                    images,n,end=m.padded(data,role,begin)
                    cache={k:np.asarray(v[role][begin:end]).copy() for k,v in previous.items()}
                    if n<8:cache={k:np.concatenate([v,np.zeros((8-n,478,478),np.uint8)]) for k,v in cache.items()}
                    ids=np.r_[data[role][2][begin:end],['__padding__']*(8-n)]
                    def physical(current,amplitudes,phase,sample_ids):
                        nonlocal phase_pixels,white,last_frame,pending
                        assert current==stage
                        if phase_pixels is None:
                            phase_pixels=phase.copy();Image.fromarray(m.phase_native(phase)).save(phasepath);bench.stage(phasepath)
                        assert np.array_equal(phase,phase_pixels)
                        result=[]
                        for j in range(n):
                            index=begin+j;stem=folder/f'{index:06d}';canonical=stem.with_name(stem.name+'_input.png');bmp=stem.with_suffix('.bmp')
                            gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8)
                            assert np.max(np.abs(gray/255-amplitudes[j]))<=1e-7
                            assert str(sample_ids[j])==str(data[role][2][index])
                            Image.fromarray(gray).save(canonical);Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                            raw,meta=bench.capture(bmp);assert meta['frame_id']>last_frame;last_frame=meta['frame_id']
                            assert abs(meta['exposure_us']-60006)<2 and meta['gain']==1 and meta['shape']==[3648,5480]
                            assert not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
                            crop=m.orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),hw['orientation'])
                            record=dict(role=role,index=index,sample_id=str(sample_ids[j]),PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,
                                phase_SHA256=m.sha(phasepath),canonical_SHA256=m.sha(canonical),BMP_SHA256=m.sha(bmp),frame_metadata=meta,
                                p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),captured_at_ns=time.time_ns())
                            try:finish()
                            except BaseException:
                                m.save(out/'following_frame_rescue.json',dict(record=record,preserved=write_images(stem,raw,crop)));raise
                            writer.submit(stem,raw,crop);pending=(stem,record);values[index]=crop;values.flush();result.append(crop);bmp.unlink()
                            white=white+1 if record['saturation_fraction']>.99 else 0
                            if white>=2:finish();raise RuntimeError('Two consecutive whole-crop white frames; preserve, no retry')
                        if n<8:result.extend([np.zeros((478,478),np.uint8)]*(8-n))
                        return np.stack(result)
                    m.layer_forward(model,images,ids,stage,cache,physical)
                finish();arrays[role]=dict(samples=len(values),SHA256=m.sha(folder/'ccd.npy'))
        assert count==2011 and protected=={p:m.sha(p) for p in protected}
        m.save(out/'report.json',dict(state='complete',PT_SHA256=digest,phase_SHA256=m.sha(phasepath),physical_ccd=count,TRAIN=1600,TEST=411,VALID=0,
            arrays=arrays,SDK_released=True,hardware_SHA256=HARDWARE,data_SHA256=dm['sha256'],protocol_SHA256=m.sha(out/'protocol.json'),elapsed_s=time.monotonic()-start))
    except BaseException as error:
        m.save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_receipts=count,automatic_retry=False));raise
    finally:
        try:finish();writer.close()
        finally:
            assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()
    loop.audit_capture(m,a,data,hw,previous,model,digest,stage,out)
    return {**previous,stage:{role:np.load(out/role/'ccd.npy',mmap_mode='r') for role in ('train','test')}}


def run(a):
    import numpy as np
    import torch
    from lab_caltech20_dvp_TRAIN_regularization_20261021 import TRAINTransforms
    m,loop,cfg,data,dm,hw,roots,proof=admission(a)
    loop.gpu_admission(m,cfg)
    assert not a.out.exists();a.out.mkdir()
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),config=cfg,
        current_stage_handoff=proof,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,
        stage_order={'4':schedule(4),'6':schedule(6)},recurring=False,no_current_stage_refit=True,
        carry_is_not_threshold_pass=True,unmet_intermediate_gate_does_not_repeat=True))
    history=[]
    try:
        for depth in (4,6):
            row=roots[depth]
            for stage in schedule(depth):
                folder=a.out/f'L{depth}'/('stage_'+stage);folder.mkdir(parents=True)
                final=int(stage)==depth-1;prefix=int(stage)+2
                m.save(a.out/'progress.json',dict(state='capturing',depth=depth,stage=stage,full_stage_frames=2011,completed=history))
                row['fields']=acquire(m,loop,a,data,dm,hw,row,depth,stage,folder/'capture')
                row['paths'][stage]=folder/'capture'
                for role in ('train','test'):row['identities'][stage+'/'+role]=m.sha(folder/'capture'/role/'ccd.npy')
                m.save(a.out/'progress.json',dict(state='bounded_suffix_fit',depth=depth,stage=stage,completed=history))
                loop.gpu_admission(m,cfg)
                fitcfg=dict(cfg,_reference=str(row['reference']),suffix_epochs=cfg['final_epochs'] if final else cfg['intermediate_epochs'],
                    suffix_fit_seconds=cfg['final_fit_seconds'] if final else cfg['intermediate_fit_seconds'])
                target=cfg['final_targets'][str(depth)] if final else cfg['historical_intermediate_targets'][str(depth)]
                transforms=TRAINTransforms(row['model'],stage,cfg)
                try:
                    chosen,correct=loop.refine(m,data,row['fields'],row['model'],dict(stage=stage,prefix=prefix,target=target),fitcfg,folder/'suffix')
                finally:transforms.remove()
                report=m.read(folder/'suffix/report.json')
                m.save(folder/'suffix/TRAIN_transform_audit.json',dict(counts=transforms.counts,eval_identity=True,original_hook_free_bestlast=True,
                    synthetic_noise_not_calibrated=True,inference_graph_changed=False))
                row['model']=chosen;row['parent']=folder/'suffix/best.pt'
                score,prob=loop.original_replay(chosen,data,row['fields'],'test');assert score['correct']==correct
                endpoint=dict(depth=depth,stage=stage,measured_stages=list(row['fields']),full_physical=final,
                    TEST_development=score,PT=str(row['parent']),PT_SHA256=m.sha(row['parent']),prefix_SHA256=row['identities'],
                    selected_E0=report['selected_e0'],target_correct=target,target_met=correct>=target,
                    user_carry_to_next_stage=not final,finite_fit_complete=True,new_physical_frames=2011,SDK_released=not m.LOCK.exists(),
                    capture_report_SHA256=m.sha(folder/'capture/report.json'),capture_audit_SHA256=m.sha(folder/'capture/independent_audit.json'),
                    suffix_report_SHA256=m.sha(folder/'suffix/report.json'),suffix_audit_SHA256=m.sha(folder/'suffix/independent_audit.json'))
                m.save(folder/'endpoint_report.json',endpoint);history.append(endpoint);m.save(a.out/'history.json',history)
                print(json.dumps(endpoint),flush=True)
                torch.cuda.empty_cache()
            m.save(a.out/f'L{depth}_full_physical_report.json',history[-1])
        m.save(a.out/'report.json',dict(state='complete',history=history,SDK_released=not m.LOCK.exists(),recurring=False,TEST_gradient=False,all_evidence_retained=True))
    except BaseException as error:
        m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),history=history,automatic_retry=False,all_evidence_preserved=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('plan','run'),required=True)
    p.add_argument('--sequence',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.mode=='plan':
        m,loop,cfg,data,dm,hw,roots,proof=admission(a)
        print(json.dumps(dict(passed=True,parents=proof,SDK_opened=False,stage_order={'4':schedule(4),'6':schedule(6)},total_new_frames=12066)),flush=True)
    else:run(a)
