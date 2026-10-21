"""Finite full-TRAIN physical-forward / digital-adjoint mask adaptation.

The camera is NOT differentiated. Observed CCD supplies the exact forward
value; the unchanged quantized optical model supplies an APPROXIMATE local
Jacobian. Every changed mask needs its OWN 1600 TRAIN -> 411 TEST acquisition.
Only a fully audited measured-prefix score can admit a following optical layer.
This supersedes the ideal-simulation prefilter for THIS explicit workflow only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

HARDWARE='0c6b75372fb5d2fc51f0cc2ae14eee3b5babac2cb8b62fc8709e9ac3a8417982'
SPECS={
 4:dict(stage='2',prefix=4,target=314,final=342,fit='2_current_response_pair_20261021',PT='8c7443c82b8218113105bfbfec505bb2a595592d69b8921b375b49502a879bfe',correct=275),
 6:dict(stage='0',prefix=2,target=322,final=350,fit='0_current_teacher_pair_20261021',PT='84d4e431832f3bac3fe927f470b64fcc3c470877893239874070bf8badbc8179',correct=291)}


def runtime():
    import torch
    import numpy as np
    import lab_caltech20_dvp_layerwise_20261019 as m
    from torch.nn import functional as F
    torch.set_num_threads(2)
    return torch,np,F,m


def mask_keys(model,stage):
    lo,hi=model.stage_slices[int(stage)]
    return {f'phase_sets.{k}' for k in range(lo,hi)}


def physical_adjoint(base,stage):
    """Training-only wrapper; no persistent state or deployment-layer changes."""
    torch,np,F,m=runtime()
    class PhysicalAdjoint(m.MeasuredPrefix):
        def capture(self,amplitude,current,weights=None,callback=None,ids=None):
            if current!=stage:return super().capture(amplitude,current,weights,callback,ids)
            assert callback is None and current in self.measured
            observed=self.measured[current].float()
            field,_,_=self.physical_fields(amplitude,current,weights)
            simulated=self.propagator(field).abs().square()[:,20:498,20:498]
            gain=(observed.mean((-2,-1),keepdim=True)/simulated.detach().mean((-2,-1),keepdim=True).clamp_min(1e-8)).detach()
            # Exactly observed in forward, approximate native optical Jacobian.
            return observed+(simulated-simulated.detach())*gain
    model=PhysicalAdjoint(base.depth).eval()
    model.load_state_dict(m.state_cpu(base));model.propagator.transfer.copy_(base.propagator.transfer)
    assert set(model.state_dict())==set(base.state_dict())
    model.requires_grad_(False)
    return model


def put_fields(model,data,fields,role,ix,device):
    torch,np,F,m=runtime()
    model.measured={k:torch.from_numpy(np.asarray(v[role][ix]).copy()).to(device) for k,v in fields.items()}
    return m.encode_images(data[role][0][ix].to(device))


def original_replay(model,data,fields,role):
    torch,np,F,m=runtime()
    model=model.cpu().eval();score,prob=m.assess(model,data,fields,role)
    return score,prob


def saved_model(m,model,path,protocol):
    import torch
    state=m.state_cpu(model)
    torch.save(dict(model=state,protocol=protocol),path)
    return m.sha(path)


def validate_state(m,model,original,allowed):
    import torch
    state=m.state_cpu(model)
    assert set(state)==set(original)
    assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed),'Frozen upstream changed'
    for p in model.fusion_logits:assert p.dtype==torch.float64 and .4001<float(p.float())<=.95
    assert sum(p.numel() for p in model.linear.parameters())==5140


def base_admission(a):
    torch,np,F,m=runtime();torch.manual_seed(73)
    assert m.session_hardware_digest(a.sequence)==HARDWARE
    cfg=m.read(a.sequence/'calibration/admission/admitted_hardware.json')
    assert cfg['exposure_us']==60006. and cfg['gain']==1. and cfg['wait_ms']==240 and cfg['orientation']=='flip_v'
    data,dm=m.data_for(a.sequence)
    assert dm['sha256']=='e6fae9878ddb8fb7b622aeb7d53d7eb2e224db4d95a69cdf20c017d25b2f7d1c'
    assert not m.LOCK.exists()
    roots={};proofs={}
    for depth,s in SPECS.items():
        session=a.sequence/f'L{depth}_response_20261020';folder=session/'fits'/s['fit']
        report,protocol,audit=[m.read(folder/name) for name in ('report.json','protocol.json','independent_audit.json')]
        assert report['state']=='complete' and audit['passed'] and audit['report_SHA256']==m.sha(folder/'report.json')
        assert audit['protocol_SHA256']==m.sha(folder/'protocol.json') and m.sha(folder/'best.pt')==s['PT']
        fields,identities=m.fields_for(session,s['prefix'],data,dm,HARDWARE)
        assert protocol['data_SHA256']==dm['sha256'] and all(protocol['prefix_SHA256'][k]==v for k,v in identities.items())
        model,_=m.model_for(folder/'best.pt',s['PT'],depth,session/'source_reference.ts')
        score,_=original_replay(model,data,fields,'test');assert score['correct']==s['correct']
        roots[depth]=(session,folder/'best.pt',fields,identities,model)
        proofs[str(depth)]=dict(PT_SHA256=s['PT'],correct=score['correct'],report_SHA256=m.sha(folder/'report.json'),audit_SHA256=m.sha(folder/'independent_audit.json'),prefix_SHA256=identities)
    return m,data,dm,cfg,roots,proofs


def gpu_admission(m,profile):
    import torch,subprocess
    from lab_caltech20_rich_tail_refine_20261018 import windows_process_names
    assert not m.LOCK.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==profile['GPU_UUID']
    assert torch.cuda.is_available() and 'A100' not in torch.cuda.get_device_name(0)
    names=windows_process_names();own={os.getpid(),os.getppid()}
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines():
        gpu,pid,exe=row.split(',',2);pid=int(pid)
        if gpu.strip()==profile['GPU_UUID']:
            assert pid in own or names.get(pid,Path(exe.strip()).name.lower()) in profile['desktop_graphics_process_allowlist'],('Foreign GPU owner',pid)
    torch.cuda.set_per_process_memory_fraction(.65,0)


def propose(m,data,fields,model,spec,profile,out):
    """ONE accumulated full1600 gradient, then one bounded smooth phase update."""
    torch,np,F,_=runtime();assert not out.exists();out.mkdir()
    original=m.state_cpu(model);keys=mask_keys(model,spec['stage']);work=physical_adjoint(model,spec['stage']).cuda()
    for name,p in work.named_parameters():p.requires_grad_(name in keys)
    baseline,bp=original_replay(model,data,fields,'test')
    cm=np.array(baseline['confusion_matrix']);support=cm.sum(1);errors=1-np.divide(cm.diagonal(),support,out=np.zeros(20),where=support>0)
    # TEST informs TRAIN-class emphasis, NEVER derivative examples/labels.
    prior=np.array([348,160,638,640,102,78,68,72,98,85,68,79,80,70,68,91,69,68,68,191],float)
    weights=torch.tensor(prior/prior.mean()*(1+.25*errors),device='cuda',dtype=torch.float32)
    total_weight=float(weights[torch.from_numpy(data['train'][1]).long().cuda()].sum())
    work.zero_grad(set_to_none=True);loss_sum=0.;used=[];start=time.monotonic();max_replay_error=0.
    for begin in range(0,1600,profile['gradient_microbatch']):
        ix=np.arange(begin,min(1600,begin+profile['gradient_microbatch']));x=put_fields(work,data,fields,'train',ix,'cuda')
        p=work(x)[0];lp=p.clamp_min(1e-12).log();labels=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
        ce=(1-profile['label_smoothing'])*F.nll_loss(lp,labels,reduction='none')-profile['label_smoothing']*lp.mean(1)
        loss=(ce*weights[labels]).sum()/total_weight;assert torch.isfinite(loss);loss.backward();loss_sum+=float(loss.detach());used.extend(ix.tolist())
        if begin==0:
            with torch.no_grad():
                model.cuda();expected=model(put_fields(model,data,fields,'train',ix,'cuda'))[0]
                max_replay_error=float((expected-p.detach()).abs().max());assert max_replay_error<=3e-6
                model.cpu()
        m.save(out/'progress.json',dict(state='full_TRAIN_mask_gradient',TRAIN_consumed=len(used),TRAIN_expected=1600,optimizer_updates=0,loss_sum=loss_sum,elapsed_s=time.monotonic()-start))
    assert used==list(range(1600)) and all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in work.named_parameters() if n in keys)
    records=[]
    with torch.no_grad():
        for name,p in work.named_parameters():
            if name not in keys:continue
            phase=2*math.pi*p.sigmoid();jac=(2*math.pi*p.sigmoid()*(1-p.sigmoid())).clamp_min(1e-4)
            gradient=p.grad/jac
            smooth=F.interpolate(F.adaptive_avg_pool2d(gradient[None,None],(16,16)),p.shape,mode='bilinear',align_corners=False)[0,0]
            norm=smooth.square().mean().sqrt();assert float(norm)>1e-14,'Zero optical gradient; stop, do not manufacture update'
            delta=(-profile['phase_step_rms_radians']*smooth/norm).clamp(-profile['phase_step_max_radians'],profile['phase_step_max_radians'])
            next_phase=(phase+delta).clamp(1e-4,2*math.pi-1e-4);p.copy_(torch.logit(next_phase/(2*math.pi)))
            records.append(dict(key=name,gradient_RMS=float(norm),phase_delta_RMS=float(delta.square().mean().sqrt()),max_phase_delta=float(delta.abs().max())))
    work.cpu();validate_state(m,work,original,keys)
    protocol=dict(method='physical_forward_native_approximate_adjoint_full_TRAIN',TRAIN=1600,TEST_gradient=False,VALID=0,physical_Jacobian_measured=False,exact_measured_forward=True,gradient_microbatch=profile['gradient_microbatch'],effective_gradient_batch=1600,mask_updates=1,allowed_keys=sorted(keys),baseline_TEST=baseline,training_loss=loss_sum,phase_steps=records,forward_replay_max_error=max_replay_error,source_git=m.git_head(),source_SHA256=m.sha(__file__),config=profile)
    m.save(out/'protocol.json',protocol);digest=saved_model(m,work,out/'candidate.pt',protocol)
    m.save(out/'report.json',dict(state='complete',candidate_PT_SHA256=digest,TRAIN_consumed=1600,mask_updates=1,original_upstream_frozen=True,new_mask_needs_new_CCD=True,approximate_adjoint_not_calibrated=True,baseline_TEST=baseline,elapsed_s=time.monotonic()-start))
    del work;torch.cuda.empty_cache()
    candidate,_=m.model_for(out/'candidate.pt',digest,model.depth,profile['_reference'])
    return candidate,digest


def acquire(m,a,data,dm,cfg,previous,model,digest,stage,out):
    """One original layer-forward and one Bench connection, TRAIN then TEST."""
    import cv2
    torch,np,F,_=runtime();from PIL import Image
    from lab_caltech20_dvp_TRAIN_response_20261020 import protected_paths
    from lab_caltech20_dvp_persistence_20261019 import FrameWriter
    assert not out.exists() and not m.LOCK.exists();out.mkdir()
    protected={str(p):m.sha(p) for p in protected_paths(cfg,a.sequence)}
    drive=np.asarray(m.read(cfg['response_lut'])['drive'],np.uint8)
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(cfg['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    nominal=a.sequence/f"L{model.depth}_response_20261020/capture/{stage}/phase.bmp"
    model.measured={}  # Physical callbacks must not be bypassed by a prior replay.
    phasepath=out/'phase.bmp';phase_pixels=None;white=0;count=0;arrays={};start=time.monotonic();last_frame=-1
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),PT_SHA256=digest,stage=stage,depth=model.depth,data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,protected=protected,TRAIN=1600,TEST=411,VALID=0,automatic_retry=False,original_layer_forward=True,no_old_failed_stage_CCD=True,phase_LUT_scope='linear voltage; NOT verified linear phase')
    m.save(out/'protocol.json',protocol)
    with m.LOCK.open('x') as stream:json.dump(dict(pid=os.getpid(),camera='DVP',kind='full_TRAIN_closedloop',out=str(out),stage=stage),stream)
    writer=FrameWriter();pending=None
    def finish():
        nonlocal pending,count
        if pending is None:return
        stem,row=pending;result=writer.finish();row['raw_SHA256']=result['raw_SHA256'];m.save(stem.with_suffix('.json'),row)
        count+=1;m.save(out/'progress.json',dict(state='capturing',stage=stage,frames=count,expected=2011,latest=row,elapsed_s=time.monotonic()-start));pending=None
    try:
        with m.Bench(cfg,out) as bench:
            try:
                for role in ('train','test'):
                    folder=out/role;folder.mkdir();values=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(len(data[role][1]),478,478))
                    for begin in range(0,len(data[role][1]),8):
                        images,n,end=m.padded(data,role,begin);cache={k:np.asarray(v[role][begin:end]).copy() for k,v in previous.items()}
                        if n<8:cache={k:np.concatenate([v,np.zeros((8-n,478,478),np.uint8)]) for k,v in cache.items()}
                        ids=np.r_[data[role][2][begin:end],['__padding__']*(8-n)]
                        def physical(current,amplitude,phase,sample_ids):
                            nonlocal phase_pixels,white,last_frame,pending
                            assert current==stage
                            if phase_pixels is None:
                                phase_pixels=phase.copy();Image.fromarray(m.phase_native(phase)).save(phasepath);bench.stage(phasepath)
                            assert np.array_equal(phase,phase_pixels)
                            result=[]
                            for j in range(n):
                                index=begin+j;stem=folder/f'{index:06d}';canonical=stem.with_name(stem.name+'_input.png');bmp=stem.with_suffix('.bmp')
                                gray=np.rint(amplitude[j].clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amplitude[j]))<=1e-7
                                Image.fromarray(gray).save(canonical);Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                                raw,meta=bench.capture(bmp);assert meta['frame_id']>last_frame;last_frame=meta['frame_id']
                                assert abs(meta['exposure_us']-60006)<2 and meta['gain']==1 and meta['shape']==[3648,5480]
                                assert not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
                                crop=m.orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),cfg['orientation'])
                                row=dict(role=role,index=index,sample_id=str(data[role][2][index]),PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,phase_SHA256=m.sha(phasepath),canonical_SHA256=m.sha(canonical),BMP_SHA256=m.sha(bmp),frame_metadata=meta,p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),captured_at_ns=time.time_ns())
                                try:finish()
                                except BaseException:
                                    from lab_caltech20_dvp_persistence_20261019 import write_images
                                    m.save(out/'following_frame_rescue.json',dict(record=row,preserved=write_images(stem,raw,crop)));raise
                                writer.submit(stem,raw,crop);pending=(stem,row);values[index]=crop;values.flush();result.append(crop);bmp.unlink()
                                white=white+1 if row['saturation_fraction']>.99 else 0
                                if white>=2:finish();raise RuntimeError('Two consecutive whole-crop white frames; preserve and diagnose, no retry')
                            if n<8:result.extend([np.zeros((478,478),np.uint8)]*(8-n))
                            return np.stack(result)
                        m.layer_forward(model,images,ids,stage,cache,physical)
                    finish();arrays[role]=dict(samples=len(values),SHA256=m.sha(folder/'ccd.npy'))
            finally:bench.stage(nominal)
        assert count==2011 and protected=={p:m.sha(p) for p in protected}
        m.save(out/'report.json',dict(state='complete',PT_SHA256=digest,phase_SHA256=m.sha(phasepath),physical_ccd=count,TRAIN=1600,TEST=411,VALID=0,arrays=arrays,SDK_released=True,nominal_restored=True,hardware_SHA256=HARDWARE,data_SHA256=dm['sha256'],protocol_SHA256=m.sha(out/'protocol.json'),elapsed_s=time.monotonic()-start))
    except BaseException as error:
        m.save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),completed_receipts=count,automatic_retry=False));raise
    finally:
        try:finish();writer.close()
        finally:
            assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()
    audit_capture(m,a,data,cfg,previous,model,digest,stage,out)
    return {**previous,stage:{role:np.load(out/role/'ccd.npy',mmap_mode='r') for role in ('train','test')}}


def audit_capture(m,a,data,cfg,previous,model,digest,stage,out):
    """Independent full ID/array/raw/input audit; regenerate ALL canonical inputs."""
    torch,np,F,_=runtime();from PIL import Image
    model.measured={}
    report=m.read(out/'report.json');protocol=m.read(out/'protocol.json');assert report['state']=='complete' and report['SDK_released'] and not m.LOCK.exists()
    assert report['protocol_SHA256']==m.sha(out/'protocol.json') and protocol['protected']=={p:m.sha(p) for p in protocol['protected']}
    phase=np.asarray(Image.open(out/'phase.bmp'));checked=0;drive=np.asarray(m.read(cfg['response_lut'])['drive'],np.uint8)
    for role in ('train','test'):
        folder=out/role;array=np.load(folder/'ccd.npy',mmap_mode='r');assert m.sha(folder/'ccd.npy')==report['arrays'][role]['SHA256']
        assert len(list(folder.glob('*_raw.png')))==len(data[role][1])
        for begin in range(0,len(data[role][1]),8):
            x,n,end=m.padded(data,role,begin);cache={k:np.asarray(v[role][begin:end]).copy() for k,v in previous.items()}
            if n<8:cache={k:np.concatenate([v,np.zeros((8-n,478,478),np.uint8)]) for k,v in cache.items()}
            def verify(current,amplitudes,pixels,ids):
                nonlocal checked
                assert current==stage and np.array_equal(m.phase_native(pixels),phase)
                for j in range(n):
                    i=begin+j;stem=folder/f'{i:06d}';row=m.read(stem.with_suffix('.json'))
                    assert row['PT_SHA256']==digest and row['sample_id']==str(data[role][2][i]) and row['index']==i and row['role']==role
                    assert row['phase_SHA256']==m.sha(out/'phase.bmp') and row['data_SHA256']==report['data_SHA256']
                    canonical=stem.with_name(stem.name+'_input.png');raw=stem.with_name(stem.name+'_raw.png')
                    assert m.sha(canonical)==row['canonical_SHA256'] and m.sha(raw)==row['raw_SHA256']
                    gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8);assert np.array_equal(gray,np.asarray(Image.open(canonical)))
                    assert np.array_equal(array[i],np.asarray(Image.open(stem.with_suffix('.png'))))
                    import io
                    buffer=io.BytesIO();Image.fromarray(m.native(drive[gray],'amplitude')).save(buffer,format='BMP')
                    import hashlib
                    assert hashlib.sha256(buffer.getvalue()).hexdigest()==row['BMP_SHA256'];checked+=1
                return np.zeros((8,478,478),np.uint8)
            m.layer_forward(model,x,data[role][2][begin:end],stage,cache,verify)
    assert checked==2011
    m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),protocol_SHA256=m.sha(out/'protocol.json'),all2011_ID_canonical_raw_array_phase_BMP_SHA_verified=True,SDK_opened=False))


def refine(m,data,fields,model,spec,profile,out):
    """Full TRAIN suffix adaptation; current physical input/mask stays frozen."""
    torch,np,F,_=runtime();out.mkdir();original=m.state_cpu(model);allowed=m.allowed_keys(model,spec['prefix'])
    assert not allowed&mask_keys(model,spec['stage'])
    baseline,prob=original_replay(model,data,fields,'test');best,best_nll=baseline['correct'],baseline['nll'];selected=0
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),config=profile,depth=model.depth,prefix=spec['prefix'],allowed_keys=sorted(allowed),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,baseline_TEST=baseline,current_physical_mask_and_all_upstream_frozen=True)
    m.save(out/'protocol.json',protocol);saved_model(m,model,out/'best.pt',protocol);history=[]
    np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=prob)
    model.cuda();model.requires_grad_(False)
    for name,p in model.named_parameters():p.requires_grad_(name in allowed)
    groups=[]
    for prefix,lr in (('phase_sets.',.0002),('electronic.',.0002),('fusion_logits.',.0002),('linear.',.0001)):
        params=[p for n,p in model.named_parameters() if n in allowed and n.startswith(prefix)]
        if params:groups.append(dict(params=params,lr=lr,weight_decay=0.))
    opt=torch.optim.AdamW(groups);start=time.monotonic();rng=np.random.default_rng(73)
    for epoch in range(1,profile['suffix_epochs']+1):
        model.cuda().train();order=rng.permutation(1600);total=0.;used=0
        cm=np.asarray(baseline['confusion_matrix']);support=cm.sum(1);errors=1-np.divide(cm.diagonal(),support,out=np.zeros(20),where=support>0)
        prior=np.array([348,160,638,640,102,78,68,72,98,85,68,79,80,70,68,91,69,68,68,191],float)
        w=torch.tensor(prior/prior.mean()*(1+.25*errors),dtype=torch.float32,device='cuda')
        for begin in range(0,1600,8):
            ix=order[begin:begin+8];x=put_fields(model,data,fields,'train',ix,'cuda');y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda()
            opt.zero_grad(set_to_none=True);p=model(x)[0];lp=p.clamp_min(1e-12).log();ce=(1-profile['label_smoothing'])*F.nll_loss(lp,y,reduction='none')-profile['label_smoothing']*lp.mean(1)
            loss=(ce*w[y]).sum()/w[y].sum();assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in model.named_parameters() if n in allowed)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
            with torch.no_grad():
                for p in model.fusion_logits:
                    if p.requires_grad:p.clamp_(.4001001,.95)
            total+=float(loss)*len(ix);used+=len(ix)
        assert used==1600;score,prob=original_replay(model,data,fields,'test');baseline=score
        if score['correct']>best or (score['correct']==best and score['nll']<best_nll-1e-8):
            best,best_nll=score['correct'],score['nll'];selected=epoch;saved_model(m,model,out/'best.pt',protocol)
        row=dict(epoch=epoch,TRAIN_loss=total/1600,TRAIN_consumed=used,TEST_correct=score['correct'],TEST_NLL=score['nll'],selected_correct=best,elapsed_s=time.monotonic()-start)
        history.append(row);m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='suffix_fit',latest=row));print(json.dumps(row),flush=True)
        if best>=spec['target'] or time.monotonic()-start>=profile['suffix_fit_seconds']:break
    model.cpu();validate_state(m,model,original,allowed);saved_model(m,model,out/'last.pt',protocol);results={}
    for name in ('best','last'):
        checked,_=m.model_for(out/(name+'.pt'),m.sha(out/(name+'.pt')),model.depth,profile['_reference']);validate_state(m,checked,original,allowed)
        score,prob=original_replay(checked,data,fields,'test');train,tp=original_replay(checked,data,fields,'train')
        np.savez_compressed(out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=prob)
        results[name]=dict(TEST=score,TRAIN=train,PT_SHA256=m.sha(out/(name+'.pt')),strict_CPU=True)
    assert results['best']['TEST']['correct']==best
    m.save(out/'report.json',dict(state='complete',selected_correct=best,selected_epoch=selected,selected_e0=selected==0,target=spec['target'],passed=best>=spec['target'],results=results,actual_epochs=epoch,intermediate=True,full_physical=False,SDK_used=False,TEST_gradient=False,frozen_verified=True,protocol_SHA256=m.sha(out/'protocol.json')))
    m.save(out/'independent_audit.json',dict(passed=True,report_SHA256=m.sha(out/'report.json'),protocol_SHA256=m.sha(out/'protocol.json'),original_CPU_bestlast_TRAIN1600_TEST411=True,all_upstream_and_current_mask_frozen=True))
    del opt;torch.cuda.empty_cache();chosen,_=m.model_for(out/'best.pt',m.sha(out/'best.pt'),model.depth,profile['_reference'])
    return chosen,best


def run(a):
    torch,np,F,m=runtime();profile=m.read(a.profile)
    assert profile['schema']=='DVP_fullTRAIN_physical_adjoint_v1' and profile['rounds_per_depth']==2
    assert (profile['phase_step_rms_radians'],profile['phase_step_max_radians'])==(.015,.04)
    m,data,dm,cfg,roots,proofs=base_admission(a);gpu_admission(m,profile)
    assert not a.out.exists();a.out.mkdir()
    protocol=dict(config=profile,source_git=m.git_head(),source_SHA256=m.sha(__file__),profile_SHA256=m.sha(a.profile),hardware_SHA256=HARDWARE,data_SHA256=dm['sha256'],parents=proofs,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,ideal_simulation_prefilter_replaced_by_actual_measured_gate=True,actual_targets={'4':314,'6':322},final_targets={'4':342,'6':350},automatic_retry=False,recurring=False)
    m.save(a.out/'protocol.json',protocol);active={};history=[]
    for depth,(session,pt,fields,identities,model) in roots.items():active[depth]=dict(model=model,fields=fields,correct=SPECS[depth]['correct'],session=session,pt=pt,closed=False)
    try:
        for round_index in range(1,profile['rounds_per_depth']+1):
            for depth in (4,6):
                row=active[depth];s=SPECS[depth]
                if row['closed']:continue
                folder=a.out/f'L{depth}_round{round_index}';folder.mkdir();pcfg=dict(profile,_reference=str(row['session']/'source_reference.ts'))
                m.save(a.out/'progress.json',dict(state='proposing_mask',depth=depth,round=round_index,retained_correct=row['correct']))
                candidate,digest=propose(m,data,row['fields'],row['model'],s,pcfg,folder/'proposal')
                previous={k:v for k,v in row['fields'].items() if k!=s['stage']}
                m.save(a.out/'progress.json',dict(state='new_mask_capture',depth=depth,round=round_index,candidate_PT_SHA256=digest,expected_frames=2011))
                torch.cuda.empty_cache();newfields=acquire(m,a,data,dm,cfg,previous,candidate,digest,s['stage'],folder/'capture')
                gpu_admission(m,profile);candidate,correct=refine(m,data,newfields,candidate,s,pcfg,folder/'suffix')
                score,prob=original_replay(candidate,data,newfields,'test')
                m.save(folder/'error_analysis.json',dict(TEST_informed_development=True,confusion_matrix=score['confusion_matrix'],support=score['support'],wrong_indices=np.flatnonzero(prob.argmax(1)!=data['test'][1]).tolist(),no_TEST_gradient=True))
                improved=correct>row['correct'];passed=correct>=s['target'];record=dict(depth=depth,round=round_index,previous_correct=row['correct'],candidate_correct=correct,improved=improved,actual_gate_passed=passed,target=s['target'],new_physical_frames=2011,full_physical=False,report_SHA256=m.sha(folder/'suffix/report.json'),audit_SHA256=m.sha(folder/'suffix/independent_audit.json'))
                history.append(record);m.save(a.out/'history.json',history)
                if improved:row.update(model=candidate,fields=newfields,correct=correct,pt=folder/'suffix/best.pt')
                row['closed']=passed or not improved  # No blind repeat of a rejected deterministic proposal.
                m.save(a.out/f'L{depth}_current.json',dict(correct=row['correct'],PT=str(row['pt']),PT_SHA256=m.sha(row['pt']),target=s['target'],passed=passed,history=history,old_parent_retained_if_no_gain=not improved,next_layer_started=False))
                print(json.dumps(record),flush=True)
        m.save(a.out/'report.json',dict(state='complete',history=history,SDK_released=not m.LOCK.exists(),rows=[dict(depth=d,correct=v['correct'],PT=str(v['pt']),PT_SHA256=m.sha(v['pt']),passed=v['correct']>=SPECS[d]['target'],next_layer_started=False) for d,v in active.items()],actual_gate_not_relaxed=True,workflow_requires_review_before_next_layer=True))
    except BaseException as error:
        m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),history=history,automatic_retry=False,all_evidence_preserved=True));raise


def selftest():
    torch,np,F,m=runtime();torch.manual_seed(73);checks=[]
    for depth,stage,prefix in ((4,'2',4),(6,'0',2)):
        base=m.MeasuredPrefix(depth).eval();base.requires_grad_(False)
        with torch.no_grad():
            for p in base.fusion_logits:p.fill_(.8)
            x=torch.rand(2,224,224)*.25;base(x);observed={k:v['intensity'].clone()*173 for k,v in base.last.items() if k in (['router']+[str(i) for i in range(depth)])[:prefix]}
        base.measured=observed;wrapped=physical_adjoint(base,stage);wrapped.measured=observed
        with torch.no_grad():assert torch.equal(base(x)[0],wrapped(x)[0])
        keys=mask_keys(wrapped,stage);before=m.state_cpu(wrapped)
        for n,p in wrapped.named_parameters():p.requires_grad_(n in keys)
        loss=-wrapped(x)[0][:,0].clamp_min(1e-12).log().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().max())>0 for n,p in wrapped.named_parameters() if n in keys)
        assert all(p.grad is None for n,p in wrapped.named_parameters() if n not in keys)
        assert set(before)==set(wrapped.state_dict()) and not keys&m.allowed_keys(wrapped,prefix)
        # A full-batch gradient equals accumulated partition gradients.
        full={n:p.grad.clone() for n,p in wrapped.named_parameters() if n in keys};wrapped.zero_grad(set_to_none=True)
        for i in range(2):
            wrapped.measured={k:v[i:i+1] for k,v in observed.items()};(-wrapped(x[i:i+1])[0][:,0].clamp_min(1e-12).log().sum()/2).backward()
        relative=max(float((p.grad-full[n]).norm()/full[n].norm().clamp_min(1e-12)) for n,p in wrapped.named_parameters() if n in keys)
        assert relative<.02,relative
        validate_state(m,wrapped,before,keys);checks.append(dict(depth=depth,exact_observed_forward=True,mask_gradients_finite_nonzero=True,partition_gradient_relative_error=relative))
    print(json.dumps(dict(passed=True,checks=checks,SDK_opened=False,TEST_gradient=False,original_state_structure=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('run','plan','selftest'),required=True);parser.add_argument('--sequence',type=Path);parser.add_argument('--profile',type=Path);parser.add_argument('--out',type=Path);a=parser.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='plan':
        m,data,dm,cfg,roots,proofs=base_admission(a);print(json.dumps(dict(passed=True,parents=proofs,SDK_opened=False,physical_frames=0,TRAIN=1600,TEST=411)),flush=True)
    else:run(a)
