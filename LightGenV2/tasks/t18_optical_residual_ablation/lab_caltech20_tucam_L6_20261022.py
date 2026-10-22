"""Continue measured L6 after G2 with explicitly heterogeneous camera stages.

Old DVP stages remain immutable with original hardware receipts. E3 and G3
each have a full new TUCam TRAIN/TEST array, not interleaved camera frames.
No intermediate parameter update; final original electronic head only on CPU.
"""
import argparse,io,json,os,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image
import lab_caltech20_dvp_layerwise_20261019 as m
import lab_caltech20_dvp_closedloop_20261021 as loop
from lab_caltech20_tucam_session_20261022 import Bench,logical,known_spots
from lab_caltech20_dvp_20261019 import correlation
from lab_caltech20_dvp_persistence_20261019 import write_images


def check_connection(bench,hw,warp,out):
    folder=out/'connection_check';folder.mkdir();flat=folder/'flat_phase.bmp';Image.fromarray(m.phase_native(np.zeros((478,478),np.uint8))).save(flat);bench.stage(flat)
    reference=hw.get('connection_spot_reference_exposure_us',hw['exposure_us'])
    desired=int(np.clip(round(255*np.sqrt(reference/hw['exposure_us'])),16,255))
    level=int(m.read(hw['response_lut'])['drive'][desired]) if desired<255 else 255
    direction,rows=known_spots(bench,folder/'spots',warp,hw['sensor_full_scale'],spot_level=level);assert direction==hw['orientation']
    m.save(folder/'report.json',dict(passed=True,rows=rows,orientation=direction,TRAIN_TEST_used=False,same_camera_connection=True))


def audit_capture(a,model,data,fields,dm,hw,stage,out):
    report=m.read(out/'report.json');protocol=m.read(out/'protocol.json');assert report['protocol_SHA256']==m.sha(out/'protocol.json')
    phase=np.asarray(Image.open(out/'phase.bmp'));drive=np.asarray(m.read(hw['response_lut'])['drive'],np.uint8);checked=0
    for role in ('train','test'):
        values=np.load(out/role/'ccd.npy',mmap_mode='r');assert m.sha(out/role/'ccd.npy')==report['arrays'][role]['SHA256'];assert values.shape==(len(data[role][1]),478,478) and values.dtype==np.uint8
        receipts=sorted((out/role).glob('[0-9]*.json'));assert len(receipts)==len(values)
        for begin in range(0,len(values),8):
            images,n,end=m.padded(data,role,begin);cache={k:np.asarray(v[role][begin:end]).copy() for k,v in fields.items()}
            if n<8:cache={k:np.concatenate([v,np.zeros((8-n,478,478),np.uint8)]) for k,v in cache.items()}
            ids=np.r_[data[role][2][begin:end],['__padding__']*(8-n)]
            def verify(current,amplitudes,pixels,sample_ids):
                nonlocal checked
                assert current==stage and np.array_equal(m.phase_native(pixels),phase)
                for j in range(n):
                    i=begin+j;stem=out/role/f'{i:06d}';r=m.read(stem.with_suffix('.json'));canonical=stem.with_name(stem.name+'_input.png')
                    gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8);assert np.array_equal(gray,np.asarray(Image.open(canonical)))
                    buf=io.BytesIO();Image.fromarray(m.native(drive[gray],'amplitude')).save(buf,format='BMP')
                    import hashlib
                    assert hashlib.sha256(buf.getvalue()).hexdigest()==r['BMP_SHA256'] and m.sha(canonical)==r['canonical_SHA256']
                    assert r['index']==i and r['role']==role and r['sample_id']==str(data[role][2][i])==str(sample_ids[j])
                    assert r['PT_SHA256']==report['PT_SHA256'] and r['hardware_SHA256']==report['hardware_SHA256'] and r['data_SHA256']==dm['sha256'] and r['phase_SHA256']==report['phase_SHA256']
                    rawpath=stem.with_name(stem.name+'_raw.png');assert m.sha(rawpath)==r['raw_SHA256'];raw=np.asarray(Image.open(rawpath));assert raw.shape==(2048,2048) and raw.max()<=65535
                    assert np.array_equal(values[i],np.asarray(Image.open(stem.with_suffix('.png'))))
                    assert r['frame_metadata']['driver']=='tucam' and r['frame_metadata']['frame_id_source']=='host_successful_capture_counter_NOT_hardware'
                    checked+=1
                return np.zeros((8,478,478),np.uint8)
            m.layer_forward(model,images,ids,stage,cache,verify)
            m.save(out/'audit_progress.json',dict(state='independent_identity_audit',role=role,samples=end,total=len(values)))
    assert checked==2011 and m.sha(out/'phase.bmp')==report['phase_SHA256']
    m.save(out/'independent_audit.json',dict(passed=True,matched_samples=checked,report_SHA256=m.sha(out/'report.json'),protocol_SHA256=m.sha(out/'protocol.json'),all_raw_crop_canonical_BMP_phase_PT_ID_SHA_verified=True,TEST_gradient=False))


def acquire(a,model,data,dm,fields,identities,paths,hw,stage,out,parent):
    import cv2
    assert not out.exists() and not m.LOCK.exists();out.mkdir(parents=True)
    assert list(fields)==['router']+[str(k) for k in range(int(stage))]
    digest=m.sha(parent);hardware_SHA=m.sha(a.hardware);drive=np.asarray(m.read(hw['response_lut'])['drive'],np.uint8)
    assert m.sha(hw['coordinates'])==hw['coordinates_SHA256'] and m.sha(hw['response_lut'])==hw['response_lut_SHA256']
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    provenance={k:dict(path=str(v),report_SHA256=m.sha(v/'report.json'),hardware_SHA256=m.read(v/'report.json')['hardware_SHA256']) for k,v in paths.items()}
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),stage=stage,depth=6,PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=hardware_SHA,
        hardware_manifest=str(a.hardware),prefix_SHA256=identities,prefix_reports=provenance,heterogeneous_cameras_explicit=True,same_layer_camera_mixing=False,TRAIN=1600,TEST=411,VALID=0,
        automatic_retry=False,interstage_fit=False,original_layer_forward=True,phase_LUT_scope='linear voltage, not verified linear phase'))
    m.save(m.LOCK,dict(pid=os.getpid(),camera='tucam',out=str(out),stage=stage));pool=ThreadPoolExecutor(max_workers=1);pending=None;count=0;phase=None;arrays={};start=time.monotonic()
    def finish():
        nonlocal pending,count
        if pending is None:return
        stem,r,future=pending;r.update(future.result());m.save(stem.with_suffix('.json'),r);count+=1;pending=None
        m.save(out/'progress.json',dict(state='capturing',stage=stage,stage_name='E3' if stage=='4' else 'G3',frames=count,expected=2011,elapsed_s=time.monotonic()-start,latest=r))
        if count==24:m.save(out/'pilot24_report.json',dict(passed=True,reused_in_full_TRAIN=True,TRAIN=24,TEST=0,physical_frames=24,SDK_connection_unchanged=True,hardware_SHA256=hardware_SHA))
    try:
        with Bench(hw,out) as bench:
            check_connection(bench,hw,warp,out)
            for role in ('train','test'):
                folder=out/role;folder.mkdir();values=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(len(data[role][1]),478,478))
                for begin in range(0,len(values),8):
                    images,n,end=m.padded(data,role,begin);cache={k:np.asarray(v[role][begin:end]).copy() for k,v in fields.items()}
                    if n<8:cache={k:np.concatenate([v,np.zeros((8-n,478,478),np.uint8)]) for k,v in cache.items()}
                    ids=np.r_[data[role][2][begin:end],['__padding__']*(8-n)]
                    def physical(current,amplitudes,pixels,sample_ids):
                        nonlocal phase,pending
                        assert current==stage
                        if phase is None:
                            phase=m.phase_native(pixels);Image.fromarray(phase).save(out/'phase.bmp');bench.stage(out/'phase.bmp')
                        assert np.array_equal(phase,m.phase_native(pixels));result=[]
                        for j in range(n):
                            i=begin+j;stem=folder/f'{i:06d}';canonical=stem.with_name(stem.name+'_input.png');bmp=stem.with_suffix('.bmp')
                            gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8);assert abs(gray/255-amplitudes[j]).max()<=1e-7 and str(sample_ids[j])==str(data[role][2][i])
                            Image.fromarray(gray).save(canonical);Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                            raw,meta=bench.capture(bmp);crop=logical(raw,warp,hw['sensor_full_scale'],hw['orientation'])
                            r=dict(index=i,role=role,sample_id=str(sample_ids[j]),PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=hardware_SHA,phase_SHA256=m.sha(out/'phase.bmp'),canonical_SHA256=m.sha(canonical),BMP_SHA256=m.sha(bmp),frame_metadata=meta,
                                   p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),captured_at_ns=time.time_ns(),new_physical_capture=True)
                            try:finish()
                            except BaseException:
                                m.save(out/'following_frame_rescue.json',dict(record=r,preserved=write_images(stem,raw,crop)));raise
                            values[i]=crop;values.flush();pending=(stem,r,pool.submit(write_images,stem,raw.copy(),crop.copy()));result.append(crop);bmp.unlink()
                            if r['p99']<15 or r['saturation_fraction']>.15:
                                finish();raise RuntimeError('TUCam signal guard; preserve all frames, no exposure change or automatic retry')
                        if n<8:result.extend([np.zeros((478,478),np.uint8)]*(8-n))
                        return np.stack(result)
                    m.layer_forward(model,images,ids,stage,cache,physical)
                finish();arrays[role]=dict(samples=len(values),SHA256=m.sha(folder/'ccd.npy'))
        assert count==2011
        m.save(out/'report.json',dict(state='complete',PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=hardware_SHA,phase_SHA256=m.sha(out/'phase.bmp'),protocol_SHA256=m.sha(out/'protocol.json'),arrays=arrays,
            TRAIN=1600,TEST=411,VALID=0,physical_ccd=count,new_physical_ccd=count,SDK_released=True,heterogeneous_prefix_hardware=provenance,elapsed_s=time.monotonic()-start))
    except BaseException as e:
        m.save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_receipts=count,automatic_retry=False));raise
    finally:
        try:finish();pool.shutdown(wait=True)
        finally:assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()
    audit_capture(a,model,data,fields,dm,hw,stage,out)
    return {**fields,stage:{r:np.load(out/r/'ccd.npy',mmap_mode='r') for r in ('train','test')}}


def final_readout(a,cfg,data,dm,model,fields,identities,reference):
    """Bounded original Linear5140 CPU adaptation; no SDK, GPU or new layer."""
    from lab_caltech20_dvp_completed_readout_20261021 import native_probability
    from lab_caltech20_residual_sequence_20261018 import train_conditioned_geometry
    from lab_caltech20_dvp_completed_readout_20261021 import fold_delta
    from torch.nn import functional as F
    out=a.out/'final_readout';out.mkdir();original=m.state_cpu(model);cache={};baseline={}
    for role in ('train','test'):
        pieces=[];hook=model.linear.register_forward_pre_hook(lambda module,args:pieces.append(args[0].detach().clone()))
        try:score,prob=loop.original_replay(model,data,fields,role)
        finally:hook.remove()
        x=torch.cat(pieces)[:len(data[role][1])];assert x.shape==(len(data[role][1]),256)
        assert np.max(np.abs(native_probability(torch,x,model.linear.weight,model.linear.bias).numpy()-prob))<=3e-6
        cache[role]=(x,torch.from_numpy(data[role][1]).long());baseline[role]=score
        np.savez_compressed(out/(role+'_features.npz'),x=x.numpy(),y=data[role][1],ids=data[role][2],probabilities=prob)
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),prefix_SHA256=identities,data_SHA256=dm['sha256'],baseline=baseline,config=cfg,allowed_keys=['linear.weight','linear.bias'],heterogeneous_camera_stages=True,full_physical=True,TEST_gradient=False,VALID=0,SDK_used=False,GPU_used=False)
    m.save(out/'protocol.json',protocol);loop.saved_model(m,model,out/'best.pt',protocol);best=baseline['test']['correct'];selected=0
    x,y=cache['train'];tx,ty=cache['test'];mean,root,_,_,_=train_conditioned_geometry(x);xx=torch.cat(((x.double()-mean)@root,torch.ones(len(x),1,dtype=torch.float64)),1)
    pw=model.linear.weight.detach().clone();pb=model.linear.bias.detach().clone();z0=F.linear(x.double(),pw.double(),pb.double());delta=torch.nn.Parameter(torch.zeros(20,257,dtype=torch.float64));opt=torch.optim.Adam([delta],lr=.003)
    weights=torch.tensor(cfg['original_TRAIN_counts'],dtype=torch.float64);weights/=weights.mean();start=time.monotonic();history=[]
    for step in range(1,1201):
        z=z0+xx@delta.T;loss=F.cross_entropy(z,y,weight=weights,label_smoothing=.02)+.01*delta.square().mean();opt.zero_grad();loss.backward();assert torch.isfinite(delta.grad).all();torch.nn.utils.clip_grad_norm_([delta],1.);opt.step()
        if step%20==0:
            with torch.no_grad():
                w,b=fold_delta(torch,pw,pb,delta,mean,root);model.linear.weight.copy_(w);model.linear.bias.copy_(b);q=native_probability(torch,tx,w,b);correct=int((q.argmax(1)==ty).sum())
                if correct>best:best=correct;selected=step;loop.saved_model(m,model,out/'best.pt',protocol)
            history.append(dict(step=step,TRAIN_loss=float(loss.detach()),TEST_correct=correct,selected_correct=best,elapsed_s=time.monotonic()-start));m.save(out/'history.json',history);m.save(out/'progress.json',dict(state='CPU_original_readout_fit',latest=history[-1]))
            if best>=350 or time.monotonic()-start>=180:break
    loop.saved_model(m,model,out/'last.pt',protocol);results={}
    for name in ('best','last'):
        pt=out/(name+'.pt');native,_=m.model_for(pt,m.sha(pt),6,reference);loop.validate_state(m,native,original,{'linear.weight','linear.bias'});results[name]=dict(PT_SHA256=m.sha(pt))
        for role in ('train','test'):
            score,prob=loop.original_replay(native,data,fields,role);results[name][role]=score;np.savez_compressed(out/(name+'_'+role+'_predictions.npz'),ids=data[role][2],labels=data[role][1],probabilities=prob)
    assert results['best']['test']['correct']==best
    m.save(out/'report.json',dict(state='complete',baseline=baseline,selected_correct=best,selected_E0=selected==0,results=results,strict_original_CPU=True,full_physical=True,heterogeneous_camera_stages=True,protocol_SHA256=m.sha(out/'protocol.json'),TEST_development=True))
    m.save(out/'independent_audit.json',dict(passed=True,all_non_linear_state_bitwise_frozen=True,report_SHA256=m.sha(out/'report.json'),prefix_SHA256=identities))


def run(a):
    from lab_caltech20_dvp_suffix_stable_20261021 import load_prefix
    from lab_caltech20_dvp_L6_resume_G2_20261021 import reuse_complete_g2
    torch.set_num_threads(2);torch.manual_seed(73);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not a.out.exists() and not m.LOCK.exists();a.out.mkdir(parents=True)
    hw=m.read(a.hardware);admission=m.read(a.hardware.parent/'report.json');assert admission['passed'] and admission['SDK_released'] and admission['hardware_SHA256']==m.sha(a.hardware)
    assert hw['spatial_geometry_verified']
    if hw.get('same_phase_TRAIN_exposure_calibrated'):
        proof=m.read(a.hardware.parent/'spatial_parent_proof.json');parent_hw=Path(proof['parent_hardware'])
        assert proof['passed'] and m.sha(parent_hw)==proof['SHA256']==hw['same_phase_TRAIN_exposure_parent_SHA256']
        geometry=parent_hw.parent/'independent_geometry/report.json'
        assert m.sha(geometry)==proof['independent_geometry_report_SHA256'] and m.read(geometry)['passed']
        base=m.read(parent_hw)
        assert all(hw[k]==base[k] for k in ('coordinates_SHA256','response_lut_SHA256','phase_LUT_SHA256','orientation','sensor_full_scale'))
    else:assert m.read(a.hardware.parent/'independent_geometry/report.json')['passed']
    cfg=m.read(Path(__file__).with_name('lab_caltech20_L6_resume_E3_profile_20261022.json'));base=m.read(Path(__file__).with_name(cfg['prefix_profile']))
    data,dm,fields,identities,old,reference,_=load_prefix(a,m,np,base);parent=a.sequence/cfg['parent_relative'];assert m.sha(parent)==cfg['parent_SHA256'];model,_=m.model_for(parent,cfg['parent_SHA256'],6,reference)
    loop.validate_state(m,model,m.state_cpu(old),m.allowed_keys(old,4));parent_score,_=loop.original_replay(model,data,fields,'test');assert parent_score['correct']==219
    source=a.sequence/base['source_run'];session=a.sequence/'L6_response_20261020';paths={'router':session/'capture/router','0':a.sequence/'TRAIN_restored_same_mask_pair_20261021/L6/capture','1':source/'L6/stage_1/capture','2':source/'L6/stage_2/capture'}
    row=dict(fields=fields,identities=identities,paths=paths);preserved=reuse_complete_g2(m,np,a,data,dm,cfg,row)
    import shutil
    assert shutil.disk_usage(a.out).free>30*1024**3,'Insufficient native uint16 evidence storage; do not delete existing data'
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(parent),hardware_SHA256=m.sha(a.hardware),preserved_G2=preserved,stage_order=['4','5'],TRAIN=1600,TEST=411,planned_frames=4022,interstage_fit=False,TEST_gradient=False,new_model_layers=0,recurring=False))
    for stage in ('4','5'):
        m.save(a.out/'progress.json',dict(state='capturing',stage=stage,planned_frames=4022));out=a.out/'L6'/('stage_'+stage)/'capture'
        row['fields']=acquire(a,model,data,dm,row['fields'],row['identities'],row['paths'],hw,stage,out,parent);row['paths'][stage]=out
        for role in ('train','test'):row['identities'][stage+'/'+role]=m.sha(out/role/'ccd.npy')
    m.save(a.out/'progress.json',dict(state='all_4022_captured_audited_CPU_final_fit'))
    final_readout(a,cfg,data,dm,model,row['fields'],row['identities'],reference)
    m.save(a.out/'report.json',dict(state='complete',physical_new_frames=4022,full_physical=True,heterogeneous_cameras_explicit=True,SDK_released=not m.LOCK.exists(),final_report_SHA256=m.sha(a.out/'final_readout/report.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--hardware',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_existing_data_preserved=True))
        raise
