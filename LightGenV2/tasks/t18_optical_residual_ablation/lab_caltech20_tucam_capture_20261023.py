"""Task-local depth-independent TUCam capture, with immutable stage identities.

No model fitting, per-image normalization, retries or cross-session frame mixing.
The currently running older L6 controller is not modified during acquisition.
"""
import io,os,time,hashlib,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np
from PIL import Image
import lab_caltech20_dvp_layerwise_20261019 as m
from lab_caltech20_tucam_session_20261022 import Bench,logical
from lab_caltech20_tucam_L6_20261022 import check_connection
from lab_caltech20_dvp_persistence_20261019 import write_images


def name(stage):return 'router' if stage=='router' else ('E' if int(stage)%2==0 else 'G')+str(int(stage)//2+1)


def validate_hardware(path):
    path=Path(path);hw=m.read(path);report=m.read(path.parent/'report.json')
    assert report['passed'] and report['SDK_released'] and report['hardware_SHA256']==m.sha(path)
    assert hw['spatial_geometry_verified']
    current=path
    for _ in range(4):
        proof=current.parent/'spatial_parent_proof.json'
        if not proof.exists():
            assert m.read(current.parent/'independent_geometry/report.json')['passed'];break
        p=m.read(proof);parent=Path(p['parent_hardware']);assert p['passed'] and m.sha(parent)==p['SHA256']
        base=m.read(parent)
        assert all(hw[k]==base[k] for k in ('coordinates_SHA256','response_lut_SHA256','phase_LUT_SHA256','orientation','sensor_full_scale'))
        current=parent
    else:raise AssertionError('Unbounded spatial proof ancestry')
    for key,digest in (('coordinates','coordinates_SHA256'),('response_lut','response_lut_SHA256'),('phase_lut','phase_LUT_SHA256')):assert m.sha(hw[key])==hw[digest]
    return hw


def audit(model,data,fields,dm,hw,stage,out):
    model.measured={};report=m.read(out/'report.json');protocol=m.read(out/'protocol.json')
    assert report['protocol_SHA256']==m.sha(out/'protocol.json') and protocol['depth']==model.depth
    phase=np.asarray(Image.open(out/'phase.bmp'));drive=np.asarray(m.read(hw['response_lut'])['drive'],np.uint8)
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]));checked=0
    for role in ('train','test'):
        values=np.load(out/role/'ccd.npy',mmap_mode='r');assert m.sha(out/role/'ccd.npy')==report['arrays'][role]['SHA256']
        assert values.shape==(len(data[role][1]),478,478) and len(list((out/role).glob('[0-9]*.json')))==len(values)
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
                    assert hashlib.sha256(buf.getvalue()).hexdigest()==r['BMP_SHA256'] and m.sha(canonical)==r['canonical_SHA256']
                    assert r['index']==i and r['role']==role and r['sample_id']==str(data[role][2][i])==str(sample_ids[j])
                    assert r['PT_SHA256']==report['PT_SHA256'] and r['hardware_SHA256']==report['hardware_SHA256'] and r['data_SHA256']==dm['sha256'] and r['phase_SHA256']==report['phase_SHA256']
                    rawpath=stem.with_name(stem.name+'_raw.png');assert m.sha(rawpath)==r['raw_SHA256'];raw=np.asarray(Image.open(rawpath));assert raw.shape==(2048,2048) and raw.max()<=65535
                    crop=logical(raw.astype(np.uint16),warp,hw['sensor_full_scale'],hw['orientation'])
                    assert np.array_equal(values[i],np.asarray(Image.open(stem.with_suffix('.png')))) and np.array_equal(values[i],crop)
                    meta=r['frame_metadata'];assert meta['driver']=='tucam' and meta['frame_id_source']=='host_successful_capture_counter_NOT_hardware'
                    assert meta['source_dtype']=='uint16' and meta['exposure_us']==hw['exposure_us'] and meta['auto_exposure'] is False;checked+=1
                return np.zeros((8,478,478),np.uint8)
            m.layer_forward(model,images,ids,stage,cache,verify);m.save(out/'audit_progress.json',dict(state='independent_identity_audit',role=role,samples=end,total=len(values)))
    assert checked==2011 and m.sha(out/'phase.bmp')==report['phase_SHA256']
    m.save(out/'independent_audit.json',dict(passed=True,matched_samples=checked,report_SHA256=m.sha(out/'report.json'),protocol_SHA256=m.sha(out/'protocol.json'),all_raw_crop_canonical_BMP_phase_PT_ID_SHA_verified=True,native_crop_recomputed=True,TEST_gradient=False))


def acquire(model,data,dm,fields,paths,hardware,stage,out,parent):
    hardware=Path(hardware);hw=validate_hardware(hardware);out=Path(out)
    assert not out.exists() and not m.LOCK.exists();out.mkdir(parents=True)
    preceding=[] if stage=='router' else ['router']+[str(k) for k in range(int(stage))]
    assert list(fields)==preceding and model.depth in (4,6);model.measured={};model.cpu().eval()
    digest=m.sha(parent);hardware_SHA=m.sha(hardware);drive=np.asarray(m.read(hw['response_lut'])['drive'],np.uint8)
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(hw['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    provenance={k:dict(path=str(v),report_SHA256=m.sha(v/'report.json'),hardware_SHA256=m.read(v/'report.json')['hardware_SHA256']) for k,v in paths.items()}
    identities={k+'/'+r:m.sha(paths[k]/r/'ccd.npy') for k in fields for r in ('train','test')}
    m.save(out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),stage=stage,stage_name=name(stage),depth=model.depth,PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=hardware_SHA,hardware_manifest=str(hardware),prefix_SHA256=identities,prefix_reports=provenance,TRAIN=1600,TEST=411,VALID=0,automatic_retry=False,interstage_fit=False,original_layer_forward=True,phase_LUT_scope='linear voltage, not verified linear phase'))
    m.save(m.LOCK,dict(pid=os.getpid(),camera='tucam',out=str(out),stage=stage));pool=ThreadPoolExecutor(max_workers=1);pending=None;count=0;phase=None;arrays={};start=time.monotonic()
    def finish():
        nonlocal pending,count
        if pending is None:return
        stem,r,future=pending;r.update(future.result());m.save(stem.with_suffix('.json'),r);count+=1;pending=None
        m.save(out/'progress.json',dict(state='capturing',stage=stage,stage_name=name(stage),frames=count,expected=2011,elapsed_s=time.monotonic()-start,latest=r))
        if count==24:m.save(out/'pilot24_report.json',dict(passed=True,reused_in_full_TRAIN=True,TRAIN=24,TEST=0,SDK_connection_unchanged=True,hardware_SHA256=hardware_SHA))
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
                            i=begin+j;stem=folder/f'{i:06d}';canonical=stem.with_name(stem.name+'_input.png');bmp=stem.with_suffix('.bmp');gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8)
                            assert abs(gray/255-amplitudes[j]).max()<=1e-7 and str(sample_ids[j])==str(data[role][2][i])
                            Image.fromarray(gray).save(canonical);Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                            raw,meta=bench.capture(bmp);crop=logical(raw,warp,hw['sensor_full_scale'],hw['orientation'])
                            r=dict(index=i,role=role,sample_id=str(sample_ids[j]),PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=hardware_SHA,phase_SHA256=m.sha(out/'phase.bmp'),canonical_SHA256=m.sha(canonical),BMP_SHA256=m.sha(bmp),frame_metadata=meta,p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),captured_at_ns=time.time_ns(),new_physical_capture=True)
                            try:finish()
                            except BaseException:
                                m.save(out/'following_frame_rescue.json',dict(record=r,preserved=write_images(stem,raw,crop)));raise
                            values[i]=crop;values.flush();pending=(stem,r,pool.submit(write_images,stem,raw.copy(),crop.copy()));result.append(crop);bmp.unlink()
                            if r['p99']<15 or r['saturation_fraction']>.15:
                                finish();raise RuntimeError('TUCam signal guard; preserve data, no automatic retry/exposure change')
                        if n<8:result.extend([np.zeros((478,478),np.uint8)]*(8-n))
                        return np.stack(result)
                    m.layer_forward(model,images,ids,stage,cache,physical)
                finish();arrays[role]=dict(samples=len(values),SHA256=m.sha(folder/'ccd.npy'))
        assert count==2011
        m.save(out/'report.json',dict(state='complete',depth=model.depth,PT_SHA256=digest,data_SHA256=dm['sha256'],hardware_SHA256=hardware_SHA,phase_SHA256=m.sha(out/'phase.bmp'),protocol_SHA256=m.sha(out/'protocol.json'),arrays=arrays,TRAIN=1600,TEST=411,VALID=0,physical_ccd=count,new_physical_ccd=count,SDK_released=True,elapsed_s=time.monotonic()-start))
    except BaseException as e:
        m.save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_receipts=count,automatic_retry=False));raise
    finally:
        try:finish();pool.shutdown(wait=True)
        finally:assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()
    audit(model,data,fields,dm,hw,stage,out)
    return {**fields,stage:{r:np.load(out/r/'ccd.npy',mmap_mode='r') for r in ('train','test')}}
