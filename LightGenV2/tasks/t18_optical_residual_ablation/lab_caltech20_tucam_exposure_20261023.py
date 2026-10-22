"""Bounded same-phase E3 TRAIN exposure calibration; no TEST or optical refit.

The initial flat-field 2ms admission was geometrically valid but the first E3
TRAIN frame was too dim. Preserve it, calibrate exposure using three fixed
TRAIN anchors, then validate all40 TRAIN(2/class). Keep the field inverse and
optical LUTs unchanged, verify native sensor exposure linearity separately.
"""
import argparse,json,os,time,traceback
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
import lab_caltech20_dvp_layerwise_20261019 as m
from lab_caltech20_tucam_session_20261022 import Bench,logical


def set_exposure(bench,requested):
    module=bench.camera._module;bench.camera._set_property(module.TUCAM_IDPROP.TUIDP_EXPOSURETM,requested/1000,'TRAIN exposure calibration')
    actual=bench.camera._get_property(module.TUCAM_IDPROP.TUIDP_EXPOSURETM)*1000
    assert abs(actual-requested)<max(2,requested*.01)
    bench.cfg={**bench.cfg,'exposure_us':actual};bench.camera.exposure_us=actual;bench.camera._info['Exposure']=actual
    return actual


def run(a):
    assert not a.out.exists() and not m.LOCK.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';a.out.mkdir(parents=True);torch.set_num_threads(2)
    hw=m.read(a.hardware);assert hw['spatial_geometry_verified'];failed=a.hardware.parents[2]/'L6_remaining_20261023/L6/stage_4/capture'
    failure=m.read(failed/'failure.json');receipt=m.read(failed/'train/000000.json');assert failure['completed_receipts']==1 and receipt['p99']<15 and receipt['role']=='train'
    assert receipt['hardware_SHA256']==m.sha(a.hardware);proof=m.read(failed/'protocol.json');data,dm=m.data_for(a.sequence);assert dm['sha256']==proof['data_SHA256']
    cfg=m.read(Path(__file__).with_name('lab_caltech20_L6_resume_E3_profile_20261022.json'));parent=a.sequence/cfg['parent_relative'];assert m.sha(parent)==cfg['parent_SHA256']==receipt['PT_SHA256']
    model,_=m.model_for(parent,cfg['parent_SHA256'],6,a.sequence/'L6_response_20261020/source_reference.ts');fields={}
    for key,value in proof['prefix_reports'].items():
        cap=Path(value['path']);assert m.sha(cap/'report.json')==value['report_SHA256'];p=cap/'train/ccd.npy';assert m.sha(p)==proof['prefix_SHA256'][key+'/train'];fields[key]=np.load(p,mmap_mode='r')
    assert list(fields)==['router','0','1','2','3'];drive=np.array(m.read(hw['response_lut'])['drive'],np.uint8);patterns=a.out/'TRAIN40_inputs';patterns.mkdir();phase=None;rows=[]
    for begin in range(0,40,8):
        images,n,end=m.padded(data,'train',begin);cache={k:np.asarray(v[begin:end]).copy() for k,v in fields.items()}
        def export(stage,amplitudes,pixels,ids):
            nonlocal phase
            assert stage=='4'
            if phase is None:phase=m.phase_native(pixels);Image.fromarray(phase).save(a.out/'phase.bmp')
            assert np.array_equal(phase,m.phase_native(pixels))
            for j in range(8):
                i=begin+j;gray=np.rint(amplitudes[j]*255).clip(0,255).astype(np.uint8);canonical=patterns/f'{i:03d}.png';bmp=patterns/f'{i:03d}.bmp';Image.fromarray(gray).save(canonical);Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                assert str(ids[j])==str(data['train'][2][i]);rows.append(dict(index=i,id=str(ids[j]),label=int(data['train'][1][i]),canonical_SHA256=m.sha(canonical),BMP_SHA256=m.sha(bmp)))
            return np.zeros((8,478,478),np.uint8)
        m.layer_forward(model,images,data['train'][2][begin:end],'4',cache,export)
    assert len(rows)==40 and np.bincount(data['train'][1][:40],minlength=20).tolist()==[2]*20
    assert m.sha(a.out/'phase.bmp')==receipt['phase_SHA256'] and rows[0]['canonical_SHA256']==receipt['canonical_SHA256'] and rows[0]['BMP_SHA256']==receipt['BMP_SHA256']
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),parent_PT_SHA256=m.sha(parent),data_SHA256=dm['sha256'],base_hardware_SHA256=m.sha(a.hardware),phase_SHA256=m.sha(a.out/'phase.bmp'),TRAIN=40,TEST=0,VALID=0,anchors=[0,13,27],exposure_probe_us=[1999.8,5000,10000,20000,40000,80000],field_inverse_unchanged=True,optical_LUTs_unchanged=True,rows=rows))
    pts=np.float32(m.corners(Path(hw['coordinates'])));warp=cv2.getPerspectiveTransform(pts,np.float32([[0,0],[477,0],[477,477],[0,477]]));ladder=[];baseline=None;selected=None;full_scale=hw['sensor_full_scale']
    m.save(m.LOCK,dict(pid=os.getpid(),camera='tucam',kind='same_phase_TRAIN_exposure',out=str(a.out)))
    def capture(bench,path,folder,name):
        raw,meta=bench.capture(path);v=logical(raw,warp,full_scale,hw['orientation']);native=cv2.warpPerspective(raw.astype(np.float32),warp,(478,478),flags=cv2.INTER_AREA)
        Image.fromarray(raw).save(folder/(name+'_raw.png'),compress_level=1);Image.fromarray(v).save(folder/(name+'_crop.png'))
        row=dict(mean=float(v.mean()),native_mean=float(native.mean()),p99=float(np.percentile(v,99)),saturation_fraction=float((v==255).mean()),metadata=meta,raw_SHA256=m.sha(folder/(name+'_raw.png')));m.save(folder/(name+'.json'),row);return row
    try:
        devices=a.out/'devices';devices.mkdir()
        with Bench(hw,devices) as bench:
            bench.stage(a.out/'phase.bmp');zero=a.out/'zero.bmp';Image.fromarray(m.native(np.zeros((478,478),np.uint8),'amplitude')).save(zero)
            for requested in (hw['exposure_us'],5000,10000,20000,40000,80000):
                exposure=set_exposure(bench,requested);folder=a.out/f'exposure_{int(requested)}';folder.mkdir();background=capture(bench,zero,folder,'background');anchors=[capture(bench,patterns/f'{i:03d}.bmp',folder,str(i)) for i in (0,13,27)]
                row=dict(requested=requested,actual_exposure_us=exposure,background=background,anchors=anchors)
                if baseline is None:baseline=row
                else:
                    ratios=[(v['native_mean']-background['native_mean'])/(b['native_mean']-baseline['background']['native_mean'])/(exposure/baseline['actual_exposure_us']) for v,b in zip(anchors,baseline['anchors'])]
                    row['native_background_corrected_exposure_ratios']=ratios
                ladder.append(row);m.save(a.out/'exposure_ladder.json',ladder);m.save(a.out/'progress.json',dict(state='TRAIN_anchor_exposure',latest=row))
                if min(v['p99'] for v in anchors)>=40 and max(v['saturation_fraction'] for v in anchors)<=.001:
                    assert max(abs(v-1) for v in row.get('native_background_corrected_exposure_ratios',[1]))<=.2,'Native camera exposure response not proportional; stop'
                    selected=row;break
            assert selected is not None,'No acceptable bounded same-phase TRAIN exposure'
            validation=a.out/'TRAIN40_validation';validation.mkdir();records=[]
            for i in range(40):
                record=capture(bench,patterns/f'{i:03d}.bmp',validation,f'{i:03d}');record.update(index=i,id=rows[i]['id'],label=rows[i]['label']);records.append(record);m.save(a.out/'progress.json',dict(state='TRAIN40_exposure_validation',samples=i+1,latest=record))
                assert record['p99']>=15 and record['saturation_fraction']<=.15,'Full TRAIN calibration guard failed; preserve, no retry'
            assert m.sha(hw['response_lut'])==hw['response_lut_SHA256'] and m.sha(hw['coordinates'])==hw['coordinates_SHA256'] and m.sha(hw['phase_lut'])==hw['phase_LUT_SHA256']
            accepted={**hw,'exposure_us':selected['actual_exposure_us'],'camera_device_info':{**bench.settings,'Exposure':selected['actual_exposure_us']},'connection_spot_reference_exposure_us':hw['exposure_us'],'same_phase_TRAIN_exposure_parent_SHA256':m.sha(a.hardware),'same_phase_TRAIN_exposure_calibrated':True,'same_phase_TRAIN_phase_SHA256':m.sha(a.out/'phase.bmp')}
        m.save(a.out/'admitted_hardware.json',accepted);m.save(a.out/'report.json',dict(passed=True,state='complete',TRAIN=40,TRAIN_each_class=2,TEST_used=False,exposure_us=selected['actual_exposure_us'],max_saturation=max(v['saturation_fraction'] for v in records),min_p99=min(v['p99'] for v in records),response_lut_unchanged=True,coordinates_unchanged=True,phase_LUT_unchanged=True,network_background_subtraction=False,hardware_SHA256=m.sha(a.out/'admitted_hardware.json'),SDK_released=True,automatic_retry=False))
        # Spatial proof is inherited by exact SHA, not re-fitted under TRAIN.
        m.save(a.out/'spatial_parent_proof.json',dict(parent_hardware=str(a.hardware),SHA256=m.sha(a.hardware),independent_geometry_report_SHA256=m.sha(a.hardware.parent/'independent_geometry/report.json'),passed=True))
    except BaseException as e:
        m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sequence',type=Path,required=True);p.add_argument('--hardware',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a)
