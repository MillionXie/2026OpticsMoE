"""Explicit human-authorized resume after the single white frame, not auto-retry.

Original capture stays read-only. A new logical capture reuses audited prefixes
and acquires only missing frames at the unchanged checkpoint/hardware contract.
Then fit the existing head, capture the other arm, and fit its existing head.
"""
import argparse,io,json,os,shutil,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from PIL import Image
from lab_caltech10_20261015 import Bench,corners,native,phase_native,save,sha
from lab_caltech20_rich_capture_20261018 import data,layer_forward,metrics
from lab_caltech20_rich_model_20261018 import load_checkpoint


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def guard(p99,saturation):return p99>=15 and saturation<=.15


def prefix_length(receipts,n):
    assert [r['index'] for r in receipts]==list(range(len(receipts)))
    assert len(receipts)<=n
    good=0
    for r in receipts:
        if not guard(r['p99'],r['saturation_fraction']):break
        good+=1
    assert good==len(receipts) or good==len(receipts)-1,'Only a terminal failed frame may be excluded'
    return good


def bmp_sha(pixels):
    stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='BMP')
    import hashlib
    return hashlib.sha256(stream.getvalue()).hexdigest()


def selftest():
    valid=lambda i:dict(index=i,p99=90,saturation_fraction=0.)
    assert prefix_length([valid(0),valid(1),dict(index=2,p99=255,saturation_fraction=1.)],5)==2
    assert prefix_length([],5)==0 and prefix_length([valid(0)],1)==1
    try:prefix_length([dict(index=0,p99=255,saturation_fraction=1.),valid(1)],2)
    except AssertionError:pass
    else:raise AssertionError('A nonterminal corrupt frame must not be accepted')
    assert guard(15,.15) and not guard(14,.0) and not guard(255,1.)
    assert bmp_sha(np.zeros((2,2),np.uint8))==bmp_sha(np.zeros((2,2),np.uint8))


def validate(a):
    a.assets=a.session/'assets';a.release=a.session/'release.json'
    release=read(a.release);digest=sha(a.release)
    assert release['capture_released'] and (release['depth'],release['TRAIN'],release['TEST'],release['VALID'])==(2,1000,411,0)
    assert [s['arm'] for s in release['models']]==['residual','control']
    assert release['capture_source_sha256']==sha(HERE/'lab_caltech20_rich_capture_20261018.py')
    assert release['model_source_sha256']==sha(HERE/'lab_caltech20_rich_model_20261018.py')
    for key,path in [('coordinates_sha256',a.coordinates),('machine_config_sha256',a.machine_config),('phase_lut_sha256',a.phase_lut),('response_lut_sha256',a.response_lut)]:assert sha(path)==release[key]
    bridge=read(a.session/'cpu_final_release_bridge/report.json')
    assert bridge['passed'] and bridge['release_sha256']==digest and bridge['source_sha256']==release['capture_source_sha256']
    pilot=read(a.session/'pilot_residual/report.json')
    assert pilot['state']=='complete' and pilot['sdk_released'] and pilot['physical_ccd']==60 and pilot['release_sha256']==digest
    diagnostic=read(a.session/'whiteframe_diagnostic_20261018/report.json')
    assert diagnostic['state']=='complete' and diagnostic['frames']==10 and diagnostic['sdk_released'] and diagnostic['original_files_unchanged']
    assert diagnostic['release_sha256']==digest and diagnostic['stop_reason']=='fixed_sequence_complete'
    assert all(r['crop']['saturation_fraction']<=.15 for r in diagnostic['rows'])
    response=read(a.response_lut);assert response['passed'] and release['radiometric_mode']=='session_field_inverse'
    drive=np.asarray(response['drive'],np.uint8);assert drive.shape==(256,) and drive[0]==0 and drive[-1]==255
    datasets,manifest=data(a,False);assert manifest['sha256']==release['data_sha256']
    return release,digest,datasets,drive


def audit(a,datasets,release,warp):
    """Verify all saved original frames, using only a contiguous valid prefix."""
    import cv2
    old=a.session/'full_residual';failure=read(old/'failure.json')
    assert failure['physical_ccd']==1706 and not (old/'report.json').exists()
    contract=read(old/'contract.json');assert contract['release_sha256']==sha(a.release) and contract['arm']=='residual'
    assert contract['exposure_us']==2000 and contract['gain']=='Gain_X4' and contract['wait_ms']==240
    spec=release['models'][0];protected={};rows=[];prefixes={};excluded=[]
    def protect(p):protected[str(p)]=sha(p)
    for p in old.rglob('*'):
        if p.is_file():protect(p)
    for stage in ('router','0','1'):
        phase=old/'phases'/('residual_'+stage+'.bmp')
        for role,(_,labels,ids) in datasets.items():
            folder=old/'residual'/role/stage
            files=sorted(p for p in folder.glob('*.json') if p.stem.isdigit()) if folder.exists() else []
            receipts=[read(p) for p in files];count=prefix_length(receipts,len(labels));prefixes[stage,role]=count
            ccd=np.load(folder/'ccd.npy',mmap_mode='r') if files else None
            if ccd is not None:assert ccd.dtype==np.uint8 and ccd.shape==(len(labels),478,478)
            for path,r in zip(files,receipts):
                index=r['index'];stem=path.with_suffix('')
                assert r['arm']=='residual' and r['stage']==stage and r['role']==role and r['sample_id']==str(ids[index])
                assert r['checkpoint_sha256']==spec['sha256'] and r['phase_sha256']==sha(phase)
                canonical=stem.with_name(stem.name+'_input.png');assert sha(canonical)==r['canonical_sha256']
                crop=np.asarray(Image.open(stem.with_suffix('.png')));raw=np.asarray(Image.open(stem.with_name(stem.name+'_raw.png')))
                assert raw.shape==(1080,1920) and crop.shape==(478,478) and np.array_equal(crop,ccd[index])
                expected=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                assert np.array_equal(expected,crop)
                assert float(np.percentile(crop,99))==r['p99'] and float((crop==255).mean())==r['saturation_fraction']
                assert float(crop.mean())==r['mean'] and int(crop.max())==r['maximum']
                row=dict(stage=stage,role=role,index=index,sample_id=str(ids[index]),receipt=str(path),receipt_sha256=sha(path),valid=index<count)
                rows.append(row)
                if index>=count:excluded.append(row)
    assert prefixes=={('router','train'):1000,('router','test'):411,('0','train'):294,('0','test'):0,('1','train'):0,('1','test'):0}
    assert len(rows)==1706 and sum(prefixes.values())==1705 and len(excluded)==1
    save(a.out/'resume_audit.json',dict(state='passed',reused_valid_CCD=1705,new_CCD_required=2528,excluded_original_frame=excluded,prefixes=[dict(stage=s,role=r,count=n) for (s,r),n in prefixes.items()],original_files_sha256=protected,frames=rows,original_capture_unchanged=True))
    return prefixes,protected


@torch.inference_mode()
def resume(a):
    import cv2
    assert not a.out.exists();a.out.mkdir(parents=True);started=time.perf_counter();torch.set_num_threads(4)
    release,digest,datasets,drive=validate(a);a.exposure_us=2000
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    prefixes,protected=audit(a,datasets,release,warp)
    if a.mode=='audit':return
    targetroot=a.out/'capture_residual';targetroot.mkdir();(targetroot/'phases').mkdir()
    spec=release['models'][0];model,_=load_checkpoint(a.assets/spec['file'],spec['sha256'],2,spec['rho'],alpha_schema=spec['alpha_schema'])
    save(targetroot/'contract.json',dict(release=release,release_sha256=digest,arm='residual',pilot=False,resumed_from=str(a.session/'full_residual'),original_read_only=True,reused_valid_CCD=1705,new_CCD_required=2528,excluded_original_index=294,exposure_us=2000,gain='Gain_X4',wait_ms=240,checkpoint_sha256=spec['sha256'],fine_tuned=False,source_sha256=sha(__file__),human_authority='2026-10-18 direct request: retain acquired data, continue at fault position, adapt head, then other arm'))
    previous={role:{} for role in datasets};final={role:[] for role in datasets};frames=0;replayed=0
    lock=HERE/'runs/hardware/.t18_sdk_owner.lock';owned=False
    try:
        with lock.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),out=str(targetroot),kind='explicit_human_resume',release_sha256=digest)))
        owned=True
        with Bench(a) as bench:
            for stage in ('router','0','1'):
                phasepath=targetroot/'phases'/('residual_'+stage+'.bmp');phase_pixels=None;displayed=False
                for role,(images,labels,ids) in datasets.items():
                    target=targetroot/'residual'/role/stage;target.mkdir(parents=True);n=len(labels);keep=prefixes[stage,role]
                    ccd=np.lib.format.open_memmap(target/'ccd.npy',mode='w+',dtype=np.uint8,shape=(n,478,478))
                    old=a.session/'full_residual/residual'/role/stage
                    if keep:ccd[:keep]=np.load(old/'ccd.npy',mmap_mode='r')[:keep];ccd.flush()
                    for begin in range(0,n,8):
                        end=min(begin+8,n);cached={k:np.asarray(v[begin:end]).copy() for k,v in previous[role].items()}
                        def acquire(current,amplitudes,phase,sample_ids):
                            nonlocal frames,replayed,phase_pixels,displayed
                            if phase_pixels is None:
                                phase_pixels=phase.copy();Image.fromarray(phase_native(phase)).save(phasepath)
                                original_phase=a.session/'full_residual/phases'/phasepath.name
                                if original_phase.exists():assert sha(original_phase)==sha(phasepath)
                            assert np.array_equal(phase_pixels,phase)
                            for index,amplitude in enumerate(amplitudes,begin):
                                gray=np.rint(amplitude.clip(0,1)*255).astype(np.uint8)
                                assert np.max(np.abs(gray/255-amplitude))<=1e-7
                                if index<keep:
                                    receipt=read(old/f'{index:06d}.json')
                                    assert np.array_equal(gray,np.asarray(Image.open(old/f'{index:06d}_input.png'))),'Cached predecessor encoding changed'
                                    assert bmp_sha(native(drive[gray],'amplitude'))==receipt['amplitude_sha256']
                                    assert receipt['phase_sha256']==sha(phasepath);replayed+=1
                                    continue
                                if not displayed:bench.stage(phasepath);displayed=True
                                prefix=target/f'{index:06d}';canonical=prefix.with_name(prefix.name+'_input.png');Image.fromarray(gray).save(canonical)
                                bmp=prefix.with_name(prefix.name+'_transient.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(bmp)
                                settings={node:bench.hw.camera.get(node) for node in ('ExposureTime','Gain','TestPattern','AcquisitionFrameRate','SyncMode','PixelFormat')}
                                assert abs(float(settings['ExposureTime'])-2000)<1 and settings['Gain']=='Gain_X4' and settings['TestPattern']=='Normal'
                                raw,meta=bench.capture(bmp);assert raw.shape==(1080,1920)
                                crop=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                                receipt=dict(arm='residual',role=role,stage=stage,index=index,sample_id=str(ids[index]),frame_id=meta['frame_id'],captured_at_UTC=datetime.now(timezone.utc).isoformat(),mean=float(crop.mean()),p99=float(np.percentile(crop,99)),maximum=int(crop.max()),saturation_fraction=float((crop==255).mean()),capture_total_ms=meta['capture_total_ms'],amplitude_sha256=sha(bmp),canonical_sha256=sha(canonical),phase_sha256=sha(phasepath),checkpoint_sha256=spec['sha256'],camera_current_readback=settings,frame_metadata=meta)
                                Image.fromarray(raw).save(prefix.with_name(prefix.name+'_raw.png'));Image.fromarray(crop).save(prefix.with_suffix('.png'));save(prefix.with_suffix('.json'),receipt)
                                ccd[index]=crop;ccd.flush();frames+=1;bmp.unlink()
                                save(targetroot/'progress.json',dict(state='capturing_explicit_resume',arm='residual',stage=stage,role=role,completed=index+1,total=n,new_physical_ccd=frames,reused_valid_ccd=1705,logical_completed_ccd=1705+frames,elapsed_s=time.perf_counter()-started,latest=receipt))
                                if not guard(receipt['p99'],receipt['saturation_fraction']):raise RuntimeError('Optical signal guard failed during explicit resume; retain evidence, no automatic retry')
                            return np.asarray(ccd[begin:end]).copy()
                        result=layer_forward(model,images[begin:end],ids[begin:end],stage,cached,acquire)
                        if result is not None:final[role].append(result)
                    previous[role][stage]=ccd
                save(targetroot/'residual'/f'{stage}_complete.json',dict(stage=stage,new_physical_ccd=frames,replayed_verified_ccd=replayed,elapsed_s=time.perf_counter()-started))
        assert frames==2528 and replayed==1705
        assert all(sha(path)==digest0 for path,digest0 in protected.items()),'Original failed run was modified'
        results={}
        for role,(_,labels,ids) in datasets.items():
            probabilities=np.concatenate([v['probabilities'] for v in final[role]]);features=np.concatenate([v['features'] for v in final[role]])
            np.savez_compressed(targetroot/'residual'/role/'predictions.npz',ids=ids,labels=labels,probabilities=probabilities,features=features)
            results[role]=dict(metrics(labels,probabilities),samples=len(labels),physical=True,fine_tuned=False,checkpoint_sha256=spec['sha256'])
            save(targetroot/'residual'/role/'metrics.json',results[role])
        save(targetroot/'report.json',dict(state='complete',arm='residual',pilot=False,physical_ccd=4233,new_physical_ccd=frames,reused_valid_ccd=replayed,sdk_released=True,release_sha256=digest,elapsed_s=time.perf_counter()-started,original_capture_files_SHA256_unchanged=True,VALID_captured=0,results=results))
        print(json.dumps(dict(event='residual_capture_complete',results=results,new_CCD=frames,reused_CCD=replayed)),flush=True)
    except BaseException as error:
        save(targetroot/'failure.json',dict(error=repr(error),new_physical_ccd=frames,reused_valid_ccd=1705,automatic_retry=False,traceback=traceback.format_exc()));raise
    finally:
        if owned:lock.unlink()


def followup(a):
    steps=[];started=time.perf_counter();release=a.session/'release.json';release_sha=sha(release)
    source_names=(Path(__file__).name,'lab_caltech20_rich_model_20261018.py','lab_caltech20_rich_capture_20261018.py','lab_caltech20_rich_readout_20261018.py')
    sources={n:sha(HERE/n) for n in source_names}
    def run(step,args):
        assert not (HERE/'runs/hardware/.t18_sdk_owner.lock').exists()
        assert sources=={n:sha(HERE/n) for n in source_names} and sha(release)==release_sha
        save(a.out/'progress.json',dict(state='followup',step=step,elapsed_s=time.perf_counter()-started))
        executable=Path(sys.executable).with_name('python.exe') if os.name=='nt' else Path(sys.executable)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (a.out/(step+'.log')).open('x',encoding='utf-8') as log:
            result=subprocess.run([str(executable),*map(str,args)],cwd=HERE.parents[2],env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(step,result.returncode)
        steps.append(dict(step=step,returncode=result.returncode,elapsed_s=time.perf_counter()-started));save(a.out/'steps.json',steps)
    def fit(arm,capture):
        run('fit_'+arm,[HERE/'lab_caltech20_rich_readout_20261018.py','--arm',arm,'--capture',capture,'--assets',a.session/'assets','--release',release,'--out',a.out/('readout_'+arm+'_120')])
        report=read(a.out/('readout_'+arm+'_120/report.json'));assert report['state']=='complete'
        print(json.dumps(dict(event='adapted_result',arm=arm,baseline_TEST=report['baseline_TEST_development']['accuracy'],best_TEST_development=report['results']['best']['TEST_development']['accuracy'],best_epoch=report['results']['best']['epoch'])),flush=True)
    fit('residual',a.out/'capture_residual')
    common=['--arm','control','--assets',a.session/'assets','--release',release,'--release-sha',release_sha,'--bridge',a.session/'cpu_final_release_bridge','--previous-arm',a.out/'capture_residual']
    for name in ('coordinates','machine_config','phase_lut','response_lut','phase_sdk','amplitude_sdk'):common+=['--'+name.replace('_','-'),getattr(a,name)]
    for mode in ('pilot','full'):
        args=[HERE/'lab_caltech20_rich_capture_20261018.py','--mode',mode,'--out',a.out/(mode+'_control'),*common]
        if mode=='full':args+=['--pilot',a.out/'pilot_control']
        run('capture_'+mode+'_control',args)
    fit('control',a.out/'full_control')
    save(a.out/'report.json',dict(state='complete',steps=steps,source_files_SHA256=sources,release_sha256=release_sha,sdk_released=True,original_failed_capture_preserved=True,reused_CCD=1705,new_residual_CCD=2528,new_control_CCD=4233,control_pilot_CCD=60,TEST_is_development_not_independent=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--mode',choices=('audit','chain'))
    for name in ('session','out','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.selftest:selftest();print('Resume prefix, guard and BMP byte identity selftest passed; SDK not opened')
    else:
        assert a.mode in ('audit','chain')
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:
            resume(a)
            if a.mode=='chain':followup(a)
        except BaseException as error:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),automatic_retry=False,traceback=traceback.format_exc()))
            raise
