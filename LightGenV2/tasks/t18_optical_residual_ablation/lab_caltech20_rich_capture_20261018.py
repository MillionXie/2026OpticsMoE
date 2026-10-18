"""Explicit-release rich20 L2 capture: one arm, one whole layer, TRAIN then TEST.

Past layers replay only immutable CCD arrays through the *same* learned fusion
graph. Only the current layer opens physical acquisition. No optimizer, new head,
VAL acquisition, electronic fitting or automatic failure/restart path is present.
"""
import argparse,hashlib,json,math,os,shutil,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from PIL import Image
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,sha
from lab_caltech10_20261015 import Bench,corners,native,phase_native,save


def metrics(labels,probabilities):
    assert probabilities.shape==(len(labels),20) and np.isfinite(probabilities).all()
    predictions=probabilities.argmax(1);cm=np.zeros((20,20),np.int64)
    np.add.at(cm,(labels,predictions),1);support=cm.sum(1)
    recalls=np.divide(cm.diagonal(),support,out=np.zeros(20),where=support>0)
    return dict(accuracy=float((labels==predictions).mean()),macro_recall=float(recalls[support>0].mean()),support=support.tolist(),confusion_matrix=cm.tolist())


class StageComplete(Exception):pass


@torch.inference_mode()
def layer_forward(model,images,ids,current,previous,acquire):
    """One exact graph prefix; no physical call for cached/future layers."""
    stages=['router']+[str(k) for k in range(model.depth)]
    assert current in stages and set(previous)==set(stages[:stages.index(current)])
    calls=[]
    def callback(stage,amplitude,phase,sample_ids):
        calls.append(stage)
        if stage in previous:return previous[stage]
        assert stage==current,'Never propagate into an uncaptured future layer'
        result=acquire(stage,amplitude,phase,sample_ids)
        if current!=stages[-1]:raise StageComplete()
        return result
    try:probability,_=model(encode_images(images),callback=callback,ids=ids)
    except StageComplete:
        assert current!=stages[-1] and calls==stages[:stages.index(current)+1]
        return None
    assert current==stages[-1] and calls==stages
    return dict(probabilities=probability.numpy().copy(),features=model.readout['features'].numpy().copy())


def pack(a):
    """Selected source TRAIN and all TEST only; IDs fixed before model evaluation."""
    assert not a.out.exists();assert sha(a.cache)==a.cache_sha and sha(a.selection)==a.selection_sha
    selection=json.loads(a.selection.read_text());assert selection['cache_sha256']==a.cache_sha
    indices=np.asarray(selection['train_indices']);assert len(indices)==1000
    with np.load(a.cache,allow_pickle=False) as z:
        assert z['train_ids'][indices].tolist()==selection['train_ids']
        assert np.bincount(z['train_labels'][indices],minlength=20).tolist()==[50]*20
        assert z['test_ids'].tolist()==selection['test_ids'] and len(z['test_ids'])==411
        data={}
        for role,index in (('train',indices),('test',np.arange(411))):
            for name in ('images','labels','ids'):data[role+'_'+name]=z[role+'_'+name][index].copy()
    assert not set(data['train_ids'].tolist())&set(data['test_ids'].tolist())
    a.out.mkdir(parents=True)
    path=a.out/'rich20_train1000_test411.npz';np.savez_compressed(path,**data)
    result=dict(state='prepared_not_capture_release',file=path.name,sha256=sha(path),source_cache_sha256=a.cache_sha,selection_sha256=a.selection_sha,TRAIN=1000,TEST=411,VALID_exported=0,TRAIN_each_class=50,optimizer_steps=0,SDK_opened=False)
    save(a.out/'data_manifest.json',result);print(json.dumps(result),flush=True)


def data(a,pilot):
    manifest=json.loads((a.assets/'data_manifest.json').read_text())
    assert manifest['TRAIN']==1000 and manifest['TEST']==411 and manifest['VALID_exported']==0
    path=a.assets/manifest['file'];assert sha(path)==manifest['sha256']
    result={}
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={r+'_'+n for r in ('train','test') for n in ('images','labels','ids')}
        for role in (('train',) if pilot else ('train','test')):
            x,y,ids=[z[role+'_'+n].copy() for n in ('images','labels','ids')]
            if pilot:
                ix=np.array([np.flatnonzero(y==k)[0] for k in range(20)]);x,y,ids=x[ix],y[ix],ids[ix]
            assert x.shape==(len(y),150,150,3)
            result[role]=(torch.from_numpy(x.transpose(0,3,1,2).copy()),y,ids)
    return result,manifest


@torch.inference_mode()
def offline(a):
    """Prove layer-major numerical replay against whole-graph simulation, TRAIN only."""
    assert not a.out.exists();torch.set_num_threads(4)
    metric_proof=metrics(np.arange(20),np.eye(20));assert metric_proof['accuracy']==1. and len(metric_proof['confusion_matrix'])==20
    datasets,manifest=data(a,True);images,_,ids=datasets['train']
    release=json.loads(a.release.read_text());rows=[]
    assert [s['arm'] for s in release['models']]==['residual','control']
    for spec in release['models']:
        model,_=load_checkpoint(a.assets/spec['file'],spec['sha256'],2,spec['rho'],alpha_schema=spec['alpha_schema'])
        baseline=model(encode_images(images))[0].numpy().copy()
        previous={};calls=[]
        for stage in ('router','0','1'):
            def simulated(current,amplitude,phase,sample_ids):
                calls.append(current)
                field=torch.from_numpy(amplitude.copy())*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
                intensity=model.propagator(torch.nn.functional.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy()
                previous[current]=intensity
                return intensity
            # Supply only predecessors; current data is added by the callback.
            actual=layer_forward(model,images,ids,stage,dict(previous),simulated)
        assert calls==['router','0','1']
        error=float(np.abs(actual['probabilities']-baseline).max());assert error<=3e-6
        rows.append(dict(arm=spec['arm'],probability_max_error=error,physical_frames=0,TRAIN_samples=20,stages=calls))
    a.out.mkdir(parents=True);save(a.out/'report.json',dict(passed=True,scope='offline_layer_major_preparation_not_SDK_release',proof=rows,SDK_opened=False,TEST_images_read=0,VALID_images_read=0,source_sha256=sha(__file__),release_sha256=sha(a.release),data_sha256=manifest['sha256']))
    print(json.dumps(rows),flush=True)


@torch.inference_mode()
def capture(a):
    import cv2
    release=json.loads(a.release.read_text());assert release['capture_released'] is True
    assert release['release_authority'] and release['depth']==2
    assert release['TRAIN']==1000 and release['TEST']==411 and release['VALID']==0
    assert [s['arm'] for s in release['models']]==['residual','control']
    assert [s['rho'] for s in release['models']]==[.3,0.]
    assert release['capture_source_sha256']==sha(__file__)
    assert release['model_source_sha256']==sha(Path(__file__).with_name('lab_caltech20_rich_model_20261018.py'))
    assert sha(a.release)==a.release_sha
    offline_proof=json.loads((a.bridge/'report.json').read_text())
    assert offline_proof['passed'] and offline_proof['release_sha256']==a.release_sha and offline_proof['source_sha256']==sha(__file__)
    assert release['coordinates_sha256']==sha(a.coordinates) and release['phase_lut_sha256']==sha(a.phase_lut)
    assert release['machine_config_sha256']==sha(a.machine_config)
    assert release['radiometric_mode'] in ('identity','session_field_inverse')
    if release['radiometric_mode']=='session_field_inverse':
        assert sha(a.response_lut)==release['response_lut_sha256']
        response=json.loads(a.response_lut.read_text());assert response['passed'] and response['coordinates_sha256']==sha(a.coordinates)
        drive=np.asarray(response['drive'],np.uint8);assert drive.shape==(256,) and drive[0]==0 and drive[-1]==255
    else:drive=np.arange(256,dtype=np.uint8)
    if a.mode=='full':
        pilot=json.loads((a.pilot/'report.json').read_text())
        assert pilot['state']=='complete' and pilot['sdk_released'] and pilot['pilot'] and pilot['release_sha256']==a.release_sha
        assert pilot['physical_ccd']==120,'Both 20-TRAIN arm pilots must pass first'
    assert not a.out.exists();assert shutil.disk_usage(a.out.parent).free>=20*(1<<30),'Need20GiB free before opening devices'
    a.out.mkdir(parents=True);torch.set_num_threads(4)
    datasets,manifest=data(a,a.mode=='pilot');assert manifest['sha256']==release['data_sha256']
    a.exposure_us=2000;start=time.perf_counter();frames=0;phase_root=a.out/'phases';phase_root.mkdir()
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    save(a.out/'contract.json',dict(release=release,release_sha256=a.release_sha,pilot=a.mode=='pilot',order='one arm; router TRAIN then TEST, expert TRAIN then TEST, global TRAIN then TEST; then next arm',exposure_us=2000,gain='Gain_X4',wait_ms=240,native_active=1016,phase_transform='hv_inverse',camera_orientation='flip_v',response_mode=release['radiometric_mode'],fine_tuned=False))
    lock=Path(__file__).parent/'runs/hardware/.t18_sdk_owner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as stream:stream.write(json.dumps(dict(pid=os.getpid(),out=str(a.out),release_sha256=a.release_sha)))
    try:
        with Bench(a) as bench:
            for spec in release['models']:
                arm=spec['arm'];model,_=load_checkpoint(a.assets/spec['file'],spec['sha256'],2,spec['rho'],alpha_schema=spec['alpha_schema']);armroot=a.out/arm;armroot.mkdir()
                previous={role:{} for role in datasets};final={role:[] for role in datasets}
                for stage in ('router','0','1'):
                    phasepath=phase_root/f'{arm}_{stage}.bmp';phase_pixels=None
                    for role,(images,labels,ids) in datasets.items():
                        target=armroot/role/stage;target.mkdir(parents=True);n=len(labels)
                        ccd=np.lib.format.open_memmap(target/'ccd.npy',mode='w+',dtype=np.uint8,shape=(n,478,478))
                        for begin in range(0,n,8):
                            end=min(begin+8,n);cached={k:np.asarray(v[begin:end]).copy() for k,v in previous[role].items()}
                            def acquire(current,amplitudes,phase,sample_ids):
                                nonlocal frames,phase_pixels
                                if phase_pixels is None:
                                    phase_pixels=phase.copy();Image.fromarray(phase_native(phase)).save(phasepath);bench.stage(phasepath)
                                assert np.array_equal(phase_pixels,phase)
                                for index,amplitude in enumerate(amplitudes,begin):
                                    gray=np.rint(amplitude.clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amplitude))<=1e-7
                                    prefix=target/f'{index:06d}';canonical=prefix.with_name(prefix.name+'_input.png');Image.fromarray(gray).save(canonical)
                                    bmp=prefix.with_name(prefix.name+'_transient.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(bmp)
                                    raw,meta=bench.capture(bmp);assert raw.shape==(1080,1920)
                                    crop=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                                    receipt=dict(arm=arm,role=role,stage=stage,index=index,sample_id=str(ids[index]),frame_id=meta['frame_id'],captured_at_UTC=datetime.now(timezone.utc).isoformat(),mean=float(crop.mean()),p99=float(np.percentile(crop,99)),maximum=int(crop.max()),saturation_fraction=float((crop==255).mean()),capture_total_ms=meta['capture_total_ms'],amplitude_sha256=sha(bmp),canonical_sha256=sha(canonical),phase_sha256=sha(phasepath),checkpoint_sha256=spec['sha256'])
                                    Image.fromarray(raw).save(prefix.with_name(prefix.name+'_raw.png'));Image.fromarray(crop).save(prefix.with_suffix('.png'));save(prefix.with_suffix('.json'),receipt)
                                    ccd[index]=crop;ccd.flush();frames+=1;bmp.unlink()
                                    save(a.out/'progress.json',dict(state='capturing',arm=arm,stage=stage,role=role,completed=index+1,total=n,physical_ccd=frames,elapsed_s=time.perf_counter()-start,latest=receipt))
                                    if receipt['p99']<15 or receipt['saturation_fraction']>.15:raise RuntimeError('Optical signal guard failed; no automatic retry; retain evidence')
                                return np.asarray(ccd[begin:end]).copy()
                            result=layer_forward(model,images[begin:end],ids[begin:end],stage,cached,acquire)
                            if result is not None:final[role].append(result)
                        previous[role][stage]=ccd
                    save(armroot/f'{stage}_complete.json',dict(stage=stage,roles=list(datasets),physical_ccd=frames,elapsed_s=time.perf_counter()-start))
                results={}
                for role,(_,labels,ids) in datasets.items():
                    probabilities=np.concatenate([v['probabilities'] for v in final[role]]);features=np.concatenate([v['features'] for v in final[role]])
                    np.savez_compressed(armroot/role/'predictions.npz',ids=ids,labels=labels,probabilities=probabilities,features=features)
                    results[role]=dict(metrics(labels,probabilities),samples=len(labels),physical=True,fine_tuned=False,checkpoint_sha256=spec['sha256']);save(armroot/role/'metrics.json',results[role])
                save(armroot/'report.json',dict(state='complete',results=results,release_sha256=a.release_sha,physical_ccd=frames,elapsed_s=time.perf_counter()-start,SDK_still_owned=True))
                print(json.dumps(dict(arm=arm,results=results)),flush=True)
        save(a.out/'report.json',dict(state='complete',pilot=a.mode=='pilot',physical_ccd=frames,elapsed_s=time.perf_counter()-start,sdk_released=True,release_sha256=a.release_sha,VALID_captured=0))
    except BaseException as error:
        save(a.out/'failure.json',dict(error=repr(error),physical_ccd=frames,automatic_retry=False,traceback=traceback.format_exc()));raise
    finally:lock.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('pack','offline','pilot','full'),required=True)
    for name in ('out','cache','selection','assets','release','bridge','pilot','coordinates','machine-config','phase-lut','phase-sdk','amplitude-sdk','response-lut'):p.add_argument('--'+name,type=Path,required=name=='out')
    for name in ('cache-sha','selection-sha','release-sha'):p.add_argument('--'+name)
    a=p.parse_args()
    if sys.stdout is None:
        sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    pack(a) if a.mode=='pack' else offline(a) if a.mode=='offline' else capture(a)
