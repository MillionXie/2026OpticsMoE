"""Read-only rich20 TRAIN pilot audit; no SDK, TEST, fitting or repair."""
import argparse,hashlib,io,json,math,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]


def pcc(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else None


def views(a):
    return {**{f'rot{k*90}':np.rot90(a,k) for k in range(4)},**{f'flip_lr_rot{k*90}':np.rot90(np.fliplr(a),k) for k in range(4)}}


def selftest():
    a=np.arange(25).reshape(5,5)
    assert abs(pcc(a,a)-1)<1e-12 and abs(pcc(a,-a)+1)<1e-12
    assert pcc(a,np.ones_like(a)) is None and len(views(a))==8


def audit(a):
    import cv2,torch
    from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,sha
    from lab_caltech20_rich_capture_20261018 import data,metrics
    from lab_caltech10_20261015 import corners,native
    assert not a.out.exists();assert not (HERE/'runs/hardware/.t18_sdk_owner.lock').exists()
    torch.set_num_threads(4);started=time.perf_counter()
    pilot=a.session/'pilot_residual';report=json.loads((pilot/'report.json').read_text())
    assert report['state']=='complete' and report['sdk_released'] and report['physical_ccd']==60
    contract=json.loads((pilot/'contract.json').read_text());release=contract['release']
    assert report['release_sha256']==sha(a.session/'release.json')
    a.assets=a.session/'assets';datasets,_=data(a,True);images,labels,ids=datasets['train']
    assert len(labels)==20 and sorted(labels.tolist())==list(range(20))
    spec=release['models'][0];assert spec['arm']=='residual'
    model,_=load_checkpoint(a.assets/spec['file'],spec['sha256'],2,spec['rho'],alpha_schema=spec['alpha_schema'])
    coordinates=Path('E:/code/guest/20261015.txt');assert sha(coordinates)==release['coordinates_sha256']
    response=HERE/'runs/hardware/caltech10_L3_20261015/response621/response_lut.json'
    assert sha(response)==release['response_lut_sha256'];inverse=json.loads(response.read_text());drive=np.asarray(inverse['drive'],np.uint8)
    warp=cv2.getPerspectiveTransform(np.float32(corners(coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    saved=np.load(pilot/'residual/train/predictions.npz',allow_pickle=False)
    assert np.array_equal(saved['ids'],ids) and np.array_equal(saved['labels'],labels)
    ccd={s:np.load(pilot/f'residual/train/{s}/ccd.npy',mmap_mode='r') for s in ('router','0','1')}
    rows=[];ideal_probs=[];actual_probs=[];max_error=0.
    with torch.inference_mode():
        for begin in range(0,20,4):
            end=begin+4;x=encode_images(images[begin:end]);ideal_probs.append(model(x)[0].numpy().copy())
            def callback(stage,amplitude,phase,sample_ids):
                folder=pilot/f'residual/train/{stage}';phasefile=pilot/'phases'/f'residual_{stage}.bmp'
                encoded=np.asarray(Image.open(phasefile));decoded=np.asarray(Image.fromarray(np.flipud(np.fliplr(255-encoded[92:1108,452:1468]))).resize((478,478),Image.Resampling.NEAREST))
                assert np.array_equal(decoded,phase)
                for index,amp in enumerate(amplitude,begin):
                    path=folder/f'{index:06d}';receipt=json.loads(path.with_suffix('.json').read_text());assert receipt['sample_id']==str(ids[index])
                    canonical=path.with_name(path.name+'_input.png');assert sha(canonical)==receipt['canonical_sha256'] and sha(phasefile)==receipt['phase_sha256']
                    gray=np.asarray(Image.open(canonical));assert np.array_equal(gray,np.rint(amp.clip(0,1)*255).astype(np.uint8))
                    buf=io.BytesIO();Image.fromarray(native(drive[gray],'amplitude')).save(buf,format='BMP');assert hashlib.sha256(buf.getvalue()).hexdigest()==receipt['amplitude_sha256']
                    raw=np.asarray(Image.open(path.with_name(path.name+'_raw.png')));observed=np.asarray(Image.open(path.with_suffix('.png')))
                    reconstructed=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                    assert np.array_equal(reconstructed,observed) and np.array_equal(ccd[stage][index],observed)
                    field=torch.from_numpy(gray.astype(np.float32)/255)*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))
                    ideal=model.propagator(torch.nn.functional.pad(field[None],(20,20,20,20))).abs().square()[0,20:498,20:498].numpy()
                    geometry={name:pcc(ideal,v) for name,v in views(observed).items()};best=max(geometry,key=lambda k:geometry[k] if geometry[k] is not None else -2)
                    unsaturated=observed<250
                    row=dict(stage=stage,index=index,label=int(labels[index]),sample_id=str(ids[index]),frame_id=receipt['frame_id'],saved_inputs_phase_BMP_raw_crop_memmap_exact=True,mean=float(observed.mean()),p01=float(np.percentile(observed,1)),p50=float(np.percentile(observed,50)),p99=float(np.percentile(observed,99)),maximum=int(observed.max()),saturation_fraction=float((observed==255).mean()),identity_PCC=geometry['rot0'],unsaturated_PCC=pcc(ideal[unsaturated],observed[unsaturated]),D4_best_view=best,D4_best_PCC=geometry[best],D4_PCC=geometry,ideal_mean=float(ideal.mean()),ideal_p99=float(np.percentile(ideal,99)))
                    rows.append(row)
                return np.asarray(ccd[stage][begin:end]).copy()
            actual=model(x,callback=callback,ids=ids[begin:end])[0].numpy();actual_probs.append(actual.copy())
            max_error=max(max_error,float(np.max(np.abs(actual-saved['probabilities'][begin:end]))))
            assert max_error<=3e-6
            if time.perf_counter()-started>180:raise TimeoutError('Read-only audit180s limit; no automatic retry')
    assert len(rows)==60
    groups=[]
    for stage in ('router','0','1'):
        selected=[r for r in rows if r['stage']==stage]
        summary={k:dict(min=float(np.min([r[k] for r in selected])),median=float(np.median([r[k] for r in selected])),max=float(np.max([r[k] for r in selected]))) for k in ('p99','maximum','mean','saturation_fraction','identity_PCC','unsaturated_PCC','D4_best_PCC')}
        groups.append(dict(stage=stage,operations=len(selected),summary=summary,best_D4_views=[r['D4_best_view'] for r in selected]))
    result=dict(state='complete',TRAIN_samples=20,TEST_images_read=0,VALID_images_read=0,SDK_opened=False,optimizer_steps=0,source_sha256=sha(__file__),release_sha256=report['release_sha256'],source_checkpoint_sha256=spec['sha256'],same_TRAIN_simulation=metrics(labels,np.concatenate(ideal_probs)),saved_physical_TRAIN_recomputed=metrics(labels,np.concatenate(actual_probs)),saved_probability_max_error=max_error,saved_operations_checked=60,all_saved_inputs_phases_BMP_crop_memmap_exact=True,exposure_us=2000,response_calibration_exposure_us=inverse['exposure_us'],inverse_definition='(measured-dark)/(full-dark); cancels exposure scale ONLY under affine unsaturated response; current clipping is evidence against full-field exposure-invariance',geometry_changes_applied=False,full_capture_released_by_this_audit=False,groups=groups,rows=rows,elapsed_s=time.perf_counter()-started)
    a.out.mkdir(parents=True);(a.out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','same_TRAIN_simulation','saved_physical_TRAIN_recomputed')},ensure_ascii=False),flush=True)
    print(json.dumps(dict(simulated_TRAIN_accuracy=result['same_TRAIN_simulation']['accuracy'],physical_TRAIN_accuracy=result['saved_physical_TRAIN_recomputed']['accuracy'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--session',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    selftest()
    if a.selftest:print('Synthetic correlation and orientation checks passed; no model/data/SDK imported')
    else:
        assert a.session and a.out;audit(a)
