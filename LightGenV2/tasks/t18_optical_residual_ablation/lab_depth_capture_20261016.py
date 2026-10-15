"""Accepted L2/L4 replay. Same group/layer TRAIN then TEST; frozen L3 utilities."""
import argparse,inspect,json,math,os,subprocess,sys,textwrap,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import lab_caltech10_20261015 as l


class DepthMoE(l.LabMoE):
    def __init__(self,depth,rho):
        l.CorrectedMoE.__init__(self,depth,rho)
        self.detectors=self.detectors[list(l.INDICES)].clone()


def load_model(assets,manifest,depth,arm):
    spec=manifest['models'][str(depth)][arm];path=assets/spec['file']
    assert l.sha(path)==spec['sha256']
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['epoch']==spec['epoch'] and ck['config']['original_indices']==list(l.INDICES) and ck['config']['input_tile']==100
    model=DepthMoE(depth,.3 if arm=='residual' else 0.).eval().requires_grad_(False)
    model.load_state_dict(ck['model'],strict=True)
    return model


@torch.inference_mode()
def bridge(a,manifest):
    out=a.out/'cpu_bridge';assert not out.exists();out.mkdir(parents=True)
    torch.set_num_threads(4);x,_,ids=l.dataset(a.assets,'train',10);inp=l.encode(x,None,100);rows=[]
    original=textwrap.dedent(inspect.getsource(l.CorrectedMoE.forward));assert original.count('+20e-12')==1
    ns={'torch':torch};exec(compile(original.replace('+20e-12','+10e-12'),'accepted_ten_window_forward','exec'),ns)
    for depth in (2,4):
        for arm in ('residual','control'):
            model=load_model(a.assets,manifest,depth,arm);expected=model(inp)[0]
            original_probability=ns['forward'](model,inp)[0];error=float((expected-original_probability).abs().max());assert error<=1e-6
            stages=[]
            def callback(stage,amp,phase,ids):
                qa=np.rint(amp.clip(0,1)*255).astype(np.uint8)
                display=l.native(qa[0],'amplitude');back=np.asarray(Image.fromarray(display[32:1048,452:1468]).resize((478,478),Image.Resampling.NEAREST))
                assert np.array_equal(back,qa[0])
                ph=l.phase_native(phase);back=np.flipud(np.fliplr(255-ph[92:1108,452:1468]))
                back=np.asarray(Image.fromarray(back).resize((478,478),Image.Resampling.NEAREST));assert np.array_equal(back,phase)
                field=torch.from_numpy(qa.astype(np.float32)/255)*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
                stages.append(stage)
                return model.propagator(torch.nn.functional.pad(field,(20,20,20,20))).abs().square()[:,20:498,20:498].numpy()
            actual=model(inp,callback,ids)[0];native_error=float((actual-expected).abs().max())
            assert native_error<=1e-6 and stages==['router']+[str(k) for k in range(depth)]
            rows.append(dict(depth=depth,arm=arm,original_forward_error=error,native_BMP_probability_error=native_error,checkpoint_sha256=manifest['models'][str(depth)][arm]['sha256'],samples=10,stages=stages))
    l.save(out/'report.json',dict(passed=True,physical_frames=0,source_sha256=l.sha(__file__),frozen_L3_helper_sha256=l.sha(l.__file__),proof=rows))


@torch.inference_mode()
def capture_pair(a,manifest,depth,pilot=False):
    import cv2
    out=a.out/(f'L{depth}_pilot' if pilot else f'L{depth}');assert not out.exists();out.mkdir()
    roles=('train',) if pilot else ('train','test');limit=10 if pilot else 0
    data={role:l.dataset(a.assets,role,limit) for role in roles};sizes={role:len(data[role][1]) for role in roles}
    response=json.loads(a.response_lut.read_text());assert response['passed'] and response['coordinates_sha256']==l.sha(a.coordinates)
    drive=np.asarray(response['drive'],np.uint8);assert drive.shape==(256,) and drive[0]==0 and drive[-1]==255
    pts=np.float32(l.corners(a.coordinates));warp=cv2.getPerspectiveTransform(pts,np.float32([[0,0],[477,0],[477,477],[0,477]]))
    checkpoints={arm:manifest['models'][str(depth)][arm]['sha256'] for arm in ('residual','control')}
    counts={role:0 for role in roles};started=time.perf_counter();phase_folder=out/'phases';phase_folder.mkdir()
    for role in roles:
        folder=out/role;folder.mkdir()
        l.save(folder/'contract.json',dict(main_depth=depth,main_stages=['expert' if k%2==0 else 'global' for k in range(depth)],role=role,samples=sizes[role],pilot=pilot,checkpoint_sha256=checkpoints,checkpoint_files={arm:manifest['models'][str(depth)][arm]['file'] for arm in checkpoints},coordinates_sha256=l.sha(a.coordinates),corners_TL_TR_BR_BL=pts.tolist(),exposure_us=2000,gain='Gain_X4',wait_ms=240,phase_transform='hv_inverse',camera_orientation='flip_v',response_lut_sha256=l.sha(a.response_lut),data_sha256=l.sha(a.assets/'caltech10.npz'),source_sha256=l.sha(__file__),frozen_helper_sha256=l.sha(l.__file__),fine_tuned=False,routing='continuous all4',native_active=1016,acquisition_order='same arm/stage TRAIN immediately followed by TEST'))
    lock=Path(l.__file__).parent/'runs/hardware/.t18_sdk_owner.lock'
    with lock.open('x') as stream:stream.write(json.dumps({'pid':os.getpid(),'out':str(out)}))
    a.out_for_queue=a.out;a.out=out
    try:
        with l.Bench(a) as bench:
            for arm in ('residual','control'):
                model=load_model(a.assets,manifest,depth,arm);previous={};routing={}
                for role in roles:(out/role/arm).mkdir()
                for stage in ['router']+[str(k) for k in range(depth)]:
                    phase_path=phase_folder/f'{arm}_{stage}.bmp';phase_set=False
                    for role in roles:
                        x,y,ids=data[role];n=len(y);target=out/role/arm/stage;target.mkdir()
                        ccd=np.lib.format.open_memmap(target/'ccd.npy',mode='w+',dtype=np.uint8,shape=(n,478,478));weights=[]
                        for start in range(0,n,8):
                            stop=min(start+8,n)
                            amp=l.encode(x[start:stop],None,100) if stage in ('router','0') else model.nonlinear(torch.from_numpy(np.asarray(previous[role][start:stop]).copy()).float(),model.stages[int(stage)])
                            route=None if stage=='router' else torch.from_numpy(routing[role][start:stop]).float()
                            amplitudes,phase=l.encode_stage(model,amp,stage,route)
                            if not phase_set:
                                Image.fromarray(l.phase_native(phase)).save(phase_path);bench.stage(phase_path);phase_set=True
                            assert np.array_equal(np.asarray(Image.open(phase_path)),l.phase_native(phase))
                            if start==0:Image.fromarray(l.phase_native(phase)).save(target/'phase.bmp')
                            for j,gray in enumerate(amplitudes,start):
                                sid=f'{j:06d}';canonical=target/(sid+'_input.png');Image.fromarray(gray).save(canonical)
                                bmp=target/(sid+'_transient.bmp');Image.fromarray(l.native(drive[gray],'amplitude')).save(bmp)
                                raw,meta=bench.capture(bmp);assert raw.shape==(1080,1920)
                                crop=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                                row=dict(depth=depth,arm=arm,role=role,stage=stage,index=j,sample_id=str(ids[j]),frame_id=meta['frame_id'],mean=float(crop.mean()),p99=float(np.percentile(crop,99)),maximum=int(crop.max()),saturation_fraction=float((crop==255).mean()),capture_total_ms=meta['capture_total_ms'],amplitude_sha256=l.sha(bmp),canonical_sha256=l.sha(canonical),phase_sha256=l.sha(target/'phase.bmp'))
                                Image.fromarray(raw).save(target/(sid+'_raw.png'));Image.fromarray(crop).save(target/(sid+'.png'));l.save(target/(sid+'.json'),row)
                                ccd[j]=crop;ccd.flush();bmp.unlink();counts[role]+=1
                                l.save(out/'progress.json',dict(state='capturing',depth=depth,pilot=pilot,arm=arm,stage=stage,role=role,completed=j+1,total=n,physical_ccd=sum(counts.values()),target_ccd=sum(sizes.values())*2*(depth+1),elapsed_s=time.perf_counter()-started))
                                if row['p99']<15 or row['saturation_fraction']>.15:raise RuntimeError('Optical signal guard failed; retain all data; no automatic retry')
                            if stage=='router':weights.append(l.routing_from_ccd(model,torch.from_numpy(np.asarray(ccd[start:stop]).copy()).float()).numpy())
                        if stage=='router':routing[role]=np.concatenate(weights);np.save(out/role/arm/'routing.npy',routing[role])
                        previous[role]=ccd
                for role in roles:
                    _,labels,ids=data[role];n=len(labels)
                    energy=np.concatenate([np.einsum('bhw,khw->bk',np.asarray(previous[role][k:k+8]).astype(np.float32),model.detectors.numpy()) for k in range(0,n,8)])
                    probability=(energy+1e-12)/(energy.sum(1,keepdims=True)+10e-12)
                    np.savez_compressed(out/role/arm/'predictions.npz',ids=ids,labels=labels,probabilities=probability)
                    l.save(out/role/arm/'metrics.json',dict(l.metrics(labels,probability),physical=True,fine_tuned=False,checkpoint_sha256=checkpoints[arm]))
        for role in roles:l.save(out/role/'report.json',dict(state='complete',main_depth=depth,samples_per_arm=sizes[role],physical_ccd=counts[role],sdk_released=True,results={arm:json.loads((out/role/arm/'metrics.json').read_text()) for arm in checkpoints}))
        l.save(out/'report.json',dict(state='complete',main_depth=depth,pilot=pilot,physical_ccd=sum(counts.values()),sdk_released=True,elapsed_s=time.perf_counter()-started))
    except BaseException as error:l.save(out/'failure.json',dict(error=repr(error),physical_ccd=sum(counts.values()),automatic_retry=False));raise
    finally:a.out=a.out_for_queue;lock.unlink()


def main(a):
    if a.mode=='bridge':
        assert not a.out.exists();a.out.mkdir(parents=True)
        bridge(a,json.loads((a.assets/'depth_manifest_20261016.json').read_text()));return
    assert json.loads((a.bridge/'cpu_bridge/report.json').read_text())['passed']
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4)
    l.save(a.out/'claim.json',dict(pid=os.getpid(),source_sha256=l.sha(__file__),frozen_L3_helper_sha256=l.sha(l.__file__),assets_manifest_sha256=l.sha(a.assets/'depth_manifest_20261016.json'),depth_order=[2,4],capture_only=True,lab_receiver_only=True))
    deadline=time.monotonic()+8*3600
    while not (a.wait_for/'report.json').exists():
        if (a.wait_for/'failure.json').exists():raise RuntimeError('Existing L3 TEST failed; no automatic new SDK owner')
        if time.monotonic()>deadline:raise TimeoutError('L3 TEST wait exceeded8h')
        time.sleep(15)
    report=json.loads((a.wait_for/'report.json').read_text());assert report['state']=='complete' and report['sdk_released'] and report['physical_ccd']==14664
    lock=Path(l.__file__).parent/'runs/hardware/.t18_sdk_owner.lock';deadline=time.monotonic()+30
    while lock.exists():
        if time.monotonic()>deadline:raise RuntimeError('SDK not released after L3 TEST')
        time.sleep(.5)
    manifest=json.loads((a.assets/'depth_manifest_20261016.json').read_text())
    for depth in (2,4):
        capture_pair(a,manifest,depth,pilot=True)
        capture_pair(a,manifest,depth,pilot=False)
    l.save(a.out/'complete.json',dict(state='complete',physical_ccd_full=41504,pilot_ccd=160,sdk_released=True,electronic_finetuning=False,waiting_for_same_simulation_heads=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['bridge','queue'],required=True)
    for name in ('out','assets','bridge','wait-for','coordinates','response-lut','machine-config','amplitude-sdk','phase-sdk','phase-lut'):parser.add_argument('--'+name,type=Path,required=name in ('out','assets'))
    a=parser.parse_args();a.exposure_us=2000
    if sys.stdout is None:
        sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    try:main(a)
    except BaseException as error:
        if a.out.exists():l.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
