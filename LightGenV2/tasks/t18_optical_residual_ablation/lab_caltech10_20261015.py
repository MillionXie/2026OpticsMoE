"""Explicit, layer-major Caltech10 L3 bench replay; no training or device-on import."""
import argparse,ast,hashlib,inspect,json,math,os,re,sys,textwrap,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import numpy as np
import torch
from PIL import Image
from caltech_repair234_model import CorrectedMoE
from caltech_deploy234_model import rms_encode
INDICES=(0,1,2,3,4,7,14,15,16,19)
EXPECTED={'residual':'b3d89f3ab2b177bd85997453cb0d0f8289d25b600243ca59c824ac529f59172d',
          'control':'62154e9067aa75494189955421a0d32e74cbe8ff14b4f3caf929262ed11445cc'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def save(p,v):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2),encoding='utf-8')
    for attempt in range(8):
        try:q.replace(p);return
        except PermissionError:
            if attempt<7:time.sleep(.1*(attempt+1));continue
            if p.name=='progress.json':
                # Windows readers may deny replacement. Immutable per-frame
                # receipts remain authoritative; never abort valid acquisition
                # just because the convenience progress pointer is busy.
                q.replace(p.with_name('progress_'+str(time.time_ns())+'.json'));return
            raise

class LabMoE(CorrectedMoE):
    def __init__(self,rho):
        super().__init__(3,rho);self.detectors=self.detectors[list(INDICES)].clone()
    def forward(self,amplitude,callback=None,ids=None):
        # Exact frozen ten-window body, avoiding training-only sklearn imports.
        self.last={};intensity=self.capture(amplitude,'router',callback=callback,ids=ids)
        weights=routing_from_ccd(self,intensity);self.routing_weights=weights.detach()
        for k in range(self.depth):
            intensity=self.capture(amplitude,str(k),weights,callback,ids)
            if k<self.depth-1:amplitude=self.nonlinear(intensity,self.stages[k+1])
        energy=torch.einsum('bhw,khw->bk',intensity,self.detectors)
        return (energy+1e-12)/(energy.sum(1,keepdim=True)+10e-12),energy.sum(1)/intensity.sum((-2,-1)).clamp_min(1e-12)

def encode(images,theta=None,tile=100):
    assert theta is None and tile==100,'Hardware uses the frozen evaluation encoder only'
    f=torch.nn.functional;x=images.float()/255
    x=f.interpolate(x,(100,100),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    mosaic=torch.cat([torch.cat([x[:,0:1],x[:,1:2]],3),torch.cat([x[:,2:3],x.mean(1,keepdim=True)],3)],2)
    return rms_encode(f.interpolate(mosaic,(224,224),mode='bilinear',align_corners=False)[:,0])

def routing_from_ccd(model,value):
    energy=torch.einsum('bhw,khw->bk',value,model.router_masks);centered=energy-energy.mean(1,keepdim=True)
    dense=torch.softmax(centered/(centered.square().mean(1,keepdim=True)+1e-8).sqrt()/2,1)
    return dense/dense.square().sum(1,keepdim=True).sqrt().clamp_min(1e-8)

def encode_stage(model,amplitude,stage,weights=None):
    _,amp,phase=model.physical_fields(amplitude,stage,weights)
    return np.rint(amp.numpy().clip(0,1)*255).astype(np.uint8),np.rint(phase.numpy()*256).astype(np.uint8)

def model_for(assets,arm):
    p=assets/(arm+'.pt');assert sha(p)==EXPECTED[arm]
    c=torch.load(p,map_location='cpu',weights_only=False)
    assert c['epoch']==(123 if arm=='residual' else 100)
    m=LabMoE(.3 if arm=='residual' else 0.).eval().requires_grad_(False)
    m.load_state_dict(c['model'],strict=True)
    assert c['config']['original_indices']==list(INDICES) and c['config']['input_tile']==100
    return m

def dataset(assets,role,limit=0):
    p=assets/'caltech10.npz';manifest=json.loads((assets/'manifest.json').read_text())
    assert sha(p)==manifest['data_sha256']
    with np.load(p,allow_pickle=False) as z:x,y,ids=[z[role+'_'+n].copy() for n in ('images','labels','ids')]
    assert len(y)=={'train':761,'val':157,'test':1833}[role]
    assert len(set(ids.tolist()))==len(ids) and x.shape==(len(y),150,150,3)
    if limit:
        # Calibration uses TRAIN only, balanced over all ten classes, no TEST selection.
        assert role=='train' and limit>=10
        order=[i for k in range(10) for i in np.flatnonzero(y==k)[:max(1,limit//10)]]
        x,y,ids=x[order],y[order],ids[order]
    return torch.from_numpy(x.transpose(0,3,1,2).copy()),y,ids

def metrics(y,p):
    pred=p.argmax(1);cm=np.zeros((10,10),np.int64);np.add.at(cm,(y,pred),1)
    support=cm.sum(1);recall=np.divide(cm.diagonal(),support,out=np.zeros(10),where=support>0)
    return dict(samples=len(y),accuracy=float((y==pred).mean()),macro_recall=float(recall[support>0].mean()),support=support.tolist(),confusion_matrix=cm.tolist())

def corners(txt):
    rows={k:[float(x),float(y)] for k,x,y in re.findall(r'(TL|TR|BL|BR)\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',txt.read_text())}
    assert set(rows)=={'TL','TR','BR','BL'}
    return [rows[k] for k in ('TL','TR','BR','BL')]

def native(gray,kind):
    h=1080 if kind=='amplitude' else 1200
    a=np.zeros((h,1920),np.uint8);v=np.asarray(Image.fromarray(gray).resize((1016,1016),Image.Resampling.NEAREST))
    a[(h-1016)//2:(h+1016)//2,452:1468]=v
    return a

def phase_native(gray):
    # Exact established machine convention, not a learned geometry correction.
    return 255-native(np.flipud(np.fliplr(gray)),'phase')

@torch.inference_mode()
def selftest(a):
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);x,y,ids=dataset(a.assets,'train',10);proof=[]
    metadata=json.loads((a.assets/'bridge_reference.json').read_text());assert sha(a.assets/'bridge_reference.npz')==metadata['sha256']
    reference=np.load(a.assets/'bridge_reference.npz',allow_pickle=False)
    assert ids.tolist()==reference['ids'].tolist() and metadata['checkpoints']==EXPECTED
    # Cross-OS CPU FFT/interpolation differences are diagnostic, not a native-BMP
    # parity gate. Compare the unchanged training bodies on THIS runtime first.
    tree=ast.parse(Path(__file__).with_name('resisc20_input_encoder.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encode')
    ns={'torch':torch,'F':torch.nn.functional,'rms_encode':rms_encode}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_training_encoder','exec'),ns)
    original_input=ns['encode'](x,None,100)
    assert torch.equal(original_input,encode(x,None,100))
    original_forward=textwrap.dedent(inspect.getsource(CorrectedMoE.forward))
    assert original_forward.count('+20e-12')==1
    original_ns={'torch':torch};exec(compile(original_forward.replace('+20e-12','+10e-12'),'frozen_training_ten_readout','exec'),original_ns)
    for arm in EXPECTED:
        model=model_for(a.assets,arm);inp=encode(x,None,100);ref=model(inp)[0];calls=[]
        assert np.allclose(inp.numpy(),reference['input'],atol=1e-6,rtol=0)
        training_error=float(np.max(np.abs(ref.numpy()-reference[arm])))
        original_probability=original_ns['forward'](model,original_input)[0]
        same_runtime_error=float((original_probability-ref).abs().max())
        assert same_runtime_error<=1e-6,'Lab graph differs from unchanged training body on this runtime'
        assert np.array_equal(ref.numpy().argmax(1),reference[arm].argmax(1)),'Cross-platform class prediction changed in calibration reference'
        def callback(stage,amp,phase,ids):
            qa=np.rint(amp.clip(0,1)*255).astype(np.uint8)
            assert np.allclose(qa.astype(np.float32)/255,amp,atol=1e-7,rtol=0)
            ph=phase_native(phase)
            decoded=np.flipud(np.fliplr(255-ph[92:1108,452:1468]))
            decoded=np.asarray(Image.fromarray(decoded).resize((478,478),Image.Resampling.NEAREST))
            assert np.array_equal(decoded,phase)
            field=torch.from_numpy(qa.astype(np.float32)/255)*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
            field=torch.nn.functional.pad(field,(20,20,20,20));calls.append(stage)
            return model.propagator(field).abs().square()[:,20:498,20:498].numpy()
        actual=model(inp,callback,ids)[0];err=float((actual-ref).abs().max())
        assert err<=1e-6 and calls==['router','0','1','2']
        proof.append(dict(arm=arm,checkpoint_sha256=EXPECTED[arm],bridge_error=err,same_runtime_training_body_error=same_runtime_error,cross_platform_probability_difference=training_error,cross_platform_argmax_equal=True,captures_per_sample=4,train_samples=10))
    save(a.out/'selftest.json',dict(passed=True,proof=proof,devices_opened=0,electronic_finetuning=False))
    print(json.dumps({'selftest':'passed','proof':proof}),flush=True)

class Bench:
    def __init__(self,a):self.a=a
    def __enter__(self):
        from contextlib import ExitStack
        from LightGenV2.hardware_common.shs.slm_camera import Controller
        from LightGenV2.hardware_common.shs.phase_hdmi import PhaseHDMI
        self.stack=ExitStack()
        try:
            c=json.loads(self.a.machine_config.read_text(encoding='utf-8-sig'))
            c['amplitude_slm']['sdk_path']=str(self.a.amplitude_sdk.resolve())
            c['camera'].update(exposure_us=self.a.exposure_us,gain='Gain_X4');c['settle_delay_ms']=240
            self.phase=self.stack.enter_context(PhaseHDMI(self.a.phase_sdk,self.a.phase_lut,settle_s=.8,pixel_format='rgba'))
            self.hw=self.stack.enter_context(Controller(c,config_base=self.a.machine_config.parent))
            save(self.a.out/'camera_readback.json',self.hw.camera_settings)
            assert abs(float(self.hw.camera.get('ExposureTime'))-self.a.exposure_us)<1
            assert self.hw.camera.get('Gain')=='Gain_X4'
            return self
        except BaseException:self.stack.close();raise
    def __exit__(self,*args):return self.stack.__exit__(*args)
    def stage(self,path):self.phase.show(path);self.hw.camera.fresh()
    def capture(self,bmp):return self.hw.capture(bmp)

def signalcheck(a):
    """Six TRAIN-pattern frames only; never normalize or turn them into accuracy."""
    import cv2
    assert not a.out.exists();a.out.mkdir(parents=True)
    pilot=a.out.parent/'pilot80';assert (pilot/'failure.json').exists()
    pts=np.float32(corners(a.coordinates));warp=cv2.getPerspectiveTransform(pts,np.float32([[0,0],[477,0],[477,477],[0,477]]))
    rows=[];lock=a.out.parents[1]/'.t18_sdk_owner.lock'
    with lock.open('x') as f:f.write(json.dumps({'pid':os.getpid(),'out':str(a.out)}))
    try:
        with Bench(a) as bench:
            for exposure in (400,2000):
                bench.hw.camera.set('ExposureTime',exposure)
                for kind,stage in (('black','0'),('router','router'),('expert','0')):
                    src=pilot/'residual'/stage;phase=src/'phase.bmp';bench.stage(phase)
                    gray=np.zeros((478,478),np.uint8) if kind=='black' else np.asarray(Image.open(src/'000000_input.png'))
                    bmp=a.out/f'{exposure}_{kind}.bmp';Image.fromarray(native(gray,'amplitude')).save(bmp)
                    raw,meta=bench.capture(bmp);image=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA)).copy()
                    Image.fromarray(raw).save(a.out/f'{exposure}_{kind}_raw.png');Image.fromarray(image).save(a.out/f'{exposure}_{kind}.png')
                    v=dict(exposure_us=float(bench.hw.camera.get('ExposureTime')),pattern=kind,frame_id=meta['frame_id'],p99=float(np.percentile(image,99)),mean=float(image.mean()),maximum=int(image.max()),saturation_fraction=float((image==255).mean()),phase_sha256=sha(phase),input_sha256=sha(bmp))
                    if kind!='black':v['ideal_pcc']=float(np.corrcoef(np.asarray(Image.open(src/'000000_ideal_diagnostic_only.png')).ravel(),image.ravel())[0,1])
                    rows.append(v);save(a.out/'signal.json',dict(rows=rows,complete=False));print(json.dumps(v),flush=True)
        save(a.out/'signal.json',dict(rows=rows,complete=True,sdk_released=True,accuracy_evaluated=False))
    except BaseException as ex:save(a.out/'failure.json',dict(error=repr(ex),automatic_retry=False));raise
    finally:lock.unlink()

def supportcheck(a):
    """Nine flat-gray diagnostic frames distinguish response and illumination."""
    import cv2
    assert not a.out.exists();a.out.mkdir(parents=True)
    phase=a.out.parent/'pilot80/residual/0/phase.bmp';assert phase.is_file()
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    patterns=[('black',0)] + [(kind,g) for kind in ('center','experts') for g in (64,128,192,255)]
    rows=[];lock=a.out.parents[1]/'.t18_sdk_owner.lock'
    with lock.open('x') as f:f.write(json.dumps({'pid':os.getpid(),'out':str(a.out)}))
    try:
        with Bench(a) as bench:
            bench.stage(phase)
            for kind,g in patterns:
                gray=np.zeros((478,478),np.uint8)
                if kind=='center':gray[127:351,127:351]=g
                elif kind=='experts':
                    for yy in (0,254):
                        for xx in (0,254):gray[yy:yy+224,xx:xx+224]=g
                bmp=a.out/f'{kind}_{g}.bmp';Image.fromarray(native(gray,'amplitude')).save(bmp)
                raw,meta=bench.capture(bmp);image=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA)).copy()
                Image.fromarray(raw).save(a.out/f'{kind}_{g}_raw.png');Image.fromarray(image).save(a.out/f'{kind}_{g}.png')
                row=dict(pattern=kind,gray=g,frame_id=meta['frame_id'],mean=float(image.mean()),p99=float(np.percentile(image,99)),maximum=int(image.max()),raw_mean=float(raw.mean()),raw_maximum=int(raw.max()),saturation_fraction=float((image==255).mean()),phase_sha256=sha(phase),input_sha256=sha(bmp))
                rows.append(row);save(a.out/'support.json',dict(rows=rows,complete=False));print(json.dumps(row),flush=True)
                if row['saturation_fraction']>.15:raise RuntimeError('Diagnostic saturated>15%; retain evidence')
        save(a.out/'support.json',dict(rows=rows,complete=True,sdk_released=True,exposure_us=a.exposure_us,accuracy_evaluated=False))
    except BaseException as ex:save(a.out/'failure.json',dict(error=repr(ex),automatic_retry=False));raise
    finally:lock.unlink()


def responsecheck(a):
    """Measured field inverse and held-out levels; never overwrite a device LUT."""
    import cv2
    from amplitude_response import inverse_field
    assert not a.out.exists();a.out.mkdir(parents=True)
    phase=a.out.parent/'pilot80/residual/0/phase.bmp';assert phase.is_file()
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    rows=[];lock=a.out.parents[1]/'.t18_sdk_owner.lock'
    with lock.open('x') as f:f.write(json.dumps({'pid':os.getpid(),'out':str(a.out)}))
    def frame(bench,kind,drive,tag,desired=None):
        gray=np.zeros((478,478),np.uint8)
        if kind=='experts':
            for yy in (0,254):
                for xx in (0,254):gray[yy:yy+224,xx:xx+224]=drive
        else:gray[127:351,127:351]=drive
        bmp=a.out/(tag+'.bmp');Image.fromarray(native(gray,'amplitude')).save(bmp)
        raw,meta=bench.capture(bmp);image=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA)).copy()
        Image.fromarray(raw).save(a.out/(tag+'_raw.png'));Image.fromarray(image).save(a.out/(tag+'.png'))
        samples=[image.astype(np.float32)];frame_ids=[meta['frame_id']];saturations=[float((image==255).mean())]
        for repeat in range(1,a.response_repeats):
            # Same persistent display; fresh camera frames diagnose temporal
            # variability without repeatedly releasing/reloading the SLM handle.
            extra,extra_meta=bench.hw.camera.fresh()
            crop=np.flipud(cv2.warpPerspective(extra,warp,(478,478),flags=cv2.INTER_AREA)).copy()
            Image.fromarray(extra).save(a.out/(tag+f'_raw_{repeat}.png'))
            samples.append(crop.astype(np.float32));frame_ids.append(extra_meta['frame_id']);saturations.append(float((crop==255).mean()))
        average=np.mean(samples,axis=0)
        row=dict(pattern=kind,drive=int(drive),desired=desired,frame_id=meta['frame_id'],frame_ids=frame_ids,mean=float(average.mean()),repeat_means=[float(v.mean()) for v in samples],p99=float(np.percentile(average,99)),maximum=float(average.max()),saturation_fraction=max(saturations),input_sha256=sha(bmp),raw_sha256=sha(a.out/(tag+'_raw.png')))
        save(a.out/(tag+'.json'),row)
        rows.append(row);save(a.out/'progress.json',dict(frames=len(rows),latest=row));print(json.dumps(row),flush=True)
        assert row['saturation_fraction']<.001,'Calibration clipped; retain data, do not fit a clipped response'
        return row['mean']
    try:
        with Bench(a) as bench:
            bench.stage(phase);grays=list(range(0,256,16))+[255];measure={g:[] for g in grays}
            for repeat in range(3):
                for g in (grays if repeat%2==0 else grays[::-1]):measure[g].append(frame(bench,'experts',g,f'fit_{repeat}_{g:03d}'))
            means=[float(np.mean(measure[g])) for g in grays];lut,response=inverse_field(grays,means)
            checks=[]
            for kind in ('experts','center'):
                low=frame(bench,kind,0,f'{kind}_zero');high=frame(bench,kind,255,f'{kind}_full')
                assert high-low>1
                for desired in (32,64,96,128,160,192,224):
                    value=frame(bench,kind,lut[desired],f'validate_{kind}_{desired}',desired)
                    normalized=(value-low)/(high-low);target=(desired/255.)**2
                    checks.append(dict(pattern=kind,desired=desired,measured=normalized,target=target,error=normalized-target))
            errors=np.array([v['error'] for v in checks]);passed=bool(np.max(np.abs(errors))<=.08 and np.sqrt(np.mean(errors**2))<=.04)
        save(a.out/'response_lut.json',dict(passed=passed,sdk_released=True,scope='this paired session only; common mapping both arms',target='field amplitude; intensity=(canonical_gray/255)^2',drive=lut,fit_grays=grays,fit_means=means,normalized_response=response,checks=checks,validation_rmse=float(np.sqrt(np.mean(errors**2))),validation_max_error=float(np.max(np.abs(errors))),exposure_us=a.exposure_us,frames_per_pattern=a.response_repeats,gain='Gain_X4',coordinates_sha256=sha(a.coordinates),phase_sha256=sha(phase),source_sha256=sha(__file__),raw_rows=rows,devices_original_lut_modified=False,dark_subtracted_in_network=False))
        if not passed:raise RuntimeError('Held-out response validation failed; do not deploy this mapping')
    except BaseException as ex:save(a.out/'failure.json',dict(error=repr(ex),frames=len(rows),automatic_retry=False));raise
    finally:lock.unlink()

@torch.inference_mode()
def capture(a):
    import cv2
    assert a.role in ('train','test') and (not a.limit or a.role=='train'),'No VAL acquisition; calibration uses TRAIN only'
    assert json.loads((a.selftest/'selftest.json').read_text())['passed']
    if a.resume:
        failure=json.loads((a.out/'failure.json').read_text())
        assert 'PermissionError' in failure['error'],'Only audited file-publication failure is resumable'
    else:
        assert not a.out.exists(),'Never overwrite or blindly restart a physical session'
        a.out.mkdir(parents=True)
    torch.set_num_threads(4);x,y,ids=dataset(a.assets,a.role,a.limit);n=len(y)
    pts=np.float32(corners(a.coordinates));dest=np.float32([[0,0],[477,0],[477,477],[0,477]])
    warp=cv2.getPerspectiveTransform(pts,dest)
    response=json.loads(a.response_lut.read_text()) if a.response_lut else None
    if response:
        assert response['passed'] and response['sdk_released'] and response['coordinates_sha256']==sha(a.coordinates)
        drive=np.asarray(response['drive'],dtype=np.uint8);assert drive.shape==(256,) and drive[0]==0 and drive[-1]==255
    contract=dict(coordinates_sha256=sha(a.coordinates),corners_TL_TR_BR_BL=pts.tolist(),checkpoint_sha256=EXPECTED,
        samples=n,role=a.role,exposure_us=a.exposure_us,gain='Gain_X4',wait_ms=240,phase_transform='hv_inverse',camera_orientation='flip_v',
        data_sha256=sha(a.assets/'caltech10.npz'),source_sha256=sha(__file__),fine_tuned=False,routing='continuous all4',main_stages=['expert','global','expert'],native_active=1016)
    contract['response_lut_sha256']=sha(a.response_lut) if response else None
    contract['response_calibration_exposure_us']=response['exposure_us'] if response else None
    if a.resume:
        before=json.loads((a.out/'contract.json').read_text())
        assert {k:v for k,v in before.items() if k!='source_sha256'}=={k:v for k,v in contract.items() if k!='source_sha256'}
        save(a.out/('resume_'+str(time.time_ns())+'.json'),dict(previous_failure=failure,source_sha256=sha(__file__),no_recapture=True))
    else:save(a.out/'contract.json',contract)
    started=time.perf_counter();rows=[];cached=0
    lock=a.out.parents[1]/'.t18_sdk_owner.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x') as f:f.write(json.dumps({'pid':os.getpid(),'out':str(a.out)}))
    try:
        with Bench(a) as bench:
            for arm in EXPECTED:
                model=model_for(a.assets,arm);folder=a.out/arm;folder.mkdir(exist_ok=a.resume);route=None;previous=None
                for stage in ('router','0','1','2'):
                    target=folder/stage;target.mkdir(exist_ok=a.resume);ccd=np.lib.format.open_memmap(target/'ccd.npy',mode='r+' if (target/'ccd.npy').exists() else 'w+',dtype=np.uint8,shape=(n,478,478))
                    assert ccd.shape==(n,478,478) and ccd.dtype==np.uint8
                    phase_path=target/'phase.bmp';phase_set=False;current_weights=[]
                    for start in range(0,n,8):
                        stop=min(start+8,n)
                        if stage in ('router','0'):amp=encode(x[start:stop],None,100)
                        else:amp=model.nonlinear(torch.from_numpy(np.asarray(previous[start:stop]).copy()).float(),model.stages[int(stage)])
                        weights=None if stage=='router' else torch.from_numpy(route[start:stop]).float()
                        amplitudes,phase=encode_stage(model,amp,stage,weights)
                        if not phase_set:
                            if phase_path.exists():assert np.array_equal(np.asarray(Image.open(phase_path)),phase_native(phase))
                            else:Image.fromarray(phase_native(phase)).save(phase_path)
                            bench.stage(phase_path);phase_set=True
                        else:assert np.array_equal(np.asarray(Image.open(phase_path)),phase_native(phase))
                        for j,gray in enumerate(amplitudes,start):
                            sid=f'{j:06d}';canonical=target/(sid+'_input.png');receipt=target/(sid+'.json')
                            if a.resume and receipt.exists():
                                row=json.loads(receipt.read_text());assert row['sample_id']==str(ids[j]) and row['index']==j and row['arm']==arm and row['stage']==stage
                                assert row['canonical_sha256']==sha(canonical) and row['phase_sha256']==sha(phase_path)
                                assert np.array_equal(np.asarray(Image.open(canonical)),gray)
                                cropped=np.asarray(Image.open(target/(sid+'.png')))
                                raw=np.asarray(Image.open(target/(sid+'_raw.png')))
                                expected=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                                assert np.array_equal(cropped,expected) and np.array_equal(ccd[j],cropped)
                                assert row['p99']>=15 and row['saturation_fraction']<=.15
                                rows.append(row);cached+=1;continue
                            Image.fromarray(gray).save(canonical)
                            bmp=target/(sid+'_transient.bmp');Image.fromarray(native(drive[gray] if response else gray,'amplitude')).save(bmp)
                            raw,meta=bench.capture(bmp);assert raw.shape==(1080,1920)
                            image=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                            row=dict(arm=arm,stage=stage,index=j,sample_id=str(ids[j]),frame_id=meta.get('frame_id'),mean=float(image.mean()),p99=float(np.percentile(image,99)),maximum=int(image.max()),saturation_fraction=float((image==255).mean()),capture_total_ms=meta['capture_total_ms'],amplitude_sha256=sha(bmp),canonical_sha256=sha(canonical),phase_sha256=sha(phase_path))
                            if a.limit:
                                field=torch.from_numpy(gray.astype(np.float32)/255)[None]*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))[None]
                                ideal=model.propagator(torch.nn.functional.pad(field,(20,20,20,20))).abs().square()[0,20:498,20:498].numpy()
                                row['ideal_ccd_pcc']=float(np.corrcoef(ideal.ravel(),image.ravel())[0,1])
                                Image.fromarray(np.uint8(ideal/(ideal.max()+1e-12)*255)).save(target/(sid+'_ideal_diagnostic_only.png'))
                            Image.fromarray(raw).save(target/(sid+'_raw.png'));Image.fromarray(image).save(target/(sid+'.png'));save(target/(sid+'.json'),row)
                            ccd[j]=image;ccd.flush();bmp.unlink() # Only this generated native input; canonical bytes retained.
                            rows.append(row)
                            save(a.out/'progress.json',dict(state='capturing',arm=arm,stage=stage,stage_completed=j+1,stage_total=n,total_ccd=len(rows),elapsed_s=time.perf_counter()-started))
                            if row['p99']<15:raise RuntimeError('Dark CCD: p99<15; retain evidence, do not retry blindly')
                            if row['saturation_fraction']>.15:raise RuntimeError('CCD saturation>15%; retain evidence')
                        if stage=='router':current_weights.append(routing_from_ccd(model,torch.from_numpy(np.asarray(ccd[start:stop]).copy()).float()).numpy())
                    if stage=='router':route=np.concatenate(current_weights);np.save(folder/'routing.npy',route)
                    previous=ccd
                    print(json.dumps(dict(arm=arm,stage=stage,samples=n,total_ccd=len(rows),elapsed_s=time.perf_counter()-started)),flush=True)
                energy=np.concatenate([np.einsum('bhw,khw->bk',np.asarray(previous[k:k+8]).astype(np.float32),model.detectors.numpy()) for k in range(0,n,8)])
                p=(energy+1e-12)/(energy.sum(1,keepdims=True)+10e-12)
                np.savez_compressed(folder/'predictions.npz',ids=ids,labels=y,probabilities=p)
                save(folder/'metrics.json',dict(metrics(y,p),physical=True,fine_tuned=False,checkpoint_sha256=EXPECTED[arm]))
        elapsed=time.perf_counter()-started
        save(a.out/'report.json',dict(state='complete',samples_per_arm=n,physical_ccd=len(rows),reused_verified_ccd=cached,new_ccd=len(rows)-cached,elapsed_s=elapsed,seconds_per_ccd=elapsed/max(1,len(rows)-cached),projected_two_arm_test_seconds=elapsed/max(1,len(rows)-cached)*14664,signal_rows=rows if a.limit else [],results={k:json.loads((a.out/k/'metrics.json').read_text()) for k in EXPECTED},sdk_released=True))
        save(a.out/'progress.json',dict(state='complete',total_ccd=len(rows),sdk_released=True))
    except BaseException as ex:save(a.out/('failure_resume_'+str(time.time_ns())+'.json' if a.resume else 'failure.json'),dict(error=repr(ex),total_ccd=len(rows),automatic_retry=False));raise
    finally:lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['selftest','capture','signalcheck','supportcheck','responsecheck'],required=True)
    for name in ('assets','out','coordinates','selftest','machine-config','amplitude-sdk','phase-sdk','phase-lut'):p.add_argument('--'+name,type=Path,required=name in ('assets','out'))
    p.add_argument('--role',choices=['train','test'],default='test');p.add_argument('--limit',type=int,default=0);p.add_argument('--exposure-us',type=float,default=400)
    p.add_argument('--response-lut',type=Path)
    p.add_argument('--response-repeats',type=int,choices=[1,9],default=1)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if sys.stdout is None:
        sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    {'selftest':selftest,'capture':capture,'signalcheck':signalcheck,'supportcheck':supportcheck,'responsecheck':responsecheck}[a.mode](a)
