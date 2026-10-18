"""One-shot frozen-control L4 then L6 queue AFTER L2 residual fully finishes.

One complete TRAIN50/class1000 then TEST411 per physical layer, same fixed bench
contract. No overlap, VAL, mask update, new layer, restart or recurring monitor.
Each arm's original5140 head is fitted after its entire capture, CPU only.
"""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path[:0]=[str(HERE),str(ROOT)]
import numpy as np
import torch
from PIL import Image
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,BOUNDED,BoundedAlphaHybridMoE,sha
from lab_caltech20_rich_capture_20261018 import data,layer_forward,metrics
from lab_caltech20_rich_readout_20261018 import existing_head,evaluate
from lab_caltech10_20261015 import Bench,corners,native,phase_native,save
LOCK=HERE/'runs/hardware/.t18_sdk_owner.lock'


def read(p):return json.loads(Path(p).read_text())


def selftest():
    from caltech20_rich_pair50_alpha_model import PairModel
    torch.set_num_threads(2);torch.manual_seed(17);rows=[]
    with torch.inference_mode():
        for depth in (4,6):
            source=PairModel(depth,0.).eval();bridge=BoundedAlphaHybridMoE(depth,0.).eval();bridge.load_state_dict(source.state_dict(),strict=True);x=torch.rand(2,224,224)
            error=float((source(x)[0]-bridge(x)[0]).abs().max());assert error==0
            rows.append(dict(depth=depth,error=error,alpha_schema=BOUNDED,captures=depth+1))
    print(json.dumps(dict(passed=True,proof=rows,SDK_opened=False,GPU_used=False)),flush=True)


def adapt_head(root,model,datasets):
    out=root/'readout120';out.mkdir();original={k:v.detach().clone() for k,v in model.state_dict().items()};head=existing_head(original);parts={};files={}
    for role,(_,y,ids) in datasets.items():
        path=root/'capture'/role/'predictions.npz';files[role]=sha(path)
        with np.load(path,allow_pickle=False) as z:
            assert np.array_equal(z['labels'],y) and np.array_equal(z['ids'],ids);x=torch.from_numpy(z['features'].copy());p=z['probabilities'].copy()
        with torch.no_grad():assert np.abs(torch.softmax(head(x),1).numpy()-p).max()<=3e-6
        parts[role]=dict(x=x,y=torch.from_numpy(y.astype(np.int64)),ids=ids)
    torch.manual_seed(17);torch.set_num_threads(2);optim=torch.optim.AdamW(head.parameters(),lr=.0005,weight_decay=.0001);baseline,_=evaluate(head,parts['test']);best=baseline['accuracy'];selected=0;history=[]
    protocol=dict(TRAIN=1000,TEST=411,VALID=0,epochs=120,seed=17,batch_size=64,lr=.0005,weight_decay=.0001,optimizer='AdamW',loss='cross entropy',trainable=5140,features_SHA256=files,all_nonterminal_frozen=True,TEST_development_selection=True,TEST_gradients=False,new_layers=0,GPU_used=False,SDK_opened=False)
    save(out/'protocol.json',protocol)
    def ckpt(name,epoch):
        state={k:v.detach().clone() for k,v in original.items()}
        for k,v in head.state_dict().items():state['linear.'+k]=v.detach().clone()
        assert all(torch.equal(v,state[k]) for k,v in original.items() if not k.startswith('linear.'))
        torch.save(dict(model=state,epoch=epoch,protocol=protocol),out/(name+'.pt'))
    ckpt('best',0)
    for epoch in range(1,121):
        order=torch.randperm(1000);total=0.
        for start in range(0,1000,64):
            ix=order[start:start+64];optim.zero_grad();loss=torch.nn.functional.cross_entropy(head(parts['train']['x'][ix]),parts['train']['y'][ix]);assert torch.isfinite(loss);loss.backward();optim.step();total+=float(loss.detach())*len(ix)
        if epoch==1 or epoch%10==0:
            train,_=evaluate(head,parts['train']);test,_=evaluate(head,parts['test']);row=dict(epoch=epoch,TRAIN_loss=total/1000,TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy']);history.append(row)
            if test['accuracy']>best:best=test['accuracy'];selected=epoch;ckpt('best',epoch)
            save(out/'progress.json',dict(state='fitting',latest=row,best=best));save(out/'history.json',history)
    ckpt('last',120);results={}
    for name in ('best','last'):
        value=torch.load(out/(name+'.pt'),map_location='cpu',weights_only=False);assert all(torch.equal(v,value['model'][k]) for k,v in original.items() if not k.startswith('linear.'))
        fresh=existing_head(value['model']);train,_=evaluate(fresh,parts['train']);test,p=evaluate(fresh,parts['test']);prediction=out/(name+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=parts['test']['ids'],labels=parts['test']['y'].numpy(),probabilities=p)
        results[name]=dict(epoch=value['epoch'],TRAIN=train,TEST_development=test,PT_SHA256=sha(out/(name+'.pt')),predictions_SHA256=sha(prediction),strict_CPU=True,nonterminal_unchanged=True)
    assert results['best']['TEST_development']['accuracy']==best and all(sha(root/'capture'/role/'predictions.npz')==digest for role,digest in files.items())
    report=dict(state='complete',results=results,baseline_TEST=baseline,selected_epoch=selected,SDK_opened=False,GPU_used=False,TEST_is_development=True);save(out/'report.json',report);return report


@torch.inference_mode()
def capture_group(a,spec):
    import cv2
    assert not LOCK.exists();root=a.session/('L'+str(spec['depth']));assert not root.exists();root.mkdir()
    a.assets=a.parent/'assets';datasets,dm=data(a,False);model,_=load_checkpoint(a.session/spec['file'],spec['sha256'],spec['depth'],0.,alpha_schema=BOUNDED)
    release=read(a.parent/'release.json')
    for k,p in (('coordinates_sha256',a.coordinates),('phase_lut_sha256',a.phase_lut),('machine_config_sha256',a.machine_config),('response_lut_sha256',a.response_lut)):assert sha(p)==release[k]
    response=read(a.response_lut);drive=np.asarray(response['drive'],np.uint8);assert response['passed'] and drive.shape==(256,)
    assert shutil.disk_usage(a.session).free>=22*(1<<30);a.out=root/'capture';a.out.mkdir();a.exposure_us=2000
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]));frames=0;started=time.perf_counter();arrays={}
    save(root/'contract.json',dict(model=spec,data_SHA256=dm['sha256'],order='one complete TRAIN1000 then TEST411 for every layer, L4 before L6; after L2 residual capture AND training complete',TRAIN=1000,TEST=411,VALID=0,exposure_us=2000,gain='Gain_X4',wait_ms=240,native_active=1016,phase='hv_inverse',camera='flip_v',source_SHA256=sha(__file__),hardware_release_SHA256=sha(a.parent/'release.json')))
    with LOCK.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),out=str(a.out),depth=spec['depth'])))
    try:
        with Bench(a) as bench:
            previous={role:{} for role in datasets};final={role:[] for role in datasets};stages=['router']+[str(k) for k in range(spec['depth'])]
            for stage in stages:
                phasepath=root/('phase_'+stage+'.bmp');phase_pixels=None
                for role,(images,y,ids) in datasets.items():
                    folder=a.out/role/stage;folder.mkdir(parents=True);ccd=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(len(y),478,478))
                    for begin in range(0,len(y),8):
                        end=min(begin+8,len(y));cached={k:np.asarray(v[begin:end]).copy() for k,v in previous[role].items()}
                        def acquire(current,amplitudes,phase,sample_ids):
                            nonlocal frames,phase_pixels
                            if phase_pixels is None:phase_pixels=phase.copy();Image.fromarray(phase_native(phase)).save(phasepath);bench.stage(phasepath)
                            assert np.array_equal(phase_pixels,phase)
                            for index,amp in enumerate(amplitudes,begin):
                                gray=np.rint(amp.clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amp))<=1e-7
                                p=folder/f'{index:06d}';canonical=p.with_name(p.name+'_input.png');Image.fromarray(gray).save(canonical);bmp=p.with_name(p.name+'_transient.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(bmp)
                                raw,meta=bench.capture(bmp);assert raw.shape==(1080,1920);crop=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                                receipt=dict(depth=spec['depth'],stage=stage,role=role,index=index,sample_id=str(ids[index]),frame_id=meta['frame_id'],captured_at_ns=time.time_ns(),mean=float(crop.mean()),p99=float(np.percentile(crop,99)),saturation_fraction=float((crop==255).mean()),capture_total_ms=meta['capture_total_ms'],input_SHA256=sha(bmp),canonical_SHA256=sha(canonical),phase_SHA256=sha(phasepath),checkpoint_SHA256=spec['sha256'])
                                Image.fromarray(raw).save(p.with_name(p.name+'_raw.png'));Image.fromarray(crop).save(p.with_suffix('.png'));save(p.with_suffix('.json'),receipt);ccd[index]=crop;ccd.flush();frames+=1;bmp.unlink()
                                save(a.session/'progress.json',dict(state='capturing',depth=spec['depth'],stage=stage,role=role,completed=index+1,total=len(y),frames=frames,elapsed_s=time.perf_counter()-started,latest=receipt))
                                if receipt['p99']<15 or receipt['saturation_fraction']>.15:raise RuntimeError('Signal guard; retain evidence; no automatic retry')
                                if frames==24:save(root/'initial24_signal_gate.json',dict(passed=True,part_of_full_capture=True))
                            return np.asarray(ccd[begin:end]).copy()
                        value=layer_forward(model,images[begin:end],ids[begin:end],stage,cached,acquire)
                        if value is not None:final[role].append(value)
                    previous[role][stage]=ccd;arrays[role+'/'+stage]=dict(sha256=sha(folder/'ccd.npy'),samples=len(y))
                save(root/('stage_'+stage+'_complete.json'),dict(state='complete',stage=stage,frames=frames,phase_SHA256=sha(phasepath)))
            results={}
            for role,(_,y,ids) in datasets.items():
                p=np.concatenate([v['probabilities'] for v in final[role]]);features=np.concatenate([v['features'] for v in final[role]]);np.savez_compressed(a.out/role/'predictions.npz',ids=ids,labels=y,probabilities=p,features=features);results[role]=dict(metrics(y,p),physical=True,samples=len(y),adapted=False);save(a.out/role/'metrics.json',results[role])
        assert frames==(spec['depth']+1)*1411
        save(root/'capture_report.json',dict(state='complete',depth=spec['depth'],physical_ccd=frames,sdk_released=True,results=results,arrays=arrays,PT_SHA256=spec['sha256'],elapsed_s=time.perf_counter()-started))
    finally:LOCK.unlink()
    return model,datasets


def main(a):
    assert not (a.session/'queue_started.json').exists();manifest=read(a.session/'models_manifest.json');assert manifest['state']=='CPU_verified'
    assert [q['depth'] for q in manifest['models']]==[4,6]
    save(a.session/'queue_started.json',dict(pid=os.getpid(),source_SHA256=sha(__file__),manifest_SHA256=sha(a.session/'models_manifest.json'),wait_for=str(a.wait_for),condition='L2 residual full pipeline report complete, SDK released and training child exited; no insertion in between layers',planned_frames=16932,TRAIN_each_class=50,TRAIN=1000,TEST=411,VALID=0,recurring_monitor=False))
    deadline=time.monotonic()+3*3600
    while not (a.wait_for/'report.json').exists():
        if (a.wait_for/'failure.json').exists():raise RuntimeError('Prior L2 failed; do not overlap or automatically restart it')
        if time.monotonic()>deadline:raise TimeoutError('L2 dependency deadline; no infinite wait')
        time.sleep(5)
    prior=read(a.wait_for/'report.json');assert prior['state']=='complete' and prior['sdk_released'] and prior['GPU_child_exited'] and not LOCK.exists()
    torch.set_num_threads(4);rows=[]
    for spec in manifest['models']:
        model,datasets=capture_group(a,spec)
        # Leave inference_mode before this optimization: capture_group returns ordinary parameters.
        fit=adapt_head(a.session/('L'+str(spec['depth'])),model,datasets);rows.append(dict(depth=spec['depth'],capture=read(a.session/('L'+str(spec['depth']))/'capture_report.json'),readout=fit))
    save(a.session/'report.json',dict(state='complete',depths=rows,physical_ccd=16932,sdk_released=not LOCK.exists(),VALID=0,TEST_development=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true')
    for name in ('session','parent','wait-for','coordinates','machine-config','phase-lut','response-lut','phase-sdk','amplitude-sdk'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=(a.session/'queue.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:main(a)
        except BaseException as error:save(a.session/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
