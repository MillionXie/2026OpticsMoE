"""Residual-only, fresh60/class, measured-prefix/downstream-mask fine-tuning.

Inference architecture is unchanged. Each complete layer's TRAIN then TEST is
captured once, its upstream graph frozen, and only its still-deployable suffix
is optimized. TEST selects development checkpoints but never supplies gradients.
No VALID, new layer, control-arm mutation, automatic signal retry or monitor.
"""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from lab_caltech20_rich_model_20261018 import DirectAlphaHybridMoE,load_checkpoint,encode_images,sha,DIRECT
from lab_caltech20_rich_capture_20261018 import metrics,layer_forward
from lab_caltech10_20261015 import Bench,corners,native,phase_native,save
from lab_caltech20_rich_tail_refine_20261018 import windows_process_names,gpu_process_allowed

STAGES=('router','0','1')
ORIGINAL='e7c140ecd11d52a91bef882a01bc947c3da7f9f72df63d7fcf23055ab503be69'
LOCK=HERE/'runs/hardware/.t18_sdk_owner.lock'


def own_sdk_controller(pid,name,actual_parent,verified_controller):
    return verified_controller is not None and pid==actual_parent==verified_controller and name.lower() in ('python.exe','pythonw.exe')


def python_parent_chain(controller_pid):
    """Windows venv launchers insert a Python shim; verify <=3 Python-only hops."""
    if os.name!='nt':return [os.getppid()] if os.getppid()==controller_pid else []
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ProcessID',wintypes.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];k.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    for n in ('Process32FirstW','Process32NextW'):getattr(k,n).argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)];getattr(k,n).restype=wintypes.BOOL
    k.CloseHandle.argtypes=[wintypes.HANDLE];handle=k.CreateToolhelp32Snapshot(2,0);assert handle not in (None,ctypes.c_void_p(-1).value)
    table={};e=Entry();e.dwSize=ctypes.sizeof(Entry)
    try:
        ok=k.Process32FirstW(handle,ctypes.byref(e))
        while ok:table[int(e.th32ProcessID)]=(int(e.th32ParentProcessID),e.szExeFile.lower());ok=k.Process32NextW(handle,ctypes.byref(e))
    finally:k.CloseHandle(handle)
    path=[];pid=os.getpid()
    for _ in range(3):
        assert pid in table and table[pid][1] in ('python.exe','pythonw.exe')
        pid=table[pid][0];path.append(pid)
        if pid==controller_pid:
            assert table.get(pid,(0,''))[1] in ('python.exe','pythonw.exe');return path
    return []


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


class MeasuredPrefix(DirectAlphaHybridMoE):
    def __init__(self):
        super().__init__(2,.3);self.measured={}

    def capture(self,amplitude,stage,weights=None,callback=None,ids=None):
        if stage in self.measured:
            value=self.measured[stage]
            assert value.shape==(len(amplitude),478,478)
            return value.float()
        return super().capture(amplitude,stage,weights,callback,ids)


def allowed_keys(model,prefix):
    assert prefix in (1,2,3)
    phase_indices=set()
    for k in range(prefix-1,2):
        lo,hi=model.stage_slices[k];phase_indices.update(range(lo,hi))
    electronics=set(range(max(0,prefix-2),2))
    return {n for n,_ in model.named_parameters() if n.startswith('linear.')
            or (n.startswith('phase_sets.') and int(n.split('.')[1]) in phase_indices)
            or (n.startswith(('electronic.','fusion_logits.')) and int(n.split('.')[1]) in electronics)}


def load_data(session):
    m=read(session/'assets/data_manifest.json');path=session/'assets'/m['file']
    assert m['TRAIN']==1200 and m['TEST']==411 and m['VALID']==0 and sha(path)==m['sha256']
    datasets={}
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={r+'_'+n for r in ('train','test') for n in ('images','labels','ids')}
        for role,n in (('train',1200),('test',411)):
            x,y,ids=[z[role+'_'+k].copy() for k in ('images','labels','ids')]
            assert x.shape==(n,150,150,3) and len(set(ids.tolist()))==n
            if role=='train':assert np.bincount(y,minlength=20).tolist()==[60]*20
            datasets[role]=(torch.from_numpy(x.transpose(0,3,1,2).copy()),y,ids)
    assert not set(datasets['train'][2].tolist())&set(datasets['test'][2].tolist())
    return datasets,m


def prepare(a):
    """Construct nested60/class from old50 plus original-TRAIN-only extra10."""
    s=a.session;assert not (s/'assets').exists();s.mkdir(parents=True,exist_ok=True)
    selection=read(s/'selection60.json');extra=read(s/'extra_manifest.json')
    assert sha(s/'extra200.npz')==extra['sha256'] and selection['TRAIN']==1200
    old=a.parent/'assets';oldmeta=read(old/'data_manifest.json');assert sha(old/oldmeta['file'])==oldmeta['sha256']
    with np.load(old/oldmeta['file'],allow_pickle=False) as z,np.load(s/'extra200.npz',allow_pickle=False) as e:
        ids=np.concatenate([z['train_ids'],e['ids']]);assert len(set(ids.tolist()))==1200
        lookup={name:i for i,name in enumerate(ids.tolist())};ix=[lookup[name] for name in selection['train_ids']]
        out={}
        for k in ('images','labels','ids'):
            out['train_'+k]=np.concatenate([z['train_'+k],e[k]])[ix]
            out['test_'+k]=z['test_'+k].copy()
        assert out['test_ids'].tolist()==selection['test_ids'] and out['train_labels'].tolist()==selection['train_labels']
    assert np.bincount(out['train_labels'],minlength=20).tolist()==[60]*20
    (s/'assets').mkdir();p=s/'assets/rich20_train1200_test411.npz';np.savez_compressed(p,**out)
    shutil.copy2(old/'residual_reference.pt',s/'assets/original.pt');assert sha(s/'assets/original.pt')==ORIGINAL
    save(s/'assets/data_manifest.json',dict(file=p.name,sha256=sha(p),TRAIN=1200,TEST=411,VALID=0,TRAIN_each_class=60,selection_sha256=sha(s/'selection60.json'),source_cache_sha256=selection['cache_sha256'],extra_sha256=extra['sha256']))
    print(json.dumps(dict(state='data_prepared',TRAIN=1200,TEST=411,VALID=0,physical_frames=4833)),flush=True)


def selftest():
    torch.set_num_threads(2);torch.manual_seed(17)
    assert own_sdk_controller(10,'pythonw.exe',10,10)
    assert not own_sdk_controller(11,'pythonw.exe',10,10)
    assert not own_sdk_controller(10,'pythonw.exe',10,None)
    assert not own_sdk_controller(10,'pythonw.exe',12,10)
    ref=DirectAlphaHybridMoE(2,.3).eval()
    with torch.no_grad():
        for p in ref.fusion_logits:p.fill_(.8)
    hybrid=MeasuredPrefix().eval();hybrid.load_state_dict(ref.state_dict())
    x=torch.rand(2,224,224)
    with torch.no_grad():
        expected=ref(x)[0];fields={k:v['intensity'].clone() for k,v in ref.last.items()}
        for prefix in (1,2,3):
            hybrid.measured={k:fields[k] for k in STAGES[:prefix]}
            actual=hybrid(x)[0];assert torch.equal(expected,actual),'Measured-prefix exact graph parity'
    checks=[]
    for prefix in (1,2,3):
        allowed=allowed_keys(hybrid,prefix);hybrid.requires_grad_(False)
        for n,p in hybrid.named_parameters():p.requires_grad_(n in allowed)
        hybrid.measured={k:fields[k] for k in STAGES[:prefix]}
        hybrid.zero_grad(set_to_none=True);p=hybrid(x)[0];F.nll_loss(p.clamp_min(1e-12).log(),torch.tensor([0,1])).backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in hybrid.named_parameters() if n in allowed)
        assert all(v.grad is None for n,v in hybrid.named_parameters() if n not in allowed)
        assert 'raw_router' not in allowed
        if prefix==2:assert all('phase_sets.'+str(k) not in allowed for k in range(4))
        if prefix==3:assert not any(n.startswith('phase_sets.') or n in ('fusion_logits.0','electronic.0.local.weight') for n in allowed)
        checks.append(dict(prefix=prefix,trainable_keys=sorted(allowed),parameters=sum(v.numel() for n,v in hybrid.named_parameters() if n in allowed)))
    print(json.dumps(dict(passed=True,exact_prefix_parity=True,gradient_and_freeze_checks=checks,SDK_opened=False)),flush=True)


def state_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(a):
    """A separate GPU child; exits before the next SDK stage is opened."""
    cfg=read(a.profile);device=a.device;assert a.prefix in (1,2,3)
    if device=='cuda':assert not LOCK.exists(),'GPU training requires SDK release; CPU read-only suffix may run alongside another capture'
    controller=None
    if a.controller_receipt is not None:
        assert a.controller_receipt.resolve().is_relative_to(a.session.resolve())
        controller=read(a.controller_receipt)
        assert controller['pid']==a.controller_pid
        ancestry=python_parent_chain(a.controller_pid);assert ancestry and ancestry[-1]==a.controller_pid,('Not own Python parent lineage',os.getpid(),os.getppid(),a.controller_pid,ancestry)
        assert controller['source_SHA256']==sha(__file__) and Path(controller['session']).resolve()==a.session.resolve()
    if device=='cuda':assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID'] and 'A100' not in torch.cuda.get_device_name(0)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True) if device=='cuda' else ''
    names=windows_process_names() if os.name=='nt' else {}
    for row in active.splitlines():
        parts=row.split(',',2)
        if parts[0].strip()!=cfg['GPU_UUID']:continue
        pid=int(parts[1]);name=names.get(pid,Path(parts[2].strip()).name.lower())
        own_idle_SDK_controller=own_sdk_controller(pid,name,ancestry[-1] if controller is not None else os.getppid(),a.controller_pid if controller is not None else None)
        assert own_idle_SDK_controller or gpu_process_allowed(pid,name,os.getpid(),os.name,cfg['desktop_graphics_process_allowlist']),('GPU occupied; do not kill/share',pid,name)
    torch.set_num_threads(cfg['cpu_threads']);torch.manual_seed(cfg['seed'])
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.70,0)
    data,dm=load_data(a.session);base,_=load_checkpoint(a.checkpoint,a.checkpoint_sha,2,.3,alpha_schema=DIRECT)
    original=state_cpu(base);model=MeasuredPrefix().to(device);model.load_state_dict(original);model.requires_grad_(False)
    allowed=allowed_keys(model,a.prefix)
    for n,p in model.named_parameters():p.requires_grad_(n in allowed)
    assert not a.out.exists();a.out.mkdir();prefix_shas={};datasets={}
    for role,(images,y,ids) in data.items():
        fields={}
        for stage in STAGES[:a.prefix]:
            p=a.session/'capture'/stage/role/'ccd.npy';receipt=read(a.session/'capture'/stage/'report.json')
            assert receipt['sdk_released'] and receipt['data_sha256']==dm['sha256'] and receipt['TRAIN']==1200 and receipt['TEST']==411
            assert sha(p)==receipt['arrays'][role]['sha256'];prefix_shas[role+'/'+stage]=sha(p)
            fields[stage]=np.load(p,mmap_mode='r')
        datasets[role]=dict(x=images,y=torch.from_numpy(y.astype(np.int64)),ids=ids,fields=fields)
    def forward(role,ix,device):
        d=datasets[role];model.measured={k:torch.from_numpy(np.asarray(v[ix]).copy()).to(device) for k,v in d['fields'].items()}
        return model(encode_images(d['x'][ix].to(device)))[0]
    def assess(role,device=device):
        parts=[];d=datasets[role]
        with torch.no_grad():
            for begin in range(0,len(d['ids']),cfg['batch_size']):parts.append(forward(role,slice(begin,begin+cfg['batch_size']),device).cpu().numpy())
        p=np.concatenate(parts);return metrics(d['y'].numpy(),p),p
    kind='measured_all_layers' if a.prefix==3 else 'measured_prefix_simulated_suffix'
    budget=cfg['tail'] if a.prefix==3 else cfg['mask']
    start=time.perf_counter();history=[];baseline,baseline_p=assess('test');best=baseline['accuracy'];selected=0
    np.savez_compressed(a.out/'baseline_TEST_predictions.npz',ids=datasets['test']['ids'],labels=datasets['test']['y'].numpy(),probabilities=baseline_p)
    git=ROOT/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    protocol=dict(config=cfg,prefix=list(STAGES[:a.prefix]),parent_PT_SHA256=a.checkpoint_sha,data_SHA256=dm['sha256'],captured_prefix_SHA256=prefix_shas,allowed_keys=sorted(allowed),source_SHA256=sha(__file__),actual_git=subprocess.check_output([str(git) if git.exists() else 'git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),evaluation_kind=kind,device=device,GPU_used=device=='cuda',SDK_opened=False,TEST_development_selection=True,TEST_gradients=False,VALID=0,control_untouched=True,baseline_TEST_development=baseline,baseline_predictions_SHA256=sha(a.out/'baseline_TEST_predictions.npz'))
    save(a.out/'protocol.json',protocol)
    save(a.out/'progress.json',dict(state='fitting',epoch=0,baseline_TEST_development=baseline,best_TEST_development=best,evaluation_kind=kind,device=device))
    def checkpoint(name,epoch):
        state=state_cpu(model);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed)
        torch.save(dict(model=state,epoch=epoch,protocol=protocol),a.out/(name+'.pt'))
    checkpoint('best',0)
    groups=[]
    for prefix,lr in (('phase_sets.',cfg['phase_lr']),('electronic.',cfg['electronic_lr']),('fusion_logits.',cfg['alpha_lr']),('linear.',cfg['head_lr'])):
        params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:groups.append(dict(params=params,lr=lr,weight_decay=0 if prefix=='fusion_logits.' else cfg['weight_decay']))
    optimizer=torch.optim.AdamW(groups);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,budget['epochs'],eta_min=1e-5)
    for epoch in range(1,budget['epochs']+1):
        model.train();total=0.;order=torch.randperm(1200).numpy()
        for begin in range(0,1200,cfg['batch_size']):
            ix=order[begin:begin+cfg['batch_size']];optimizer.zero_grad(set_to_none=True);p=forward('train',ix,device);logp=p.clamp_min(1e-12).log();y=datasets['train']['y'][ix].to(device)
            smooth=cfg['label_smoothing'];loss=(1-smooth)*F.nll_loss(logp,y)-smooth*logp.mean()
            assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
            with torch.no_grad():
                for k,p in enumerate(model.fusion_logits):
                    if p.requires_grad:p.clamp_(.4001001,.95)
            total+=float(loss.detach())*len(ix)
        schedule.step();model.eval()
        if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==budget['epochs'] or time.perf_counter()-start>=budget['seconds']:
            train,_=assess('train');test,_=assess('test');row=dict(epoch=epoch,TRAIN_loss=total/1200,TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],evaluation_kind=kind,alphas=model.alphas(),elapsed_s=time.perf_counter()-start)
            if test['accuracy']>best:best=test['accuracy'];selected=epoch;checkpoint('best',epoch)
            history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting',latest=row,best_TEST_development=best,selected_epoch=selected));print(json.dumps(row),flush=True)
        if time.perf_counter()-start>=budget['seconds']:break
    checkpoint('last',epoch);model.cpu().eval();results={}
    for name in ('best','last'):
        state=torch.load(a.out/(name+'.pt'),map_location='cpu',weights_only=False)['model'];model.load_state_dict(state,strict=True)
        assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed)
        test,p=assess('test','cpu');train,_=assess('train','cpu');np.savez_compressed(a.out/(name+'_TEST_predictions.npz'),ids=datasets['test']['ids'],labels=datasets['test']['y'].numpy(),probabilities=p)
        results[name]=dict(TRAIN=train,TEST_development=test,PT_SHA256=sha(a.out/(name+'.pt')),prediction_SHA256=sha(a.out/(name+'_TEST_predictions.npz')),strict_CPU=True)
    assert results['best']['TEST_development']['accuracy']==best,'CPU selected-score mismatch'
    save(a.out/'report.json',dict(state='complete',results=results,baseline_TEST_development=baseline,baseline_predictions_SHA256=sha(a.out/'baseline_TEST_predictions.npz'),selected_epoch=selected,actual_epochs=epoch,elapsed_s=time.perf_counter()-start,prefix=list(STAGES[:a.prefix]),evaluation_kind=kind,device=device,GPU_used=device=='cuda',frozen_parameters_unchanged=True,data_sha256=dm['sha256'],SDK_opened=False,TEST_development=True))
    print(json.dumps(dict(state='complete',prefix=a.prefix,selected_epoch=selected,best=best,elapsed_s=time.perf_counter()-start)),flush=True)


@torch.inference_mode()
def acquire_layer(a,stage,checkpoint,checkpoint_sha):
    import cv2
    cfg=read(a.profile);data,dm=load_data(a.session);model,_=load_checkpoint(checkpoint,checkpoint_sha,2,.3,alpha_schema=DIRECT)
    out=a.session/'capture'/stage;assert not out.exists();assert shutil.disk_usage(a.session).free>=12*(1<<30)
    out.mkdir(parents=True);phasepath=out/'phase.bmp';phase_pixels=None;frames=0;start=time.perf_counter();prefix={};arrays={}
    for prior in STAGES[:STAGES.index(stage)]:
        r=read(a.session/'capture'/prior/'report.json');assert r['state']=='complete' and r['sdk_released'] and r['data_sha256']==dm['sha256']
        prefix[prior]={}
        for role in data:
            p=a.session/'capture'/prior/role/'ccd.npy';assert sha(p)==r['arrays'][role]['sha256'];prefix[prior][role]=np.load(p,mmap_mode='r')
    parent=read(a.parent/'release.json')
    for key,path in (('coordinates_sha256',a.coordinates),('phase_lut_sha256',a.phase_lut),('machine_config_sha256',a.machine_config),('response_lut_sha256',a.response_lut)):
        assert sha(path)==parent[key]
    response=read(a.response_lut);drive=np.asarray(response['drive'],np.uint8);assert response['passed'] and drive.shape==(256,)
    warp=cv2.getPerspectiveTransform(np.float32(corners(a.coordinates)),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    save(out/'contract.json',dict(stage=stage,checkpoint_SHA256=checkpoint_sha,data_SHA256=dm['sha256'],TRAIN=1200,TEST=411,VALID=0,exposure_us=2000,gain='Gain_X4',wait_ms=240,native_active=1016,phase_transform='hv_inverse',camera_orientation='flip_v',response='same_session_field_inverse',source_SHA256=sha(__file__),configuration_SHA256=sha(a.profile),prefix_reports={k:sha(a.session/'capture'/k/'report.json') for k in prefix},parent_hardware_release_SHA256=sha(a.parent/'release.json')))
    a.out=out;a.exposure_us=2000
    with LOCK.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),out=str(out),stage=stage,checkpoint=checkpoint_sha)))
    try:
        with Bench(a) as bench:
            for role,(images,y,ids) in data.items():
                folder=out/role;folder.mkdir();ccd=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(len(y),478,478));probabilities=[];features=[]
                for begin in range(0,len(y),8):
                    end=min(begin+8,len(y));cached={k:np.asarray(v[role][begin:end]).copy() for k,v in prefix.items()}
                    def physical(current,amplitudes,phase,sample_ids):
                        nonlocal phase_pixels,frames
                        if phase_pixels is None:
                            phase_pixels=phase.copy();Image.fromarray(phase_native(phase)).save(phasepath);bench.stage(phasepath)
                        assert np.array_equal(phase_pixels,phase)
                        for index,amplitude in enumerate(amplitudes,begin):
                            gray=np.rint(amplitude.clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amplitude))<=1e-7
                            p=folder/f'{index:06d}';canonical=p.with_name(p.name+'_input.png');Image.fromarray(gray).save(canonical)
                            bmp=p.with_name(p.name+'_transient.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(bmp)
                            raw,meta=bench.capture(bmp);assert raw.shape==(1080,1920)
                            crop=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                            receipt=dict(stage=stage,role=role,index=index,sample_id=str(ids[index]),frame_id=meta['frame_id'],captured_at_ns=time.time_ns(),mean=float(crop.mean()),p99=float(np.percentile(crop,99)),maximum=int(crop.max()),saturation_fraction=float((crop==255).mean()),capture_total_ms=meta['capture_total_ms'],input_SHA256=sha(bmp),canonical_SHA256=sha(canonical),phase_SHA256=sha(phasepath),checkpoint_SHA256=checkpoint_sha)
                            Image.fromarray(raw).save(p.with_name(p.name+'_raw.png'));Image.fromarray(crop).save(p.with_suffix('.png'));save(p.with_suffix('.json'),receipt)
                            ccd[index]=crop;ccd.flush();frames+=1;bmp.unlink()
                            save(a.session/'progress.json',dict(state='capturing',stage=stage,role=role,completed=index+1,total=len(y),stage_frames=frames,elapsed_s=time.perf_counter()-start,latest=receipt))
                            if receipt['p99']<15 or receipt['saturation_fraction']>.15:raise RuntimeError('Optical signal guard; retain evidence; no automatic retry')
                            if frames==24:save(out/'initial24_signal_gate.json',dict(passed=True,frames=24,part_of_full_capture=True,extra_frames=0))
                        return np.asarray(ccd[begin:end]).copy()
                    result=layer_forward(model,images[begin:end],ids[begin:end],stage,cached,physical)
                    if result is not None:probabilities.append(result['probabilities']);features.append(result['features'])
                arrays[role]=dict(sha256=sha(folder/'ccd.npy'),samples=len(y))
                if probabilities:
                    p=np.concatenate(probabilities);np.savez_compressed(folder/'predictions.npz',ids=ids,labels=y,probabilities=p,features=np.concatenate(features));save(folder/'metrics.json',dict(metrics(y,p),physical=True,samples=len(y),TEST_is_development=role=='test'))
        assert frames==1611
        save(out/'report.json',dict(state='complete',stage=stage,physical_ccd=frames,TRAIN=1200,TEST=411,VALID=0,sdk_released=True,arrays=arrays,data_sha256=dm['sha256'],checkpoint_sha256=checkpoint_sha,phase_sha256=sha(phasepath),elapsed_s=time.perf_counter()-start))
    finally:LOCK.unlink()


def run(a):
    assert not (a.session/'pipeline_started.json').exists() and not LOCK.exists()
    cfg=read(a.profile);torch.set_num_threads(cfg['cpu_threads']);selftest();data,dm=load_data(a.session)
    assert sha(a.session/'assets/original.pt')==ORIGINAL
    for name in ('residual_tail5150_lab_processnames_20261018/report.json','resume_chain_20261018/report.json'):
        assert read(a.parent/name)['state']=='complete'
    save(a.session/'pipeline_started.json',dict(pid=os.getpid(),session=str(a.session.resolve()),time_ns=time.time_ns(),TRAIN=1200,TEST=411,VALID=0,TRAIN_each_class=60,planned_physical_frames=4833,source_SHA256=sha(__file__),profile_SHA256=sha(a.profile),original_PT_SHA256=ORIGINAL,control_frozen=True,recurring_monitor=False,order='router TRAIN/TEST -> suffix fit -> expert TRAIN/TEST -> suffix fit -> global TRAIN/TEST -> original terminal fit',TEST_selects_development=True))
    checkpoint=a.session/'assets/original.pt';checkpoint_sha=ORIGINAL;stages=[]
    for i,stage in enumerate(STAGES,1):
        acquire_layer(a,stage,checkpoint,checkpoint_sha)
        out=a.session/'fits'/stage;out.parent.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['GPU_UUID'])
        cli=[sys.executable,str(Path(__file__).resolve()),'--mode','fit','--session',str(a.session),'--parent',str(a.parent),'--profile',str(a.profile),'--prefix',str(i),'--checkpoint',str(checkpoint),'--checkpoint-sha',checkpoint_sha,'--out',str(out),'--controller-pid',str(os.getpid()),'--controller-receipt',str(a.session/'pipeline_started.json')]
        save(a.session/'progress.json',dict(state='suffix_fitting',after_stage=stage,planned_epochs=(cfg['tail'] if i==3 else cfg['mask'])['epochs']))
        with out.with_suffix('.log').open('x',encoding='utf-8',buffering=1) as log:
            result=subprocess.run(cli,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,('Suffix fit failed; stop next capture',stage,result.returncode)
        report=read(out/'report.json');assert report['state']=='complete' and report['frozen_parameters_unchanged']
        checkpoint=out/'best.pt';checkpoint_sha=report['results']['best']['PT_SHA256'];assert sha(checkpoint)==checkpoint_sha
        stages.append(dict(stage=stage,capture=read(a.session/'capture'/stage/'report.json'),fit=report,selected_PT_SHA256=checkpoint_sha))
    score=stages[-1]['fit']['results']['best']['TEST_development']['accuracy']
    save(a.session/'report.json',dict(state='complete',stages=stages,physical_ccd=4833,TRAIN=1200,TEST=411,VALID=0,final_PT=str(checkpoint),final_PT_SHA256=checkpoint_sha,TEST_development_accuracy=score,simulation_reference=329/411,gap_pp=100*(329/411-score),target_within_1pp=abs(329/411-score)<=.01,sdk_released=not LOCK.exists(),GPU_child_exited=True,control_untouched=True,source_SHA256=sha(__file__)))
    print(json.dumps(dict(state='complete',TEST_development_accuracy=score,physical_ccd=4833)),flush=True)


def resume_after_router(a):
    """Explicit engineering recovery, preserving the exact1611 valid router CCDs.

    No old failure/report/log is overwritten and no failed optimizer is replayed:
    prior child rejected the controller PID before out/model/optimizer creation.
    """
    assert not LOCK.exists() and not a.out.exists()
    failure=read(a.session/'failure.json');assert 'Suffix fit failed' in failure['error']
    old=read(a.session/'pipeline_started.json');log=(a.session/'fits/router.log').read_text()
    assert "'GPU occupied; do not kill/share', "+str(old['pid']) in log
    assert not (a.session/'fits/router').exists() and not (a.session/'capture/0').exists() and not (a.session/'capture/1').exists()
    router=a.session/'capture/router';report=read(router/'report.json');assert report['state']=='complete' and report['physical_ccd']==1611 and report['sdk_released'] and report['checkpoint_sha256']==ORIGINAL
    for role in ('train','test'):assert sha(router/role/'ccd.npy')==report['arrays'][role]['sha256']
    a.out.mkdir();state=a.out;cfg=read(a.profile);torch.set_num_threads(cfg['cpu_threads'])
    receipt=state/'controller.json';save(receipt,dict(pid=os.getpid(),session=str(a.session.resolve()),source_SHA256=sha(__file__),time_ns=time.time_ns(),reason='Own SDK controller graphics context rejected before optimizer; exact-parent identity fix',old_failure_SHA256=sha(a.session/'failure.json'),old_router_report_SHA256=sha(router/'report.json'),reused_valid_router_frames=1611,recaptured_router_frames=0))
    checkpoint=a.session/'assets/original.pt';digest=ORIGINAL;rows=[]
    for i,stage in enumerate(STAGES,1):
        if stage!='router':acquire_layer(a,stage,checkpoint,digest)
        target=state/'fits'/stage;target.parent.mkdir(exist_ok=True)
        cli=[str(Path(sys.executable).with_name('python.exe')),str(Path(__file__).resolve()),'--mode','fit','--session',str(a.session),'--parent',str(a.parent),'--profile',str(a.profile),'--prefix',str(i),'--checkpoint',str(checkpoint),'--checkpoint-sha',digest,'--out',str(target),'--controller-pid',str(os.getpid()),'--controller-receipt',str(receipt)]
        save(state/'progress.json',dict(state='suffix_fitting',after_stage=stage,prefix=i))
        with target.with_suffix('.log').open('x',encoding='utf-8') as stream:
            child=subprocess.run(cli,env=dict(os.environ,CUDA_VISIBLE_DEVICES=cfg['GPU_UUID']),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        assert child.returncode==0,('Recovered suffix failed; no next capture',stage,child.returncode)
        fitted=read(target/'report.json');assert fitted['state']=='complete' and fitted['frozen_parameters_unchanged']
        checkpoint=target/'best.pt';digest=fitted['results']['best']['PT_SHA256'];assert sha(checkpoint)==digest
        rows.append(dict(stage=stage,fit=fitted,capture=read(a.session/'capture'/stage/'report.json'),selected_PT_SHA256=digest))
    score=rows[-1]['fit']['results']['best']['TEST_development']['accuracy']
    assert sha(router/'report.json')==read(receipt)['old_router_report_SHA256']
    save(state/'report.json',dict(state='complete',stages=rows,physical_ccd=4833,new_CCD_after_resume=3222,reused_valid_router_frames=1611,TRAIN=1200,TEST=411,VALID=0,final_PT=str(checkpoint),final_PT_SHA256=digest,TEST_development_accuracy=score,gap_pp=100*(329/411-score),target_within_1pp=abs(329/411-score)<=.01,sdk_released=not LOCK.exists(),GPU_child_exited=True,all_old_failure_and_router_data_preserved=True,source_SHA256=sha(__file__)))


def resume_after_expert(a):
    """New explicit human authorization: preserve3222 prefix frames; capture global.

    Use the completed CPU-trained expert suffix, not the stopped old controller.
    Only the new downstream global is captured; originals/failures stay intact.
    """
    assert a.device=='cpu' and not LOCK.exists() and not a.out.exists()
    assert not (a.session/'capture/1').exists(),'Existing global data must be diagnosed, never overwritten'
    cfg=read(a.profile);torch.set_num_threads(cfg['cpu_threads']);datasets,dm=load_data(a.session)
    handoff=read(a.session/'expert_handoff_receipt_20261018.json');assert handoff['residual_controller_stopped'] and handoff['global_not_started']
    parent=a.session/'expert_suffix_cpu_20261018';fit_report=read(parent/'report.json');checkpoint=parent/'best.pt';digest=fit_report['results']['best']['PT_SHA256']
    assert fit_report['state']=='complete' and fit_report['frozen_parameters_unchanged'] and fit_report['results']['best']['strict_CPU'] and sha(checkpoint)==digest
    protocol=read(parent/'protocol.json');before=torch.load(a.session/'resume_parent_lineage_20261018/fits/router/best.pt',map_location='cpu',weights_only=False)['model'];after=torch.load(checkpoint,map_location='cpu',weights_only=False)['model'];allowed=set(protocol['allowed_keys'])
    assert all(torch.equal(v,after[k]) for k,v in before.items() if k not in allowed)
    prefix_reports={}
    for stage in STAGES[:2]:
        root=a.session/'capture'/stage;r=read(root/'report.json');assert r['state']=='complete' and r['sdk_released'] and r['physical_ccd']==1611 and r['data_sha256']==dm['sha256']
        assert sha(root/'phase.bmp')==r['phase_sha256']
        for role in datasets:assert sha(root/role/'ccd.npy')==r['arrays'][role]['sha256']
        prefix_reports[stage]=sha(root/'report.json')
    a.out.mkdir();state=a.out;receipt=state/'controller.json'
    save(receipt,dict(pid=os.getpid(),session=str(a.session.resolve()),source_SHA256=sha(__file__),profile_SHA256=sha(a.profile),time_ns=time.time_ns(),human_authorization='2026-10-18 continue L2 residual capture and adaptation; overrides earlier global deferral',selected_expert_suffix_PT_SHA256=digest,reused_valid_prefix_CCD=3222,new_planned_CCD=1611,prefix_reports_SHA256=prefix_reports))
    acquire_layer(a,'1',checkpoint,digest)
    target=state/'final_CPU_fit';cli=[str(Path(sys.executable).with_name('python.exe')),str(Path(__file__).resolve()),'--mode','fit','--device','cpu','--session',str(a.session),'--parent',str(a.parent),'--profile',str(a.profile),'--prefix','3','--checkpoint',str(checkpoint),'--checkpoint-sha',digest,'--out',str(target),'--controller-pid',str(os.getpid()),'--controller-receipt',str(receipt)]
    save(state/'progress.json',dict(state='final_CPU_fitting',global_capture_complete=True,new_CCD=1611))
    with target.with_suffix('.log').open('x',encoding='utf-8') as stream:
        child=subprocess.run(cli,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert child.returncode==0,('Final CPU fit failed; no automatic restart',child.returncode)
    result=read(target/'report.json');assert result['state']=='complete' and result['evaluation_kind']=='measured_all_layers' and result['frozen_parameters_unchanged']
    assert all(sha(a.session/'capture'/k/'report.json')==v for k,v in prefix_reports.items())
    score=result['results']['best']['TEST_development']['accuracy']
    save(state/'report.json',dict(state='complete',new_CCD=1611,total_physical_CCD=4833,reused_prefix_CCD=3222,TRAIN=1200,TEST=411,VALID=0,baseline_full_physical_TEST=result['baseline_TEST_development'],final=result,TEST_development_accuracy=score,gap_pp=100*(329/411-score),target_within_1pp=abs(329/411-score)<=.01,sdk_released=not LOCK.exists(),CPU_child_exited=True,prefix_reports_unchanged=True,source_SHA256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('prepare','selftest','run','fit','resume-after-router','resume-after-expert'),required=True)
    for n in ('session','parent','profile','out','checkpoint','controller-receipt','coordinates','machine-config','phase-lut','phase-sdk','amplitude-sdk','response-lut'):p.add_argument('--'+n,type=Path)
    p.add_argument('--checkpoint-sha');p.add_argument('--prefix',type=int);p.add_argument('--controller-pid',type=int);p.add_argument('--device',choices=('cuda','cpu'),default='cuda');a=p.parse_args()
    if sys.stdout is None:
        log=a.session/'pipeline.log' if a.mode=='run' else a.out.with_suffix('.log');sys.stdout=log.open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
    failure_folder=a.out if a.mode in ('fit','resume-after-router','resume-after-expert') else a.session
    try:
        selftest() if a.mode=='selftest' else prepare(a) if a.mode=='prepare' else fit(a) if a.mode=='fit' else resume_after_router(a) if a.mode=='resume-after-router' else resume_after_expert(a) if a.mode=='resume-after-expert' else run(a)
    except BaseException as e:
        folder=failure_folder
        if folder is not None and folder.exists():save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,source_SHA256=sha(__file__)))
        raise
