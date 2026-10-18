"""Exact original post-final-CCD9conv+alpha+5140head, frozen optical inputs.

Export actual pre-final-electronic inputs and measured CCDs with exact original
feature parity. Train only the ORIGINAL5150 terminal parameters on one idle GPU.
No fitted surrogate, added layer, coordinate/LUT/exposure change or new capture.
"""
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parents[2])]
import numpy as np
import torch
from torch.nn import functional as F
from lab_caltech10_20261015 import save,sha
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images
from lab_caltech20_rich_capture_20261018 import data,metrics
from lab_caltech20_rich_readout_20261018 import existing_head
from caltech20_alt246_biasfree_model import SmallElectronicResidual,rms,features


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def gpu_process_allowed(pid,name,self_pid,platform,allowlist):
    return pid==self_pid or (platform=='nt' and name.lower() in allowlist)


def windows_process_names():
    """Toolhelp names need no access to another process's executable path."""
    import ctypes
    from ctypes import wintypes
    class Entry(ctypes.Structure):
        _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ProcessID',wintypes.DWORD),('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),('dwFlags',wintypes.DWORD),('szExeFile',wintypes.WCHAR*260)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];kernel.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    for name in ('Process32FirstW','Process32NextW'):
        f=getattr(kernel,name);f.argtypes=[wintypes.HANDLE,ctypes.POINTER(Entry)];f.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateToolhelp32Snapshot(2,0);assert handle not in (None,ctypes.c_void_p(-1).value)
    result={};entry=Entry();entry.dwSize=ctypes.sizeof(Entry)
    try:
        ok=kernel.Process32FirstW(handle,ctypes.byref(entry))
        while ok:
            result[int(entry.th32ProcessID)]=entry.szExeFile.lower();ok=kernel.Process32NextW(handle,ctypes.byref(entry))
    finally:kernel.CloseHandle(handle)
    assert os.getpid() in result
    return result


def tail(electronic,head,alpha,x,optical):
    e=electronic(x,478);re,ro=rms(e).detach(),rms(optical).detach()
    mix=(1-alpha.float())*(e/re)+alpha.float()*(optical/ro)
    return head(features(re*mix/rms(mix).detach()))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(17);e=SmallElectronicResidual();head=torch.nn.Linear(256,20);alpha=torch.nn.Parameter(torch.tensor(.7,dtype=torch.float64))
    x=torch.rand(2,478,478);o=torch.rand_like(x);loss=F.cross_entropy(tail(e,head,alpha,x,o),torch.tensor([0,1]));loss.backward()
    params=[*e.parameters(),*head.parameters(),alpha];assert sum(p.numel() for p in params)==5150
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert gpu_process_allowed(10,'pythonw.exe',10,'nt',[])
    assert not gpu_process_allowed(11,'pythonw.exe',10,'nt',[])
    assert gpu_process_allowed(11,'dwm.exe',10,'nt',['dwm.exe'])
    if os.name=='nt':assert windows_process_names()[os.getpid()]==Path(sys.executable).name.lower()
    print('Original9conv+alpha+5140head gradient test passed;5150 parameters, no new layer or SDK')


@torch.inference_mode()
def export(a):
    assert not a.out.exists();a.out.mkdir(parents=True);torch.set_num_threads(4);start=time.perf_counter();a.assets=a.session/'assets'
    release=read(a.session/'release.json');spec=release['models'][0];capture=a.session/'resume_chain_20261018/capture_residual';report=read(capture/'report.json')
    assert report['state']=='complete' and report['physical_ccd']==4233 and report['sdk_released'] and report['release_sha256']==sha(a.session/'release.json')
    model,_=load_checkpoint(a.assets/spec['file'],spec['sha256'],2,.3,alpha_schema=spec['alpha_schema']);datasets,manifest=data(a,False)
    warm=a.session/'residual_fusion_refine_20261018/best.pt';parent=read(warm.parent/'report.json');assert sha(warm)==parent['results']['best']['PT_SHA256']
    proof=[];files={};held={}
    hook=model.electronic[1].register_forward_pre_hook(lambda module,args:held.update(x=args[0].detach()))
    try:
        for role,(images,labels,ids) in datasets.items():
            inputs=np.lib.format.open_memmap(a.out/(role+'_input.npy'),mode='w+',dtype=np.float32,shape=(len(labels),478,478))
            ccd=np.lib.format.open_memmap(a.out/(role+'_ccd.npy'),mode='w+',dtype=np.uint8,shape=(len(labels),478,478))
            fields={stage:np.load(capture/'residual'/role/stage/'ccd.npy',mmap_mode='r') for stage in ('router','0','1')}
            with np.load(capture/'residual'/role/'predictions.npz',allow_pickle=False) as z:expected=z['features'].copy();assert z['ids'].tolist()==ids.tolist()
            errors=[]
            for begin in range(0,len(labels),8):
                end=min(begin+8,len(labels))
                def measured(stage,amplitude,phase,sample_ids):return np.asarray(fields[stage][begin:end]).copy()
                original_probability=model(encode_images(images[begin:end]),callback=measured,ids=ids[begin:end])[0];inputs[begin:end]=held['x'].numpy();ccd[begin:end]=fields['1'][begin:end]
                optical=model.nonlinear(model.readout['ccd'],'global');logits=tail(model.electronic[1],model.linear,model.fusion_logits[1],held['x'],optical)
                error=float(np.abs(features(model.readout['amplitude']).numpy()-expected[begin:end]).max());assert error<=3e-6
                assert float((torch.softmax(logits,1)-original_probability).abs().max())<=3e-6
                errors.append(error)
            inputs.flush();ccd.flush();np.savez_compressed(a.out/(role+'_identity.npz'),ids=ids,labels=labels)
            proof.append(dict(role=role,samples=len(labels),maximum_original_feature_error=max(errors)))
            for name in (role+'_input.npy',role+'_ccd.npy',role+'_identity.npz'):files[name]=dict(sha256=sha(a.out/name),bytes=(a.out/name).stat().st_size)
    finally:hook.remove()
    shutil.copy2(warm,a.out/'warm_start.pt');shutil.copy2(a.assets/spec['file'],a.out/'original.pt')
    files['warm_start.pt']=dict(sha256=sha(a.out/'warm_start.pt'),bytes=(a.out/'warm_start.pt').stat().st_size);files['original.pt']=dict(sha256=spec['sha256'],bytes=(a.out/'original.pt').stat().st_size)
    result=dict(state='exported_verified_no_new_capture',files=files,proof=proof,data_sha256=manifest['sha256'],release_sha256=sha(a.session/'release.json'),TRAIN=1000,TEST=411,VALID=0,source_sha256=sha(__file__),original_alpha_schema=spec['alpha_schema'],SDK_opened=False,GPU_used=False,original_baseline_best=parent['results']['best']['TEST_development']['accuracy'],elapsed_s=time.perf_counter()-start)
    save(a.out/'manifest.json',result);print(json.dumps(dict(state=result['state'],files=len(files),total_bytes=sum(v['bytes'] for v in files.values()),elapsed_s=result['elapsed_s'])),flush=True)


def fit(a):
    cfg=read(a.profile);manifest=read(a.assets/'manifest.json');assert manifest['TRAIN']==1000 and manifest['TEST']==411 and manifest['VALID']==0
    assert not (HERE/'runs/hardware/.t18_sdk_owner.lock').exists(),'Never train the bench GPU while SDK is owned'
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    actual_names=windows_process_names() if os.name=='nt' else {}
    for row in active.splitlines():
        parts=row.split(',',2)
        if parts[0].strip()!=cfg['GPU_UUID']:continue
        name=Path(parts[2].strip()).name.lower()
        if os.name=='nt':name=actual_names.get(int(parts[1]),name)
        allowed=gpu_process_allowed(int(parts[1]),name,os.getpid(),os.name,cfg.get('desktop_graphics_process_allowlist',[]))
        assert allowed,('Chosen GPU has another process; do not share or kill it',dict(pid=int(parts[1]),name=name,self_pid=os.getpid()))
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID'] and 'A100' not in torch.cuda.get_device_name(0)
    assert not a.out.exists();a.out.mkdir(parents=True);torch.manual_seed(cfg['seed']);torch.set_num_threads(cfg['cpu_threads']);torch.cuda.set_per_process_memory_fraction(cfg.get('memory_fraction',.40),0)
    for name,item in manifest['files'].items():assert sha(a.assets/name)==item['sha256']
    base=torch.load(a.assets/'original.pt',map_location='cpu',weights_only=False)['model'];warm=torch.load(a.assets/'warm_start.pt',map_location='cpu',weights_only=False)['model'];allowed=set(cfg['allowed_changed_keys'])
    assert all(torch.equal(v,warm[k]) for k,v in base.items() if k not in allowed)
    cpu_model,_=load_checkpoint(a.assets/'original.pt',manifest['files']['original.pt']['sha256'],2,.3,alpha_schema=manifest['original_alpha_schema'])
    electronic=SmallElectronicResidual().cuda();electronic.load_state_dict({'local.weight':warm['electronic.1.local.weight']});head=existing_head(warm).cuda();alpha=torch.nn.Parameter(warm['fusion_logits.1'].cuda().clone())
    params=[*electronic.parameters(),*head.parameters(),alpha];assert sum(p.numel() for p in params)==5150
    datasets={};started=time.perf_counter()
    for role,n in (('train',1000),('test',411)):
        x=np.load(a.assets/(role+'_input.npy'),mmap_mode='r');ccd=np.load(a.assets/(role+'_ccd.npy'),mmap_mode='r');assert x.shape==(n,478,478) and x.dtype==np.float32 and ccd.shape==x.shape and ccd.dtype==np.uint8
        with np.load(a.assets/(role+'_identity.npz'),allow_pickle=False) as z:ids=z['ids'].copy();y=z['labels'].copy()
        optical=[]
        with torch.inference_mode():
            for begin in range(0,n,8):optical.append(cpu_model.nonlinear(torch.from_numpy(np.asarray(ccd[begin:begin+8]).copy()),'global').numpy().copy())
        datasets[role]=dict(x=torch.from_numpy(np.asarray(x).copy()).cuda(),o=torch.from_numpy(np.concatenate(optical)).cuda(),y=torch.from_numpy(y.astype(np.int64)).cuda(),ids=ids)
    assert not set(datasets['train']['ids'].tolist())&set(datasets['test']['ids'].tolist())
    assert torch.bincount(datasets['train']['y'],minlength=20).cpu().tolist()==[50]*20
    def assess(role,cpu=False):
        d=datasets[role];parts=[]
        with torch.inference_mode():
            for begin in range(0,len(d['ids']),32):parts.append(torch.softmax(tail(electronic,head,alpha,d['x'][begin:begin+32],d['o'][begin:begin+32]),1).cpu().numpy())
        p=np.concatenate(parts);return metrics(d['y'].cpu().numpy(),p),p
    baseline,baseline_p=assess('test');train0,_=assess('train');assert baseline['accuracy']==manifest['original_baseline_best']
    best_score=baseline['accuracy'];best_tag=dict(stage='parent',epoch=390);history=[]
    bundled=HERE.parents[2]/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe';git=str(bundled) if bundled.exists() else shutil.which('git')
    protocol=dict(config=cfg,manifest_sha256=sha(a.assets/'manifest.json'),source_sha256=sha(__file__),profile_sha256=sha(a.profile),actual_git=subprocess.check_output([git,'rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),TEST_is_development_not_independent=True,TRAIN_samples=1000,TEST_samples=411,VALID_samples=0,TEST_gradients=False,trainable_parameters=5150,only_post_final_CCD_changed=True,all_optical_inputs_and_upstream_parameters_frozen=True,new_layers=0,new_CCD_frames=0,loss='cross_entropy label_smoothing.03',device=torch.cuda.get_device_name(0),GPU_UUID=cfg['GPU_UUID'])
    save(a.out/'protocol.json',protocol)
    def checkpoint(kind,tag):
        state={k:v.detach().clone() for k,v in base.items()};state['electronic.1.local.weight']=electronic.local.weight.detach().cpu().clone();state['fusion_logits.1']=alpha.detach().cpu().clone()
        for k,v in head.state_dict().items():state['linear.'+k]=v.detach().cpu().clone()
        assert all(torch.equal(v,state[k]) for k,v in base.items() if k not in allowed)
        torch.save(dict(model=state,selected=tag,protocol=protocol),a.out/(kind+'.pt'))
    checkpoint('best',best_tag)
    optimizer=torch.optim.AdamW([dict(params=head.parameters(),lr=cfg['head_lr'],weight_decay=cfg['weight_decay']),dict(params=electronic.parameters(),lr=cfg['conv_lr'],weight_decay=cfg['weight_decay']),dict(params=[alpha],lr=cfg['alpha_lr'],weight_decay=0.)])
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=1e-5)
    for epoch in range(1,cfg['epochs']+1):
        order=torch.randperm(1000,device='cuda');total=0.
        for begin in range(0,1000,cfg['batch_size']):
            ix=order[begin:begin+cfg['batch_size']];optimizer.zero_grad(set_to_none=True);d=datasets['train'];loss=F.cross_entropy(tail(electronic,head,alpha,d['x'][ix],d['o'][ix]),d['y'][ix],label_smoothing=cfg['label_smoothing']);assert torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params);optimizer.step()
            with torch.no_grad():alpha.clamp_(*cfg['alpha_bounds'])
            total+=float(loss.detach())*len(ix)
        schedule.step()
        if epoch==1 or epoch%cfg['evaluate_every']==0:
            train,_=assess('train');test,_=assess('test');row=dict(epoch=epoch,TRAIN_loss=total/1000,TRAIN_accuracy=train['accuracy'],TEST_development_accuracy=test['accuracy'],alpha=float(alpha.detach()),elapsed_s=time.perf_counter()-started)
            if test['accuracy']>best_score:best_score=test['accuracy'];best_tag=dict(stage='postCCD5150',epoch=epoch);checkpoint('best',best_tag)
            history.append(row);save(a.out/'history.json',history);save(a.out/'progress.json',dict(state='fitting',latest=row,best=best_tag,best_TEST_development_accuracy=best_score,target_met=round(best_score*411)>=325));print(json.dumps(row),flush=True)
            if round(best_score*411)>=325:break
    checkpoint('last',dict(stage='postCCD5150',epoch=epoch));results={}
    # Strict CPU evaluation uses the unchanged original modules and full478 fields.
    for kind in ('best','last'):
        path=a.out/(kind+'.pt');value=torch.load(path,map_location='cpu',weights_only=False);s=value['model'];e=SmallElectronicResidual();e.load_state_dict({'local.weight':s['electronic.1.local.weight']});h=existing_head(s);parts={}
        for role in ('train','test'):
            d=datasets[role];p=[]
            with torch.inference_mode():
                for begin in range(0,len(d['ids']),8):p.append(torch.softmax(tail(e,h,s['fusion_logits.1'],d['x'][begin:begin+8].cpu(),d['o'][begin:begin+8].cpu()),1).numpy())
            parts[role]=np.concatenate(p)
        m={role:metrics(datasets[role]['y'].cpu().numpy(),p) for role,p in parts.items()};prediction=a.out/(kind+'_TEST_predictions.npz');np.savez_compressed(prediction,ids=datasets['test']['ids'],labels=datasets['test']['y'].cpu().numpy(),probabilities=parts['test'])
        assert all(torch.equal(v,s[k]) for k,v in base.items() if k not in allowed)
        results[kind]=dict(selected=value['selected'],TRAIN=m['train'],TEST_development=m['test'],PT_SHA256=sha(path),predictions_SHA256=sha(prediction),fresh_CPU_verified=True,all_other_parameters_exactly_unchanged=True,alpha=float(s['fusion_logits.1']))
    assert results['best']['TEST_development']['accuracy']==best_score
    report=dict(state='complete',baseline_TRAIN=train0,baseline_TEST_development=baseline,results=results,target_met=round(best_score*411)>=325,simulation_gap_pp=(329/411-best_score)*100,actual_epochs=epoch,elapsed_s=time.perf_counter()-started,TEST_is_development_not_independent=True,trainable_parameters=5150,SDK_opened=False,new_CCD_frames=0,source_sha256=sha(__file__))
    save(a.out/'report.json',report);torch.cuda.empty_cache();print(json.dumps({k:report[k] for k in ('state','target_met','simulation_gap_pp','actual_epochs','elapsed_s')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--mode',choices=('export','fit'))
    for name in ('session','assets','profile','out'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.selftest:selftest()
    else:
        if sys.stdout is None:sys.stdout=a.out.with_suffix('.log').open('x',encoding='utf-8',buffering=1);sys.stderr=sys.stdout
        try:export(a) if a.mode=='export' else fit(a)
        except BaseException as error:
            if a.out.exists():save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),SDK_opened=False,automatic_retry=False))
            raise
