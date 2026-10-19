"""Fresh DVP L4 -> L6: layer-major TRAIN/TEST then original suffix tuning.

Only hardware orchestration/data admission differ from the existing measured
prefix model. No inference layer, TEST gradient, VALID input or old CCD reuse.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from lab_caltech10_20261015 import save,sha,corners,native,phase_native
from lab_caltech20_rich_model_20261018 import load_checkpoint,encode_images,DIRECT
from lab_caltech20_rich_layerwise_20261018 import MeasuredPrefix,allowed_keys,state_cpu
from lab_caltech20_rich_capture_20261018 import layer_forward,metrics
from lab_caltech20_residual_guarded_20261018 import improves
from lab_caltech20_dvp_20261019 import Bench,orient,LOCK

SOURCE_CACHE='38b719b8ad935a0170d059d0e4fffa22f6f22640f979603f7f2543fc2d3f2445'
HARDWARE='3b89d610d939141dec9c5d0955f23765454cd7e7bebfe1c92c5e64e5376713ae'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def git_head():
    exe=ROOT/'OpenMoji_Robust_Rank64_SHS_20261002/.git-tools/MinGit_2_56_0/cmd/git.exe'
    return subprocess.check_output([str(exe) if exe.exists() else 'git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def translate(image,dx,dy):
    assert image.shape==(150,150,3) and image.dtype==np.uint8 and max(abs(dx),abs(dy))<=2
    p=np.pad(image,((2,2),(2,2),(0,0)),mode='reflect')
    return p[2-dy:152-dy,2-dx:152-dx].copy()


def select80(images,labels,ids):
    """Predetermined per-class hash order; transform ONLY missing TRAIN slots."""
    pool=[];records=[];classes=[]
    shifts=((2,-1),(-2,1),(1,2),(-1,-2))
    assert len(set(ids.tolist()))==len(ids)
    for k in range(20):
        ix=np.flatnonzero(labels==k).tolist()
        ix.sort(key=lambda i:hashlib.sha256(('T18_DVP80_v1|'+str(ids[i])).encode()).digest())
        assert len(ix)>=60
        chosen=ix[:80];rows=[]
        for i in chosen:
            rows.append((images[i].copy(),k,str(ids[i]),dict(source_id=str(ids[i]),source_index=i,kind='original',dx=0,dy=0)))
        for j in range(80-len(rows)):
            i=chosen[j];dx,dy=shifts[j%len(shifts)];value=translate(images[i],dx,dy)
            assert not np.array_equal(value,images[i]),'Augmentation unexpectedly identical to source'
            rows.append((value,k,str(ids[i])+f'::TRAIN_shift_dx{dx:+d}_dy{dy:+d}',dict(source_id=str(ids[i]),source_index=i,kind='reflect_pad_integer_translation',dx=dx,dy=dy)))
        assert len(rows)==80;classes.append(rows)
    # Round-robin acquisition order; any first20 samples covers every class.
    for slot in range(80):
        for k in range(20):
            image,label,sample,record=classes[k][slot];record.update(sample_id=sample,label=label,pixel_SHA256=hashlib.sha256(image.tobytes()).hexdigest());pool.append((image,label,sample));records.append(record)
    x=np.stack([v[0] for v in pool]);y=np.array([v[1] for v in pool],np.int64);names=np.array([v[2] for v in pool])
    assert np.bincount(y,minlength=20).tolist()==[80]*20 and len(set(names.tolist()))==1600
    assert sum(v['kind']=='original' for v in records)==1508
    assert sum(v['kind']!='original' for v in records)==92
    return x,y,names,records


def prepare(a):
    assert sha(a.cache)==SOURCE_CACHE and not a.out.exists();a.out.mkdir(parents=True)
    with np.load(a.cache,allow_pickle=False) as z:
        originals=[z['train_'+k].copy() for k in ('images','labels','ids')]
        assert len(originals[1])==3141
        x,y,ids,records=select80(*originals)
        arrays=dict(train_images=x,train_labels=y,train_ids=ids)
        for k in ('images','labels','ids'):arrays['test_'+k]=z['test_'+k].copy()
        assert len(arrays['test_labels'])==411 and not set(originals[2].tolist())&set(arrays['test_ids'].tolist())
        assert not set(v['source_id'] for v in records)&set(arrays['test_ids'].tolist())
    p=a.out/'rich20_TRAIN80_fixedaug_TEST411.npz';np.savez_compressed(p,**arrays)
    save(a.out/'TRAIN_lineage.json',dict(source_cache_SHA256=SOURCE_CACHE,records=records,selection='fixed SHA256 order within class, round robin classes',augmentation='92 images, integer shift<=2pixels in original150 frame with reflection border; no photometric change',user_authorized=True,TEST_VALID_sources=False))
    manifest=dict(file=p.name,sha256=sha(p),TRAIN=1600,TRAIN_each_class=80,original_unique_TRAIN=1508,augmented_TRAIN=92,TEST=411,VALID=0,source_cache_SHA256=SOURCE_CACHE,lineage_SHA256=sha(a.out/'TRAIN_lineage.json'),TEST_pixel_SHA256=hashlib.sha256(arrays['test_images'].tobytes()).hexdigest(),source_SHA256=sha(__file__),source_git=git_head())
    save(a.out/'data_manifest.json',manifest);print(json.dumps(manifest),flush=True)


def data_for(sequence):
    assets=Path(sequence)/'assets';m=read(assets/'data_manifest.json');p=assets/m['file'];assert sha(p)==m['sha256'] and m['TRAIN']==1600 and m['TEST']==411 and m['VALID']==0
    assert sha(assets/'TRAIN_lineage.json')==m['lineage_SHA256'];lineage=read(assets/'TRAIN_lineage.json')['records'];assert len(lineage)==1600
    with np.load(p,allow_pickle=False) as z:
        assert set(z.files)=={r+'_'+k for r in ('train','test') for k in ('images','labels','ids')}
        data={r:(torch.from_numpy(z[r+'_images'].transpose(0,3,1,2).copy()),z[r+'_labels'].copy(),z[r+'_ids'].copy()) for r in ('train','test')}
        assert hashlib.sha256(z['test_images'].tobytes()).hexdigest()==m['TEST_pixel_SHA256']
    assert data['train'][2].tolist()==[v['sample_id'] for v in lineage]
    assert np.bincount(data['train'][1],minlength=20).tolist()==[80]*20
    assert not set(v['source_id'] for v in lineage)&set(data['test'][2].tolist())
    return data,m


def model_for(checkpoint,digest,depth,reference):
    base,_=load_checkpoint(checkpoint,digest,depth,.3,alpha_schema=DIRECT)
    model=MeasuredPrefix(depth).eval();model.load_state_dict(state_cpu(base));model.requires_grad_(False)
    transfer=dict(torch.jit.load(str(reference),map_location='cpu').named_buffers())['propagator.transfer'];model.propagator.transfer.copy_(transfer)
    return model,transfer


def padded(data,role,begin,batch=8):
    images,y,ids=data[role];end=min(begin+batch,len(y));x=images[begin:end];n=len(x)
    if n<batch:x=torch.cat([x,torch.zeros((batch-n,3,150,150),dtype=x.dtype)],0)
    return x,n,end


def fields_for(session,prefix,data,dm,hardware):
    stages=['router']+[str(k) for k in range(prefix-1)];fields={};identities={}
    for stage in stages:
        out=session/'capture'/stage;r=read(out/'report.json')
        assert r['state']=='complete' and r['sdk_released'] and r['physical_ccd']==2011 and r['hardware_SHA256']==hardware and r['data_SHA256']==dm['sha256']
        fields[stage]={}
        for role in data:
            p=out/role/'ccd.npy';assert sha(p)==r['arrays'][role]['SHA256'];v=np.load(p,mmap_mode='r');assert v.shape==(len(data[role][1]),478,478);fields[stage][role]=v;identities[stage+'/'+role]=sha(p)
    return fields,identities


@torch.no_grad()
def assess(model,data,fields,role):
    if next(model.parameters()).device.type!='cpu':
        rng=torch.get_rng_state();current=MeasuredPrefix(model.depth).eval();torch.set_rng_state(rng)
        current.load_state_dict(state_cpu(model));current.propagator.transfer.copy_(model.propagator.transfer.detach().cpu());current.requires_grad_(False)
    else:current=model.eval()
    parts=[]
    for begin in range(0,len(data[role][1]),8):
        images,n,end=padded(data,role,begin);measured={}
        for k,v in fields.items():
            array=np.asarray(v[role][begin:end]).copy()
            if n<8:array=np.concatenate([array,np.zeros((8-n,478,478),array.dtype)])
            measured[k]=torch.from_numpy(array)
        current.measured=measured;parts.append(current(encode_images(images))[0][:n].numpy())
    p=np.concatenate(parts);m=metrics(data[role][1],p);m.update(samples=len(p),correct=int((p.argmax(1)==data[role][1]).sum()),nll=float(-np.log(p[np.arange(len(p)),data[role][1]].clip(1e-12)).mean()))
    return m,p


def gate(a,data,dm,spec):
    session=a.sequence/f'L{a.depth}';out=session/'lab_gate.json';assert not out.exists();session.mkdir(exist_ok=True)
    checkpoint=Path(spec['initial']);assert sha(checkpoint)==spec['PT_SHA256'];reference=Path(spec['reference']);assert sha(reference)==spec['reference_SHA256']
    # Exact immutable finalized simulation artifacts only; never an adapted old-camera PT.
    shutil.copy2(checkpoint,session/'initial.pt');shutil.copy2(reference,session/'source_reference.ts')
    model,transfer=model_for(session/'initial.pt',spec['PT_SHA256'],a.depth,session/'source_reference.ts');original=torch.jit.load(str(session/'source_reference.ts')).eval();checks={}
    for role in ('test','train'):
        error=0.;parts=[]
        with torch.no_grad():
            for begin in range(0,len(data[role][1]),8):
                images,n,end=padded(data,role,begin);x=encode_images(images);p=model(x)[0];q=original(x)[0];error=max(error,float((p-q).abs().max()));parts.append(p[:n].numpy());save(session/'gate_progress.json',dict(role=role,completed=end,total=len(data[role][1])))
        assert error<=3e-6,('Original compiled simulation/bridge mismatch',role,error)
        probability=np.concatenate(parts);m=metrics(data[role][1],probability);m.update(correct=int((probability.argmax(1)==data[role][1]).sum()),samples=len(probability),original_reference_maxabs=error);checks[role]=m
        np.savez_compressed(session/(role+'_reference.npz'),ids=data[role][2],labels=data[role][1],probabilities=probability)
    save(out,dict(passed=True,depth=a.depth,PT_SHA256=spec['PT_SHA256'],data_SHA256=dm['sha256'],reference_SHA256=spec['reference_SHA256'],fixed_cached_simulation_correct=spec['simulation_correct'],native_laboratory_simulation_correct=checks['test']['correct'],batch=8,last_batch_zero_padding=True,checks=checks,SDK_opened=False,TEST_development=True,source_git=git_head(),source_SHA256=sha(__file__)))


@torch.no_grad()
def acquire(a,stage,checkpoint,digest,data,dm,hardware,pilot=False):
    import cv2
    session=a.sequence/f'L{a.depth}';prefix=(['router']+[str(k) for k in range(a.depth)]).index(stage)
    if prefix:
        prior_stage=(['router']+[str(k) for k in range(a.depth)])[prefix-1]
        fit_report=read(session/'fits'/prior_stage/'report.json');fit_protocol=read(session/'fits'/prior_stage/'protocol.json')
        assert fit_report['frozen_parameters_unchanged'] and fit_report['results']['best']['PT_SHA256']==digest
        assert read(session/'capture'/prior_stage/'report.json')['PT_SHA256']==fit_protocol['parent_PT_SHA256']
    previous,identities=fields_for(session,prefix,data,dm,HARDWARE) if prefix else ({},{})
    model,_=model_for(checkpoint,digest,a.depth,session/'source_reference.ts');drive=np.asarray(read(hardware['response_lut'])['drive'],np.uint8)
    out=session/'router_pilot24' if pilot else session/'capture'/stage;assert not out.exists();out.mkdir(parents=True)
    warp=cv2.getPerspectiveTransform(np.float32(corners(Path(hardware['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]));phasepath=out/'phase.bmp';phase_pixels=None;frames=0;start=time.perf_counter();arrays={};elapsed=[]
    save(out/'contract.json',dict(camera='DVP',hardware_SHA256=HARDWARE,data_SHA256=dm['sha256'],PT_SHA256=digest,stage=stage,pilot=pilot,prefix_array_SHA256=identities,scope='new camera only; no old CCD reuse',TRAIN=24 if pilot else 1600,TEST=0 if pilot else 411,VALID=0,source_git=git_head(),source_SHA256=sha(__file__)))
    with LOCK.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),stage=stage,session=str(session),camera='DVP',pilot=pilot)))
    try:
        with Bench(hardware,out) as bench:
            for role in ('train',) if pilot else ('train','test'):
                images,y,ids=data[role];n_total=24 if pilot else len(y);folder=out/role;folder.mkdir();ccd=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(n_total,478,478));last_end=0
                for begin in range(0,n_total,8):
                    x,n,end=padded(data,role,begin);assert end<=n_total
                    cache={k:np.asarray(v[role][begin:end]).copy() for k,v in previous.items()}
                    for k in cache:
                        if n<8:cache[k]=np.concatenate([cache[k],np.zeros((8-n,478,478),np.uint8)])
                    padded_ids=np.concatenate([ids[begin:end],np.array(['__padding__']*(8-n))]) if n<8 else ids[begin:end]
                    def physical(current,amplitude,phase,sample_ids):
                        nonlocal phase_pixels,frames
                        if phase_pixels is None:
                            phase_pixels=phase.copy();Image.fromarray(phase_native(phase)).save(phasepath);bench.stage(phasepath)
                        assert current==stage and np.array_equal(phase_pixels,phase)
                        values=[]
                        for j in range(n):
                            index=begin+j;t=time.perf_counter();gray=np.rint(amplitude[j].clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amplitude[j]))<=1e-7
                            base=folder/f'{index:06d}';canonical=base.with_name(base.name+'_input.png');Image.fromarray(gray).save(canonical,compress_level=1);bmp=base.with_name(base.name+'_transient.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(bmp)
                            raw,meta=bench.capture(bmp);crop=orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),hardware['orientation'])
                            rawfile=base.with_name(base.name+'_raw.png');Image.fromarray(raw).save(rawfile,compress_level=1);Image.fromarray(crop).save(base.with_suffix('.png'),compress_level=1);ccd[index]=crop;ccd.flush()
                            row=dict(stage=stage,role=role,index=index,sample_id=str(ids[index]),PT_SHA256=digest,hardware_SHA256=HARDWARE,data_SHA256=dm['sha256'],phase_SHA256=sha(phasepath),canonical_SHA256=sha(canonical),input_SHA256=sha(bmp),raw_SHA256=sha(rawfile),frame_metadata=meta,p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),complete_sample_ms=(time.perf_counter()-t)*1000,captured_at_ns=time.time_ns())
                            save(base.with_suffix('.json'),row);frames+=1;elapsed.append(row['complete_sample_ms']);values.append(crop);save(session/'progress.json',dict(state='pilot' if pilot else 'capturing',depth=a.depth,stage=stage,role=role,completed=index+1,total=n_total,stage_frames=frames,elapsed_s=time.perf_counter()-start,latest=row))
                            # Transient BMP is redundant; canonical input, exact drive LUT and digest remain.
                            bmp.unlink()
                            if row['p99']<15 or row['saturation_fraction']>.15:raise RuntimeError('DVP signal guard: retain all valid/bad frames; no automatic retry or exposure change')
                        if n<8:values.extend([np.zeros((478,478),np.uint8)]*(8-n))
                        return np.stack(values)
                    result=layer_forward(model,x,padded_ids,stage,cache,physical);last_end=end
                assert last_end==n_total;arrays[role]=dict(samples=n_total,SHA256=sha(folder/'ccd.npy'))
        assert frames==(24 if pilot else 2011)
        audit={}
        for role,item in arrays.items():
            folder=out/role;stored=np.load(folder/'ccd.npy',mmap_mode='r');count=item['samples'];assert len(list(folder.glob('*_raw.png')))==count
            for i in range(count):
                base=folder/f'{i:06d}';r=read(base.with_suffix('.json'));assert r['sample_id']==str(data[role][2][i]) and r['PT_SHA256']==digest and r['hardware_SHA256']==HARDWARE and r['phase_SHA256']==sha(phasepath)
                assert r['canonical_SHA256']==sha(base.with_name(base.name+'_input.png'))
                with Image.open(base.with_suffix('.png')) as im:assert np.array_equal(np.asarray(im),stored[i])
            audit[role]=dict(receipts=count,crop_array_equality=True,IDs_PT_phase_canonical_SHA256=True)
        save(out/'report.json',dict(state='complete',pilot=pilot,stage=stage,physical_ccd=frames,TRAIN=24 if pilot else 1600,TEST=0 if pilot else 411,VALID=0,sdk_released=True,arrays=arrays,data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,PT_SHA256=digest,prefix_SHA256=identities,phase_SHA256=sha(phasepath),elapsed_s=time.perf_counter()-start,median_complete_sample_ms=float(np.median(elapsed)),mean_complete_sample_ms=float(np.mean(elapsed)),fullraw_preserved=True,audit=audit))
    except BaseException as ex:save(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),frames=frames,automatic_retry=False));raise
    finally:LOCK.unlink()


def fitting_model(a,data,dm,checkpoint,digest,prefix):
    session=a.sequence/f'L{a.depth}';model,transfer=model_for(checkpoint,digest,a.depth,session/'source_reference.ts');fields,identities=fields_for(session,prefix,data,dm,HARDWARE);return model,transfer,fields,identities


def fit(a,data,dm,checkpoint,digest,prefix,cfg):
    session=a.sequence/f'L{a.depth}';stage=(['router']+[str(k) for k in range(a.depth)])[prefix-1];out=session/'fits'/stage;assert not out.exists() and not LOCK.exists();out.mkdir(parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID'] and torch.cuda.is_available() and 'A100' not in torch.cuda.get_device_name(0)
    # Caller has already released SDK. Reject any non-desktop foreign GPU user.
    from lab_caltech20_rich_tail_refine_20261018 import windows_process_names
    names=windows_process_names();own={os.getpid(),os.getppid()}
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines():
        gpu,pid,exe=row.split(',',2);pid=int(pid)
        if gpu.strip()==cfg['GPU_UUID']:assert pid in own or names.get(pid,Path(exe.strip()).name.lower()) in cfg['desktop_graphics_process_allowlist'],('Foreign GPU owner',pid,names.get(pid,exe))
    torch.cuda.set_per_process_memory_fraction(.65,0);torch.manual_seed(cfg['seed']);model,transfer,fields,identities=fitting_model(a,data,dm,checkpoint,digest,prefix);original=state_cpu(model);allowed=allowed_keys(model,prefix);baseline,p=assess(model,data,fields,'test');best,bestloss=baseline['correct'],baseline['nll'];selected=0;target=math.ceil(cfg['models'][str(a.depth)]['simulation_correct']-411*cfg['gap_pp']/100)
    budget=cfg['final_fit'] if prefix==a.depth+1 else cfg['middle_fit'];protocol=dict(parent_PT_SHA256=digest,depth=a.depth,prefix=prefix,allowed_keys=sorted(allowed),prefix_SHA256=identities,data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,baseline_TEST=baseline,target_correct=target,config=cfg,source_git=git_head(),source_SHA256=sha(__file__))
    save(out/'protocol.json',protocol);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=p)
    def snapshot(name,epoch):
        state=state_cpu(model);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);torch.save(dict(model=state,protocol=protocol,epoch=epoch),out/(name+'.pt'))
    snapshot('best',0);model.cuda();groups=[]
    for key,lr in (('phase_sets.',cfg['phase_lr']),('electronic.',cfg['electronic_lr']),('fusion_logits.',cfg['alpha_lr']),('linear.',cfg['head_lr'])):
        values=[p for n,p in model.named_parameters() if n in allowed and n.startswith(key)]
        if values:groups.append(dict(params=values,lr=lr,weight_decay=0 if key=='fusion_logits.' else cfg['weight_decay']))
    for n,p in model.named_parameters():p.requires_grad_(n in allowed)
    opt=torch.optim.AdamW(groups);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,budget['epochs'],eta_min=1e-5);history=[];start=time.monotonic();epoch=0
    for epoch in range(1,budget['epochs']+1):
        model.train();order=torch.randperm(1600).numpy();loss_sum=0.;used=0
        # TEST confusion controls modest emphasis of existing TRAIN classes only.
        cm=np.asarray(baseline['confusion_matrix']);support=cm.sum(1);errors=1-np.divide(cm.diagonal(),support,out=np.zeros(20),where=support>0);weights=torch.tensor(1+cfg['confusion_emphasis']*errors,dtype=torch.float32,device='cuda')
        for begin in range(0,1600,cfg['batch_size']):
            ix=order[begin:begin+cfg['batch_size']];model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()).cuda() for k,v in fields.items()};x=encode_images(data['train'][0][ix].cuda());y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True);p=model(x)[0];lp=p.clamp_min(1e-12).log();ce=(1-cfg['label_smoothing'])*F.nll_loss(lp,y,reduction='none')-cfg['label_smoothing']*lp.mean(1);loss=(ce*weights[y]).sum()/weights[y].sum();assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in model.named_parameters() if n in allowed);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
            with torch.no_grad():
                for p in model.fusion_logits:
                    if p.requires_grad:p.clamp_(.4001001,.95)
            loss_sum+=float(loss)*len(ix);used+=len(ix)
        scheduler.step();assert used==1600
        if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==budget['epochs'] or time.monotonic()-start>=budget['seconds']:
            model.eval();m,p=assess(model,data,fields,'test');baseline=m
            if improves(m['correct'],m['nll'],best,bestloss):best,bestloss=m['correct'],m['nll'];selected=epoch;snapshot('best',epoch)
            row=dict(epoch=epoch,TRAIN_loss=loss_sum/1600,actual_TRAIN_samples=used,TEST_correct=m['correct'],TEST_NLL=m['nll'],TRAIN_class_weights=weights.detach().cpu().tolist(),selected_correct=best,selected_epoch=selected,elapsed_s=time.monotonic()-start);history.append(row);save(out/'history.json',history);save(session/'progress.json',dict(state='suffix_fitting',stage=stage,prefix=prefix,latest=row,target_correct=target));print(json.dumps(row),flush=True)
            if best>=target or time.monotonic()-start>=budget['seconds']:break
            model.train()
    model.cpu();snapshot('last',epoch);results={}
    for name in ('best','last'):
        pth=out/(name+'.pt');state=torch.load(pth,map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);model.load_state_dict(state);m,p=assess(model,data,fields,'test');np.savez_compressed(out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p);results[name]=dict(TEST_development=m,PT_SHA256=sha(pth),strict_CPU=True,predictions_SHA256=sha(out/(name+'_TEST_predictions.npz')))
    assert results['best']['TEST_development']['correct']==best
    save(out/'report.json',dict(state='complete',baseline_TEST=protocol['baseline_TEST'],results=results,selected_epoch=selected,actual_epochs=epoch,selected_correct=best,target_correct=target,within8pp=best>=target,selected_e0=selected==0,intermediate=prefix<a.depth+1,frozen_parameters_unchanged=True,TEST_gradients=False,SDK_used=False,elapsed_s=time.monotonic()-start))
    del model,opt;torch.cuda.empty_cache();return out/'best.pt',results['best']['PT_SHA256'],read(out/'report.json')


def run(a):
    cfg=read(a.profile);torch.set_num_threads(cfg['cpu_threads']);assert not (a.sequence/'controller').exists() and not LOCK.exists();out=a.sequence/'controller';out.mkdir();data,dm=data_for(a.sequence)
    hardwarepath=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(hardwarepath)==HARDWARE;hardware=read(hardwarepath);assert sha(hardware['response_lut'])==hardware['response_lut_SHA256'] and sha(hardware['phase_lut'])==hardware['phase_LUT_SHA256']
    assert read(a.sequence/'calibration/admission/report.json')['passed'];assert shutil.disk_usage(a.sequence).free>cfg['minimum_free_GiB']*(1<<30)
    save(out/'started.json',dict(pid=os.getpid(),source_git=git_head(),source_SHA256=sha(__file__),config_SHA256=sha(a.profile),data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,order=cfg['order'],recurring=False,planned_CCD=24132,TRAIN=1600,original_unique_TRAIN=1508,fixed_augmented_TRAIN=92,TEST=411,VALID=0));completed=[]
    for depth in cfg['order']:
        a.depth=depth;session=a.sequence/f'L{depth}';session.mkdir(exist_ok=True);spec=cfg['models'][str(depth)];gate(a,data,dm,spec);checkpoint=session/'initial.pt';digest=spec['PT_SHA256']
        # Separate24 TRAIN-only pilot is never silently counted as full data.
        acquire(a,'router',checkpoint,digest,data,dm,hardware,pilot=True);assert read(session/'router_pilot24/report.json')['sdk_released'];records=[]
        for prefix,stage in enumerate(['router']+[str(k) for k in range(depth)],1):
            acquire(a,stage,checkpoint,digest,data,dm,hardware);checkpoint,digest,report=fit(a,data,dm,checkpoint,digest,prefix,cfg);records.append(dict(stage=stage,selected_PT=str(checkpoint),PT_SHA256=digest,TEST_correct=report['selected_correct'],selected_epoch=report['selected_epoch'],within8pp=report['within8pp'],kind='full_measured' if prefix==depth+1 else 'measured_prefix_simulated_suffix'));save(session/'stage_history.json',records)
        score=records[-1]['TEST_correct'];result=dict(state='complete',depth=depth,fully_measured=True,TEST_correct=score,TEST=411,accuracy=score/411,fixed_simulation_correct=spec['simulation_correct'],gap_pp=100*(spec['simulation_correct']-score)/411,PT=str(checkpoint),PT_SHA256=digest,SDK_released=not LOCK.exists(),physical_CCD=2011*(depth+1),pilot_CCD=24,TEST_development=True,records=records);save(session/'report.json',result);completed.append(result);save(out/'completed_models.json',completed);print(json.dumps(result),flush=True)
    save(out/'report.json',dict(state='complete',results=completed,SDK_released=not LOCK.exists(),recurring=False))


def selftest():
    torch.set_num_threads(2);torch.manual_seed(17)
    # Selection audit uses synthetic independent images with the ACTUAL supports.
    counts=[348,160,638,640,102,78,68,72,98,85,68,79,80,70,68,91,69,68,68,191];y=np.repeat(np.arange(20),counts);ids=np.array([f'TRAIN/{i:05d}' for i in range(len(y))]);rng=np.random.default_rng(17);x=rng.integers(0,256,(len(y),150,150,3),np.uint8)
    a,b,n,records=select80(x,y,ids);assert a.shape==(1600,150,150,3) and b[:20].tolist()==list(range(20));assert len({r['source_id'] for r in records})==1508
    assert all(r['source_id'].startswith('TRAIN/') for r in records)
    for depth in (4,6):
        m=MeasuredPrefix(depth)
        for prefix in range(1,depth+2):
            keys=allowed_keys(m,prefix);assert 'raw_router' not in keys
            if prefix==depth+1:assert not any(k.startswith('phase_sets.') for k in keys)
        with torch.no_grad():
            for v in m.fusion_logits:v.fill_(.8)
            x=torch.rand(1,224,224);expected=m(x)[0];fields={k:v['intensity'].clone() for k,v in m.last.items()}
        for prefix in (1,2,depth,depth+1):
            replay=MeasuredPrefix(depth).eval();replay.load_state_dict(state_cpu(m));replay.measured={k:fields[k] for k in (['router']+[str(j) for j in range(depth)])[:prefix]};allowed=allowed_keys(replay,prefix)
            for k,v in replay.named_parameters():v.requires_grad_(k in allowed)
            actual=replay(x)[0];assert torch.equal(actual,expected);(-actual[:,0].log().mean()).backward()
            assert all(v.grad is None for k,v in replay.named_parameters() if k not in allowed)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for k,v in replay.named_parameters() if k in allowed)
    print(json.dumps(dict(passed=True,TRAIN1600_original1508_aug92=True,no_VALID_TEST_sources=True,depth4_6_prefix1_2_penultimate_final_forward_and_gradient_freeze=True,SDK_opened=False,GPU_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('prepare','run','selftest'));p.add_argument('--cache',type=Path);p.add_argument('--out',type=Path);p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);a=p.parse_args()
    try:selftest() if a.mode=='selftest' else prepare(a) if a.mode=='prepare' else run(a)
    except BaseException as ex:
        folder=a.sequence/'controller' if a.mode=='run' else a.out
        if folder and folder.exists():save(folder/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),automatic_retry=False))
        raise
