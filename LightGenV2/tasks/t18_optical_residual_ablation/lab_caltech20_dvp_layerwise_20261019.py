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
from lab_caltech20_dvp_20261019 import Bench,orient,LOCK,correlation,variants

SOURCE_CACHE='38b719b8ad935a0170d059d0e4fffa22f6f22640f979603f7f2543fc2d3f2445'
HARDWARE='3b89d610d939141dec9c5d0955f23765454cd7e7bebfe1c92c5e64e5376713ae'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def session_for(a):
    """Optional run namespace; never change the depth or physical model."""
    attempt=getattr(a,'attempt',None)
    assert attempt is None or attempt in ('strict8pp_20261020','response_20261020'),'Unreviewed attempt namespace'
    return a.sequence/(f'L{a.depth}'+('_'+attempt if attempt else ''))


def session_hardware_digest(sequence):
    """A changed coordinate file is a fresh explicit session, never an old-prefix patch."""
    sequence=Path(sequence);identity=sequence/'hardware_identity.json'
    if not identity.exists():return HARDWARE
    record=read(identity);assert record['schema']=='fresh_DVP_geometry_v1'
    old=Path(record['source_admission']);assert sha(old)==HARDWARE==record['source_admission_SHA256']
    base=sequence/'calibration/admission/admitted_hardware.json';assert sha(base)==record['hardware_SHA256']
    previous,current=read(old),read(base);assert set(previous)==set(current)
    changed={'coordinates','coordinates_SHA256','exposure_us'}
    assert all(current[k]==previous[k] for k in current if k not in changed)
    assert current['exposure_us']==60006. and current['gain']==1.
    assert sha(current['coordinates'])==current['coordinates_SHA256']==record['coordinates_SHA256']
    assert Path(current['coordinates']).resolve().is_relative_to(sequence.resolve())
    assert record['reuse_previous_CCD'] is False and record['fresh_layerwise_capture'] is True
    return record['hardware_SHA256']


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
        assert r['state']=='complete' and r['sdk_released'] and r['physical_ccd']==2011 and r['data_SHA256']==dm['sha256']
        if r['hardware_SHA256']!=hardware:
            h=validate_stage_hardware(out/'hardware.json',session.parent)
            assert sha(out/'hardware.json')==r['hardware_SHA256'] and h['stage']==stage and h['PT_SHA256']==r['PT_SHA256'] and h['data_SHA256']==dm['sha256'] and h['phase_SHA256']==r['phase_SHA256']
        fields[stage]={}
        for role in data:
            p=out/role/'ccd.npy';assert sha(p)==r['arrays'][role]['SHA256'];v=np.load(p,mmap_mode='r');assert v.shape==(len(data[role][1]),478,478);fields[stage][role]=v;identities[stage+'/'+role]=sha(p)
    return fields,identities


def validate_stage_hardware(path,sequence):
    base_digest=session_hardware_digest(sequence);base=Path(sequence)/'calibration/admission/admitted_hardware.json';assert sha(base)==base_digest
    original=read(base);hardware=read(path);assert hardware['base_admission_SHA256']==base_digest
    for key in ('camera_dll','camera_dll_SHA256','coordinates','coordinates_SHA256','sensor_shape','machine_config','amplitude_sdk','phase_sdk','phase_lut','phase_LUT_SHA256','gain','wait_ms','discard_frames','phase_transform','orientation','response_lut','response_lut_SHA256'):
        assert hardware[key]==original[key],('Per-layer exposure may not change other hardware/encoding',key)
    assert 0<hardware['exposure_us']<=80005 and hardware['gain']==1
    if hardware.get('quality_mode')=='shared_fixed_record_only':
        policy=validate_capture_policy(hardware['capture_policy'],hardware,sequence)
        assert sha(hardware['capture_policy'])==hardware['capture_policy_SHA256']
        assert hardware['shared_source_hardware_SHA256']==policy['source_hardware_SHA256']
        assert hardware['probe_indices']==[] and hardware['network_background_subtraction'] is False
        return hardware
    report=Path(hardware['stage_admission_report']);assert sha(report)==hardware['stage_admission_report_SHA256'];r=read(report)
    assert r['passed'] and r['SDK_released'] and not r['TEST_used'] and r['TRAIN']==40 and r['selected']['actual']['exposure_us']==hardware['exposure_us']
    assert r['PT_SHA256']==hardware['PT_SHA256'] and r['phase_SHA256']==hardware['phase_SHA256']
    assert r['data_SHA256']==hardware['data_SHA256'] and r['selected']['probe_directory']==hardware['probe_directory'] and hardware['probe_indices']==list(range(40))
    if hardware.get('quality_mode')=='same_phase_contrast':
        b=hardware['quality_background'];assert b==r['selected']['background'] and sha(b['path'])==b['SHA256'] and b['phase_SHA256']==hardware['phase_SHA256']
        assert hardware['network_background_subtraction'] is False and hardware['capture_min_contrast_p99']==r['protocol']['capture_min_contrast_p99']==15
    return hardware


def validate_capture_policy(path,hardware,sequence):
    cfg=read(path)
    assert cfg['policy']=='shared_fixed_exposure_record_only' and cfg['depths']==[4,6]
    assert cfg['calibrate_each_stage'] is False and cfg['brightness_threshold_action']==cfg['saturation_threshold_action']=='record_only'
    assert cfg['network_background_subtraction'] is False and cfg['base_admission_SHA256']==session_hardware_digest(sequence)
    if 'coordinates_SHA256' in cfg:assert cfg['coordinates_SHA256']==hardware['coordinates_SHA256']
    source=Path(sequence)/cfg['source_hardware'];assert sha(source)==cfg['source_hardware_SHA256']
    original=validate_stage_hardware(source,cfg.get('source_sequence',sequence))
    assert original['quality_mode']=='same_phase_contrast' and original['exposure_us']==cfg['exposure_us']==hardware['exposure_us']==60006.
    assert original['gain']==cfg['gain']==hardware['gain']==1.
    return cfg


def signal_flags(row,hardware):
    flags=dict(raw_p99_below15=row['p99']<15,saturation_above15percent=row['saturation_fraction']>.15)
    if 'contrast_p99' in row:flags['net_p99_below_limit']=row['contrast_p99']<hardware['capture_min_contrast_p99']
    return flags


@torch.no_grad()
def prepare_fixed_hardware(a):
    """Bind one admitted shared exposure to new PT/phase; no camera or new calibration."""
    assert a.capture_policy and not a.out.exists() and sha(a.checkpoint)==a.digest
    base_digest=session_hardware_digest(a.sequence);base=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(base)==base_digest
    hardware=read(base);cfg=read(a.capture_policy);hardware['exposure_us']=cfg['exposure_us'];validate_capture_policy(a.capture_policy,hardware,a.sequence)
    data,dm=data_for(a.sequence);session=session_for(a);prefix=(['router']+[str(k) for k in range(a.depth)]).index(a.stage)
    previous,identities=fields_for(session,prefix,data,dm,base_digest) if prefix else ({},{})
    model,_=model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');x,n,end=padded(data,'train',0);packet={}
    def collect(stage,amplitude,phase,ids):
        assert stage==a.stage;packet['phase']=phase.copy();return np.zeros((n,478,478),np.uint8)
    layer_forward(model,x,data['train'][2][:end],a.stage,{k:np.asarray(v['train'][:end]).copy() for k,v in previous.items()},collect)
    a.out.mkdir(parents=True);phase=a.out/'model_phase.bmp';Image.fromarray(phase_native(packet['phase'])).save(phase)
    hardware.update(base_admission_SHA256=base_digest,quality_mode='shared_fixed_record_only',PT_SHA256=a.digest,stage=a.stage,data_SHA256=dm['sha256'],phase_SHA256=sha(phase),probe_indices=[],network_background_subtraction=False,capture_policy=str(a.capture_policy.resolve()),capture_policy_SHA256=sha(a.capture_policy),shared_source_hardware_SHA256=cfg['source_hardware_SHA256'])
    save(a.out/'hardware.json',hardware);save(a.out/'report.json',dict(prepared=True,not_new_calibration=True,SDK_opened=False,source_git=git_head(),source_SHA256=sha(__file__),PT_SHA256=a.digest,phase_SHA256=sha(phase),hardware_SHA256=sha(a.out/'hardware.json'),prefix_SHA256=identities,shared_exposure_us=cfg['exposure_us'],signal_thresholds_record_only=True))
    validate_stage_hardware(a.out/'hardware.json',a.sequence)


@torch.no_grad()
def admit_stage(a):
    from lab_caltech20_dvp_stage_exposure_20261019 import calibration
    cfg=read(a.profile);torch.set_num_threads(2);data,dm=data_for(a.sequence);session=session_for(a);stages=['router']+[str(k) for k in range(a.depth)];prefix=stages.index(a.stage)
    assert sha(a.checkpoint)==a.digest;model,_=model_for(a.checkpoint,a.digest,a.depth,session/'source_reference.ts');previous,identities=fields_for(session,prefix,data,dm,HARDWARE) if prefix else ({},{})
    packet=dict(stage=a.stage,PT_SHA256=a.digest,data_SHA256=dm['sha256'],prefix_SHA256=identities,source_git=git_head(),layerwise_source_SHA256=sha(__file__),records=[],inputs={});phase_pixels=None
    for begin in range(0,cfg['TRAIN_probe_samples'],8):
        x,n,end=padded(data,'train',begin);assert end<=cfg['TRAIN_probe_samples']
        def collect(stage,amplitudes,phase,ids):
            nonlocal phase_pixels
            if phase_pixels is None:phase_pixels=phase.copy()
            assert stage==a.stage and np.array_equal(phase_pixels,phase)
            for j in range(n):
                index=begin+j;gray=np.rint(amplitudes[j].clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amplitudes[j]))<=1e-7
                packet['inputs'][index]=gray;packet['records'].append(dict(index=index,role='train',sample_id=str(data['train'][2][index]),label=int(data['train'][1][index]),stage=stage,PT_SHA256=a.digest,data_SHA256=dm['sha256']))
            return np.zeros((n,478,478),np.uint8)
        layer_forward(model,x,data['train'][2][begin:end],a.stage,{k:np.asarray(v['train'][begin:end]).copy() for k,v in previous.items()},collect)
    packet['phase']=phase_pixels;hardwarepath=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(hardwarepath)==HARDWARE
    calibration(a.out,packet,read(hardwarepath),HARDWARE,cfg,revalidate_replug)


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
    session=session_for(a);out=session/'lab_gate.json';assert not out.exists();session.mkdir(exist_ok=True)
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


def revalidate_replug(bench,out,hardware,warp):
    """Known patterns at the ORIGINAL settings; no score-based calibration/refit."""
    import cv2
    folder=out/'replug_validation';folder.mkdir();zero=np.zeros((478,478),np.uint8)
    phase=folder/'flat_phase.bmp';Image.fromarray(phase_native(zero)).save(phase);bench.stage(phase)
    marker=np.full_like(zero,24);marker[35:215,35:215]=210;marker[35:215,263:443]=90;marker[263:443,35:215]=150;marker[290:390,290:390]=255
    rows=[];images=[]
    for name,pixels in (('dark',zero),('white',zero+255),('marker',marker),('toggle',np.fliplr(marker).copy())):
        path=folder/(name+'.bmp');Image.fromarray(native(pixels,'amplitude')).save(path)
        raw,meta=bench.capture(path);crop=cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        Image.fromarray(raw).save(folder/(name+'_raw.png'),compress_level=1);Image.fromarray(crop).save(folder/(name+'_crop.png'))
        v=orient(crop,hardware['orientation']);images.append(crop);rows.append(dict(name=name,mean=float(v.mean()),p99=float(np.percentile(v,99)),saturation=float((v==255).mean()),meta=meta))
    # Fixed admission tolerances reject an altered scene rather than refit the LUT.
    response=read(hardware['response_lut']);old=response['fit_means'];span=old[-1]-old[0]
    assert abs(rows[0]['mean']-old[0])<=max(3.,span*.1)
    assert abs(rows[1]['mean']-old[-1])<=span*.1 and rows[1]['saturation']<.001 and rows[1]['p99']>=40
    scores={k:correlation(v,marker.astype(float)**2) for k,v in variants(images[2]).items()};rank=sorted(scores,key=scores.get,reverse=True)
    # The ideal marker itself has only .0847 separation from rot270.
    assert rank[0]==hardware['orientation'] and scores[rank[0]]>=.85 and scores[rank[0]]-scores[rank[1]]>=.05
    current=orient(images[3],hardware['orientation']);new=correlation(current,np.fliplr(marker).astype(float)**2);stale=correlation(current,marker.astype(float)**2)
    assert new>=.85 and new>stale+.1
    save(folder/'report.json',dict(passed=True,settings=bench.settings,rows=rows,orientation_scores=scores,toggle_current_PCC=new,toggle_previous_PCC=stale,hardware_SHA256=HARDWARE,LUT_unchanged=True,TEST_used=False))


def admitted_pilot(session,data,dm,digest):
    pilot=session/'router_pilot24';r=read(pilot/'report.json')
    assert r['state']=='complete' and r['pilot'] and r['physical_ccd']==24 and r['sdk_released']
    assert r['PT_SHA256']==digest and r['hardware_SHA256']==HARDWARE and r['data_SHA256']==dm['sha256']
    assert sha(pilot/'train/ccd.npy')==r['arrays']['train']['SHA256']
    array=np.load(pilot/'train/ccd.npy',mmap_mode='r');assert array.shape==(24,478,478)
    for i in range(24):
        base=pilot/'train'/f'{i:06d}';row=read(base.with_suffix('.json'))
        assert row['sample_id']==str(data['train'][2][i]) and row['index']==i and row['role']=='train' and row['stage']=='router'
        assert row['PT_SHA256']==digest and row['hardware_SHA256']==HARDWARE and row['data_SHA256']==dm['sha256'] and row['phase_SHA256']==sha(pilot/'phase.bmp')
        assert row['canonical_SHA256']==sha(base.with_name(base.name+'_input.png')) and row['raw_SHA256']==sha(base.with_name(base.name+'_raw.png'))
        meta=row['frame_metadata'];assert abs(meta['exposure_us']-5000.5)<2 and meta['gain']==1 and meta['shape']==[3648,5480]
        assert meta['format']==0 and meta['bits']==0 and not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
        with Image.open(base.with_suffix('.png')) as im:assert np.array_equal(np.asarray(im),array[i])
        assert row['p99']>=15 and row['saturation_fraction']<=.15
    return pilot,array


def validate_fit_ancestry(session,parent_name,digest,prefix,capture_digest,identities,data_digest):
    """Follow exact SHA parents, never directory age or a fixed canonical alias.

Several independent bounded refinements can follow one measured prefix. Every
node must still freeze that prefix, use its exact data/arrays and have a strict
CPU best. All paths stay in this session's fits directory. No PT substitution.
"""
    fits=Path(session)/'fits';current=fits/parent_name;chain=[];seen=set()
    for _ in range(12):
        assert current.resolve().is_relative_to(fits.resolve()) and digest not in seen
        seen.add(digest);report=current/'report.json';protocol=current/'protocol.json';r=read(report);p=read(protocol)
        assert r['state']=='complete' and r['frozen_parameters_unchanged'] and r['results']['best']['strict_CPU']
        assert sha(current/'best.pt')==r['results']['best']['PT_SHA256']==digest
        assert p['prefix']==prefix and p['prefix_SHA256']==identities and p['data_SHA256']==data_digest
        ancestor=p['parent_PT_SHA256'];chain.append(dict(fit=current.name,best_PT_SHA256=digest,parent_PT_SHA256=ancestor,report_SHA256=sha(report),protocol_SHA256=sha(protocol)))
        if ancestor==capture_digest:return chain
        candidates=[]
        for folder in fits.iterdir():
            if folder.is_dir() and (folder/'report.json').exists() and (folder/'protocol.json').exists():
                q=read(folder/'report.json')
                if q.get('results',{}).get('best',{}).get('PT_SHA256')==ancestor:candidates.append(folder)
        assert len(candidates)==1,('Missing/ambiguous exact parent; no timestamp fallback',ancestor)
        current=candidates[0];digest=ancestor
    raise AssertionError('Bounded ancestry depth exceeded; review exact lineage')


def same_connection_replug_check(bench,out,hardware,original,warp):
    """Recheck a VALID existing exposure admission without closing/reopening camera.

    No new exposure search, background subtraction or model change.  Two fresh
    zero-input frames only verify that the existing diagnostic background still
    applies after USB power cycling; they never replace model inputs.
    """
    import cv2
    folder=out/'same_connection_replug_check';assert not folder.exists();folder.mkdir()
    selected=bench.settings.copy();assert abs(selected['exposure_us']-hardware['exposure_us'])<2 and selected['gain']==1
    bench.settings=bench.camera.settings(exposure=original['exposure_us'],gain=1.)
    for _ in range(original['discard_frames']):bench.read(check_settings=False)
    revalidate_replug(bench,folder,original,warp)
    bench.settings=bench.camera.settings(exposure=selected['exposure_us'],gain=1.)
    assert abs(bench.settings['exposure_us']-hardware['exposure_us'])<2
    for _ in range(original['discard_frames']):bench.read(check_settings=False)
    phase=Path(hardware['stage_admission_report']).parent/'model_phase.bmp'
    assert sha(phase)==hardware['phase_SHA256'];bench.stage(phase)
    drive=np.asarray(read(hardware['response_lut'])['drive'],np.uint8)
    zero=folder/'zero_input.bmp';Image.fromarray(native(drive[np.zeros((478,478),np.uint8)],'amplitude')).save(zero)
    dark=[];records=[]
    for j in range(2):
        raw,meta=bench.capture(zero);crop=orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),hardware['orientation'])
        rawpath=folder/f'dark{j}_raw.png';path=folder/f'dark{j}.png';Image.fromarray(raw).save(rawpath,compress_level=1);Image.fromarray(crop).save(path,compress_level=1)
        dark.append(crop.astype(np.float32));records.append(dict(raw_SHA256=sha(rawpath),crop_SHA256=sha(path),frame_metadata=meta))
    current=(dark[0]+dark[1])/2;prior=np.load(hardware['quality_background']['path'],allow_pickle=False)
    rmse=float(np.sqrt(np.mean((current-prior)**2)));p99=float(np.percentile(current,99));prior_p99=float(np.percentile(prior,99));noise=float(np.sqrt(np.mean((dark[0]-dark[1])**2)/2));clipping=float((current>=255).mean())
    limits=dict(background_RMSE_max=3.,background_p99_delta_max=5.,background_p99_max=64.,background_saturation_max=.001,noise_sigma_max=2*hardware['quality_background']['noise_sigma']+.5)
    q=dict(passed=rmse<=3 and abs(p99-prior_p99)<=5 and p99<=64 and clipping<.001 and noise<=limits['noise_sigma_max'],settings=bench.settings.copy(),background_RMSE=rmse,background_p99=p99,prior_background_p99=prior_p99,noise_sigma=noise,saturation_fraction=clipping,limits=limits,frames=records,last_recheck_frame_id=records[-1]['frame_metadata']['frame_id'],same_Bench_instance=True,camera_reopened=False,network_background_subtraction=False,prior_background_unchanged=True,prior_background_SHA256=sha(hardware['quality_background']['path']),original_settings_check_SHA256=sha(folder/'replug_validation/report.json'))
    save(folder/'report.json',q);assert q['passed'],'Replug changed response/background: preserve evidence, no automatic retry or normalization'
    return q


def shared_geometry_check(bench,folder,hardware,warp):
    """Two known patterns once per geometry, same open camera as full router capture."""
    import cv2
    assert not folder.exists();folder.mkdir(parents=True)
    phase=folder/'flat_phase.bmp';Image.fromarray(phase_native(np.zeros((478,478),np.uint8))).save(phase);bench.stage(phase)
    marker=np.full((478,478),6,np.uint8);marker[35:215,35:215]=52;marker[35:215,263:443]=22;marker[263:443,35:215]=37;marker[290:390,290:390]=64
    drive=np.asarray(read(hardware['response_lut'])['drive'],np.uint8);rows=[];images=[]
    for name,gray in (('marker',marker),('toggle',np.fliplr(marker).copy())):
        path=folder/(name+'.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(path);raw,meta=bench.capture(path)
        crop=cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0);images.append(crop)
        rawpath=folder/(name+'_raw.png');croppath=folder/(name+'_crop.png');Image.fromarray(raw).save(rawpath,compress_level=1);Image.fromarray(crop).save(croppath,compress_level=1)
        rows.append(dict(name=name,raw_SHA256=sha(rawpath),crop_SHA256=sha(croppath),meta=meta,p99=float(np.percentile(crop,99)),saturation_fraction=float((crop==255).mean())))
    scores={k:correlation(v,marker.astype(float)**2) for k,v in variants(images[0]).items()};rank=sorted(scores,key=scores.get,reverse=True)
    current=orient(images[1],hardware['orientation']);new=correlation(current,np.fliplr(marker).astype(float)**2);stale=correlation(current,marker.astype(float)**2)
    passed=rank[0]==hardware['orientation'] and scores[rank[0]]>=.4 and scores[rank[0]]-scores[rank[1]]>=.05 and new>=.4 and new>stale+.1
    q=dict(passed=passed,orientation=hardware['orientation'],orientation_scores=scores,toggle_current_PCC=new,toggle_previous_PCC=stale,rows=rows,coordinates_SHA256=hardware['coordinates_SHA256'],settings=bench.settings.copy(),same_camera_connection=True,camera_reopened=False,TRAIN_TEST_used=False,exposure_search=False,LUT_refitted=False,brightness_and_saturation_record_only=True)
    save(folder/'report.json',q);assert passed,'New geometry direction/timing mismatch; preserve check frames, no automatic adjustment'
    return q


@torch.no_grad()
def acquire(a,stage,checkpoint,digest,data,dm,hardware,pilot=False,resume_router=False):
    import cv2
    base_digest=session_hardware_digest(a.sequence);session=session_for(a);prefix=(['router']+[str(k) for k in range(a.depth)]).index(stage)
    previous,identities=fields_for(session,prefix,data,dm,base_digest) if prefix else ({},{})
    ancestry=[]
    if prefix:
        prior_stage=(['router']+[str(k) for k in range(a.depth)])[prefix-1]
        parent_name=getattr(a,'parent_fit',None) or prior_stage
        ancestry=validate_fit_ancestry(session,parent_name,digest,prefix,read(session/'capture'/prior_stage/'report.json')['PT_SHA256'],identities,dm['sha256'])
        if getattr(a,'attempt',None):
            from lab_caltech20_dvp_strict_retake_20261020 import check_score
            parent=session/'fits'/parent_name
            assert read(parent/'independent_audit.json')['passed']
            assert check_score(read(parent/'report.json'),a.depth,prefix)==digest
    model,_=model_for(checkpoint,digest,a.depth,session/'source_reference.ts');drive=np.asarray(read(hardware['response_lut'])['drive'],np.uint8)
    out=session/'router_pilot24' if pilot else session/'capture'/stage;reused=None;hardware_digest=sha(a.hardware_manifest) if getattr(a,'hardware_manifest',None) else base_digest
    quality_source=getattr(a,'reuse_capture',None);quality_count=getattr(a,'reuse_train_prefix',0);quality_array=None
    if quality_source:
        assert getattr(a,'attempt',None)=='strict8pp_20261020' and not pilot and 0<quality_count<=1600
        quality_source=Path(quality_source);assert quality_source.resolve().is_relative_to((session/'rejected_signal_20261020').resolve())
        qr=read(quality_source/'report.json');assert qr['state']=='complete' and qr['sdk_released'] and qr['PT_SHA256']==digest and qr['data_SHA256']==dm['sha256'] and qr['hardware_SHA256']==hardware_digest and qr['prefix_SHA256']==identities
        assert sha(quality_source/'train/ccd.npy')==qr['arrays']['train']['SHA256'];quality_array=np.load(quality_source/'train/ccd.npy',mmap_mode='r')
        for i in range(quality_count):
            row=read(quality_source/'train'/f'{i:06d}.json');assert row['index']==i and row['sample_id']==str(data['train'][2][i]) and row['saturation_fraction']<.15 and row['phase_SHA256']==qr['phase_SHA256']
    else:assert quality_count==0
    if out.exists() and getattr(a,'archive_empty_start',False):
        assert a.depth==4 and stage=='2' and (getattr(a,'recheck_before_capture',False) or getattr(a,'capture_policy',None)) and not LOCK.exists()
        failure=read(out/'failure.json');prior_contract=read(out/'contract.json')
        assert failure['frames']==0 and 'DVP Start: -1001' in failure['error'] and not (out/'report.json').exists()
        assert prior_contract['PT_SHA256']==digest and prior_contract['hardware_SHA256']==hardware_digest
        assert not list(out.rglob('ccd.npy')) and not list(out.rglob('*_raw.png'))
        archive=out.with_name('2_failed_reopen_start_20261019');assert out.resolve().parent==(session/'capture').resolve() and not archive.exists();out.rename(archive)
        save(archive/'archive_receipt.json',dict(original_path=str(out),preserved_path=str(archive),reason='Zero-frame DVP reopen failure, preserved before same-connection known-pattern capture',original_frames=0,deletion=False))
    if out.exists() and getattr(a,'archive_failed',False):
        assert a.depth==4 and stage=='0' and not pilot and not LOCK.exists() and not (out/'report.json').exists()
        failure=read(out/'failure.json');assert failure['frames']==1 and len(list((out/'train').glob('[0-9]*.json')))==1 and not (out/'test').exists()
        archive=out.with_name('0_failed_low_signal_20261019');assert out.resolve().parent==(session/'capture').resolve() and not archive.exists();out.rename(archive)
        save(archive/'archive_receipt.json',dict(original_path=str(out),preserved_path=str(archive),reason='Old1frame failed signal guard under superseded PT/exposure; retained, never reused',original_frames=1,deletion=False))
    if resume_router:
        assert a.depth==4 and stage=='router' and not pilot
        source,reused=admitted_pilot(session,data,dm,digest)
        assert out.is_dir() and read(out/'failure.json')['frames']==0 and not (out/'report.json').exists()
        assert not list(out.rglob('ccd.npy')) and not list(out.rglob('*_raw.png'))
        archived=out.with_name('router_failed_start_20261019');assert not archived.exists()
        # Exact verified empty failed attempt, preserved within the same run.
        out.rename(archived)
    assert not out.exists();out.mkdir(parents=True)
    record_only=bool(getattr(a,'capture_policy',None))
    capture_policy=validate_capture_policy(a.capture_policy,hardware,a.sequence) if record_only else {}
    if not record_only:assert hardware.get('quality_mode')!='shared_fixed_record_only','Shared settings require the explicit published record-only policy'
    probe_directory=Path(hardware['probe_directory']) if hardware_digest!=base_digest and hardware.get('probe_directory') else None
    quality_background=None
    if hardware.get('quality_mode')=='same_phase_contrast':
        b=hardware['quality_background'];assert sha(b['path'])==b['SHA256'] and hardware['network_background_subtraction'] is False
        quality_background=np.load(b['path'],allow_pickle=False);assert quality_background.shape==(478,478) and quality_background.dtype==np.float32
    if hardware_digest!=base_digest:
        assert hardware['PT_SHA256']==digest and hardware['data_SHA256']==dm['sha256'] and hardware['stage']==stage and not pilot
        shutil.copy2(a.hardware_manifest,out/'hardware.json');assert sha(out/'hardware.json')==hardware_digest
    warp=cv2.getPerspectiveTransform(np.float32(corners(Path(hardware['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]));phasepath=out/'phase.bmp';phase_pixels=None;frames=0;start=time.perf_counter();arrays={};elapsed=[]
    writer=None;pending_record=None;replug_check=None
    if getattr(a,'async_png',False):
        from lab_caltech20_dvp_persistence_20261019 import FrameWriter
        writer=FrameWriter()
    persistence_mode='bounded_one_frame_PNG_overlap' if writer is not None else 'serial'
    def record_frame(base,row,index,total,role):
        nonlocal frames
        save(base.with_suffix('.json'),row);frames+=1;elapsed.append(row['complete_sample_ms'])
        save(session/'progress.json',dict(state='pilot' if pilot else 'capturing',depth=a.depth,stage=stage,role=role,completed=index+1,total=total,stage_frames=frames,elapsed_s=time.perf_counter()-start,latest=row,persistence_mode=persistence_mode))
    def finish_pending():
        nonlocal pending_record
        if pending_record is None:return
        base,row,index,total,role,sample_start=pending_record;pending_record=None
        try:result=writer.finish()
        except BaseException as error:
            save(out/'persistence_failure.json',dict(error=repr(error),receipt_candidate=row,index=index,role=role,raw_path=str(base.with_name(base.name+'_raw.png')),captured_data_may_be_partial=True,automatic_retry=False));raise
        row.update(raw_SHA256=result['raw_SHA256'],complete_sample_ms=(time.perf_counter()-sample_start)*1000)
        record_frame(base,row,index,total,role)
    save(out/'contract.json',dict(camera='DVP',hardware_SHA256=hardware_digest,data_SHA256=dm['sha256'],PT_SHA256=digest,stage=stage,pilot=pilot,prefix_array_SHA256=identities,parent_fit_ancestry=ancestry,scope='new camera only; no old CCD reuse',TRAIN=24 if pilot else 1600,TEST=0 if pilot else 411,VALID=0,source_git=git_head(),source_SHA256=sha(__file__),persistence_mode=persistence_mode,max_uncommitted_frames=1 if writer is not None else 0,same_connection_replug_recheck=getattr(a,'recheck_before_capture',False),signal_threshold_action='record_only' if record_only else 'stop',capture_policy_SHA256=sha(a.capture_policy) if record_only else None))
    with LOCK.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),stage=stage,session=str(session),camera='DVP',pilot=pilot)))
    try:
        with Bench(hardware,out) as bench:
            if capture_policy.get('geometry_check_once'):
                check=a.sequence/'calibration/geometry_once'
                if (check/'report.json').exists():
                    proof=read(check/'report.json');assert proof['passed'] and proof['coordinates_SHA256']==hardware['coordinates_SHA256']
                else:
                    assert a.depth==4 and stage=='router' and not pilot
                    shared_geometry_check(bench,check,hardware,warp)
            if resume_router:revalidate_replug(bench,out,hardware,warp)
            if getattr(a,'recheck_before_capture',False):
                assert not record_only,'User shared policy skips additional optical recalibration; transport checks remain in Bench'
                assert not pilot and not resume_router and hardware_digest!=base_digest and hardware.get('quality_mode')=='same_phase_contrast'
                originalpath=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(originalpath)==base_digest
                replug_check=same_connection_replug_check(bench,out,hardware,read(originalpath),warp)
            if getattr(a,'attempt',None) and getattr(a,'quality_recheck',False):shared_geometry_check(bench,out/'same_connection_quality_check',hardware,warp)
            white_run=0;compatibility_checked=False
            for role in ('train',) if pilot else ('train','test'):
                images,y,ids=data[role];n_total=24 if pilot else len(y);folder=out/role;folder.mkdir();ccd=np.lib.format.open_memmap(folder/'ccd.npy',mode='w+',dtype=np.uint8,shape=(n_total,478,478));last_end=0
                for begin in range(0,n_total,8):
                    x,n,end=padded(data,role,begin);assert end<=n_total
                    cache={k:np.asarray(v[role][begin:end]).copy() for k,v in previous.items()}
                    for k in cache:
                        if n<8:cache[k]=np.concatenate([cache[k],np.zeros((8-n,478,478),np.uint8)])
                    padded_ids=np.concatenate([ids[begin:end],np.array(['__padding__']*(8-n))]) if n<8 else ids[begin:end]
                    def physical(current,amplitude,phase,sample_ids):
                        nonlocal phase_pixels,frames,pending_record,white_run,compatibility_checked
                        if phase_pixels is None:
                            phase_pixels=phase.copy();Image.fromarray(phase_native(phase)).save(phasepath);bench.stage(phasepath)
                            if hardware.get('quality_mode')=='shared_fixed_record_only':assert sha(phasepath)==hardware['phase_SHA256']
                        assert current==stage and np.array_equal(phase_pixels,phase)
                        if quality_source and not compatibility_checked:
                            checks=[];assert sha(phasepath)==qr['phase_SHA256']
                            for retained in sorted(set((0,min(640,quality_count-1),quality_count-1))):
                                old=quality_source/'train'/f'{retained:06d}';source_row=read(old.with_suffix('.json'))
                                with Image.open(old.with_name(old.name+'_input.png')) as im:gray0=np.asarray(im).copy()
                                samplebmp=out/f'reuse_check_{retained:06d}.bmp';Image.fromarray(native(drive[gray0],'amplitude')).save(samplebmp);assert sha(samplebmp)==source_row['input_SHA256']
                                raw0,meta0=bench.capture(samplebmp);v=orient(cv2.warpPerspective(raw0,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),hardware['orientation']);oldv=np.asarray(quality_array[retained]).astype(float)
                                pcc=correlation(cv2.resize(v.astype(float),(16,16),interpolation=cv2.INTER_AREA),cv2.resize(oldv,(16,16),interpolation=cv2.INTER_AREA));ratio=float(v.mean()/max(oldv.mean(),1e-12));passed=pcc>=.95 and .8<=ratio<=1.2 and float((v==255).mean())<.15
                                Image.fromarray(raw0).save(out/f'reuse_check_{retained:06d}_raw.png',compress_level=1);Image.fromarray(v).save(out/f'reuse_check_{retained:06d}.png');checks.append(dict(index=retained,coarse16_PCC=pcc,mean_ratio=ratio,passed=passed,frame_metadata=meta0));save(out/'TRAIN_reuse_compatibility.json',dict(TRAIN_only=True,TEST_used=False,checks=checks,passed=all(c['passed'] for c in checks),thresholds=dict(coarse16_PCC=.95,mean_ratio=[.8,1.2])))
                                assert passed,'Retained TRAIN response changed: do not mix regimes or refit exposure'
                            compatibility_checked=True
                        values=[]
                        for j in range(n):
                            index=begin+j;t=time.perf_counter();gray=np.rint(amplitude[j].clip(0,1)*255).astype(np.uint8);assert np.max(np.abs(gray/255-amplitude[j]))<=1e-7
                            base=folder/f'{index:06d}';canonical=base.with_name(base.name+'_input.png');Image.fromarray(gray).save(canonical,compress_level=1);bmp=base.with_name(base.name+'_transient.bmp');Image.fromarray(native(drive[gray],'amplitude')).save(bmp)
                            if quality_source and role=='train' and index<quality_count:
                                old=quality_source/'train'/f'{index:06d}';row=read(old.with_suffix('.json'));assert row['canonical_SHA256']==sha(canonical) and row['input_SHA256']==sha(bmp) and row['phase_SHA256']==sha(phasepath)
                                crop=np.asarray(quality_array[index]).copy()
                                for suffix in ('_raw.png','.png'):os.link(old.with_name(old.name+suffix),base.with_name(base.name+suffix))
                                row=dict(row,reused_from=str(old.with_suffix('.json')),source_receipt_SHA256=sha(old.with_suffix('.json')),new_physical_capture=False)
                            elif reused is not None and role=='train' and index<24:
                                old=source/'train'/f'{index:06d}';row=read(old.with_suffix('.json'))
                                assert row['canonical_SHA256']==sha(canonical) and row['input_SHA256']==sha(bmp) and row['phase_SHA256']==sha(phasepath)
                                crop=np.asarray(reused[index]).copy()
                                for suffix in ('_raw.png','.png'):
                                    src=old.with_name(old.name+suffix);dest=base.with_name(base.name+suffix)
                                    os.link(src,dest)
                                row=dict(row,reused_from=str(old.with_suffix('.json')),source_receipt_SHA256=sha(old.with_suffix('.json')),new_physical_capture=False)
                            elif probe_directory is not None and role=='train' and index in hardware['probe_indices']:
                                old=probe_directory/f'{index:06d}';r=read(old.with_suffix('.json'));record=r['record'];meta=r['frame_metadata']
                                assert record['sample_id']==str(ids[index]) and record['index']==index and record['PT_SHA256']==digest and record['data_SHA256']==dm['sha256'] and record['stage']==stage
                                assert r['canonical_SHA256']==sha(canonical) and r['input_SHA256']==sha(bmp) and sha(phasepath)==hardware['phase_SHA256'] and r['raw_SHA256']==sha(old.with_name(old.name+'_raw.png'))
                                assert abs(meta['exposure_us']-hardware['exposure_us'])<2 and meta['gain']==1 and meta['shape']==hardware['sensor_shape'] and not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
                                with Image.open(old.with_suffix('.png')) as image:crop=np.asarray(image).copy()
                                for suffix in ('_raw.png','.png'):os.link(old.with_name(old.name+suffix),base.with_name(base.name+suffix))
                                row=dict(stage=stage,role=role,index=index,sample_id=str(ids[index]),PT_SHA256=digest,hardware_SHA256=hardware_digest,data_SHA256=dm['sha256'],phase_SHA256=sha(phasepath),canonical_SHA256=sha(canonical),input_SHA256=sha(bmp),raw_SHA256=r['raw_SHA256'],frame_metadata=meta,p99=r['p99'],mean=r['mean'],saturation_fraction=r['saturation_fraction'],complete_sample_ms=meta['capture_total_ms'],captured_at_ns=r['captured_at_ns'],new_physical_capture=False,reused_from=str(old.with_suffix('.json')),source_receipt_SHA256=sha(old.with_suffix('.json')))
                            else:
                                raw,meta=bench.capture(bmp);crop=orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),hardware['orientation'])
                                if replug_check is not None:assert meta['frame_id']>replug_check['last_recheck_frame_id']
                                rawfile=base.with_name(base.name+'_raw.png');raw_digest=None
                                if writer is None:
                                    Image.fromarray(raw).save(rawfile,compress_level=1);Image.fromarray(crop).save(base.with_suffix('.png'),compress_level=1);raw_digest=sha(rawfile)
                                else:
                                    try:finish_pending()
                                    except BaseException as error:
                                        # A previous disk failure must not silently discard the already-read NEXT frame.
                                        from lab_caltech20_dvp_persistence_20261019 import write_images
                                        rescue=write_images(base,raw,crop)
                                        save(out/'following_frame_rescue.json',dict(previous_error=repr(error),stage=stage,role=role,index=index,sample_id=str(ids[index]),PT_SHA256=digest,phase_SHA256=sha(phasepath),canonical_SHA256=sha(canonical),frame_metadata=meta,preserved=rescue,not_a_complete_capture_receipt=True));raise
                                row=dict(stage=stage,role=role,index=index,sample_id=str(ids[index]),PT_SHA256=digest,hardware_SHA256=hardware_digest,data_SHA256=dm['sha256'],phase_SHA256=sha(phasepath),canonical_SHA256=sha(canonical),input_SHA256=sha(bmp),raw_SHA256=raw_digest,frame_metadata=meta,p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=float((crop==255).mean()),complete_sample_ms=(time.perf_counter()-t)*1000,captured_at_ns=time.time_ns(),new_physical_capture=True)
                            if quality_background is not None:
                                from lab_caltech20_dvp_stage_exposure_20261019 import contrast_statistics
                                row.update(contrast_statistics(crop,quality_background,hardware['quality_background']['noise_sigma']))
                                row['background_quality_only_no_network_subtraction']=True
                            row.update(signal_threshold_action='record_only' if record_only else 'stop',signal_threshold_flags=signal_flags(row,hardware))
                            if writer is not None and row['new_physical_capture']:
                                writer.submit(base,raw,crop);pending_record=(base,row,index,n_total,role,t)
                            ccd[index]=crop;ccd.flush()
                            if not (writer is not None and row['new_physical_capture']):
                                finish_pending();record_frame(base,row,index,n_total,role)
                            values.append(crop)
                            # Transient BMP is redundant; canonical input, exact drive LUT and digest remain.
                            bmp.unlink()
                            if getattr(a,'attempt',None):
                                white_run=white_run+1 if row['saturation_fraction']>.99 else 0
                                if white_run>=2:
                                    finish_pending();raise RuntimeError('Whole-frame white event: two consecutive >99% clipped crops; preserve frames and stop for diagnosis, never train or auto-retry')
                            if not record_only and (row['p99']<15 or row['saturation_fraction']>.15):
                                finish_pending();raise RuntimeError('DVP signal guard: retain all valid/bad frames; no automatic retry or exposure change')
                            if not record_only and quality_background is not None and row['contrast_p99']<hardware['capture_min_contrast_p99']:
                                finish_pending();raise RuntimeError('DVP net-signal quality guard: raw CCD preserved; no background subtraction or automatic retry')
                        if n<8:values.extend([np.zeros((478,478),np.uint8)]*(8-n))
                        return np.stack(values)
                    result=layer_forward(model,x,padded_ids,stage,cache,physical);last_end=end
                finish_pending();assert last_end==n_total;arrays[role]=dict(samples=n_total,SHA256=sha(folder/'ccd.npy'))
        assert frames==(24 if pilot else 2011)
        audit={}
        for role,item in arrays.items():
            folder=out/role;stored=np.load(folder/'ccd.npy',mmap_mode='r');count=item['samples'];assert len(list(folder.glob('*_raw.png')))==count
            for i in range(count):
                base=folder/f'{i:06d}';r=read(base.with_suffix('.json'));assert r['sample_id']==str(data[role][2][i]) and r['PT_SHA256']==digest and r['hardware_SHA256']==hardware_digest and r['phase_SHA256']==sha(phasepath)
                assert r['canonical_SHA256']==sha(base.with_name(base.name+'_input.png'))
                with Image.open(base.with_suffix('.png')) as im:assert np.array_equal(np.asarray(im),stored[i])
            audit[role]=dict(receipts=count,crop_array_equality=True,IDs_PT_phase_canonical_SHA256=True)
        reused_calibration=len(hardware['probe_indices']) if probe_directory is not None else 0
        if replug_check is not None:
            save(out/'same_connection_release.json',dict(SDK_released=True,check_SHA256=sha(out/'same_connection_replug_check/report.json'),formal_capture_used_same_Bench=True,new_frame_IDs_after_recheck=True))
        save(out/'report.json',dict(state='complete',pilot=pilot,stage=stage,physical_ccd=frames,new_physical_ccd=frames-(24 if resume_router else 0)-reused_calibration-quality_count,reused_pilot_ccd=24 if resume_router else 0,reused_calibration_ccd=reused_calibration,reused_verified_TRAIN_prefix=quality_count,TRAIN=24 if pilot else 1600,TEST=0 if pilot else 411,VALID=0,sdk_released=True,arrays=arrays,data_SHA256=dm['sha256'],hardware_SHA256=hardware_digest,PT_SHA256=digest,prefix_SHA256=identities,phase_SHA256=sha(phasepath),elapsed_s=time.perf_counter()-start,median_complete_sample_ms=float(np.median(elapsed)),mean_complete_sample_ms=float(np.mean(elapsed)),fullraw_preserved=True,audit=audit))
        save(out/'persistence_report.json',dict(mode=persistence_mode,ordered_receipts=frames,TRAIN_TEST_boundaries_drained=True,SDK_calls_in_writer=False,max_uncommitted_frames=1 if writer is not None else 0,physical_timing_changed=False,completed_sample_rate_per_s=frames/(time.perf_counter()-start)))
    except BaseException as ex:
        persistence_error=None
        try:finish_pending()
        except BaseException as error:persistence_error=repr(error)
        save(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),frames=frames,persistence_error=persistence_error,automatic_retry=False));raise
    finally:
        try:
            if writer is not None:writer.close()
        finally:LOCK.unlink()


def fitting_model(a,data,dm,checkpoint,digest,prefix):
    session=session_for(a);model,transfer=model_for(checkpoint,digest,a.depth,session/'source_reference.ts');fields,identities=fields_for(session,prefix,data,dm,session_hardware_digest(a.sequence));return model,transfer,fields,identities


def fitting_keys(model,prefix,cfg):
    """Optional stricter freeze; never expand the original suffix scope."""
    keys=allowed_keys(model,prefix)
    assert isinstance(cfg.get('freeze_readout',False),bool)
    if cfg.get('freeze_readout',False):keys=keys-{'linear.weight','linear.bias'}
    assert keys and keys<=allowed_keys(model,prefix)
    return keys


def fitting_TRAIN_prior(cfg,dm):
    """Reuse a SHA-bound published TRAIN-only prior, never TEST frequencies."""
    name=cfg.get('TRAIN_prior_profile')
    if name is None:
        assert cfg.get('TRAIN_prior_profile_SHA256') is None
        return None
    path=HERE/name
    assert path.parent.resolve()==HERE.resolve() and sha(path)==cfg['TRAIN_prior_profile_SHA256']
    prior=read(path)
    assert prior['original_TRAIN_cache_SHA256']==dm['source_cache_SHA256']==SOURCE_CACHE
    counts=np.asarray(prior['original_TRAIN_counts'],dtype=np.float64)
    assert counts.shape==(20,) and np.isfinite(counts).all() and (counts>=60).all() and (counts==np.rint(counts)).all() and counts.sum()==3141
    return counts/counts.mean()


def fit(a,data,dm,checkpoint,digest,prefix,cfg):
    assert cfg.get('fit_depth',a.depth)==a.depth and cfg.get('fit_prefix',prefix)==prefix,'Profile is scoped to another depth/prefix'
    session=session_for(a);stage=(['router']+[str(k) for k in range(a.depth)])[prefix-1];out=session/'fits'/(stage+cfg.get('fit_name_suffix',''));assert not out.exists() and not LOCK.exists();out.mkdir(parents=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==cfg['GPU_UUID'] and torch.cuda.is_available() and 'A100' not in torch.cuda.get_device_name(0)
    # Caller has already released SDK. Reject any non-desktop foreign GPU user.
    from lab_caltech20_rich_tail_refine_20261018 import windows_process_names
    names=windows_process_names();own={os.getpid(),os.getppid()}
    for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).splitlines():
        gpu,pid,exe=row.split(',',2);pid=int(pid)
        if gpu.strip()==cfg['GPU_UUID']:assert pid in own or names.get(pid,Path(exe.strip()).name.lower()) in cfg['desktop_graphics_process_allowlist'],('Foreign GPU owner',pid,names.get(pid,exe))
    torch.cuda.set_per_process_memory_fraction(.65,0);torch.manual_seed(cfg['seed']);model,transfer,fields,identities=fitting_model(a,data,dm,checkpoint,digest,prefix);original=state_cpu(model);allowed=fitting_keys(model,prefix,cfg);prior=fitting_TRAIN_prior(cfg,dm);baseline,p=assess(model,data,fields,'test');best,bestloss=baseline['correct'],baseline['nll'];selected=0;target=math.ceil(cfg['models'][str(a.depth)]['simulation_correct']-411*cfg['gap_pp']/100)
    target=cfg.get('target_correct',target);budget=cfg['final_fit'] if prefix==a.depth+1 else cfg['middle_fit'];protocol=dict(parent_PT_SHA256=digest,depth=a.depth,prefix=prefix,allowed_keys=sorted(allowed),prefix_SHA256=identities,data_SHA256=dm['sha256'],hardware_SHA256=session_hardware_digest(a.sequence),TRAIN=1600,TEST=411,VALID=0,TEST_gradient=False,TEST_informed_development=True,baseline_TEST=baseline,target_correct=target,config=cfg,source_git=git_head(),source_SHA256=sha(__file__),teacher_scope='Frozen parent predictions on TRAIN only; no TEST teacher' if cfg.get('teacher_weight',0) else None)
    protocol['prefix_hardware_SHA256']={stage:read(session/'capture'/stage/'report.json')['hardware_SHA256'] for stage in fields}
    protocol['readout_frozen_by_config']=cfg.get('freeze_readout',False)
    protocol['original_TRAIN_prior_weights']=None if prior is None else prior.tolist()
    save(out/'protocol.json',protocol);np.savez_compressed(out/'baseline_TEST_predictions.npz',ids=data['test'][2],labels=data['test'][1],probabilities=p)
    def snapshot(name,epoch):
        state=state_cpu(model);assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);torch.save(dict(model=state,protocol=protocol,epoch=epoch),out/(name+'.pt'))
    snapshot('best',0);model.cuda();groups=[]
    for key,lr in (('phase_sets.',cfg['phase_lr']),('electronic.',cfg['electronic_lr']),('fusion_logits.',cfg['alpha_lr']),('linear.',cfg['head_lr'])):
        values=[p for n,p in model.named_parameters() if n in allowed and n.startswith(key)]
        if values:groups.append(dict(params=values,lr=lr,weight_decay=0 if key=='fusion_logits.' else cfg['weight_decay']))
    for n,p in model.named_parameters():p.requires_grad_(n in allowed)
    teacher=None
    if cfg.get('teacher_weight',0):
        model.eval();teacher=torch.empty((1600,20),device='cuda');temperature=cfg['teacher_temperature']
        offset=cfg.get('teacher_simulated_newest_layers',0);assert offset in (0,1) and offset<=len(fields)
        teacher_fields={k:fields[k] for k in list(fields)[:len(fields)-offset]};protocol['teacher_measured_prefix_stages']=list(teacher_fields)
        with torch.no_grad():
            for begin in range(0,1600,cfg['batch_size']):
                ix=np.arange(begin,min(1600,begin+cfg['batch_size']));model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()).cuda() for k,v in teacher_fields.items()}
                teacher[ix]=torch.softmax(model(encode_images(data['train'][0][ix].cuda()))[0].clamp_min(1e-12).log()/temperature,1)
        assert torch.isfinite(teacher).all() and not teacher.requires_grad
        np.save(out/'teacher_TRAIN_Tsoft.npy',teacher.detach().cpu().numpy());protocol['teacher_TRAIN_SHA256']=sha(out/'teacher_TRAIN_Tsoft.npy');save(out/'protocol.json',protocol)
    opt=torch.optim.AdamW(groups);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,budget['epochs'],eta_min=1e-5);history=[];start=time.monotonic();epoch=0
    for epoch in range(1,budget['epochs']+1):
        model.train();order=torch.randperm(1600).numpy();loss_sum=0.;ce_sum=0.;kl_sum=0.;used=0
        # TEST confusion controls modest emphasis of existing TRAIN classes only.
        cm=np.asarray(baseline['confusion_matrix']);support=cm.sum(1);errors=1-np.divide(cm.diagonal(),support,out=np.zeros(20),where=support>0);weights=torch.tensor(1+cfg['confusion_emphasis']*errors,dtype=torch.float32,device='cuda')
        if prior is not None:weights=weights*torch.tensor(prior,dtype=torch.float32,device='cuda')
        for begin in range(0,1600,cfg['batch_size']):
            ix=order[begin:begin+cfg['batch_size']];model.measured={k:torch.from_numpy(np.asarray(v['train'][ix]).copy()).cuda() for k,v in fields.items()};x=encode_images(data['train'][0][ix].cuda());y=torch.from_numpy(data['train'][1][ix].astype(np.int64)).cuda();opt.zero_grad(set_to_none=True);p=model(x)[0];lp=p.clamp_min(1e-12).log();ce=(1-cfg['label_smoothing'])*F.nll_loss(lp,y,reduction='none')-cfg['label_smoothing']*lp.mean(1);ce_loss=(ce*weights[y]).sum()/weights[y].sum();kl=lp.new_zeros(())
            if teacher is not None:kl=F.kl_div(torch.log_softmax(lp/temperature,1),teacher[ix],reduction='batchmean')*temperature**2
            loss=ce_loss+cfg.get('teacher_weight',0)*kl;assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for n,v in model.named_parameters() if n in allowed);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
            with torch.no_grad():
                for p in model.fusion_logits:
                    if p.requires_grad:p.clamp_(.4001001,.95)
            loss_sum+=float(loss)*len(ix);ce_sum+=float(ce_loss)*len(ix);kl_sum+=float(kl)*len(ix);used+=len(ix)
        scheduler.step();assert used==1600
        if epoch==1 or epoch%cfg['evaluate_every']==0 or epoch==budget['epochs'] or time.monotonic()-start>=budget['seconds']:
            model.eval();m,p=assess(model,data,fields,'test');baseline=m
            if improves(m['correct'],m['nll'],best,bestloss):best,bestloss=m['correct'],m['nll'];selected=epoch;snapshot('best',epoch)
            row=dict(epoch=epoch,TRAIN_loss=loss_sum/1600,TRAIN_CE=ce_sum/1600,TRAIN_parent_KL=kl_sum/1600,actual_TRAIN_samples=used,TEST_correct=m['correct'],TEST_NLL=m['nll'],TRAIN_class_weights=weights.detach().cpu().tolist(),selected_correct=best,selected_epoch=selected,elapsed_s=time.monotonic()-start);history.append(row);save(out/'history.json',history);save(session/'progress.json',dict(state='suffix_fitting',stage=stage,prefix=prefix,latest=row,target_correct=target));print(json.dumps(row),flush=True)
            if (best>=target and epoch>=cfg.get('minimum_epochs',1)) or time.monotonic()-start>=budget['seconds']:break
            model.train()
    model.cpu();snapshot('last',epoch);results={}
    for name in ('best','last'):
        pth=out/(name+'.pt');state=torch.load(pth,map_location='cpu',weights_only=False)['model'];assert all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed);model.load_state_dict(state);m,p=assess(model,data,fields,'test');np.savez_compressed(out/(name+'_TEST_predictions.npz'),ids=data['test'][2],labels=data['test'][1],probabilities=p);results[name]=dict(TEST_development=m,PT_SHA256=sha(pth),strict_CPU=True,predictions_SHA256=sha(out/(name+'_TEST_predictions.npz')))
    assert results['best']['TEST_development']['correct']==best
    gap_target=math.ceil(cfg['models'][str(a.depth)]['simulation_correct']-411*cfg['gap_pp']/100)
    save(out/'report.json',dict(state='complete',baseline_TEST=protocol['baseline_TEST'],results=results,selected_epoch=selected,actual_epochs=epoch,selected_correct=best,target_correct=target,within8pp=best>=gap_target,refinement_target_reached=best>=target,selected_e0=selected==0,intermediate=prefix<a.depth+1,frozen_parameters_unchanged=True,TEST_gradients=False,SDK_used=False,elapsed_s=time.monotonic()-start))
    del model,opt;torch.cuda.empty_cache();return out/'best.pt',results['best']['PT_SHA256'],read(out/'report.json')


def capture_worker(a,cfg,stage,checkpoint,digest,pilot=False,resume_router=False):
    """One SDK lifetime per fresh desktop child; no automatic hardware retry."""
    log=session_for(a)/(('router_pilot24' if pilot else 'capture_'+stage)+'_worker.log')
    assert not log.exists() and not LOCK.exists()
    command=[sys.executable,str(Path(__file__).resolve()),'--mode','capture','--sequence',str(a.sequence),'--profile',str(a.profile),'--depth',str(a.depth),'--stage',stage,'--checkpoint',str(checkpoint),'--digest',digest]
    if pilot:command.append('--pilot')
    if resume_router:command.append('--resume-router')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with log.open('x',encoding='utf-8',buffering=1) as stream:
        worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
        save(log.with_suffix('.launch.json'),dict(pid=worker.pid,command=command,source_git=git_head(),source_SHA256=sha(__file__),fresh_SDK_process=True,automatic_retry=False))
        rc=worker.wait()
    save(log.with_suffix('.exit.json'),dict(pid=worker.pid,returncode=rc,SDK_lock=LOCK.exists()))
    assert rc==0 and not LOCK.exists(),('Capture child failed; preserve all evidence, no retry',stage,rc,str(log))


def run(a):
    cfg=read(a.profile);torch.set_num_threads(cfg['cpu_threads']);resuming=a.mode=='resume';out=a.sequence/('controller_resume_USB_20261019' if resuming else 'controller')
    assert not out.exists() and not LOCK.exists()
    if resuming:
        import psutil
        original=read(a.sequence/'controller/started.json');assert not psutil.pid_exists(original['pid'])
        assert read(a.sequence/'controller/failure.json')['automatic_retry'] is False
        probe=read(a.sequence/'calibration/user_replug_settings_20261019/report.json')
        assert probe['state']=='original_admitted_settings_restored' and probe['SDK_released'] and probe['hardware_SHA256']==HARDWARE
        assert abs(probe['selected']['exposure_us']-5000.5)<2 and probe['selected']['gain']==1
    out.mkdir();data,dm=data_for(a.sequence)
    hardwarepath=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(hardwarepath)==HARDWARE;hardware=read(hardwarepath);assert sha(hardware['response_lut'])==hardware['response_lut_SHA256'] and sha(hardware['phase_lut'])==hardware['phase_LUT_SHA256']
    assert read(a.sequence/'calibration/admission/report.json')['passed'];assert shutil.disk_usage(a.sequence).free>cfg['minimum_free_GiB']*(1<<30)
    save(out/'started.json',dict(pid=os.getpid(),source_git=git_head(),source_SHA256=sha(__file__),config_SHA256=sha(a.profile),data_SHA256=dm['sha256'],hardware_SHA256=HARDWARE,order=cfg['order'],recurring=False,planned_CCD=24132,planned_reused_pilot_CCD=24 if resuming else 0,explicit_resume=resuming,TRAIN=1600,original_unique_TRAIN=1508,fixed_augmented_TRAIN=92,TEST=411,VALID=0));completed=[]
    for depth in cfg['order']:
        a.depth=depth;session=a.sequence/f'L{depth}';session.mkdir(exist_ok=True);spec=cfg['models'][str(depth)]
        if resuming and depth==4:
            previous_gate=read(session/'lab_gate.json');assert previous_gate['passed'] and previous_gate['PT_SHA256']==spec['PT_SHA256'] and previous_gate['reference_SHA256']==spec['reference_SHA256'] and previous_gate['data_SHA256']==dm['sha256']
            assert sha(session/'initial.pt')==spec['PT_SHA256'] and sha(session/'source_reference.ts')==spec['reference_SHA256']
            admitted_pilot(session,data,dm,spec['PT_SHA256'])
        else:gate(a,data,dm,spec)
        checkpoint=session/'initial.pt';digest=spec['PT_SHA256']
        # Recovery explicitly validates and reuses the existing pilot; no re-shoot.
        if not (resuming and depth==4):capture_worker(a,cfg,'router',checkpoint,digest,pilot=True)
        assert read(session/'router_pilot24/report.json')['sdk_released'];records=[]
        for prefix,stage in enumerate(['router']+[str(k) for k in range(depth)],1):
            capture_worker(a,cfg,stage,checkpoint,digest,resume_router=resuming and depth==4 and prefix==1);checkpoint,digest,report=fit(a,data,dm,checkpoint,digest,prefix,cfg);records.append(dict(stage=stage,selected_PT=str(checkpoint),PT_SHA256=digest,TEST_correct=report['selected_correct'],selected_epoch=report['selected_epoch'],within8pp=report['within8pp'],kind='full_measured' if prefix==depth+1 else 'measured_prefix_simulated_suffix'));save(session/'stage_history.json',records)
        score=records[-1]['TEST_correct'];reused_count=24 if resuming and depth==4 else 0;result=dict(state='complete',depth=depth,fully_measured=True,TEST_correct=score,TEST=411,accuracy=score/411,fixed_simulation_correct=spec['simulation_correct'],gap_pp=100*(spec['simulation_correct']-score)/411,PT=str(checkpoint),PT_SHA256=digest,SDK_released=not LOCK.exists(),physical_CCD=2011*(depth+1),new_physical_CCD=2011*(depth+1)-reused_count,pilot_CCD=24,reused_pilot_CCD=reused_count,TEST_development=True,records=records);save(session/'report.json',result);completed.append(result);save(out/'completed_models.json',completed);print(json.dumps(result),flush=True)
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
            if prefix in (2,depth):
                frozen=MeasuredPrefix(depth).eval();frozen.load_state_dict(state_cpu(m));frozen.measured=replay.measured
                narrow=fitting_keys(frozen,prefix,dict(freeze_readout=True));before=state_cpu(frozen)
                assert narrow==allowed-{'linear.weight','linear.bias'}
                for key,value in frozen.named_parameters():value.requires_grad_(key in narrow)
                observed=frozen(x)[0];assert torch.equal(observed,expected)
                optimizer=torch.optim.AdamW([value for value in frozen.parameters() if value.requires_grad],lr=.0002)
                (-observed[:,0].log().mean()).backward();optimizer.step();after=state_cpu(frozen)
                assert all(value.grad is None for key,value in frozen.named_parameters() if key not in narrow)
                assert all(value.grad is not None and torch.isfinite(value.grad).all() for key,value in frozen.named_parameters() if key in narrow)
                assert all(torch.equal(value,after[key]) for key,value in before.items() if key not in narrow)
    print(json.dumps(dict(passed=True,TRAIN1600_original1508_aug92=True,no_VALID_TEST_sources=True,depth4_6_prefix1_2_penultimate_final_forward_and_gradient_freeze=True,optional_frozen_readout_native_forward_identical_and_AdamW_freeze=True,SDK_opened=False,GPU_used=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('prepare','run','resume','capture','refine','recheck','admit-stage','fixed-hardware','selftest'));p.add_argument('--cache',type=Path);p.add_argument('--out',type=Path);p.add_argument('--sequence',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--depth',type=int,choices=(4,6));p.add_argument('--stage');p.add_argument('--prefix',type=int);p.add_argument('--checkpoint',type=Path);p.add_argument('--digest');p.add_argument('--hardware-manifest',type=Path);p.add_argument('--parent-fit');p.add_argument('--archive-failed',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--resume-router',action='store_true');p.add_argument('--async-png',action='store_true');p.add_argument('--recheck-before-capture',action='store_true');p.add_argument('--archive-empty-start',action='store_true');p.add_argument('--capture-policy',type=Path)
    p.add_argument('--attempt',choices=('strict8pp_20261020','response_20261020'));p.add_argument('--reuse-capture',type=Path);p.add_argument('--reuse-train-prefix',type=int,default=0);p.add_argument('--quality-recheck',action='store_true');a=p.parse_args()
    assert not (a.reuse_capture or a.reuse_train_prefix or a.quality_recheck) or (a.attempt and a.mode=='capture'), 'Quality recovery belongs only to explicit strict capture'
    assert not a.attempt or a.mode in ('capture','refine','fixed-hardware'), 'Explicit stages only; no legacy automatic chaining in a strict attempt'
    try:
        if a.mode=='selftest':selftest()
        elif a.mode=='fixed-hardware':prepare_fixed_hardware(a)
        elif a.mode=='prepare':prepare(a)
        elif a.mode=='admit-stage':admit_stage(a)
        elif a.mode=='refine':
            cfg=read(a.profile)
            if 'base_profile' in cfg:
                base=HERE/cfg['base_profile'];assert base.parent.resolve()==HERE.resolve() and sha(base)==cfg['base_profile_SHA256'];cfg=dict(read(base),**cfg['overrides'],refinement_scope=cfg['scope'],refinement_profile_SHA256=sha(a.profile))
            torch.set_num_threads(cfg['cpu_threads']);data,dm=data_for(a.sequence);assert cfg.get('fit_name_suffix') and sha(a.checkpoint)==a.digest
            fit(a,data,dm,a.checkpoint,a.digest,a.prefix,cfg)
        elif a.mode=='recheck':
            import cv2
            assert not LOCK.exists() and not a.out.exists();a.out.mkdir(parents=True);hardwarepath=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(hardwarepath)==HARDWARE;hardware=read(hardwarepath)
            warp=cv2.getPerspectiveTransform(np.float32(corners(Path(hardware['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
            with LOCK.open('x') as stream:stream.write(json.dumps(dict(pid=os.getpid(),kind='same_settings_known_pattern_recheck',out=str(a.out))))
            try:
                with Bench(hardware,a.out) as bench:revalidate_replug(bench,a.out,hardware,warp)
            finally:LOCK.unlink()
        elif a.mode=='capture':
            cfg=read(a.profile);torch.set_num_threads(cfg['cpu_threads']);data,dm=data_for(a.sequence);hardwarepath=a.sequence/'calibration/admission/admitted_hardware.json';assert sha(hardwarepath)==session_hardware_digest(a.sequence);hardware=read(hardwarepath)
            if a.hardware_manifest:hardware=validate_stage_hardware(a.hardware_manifest,a.sequence)
            assert sha(hardware['response_lut'])==hardware['response_lut_SHA256'] and sha(hardware['phase_lut'])==hardware['phase_LUT_SHA256']
            assert sha(a.checkpoint)==a.digest;acquire(a,a.stage,a.checkpoint,a.digest,data,dm,hardware,pilot=a.pilot,resume_router=a.resume_router)
        else:run(a)
    except BaseException as ex:
        folder=a.sequence/('controller_resume_USB_20261019' if a.mode=='resume' else 'controller') if a.mode in ('run','resume') else a.out
        if folder and folder.exists() and not (folder/'failure.json').exists():save(folder/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc(),automatic_retry=False))
        raise
