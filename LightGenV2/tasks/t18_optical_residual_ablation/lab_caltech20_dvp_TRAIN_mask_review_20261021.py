"""One TRAIN-only physical mask comparison; never a stage/deployment admission.

The response-trained mask's native-prefix score is below the deployment gate.
This diagnostic can measure its response, but cannot waive that gate, write PTs,
fit TEST, change a LUT, or capture any next optical layer.
"""
import argparse,json,os,time,traceback
from pathlib import Path

CASES=(
 ('reference_before','L4_response_20261020','2_current_response_pair_20261021','8c7443c82b8218113105bfbfec505bb2a595592d69b8921b375b49502a879bfe'),
 ('response_trained','L4_strict8pp_20261020','1_spatial_last_binary_continue_20261020','1eb95f96da619ac2be04b11c6647167c363a5bc4b7fc727c90f5523df3b688bc'),
 ('reference_after','L4_response_20261020','2_current_response_pair_20261021','8c7443c82b8218113105bfbfec505bb2a595592d69b8921b375b49502a879bfe'))
HARDWARE='0c6b75372fb5d2fc51f0cc2ae14eee3b5babac2cb8b62fc8709e9ac3a8417982'


def prepare_cases(sequence):
    import numpy as np
    import torch
    from PIL import Image
    import lab_caltech20_dvp_layerwise_20261019 as m
    torch.set_num_threads(2)
    assert m.session_hardware_digest(sequence)==HARDWARE
    data,dm=m.data_for(sequence);lineage=m.read(sequence/'assets/TRAIN_lineage.json')['records']
    assert all(row['kind']=='original' and row['label']==i for i,row in enumerate(lineage[:20]))
    session=sequence/'L4_response_20261020';fields,identities=m.fields_for(session,3,data,dm,HARDWARE)
    parent=session/'fits/1_binary_repaired_20261020/best.pt'
    assert m.sha(parent)=='15d290f1d59ead1a09db37781891a34c5d76943c4a62189ab25a1544c8758fc8'
    base,_=m.model_for(parent,m.sha(parent),4,session/'source_reference.ts');original=m.state_cpu(base)
    prepared=[]
    for tag,namespace,fit,digest in CASES:
        folder=sequence/namespace/'fits'/fit;r,p,a=[m.read(folder/n) for n in ('report.json','protocol.json','independent_audit.json')]
        assert r['state']=='complete' and a['passed'] and a['report_SHA256']==m.sha(folder/'report.json')
        assert a['protocol_SHA256']==m.sha(folder/'protocol.json') and m.sha(folder/'best.pt')==digest==r['results']['best']['PT_SHA256']
        assert p['data_SHA256']==dm['sha256'] and all(p['prefix_SHA256'][k]==v for k,v in identities.items())
        model,_=m.model_for(folder/'best.pt',digest,4,session/'source_reference.ts');state=m.state_cpu(model);allowed=m.allowed_keys(model,3)
        assert set(state)==set(original) and all(torch.equal(v,state[k]) for k,v in original.items() if k not in allowed)
        inputs=[];phase_pixels=[]
        for begin in (0,8,16):
            x,n,end=m.padded(data,'train',begin);n=min(n,20-begin);ids=data['train'][2][begin:begin+8]
            previous={k:np.asarray(v['train'][begin:begin+8]).copy() for k,v in fields.items()}
            def planned(stage,amplitude,phase,sample_ids):
                assert stage=='2' and phase.shape==(478,478)
                phase_pixels.append(phase.copy())
                gray=np.rint(amplitude[:n].clip(0,1)*255).astype(np.uint8)
                assert np.max(np.abs(gray/255-amplitude[:n]))<=1e-7
                inputs.extend(gray)
                return np.zeros((8,478,478),np.uint8)
            with torch.no_grad():m.layer_forward(model,x,ids,'2',previous,planned)
        assert len(inputs)==20 and len(phase_pixels)==3 and all(np.array_equal(v,phase_pixels[0]) for v in phase_pixels)
        if tag.startswith('reference'):
            capture=session/'capture/2';assert np.array_equal(m.phase_native(phase_pixels[0]),np.asarray(Image.open(capture/'phase.bmp')))
            assert all(np.array_equal(gray,np.asarray(Image.open(capture/'train'/f'{i:06d}_input.png'))) for i,gray in enumerate(inputs))
        prepared.append(dict(tag=tag,folder=folder,digest=digest,model=model,inputs=inputs,phase=phase_pixels[0],source_report_SHA256=m.sha(folder/'report.json'),source_audit_SHA256=m.sha(folder/'independent_audit.json')))
    assert not np.array_equal(prepared[0]['phase'],prepared[1]['phase'])
    return m,data,dm,fields,identities,prepared


def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import cv2,numpy as np
    from PIL import Image
    m,data,dm,fields,identities,cases=prepare_cases(a.sequence)
    from lab_caltech20_dvp_TRAIN_response_20261020 import protected_paths
    assert not a.out.exists() and not m.LOCK.exists()
    cfg=m.read(a.sequence/'calibration/admission/admitted_hardware.json')
    assert cfg['exposure_us']==60006. and cfg['gain']==1. and cfg['wait_ms']==240 and cfg['orientation']=='flip_v'
    protected={str(p):m.sha(p) for p in protected_paths(cfg,a.sequence)}
    geometry=m.read(a.sequence/'calibration/geometry_once/report.json');assert geometry['passed'] and geometry['coordinates_SHA256']==cfg['coordinates_SHA256']
    a.out.mkdir(parents=True);drive=np.asarray(m.read(cfg['response_lut'])['drive'],np.uint8)
    warp=cv2.getPerspectiveTransform(np.float32(m.corners(Path(cfg['coordinates']))),np.float32([[0,0],[477,0],[477,477],[0,477]]))
    protocol=dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),hardware_SHA256=HARDWARE,protected=protected,
        data_SHA256=dm['sha256'],prefix_SHA256=identities,TRAIN_indices=list(range(20)),TRAIN_unique=20,
        TEST_inference=False,VALID=0,new_PT_written=False,optimizer_steps=0,physical_frames=60,
        diagnostic_only=True,stage_admission=False,next_layer_capture=False,automatic_retry=False,
        exposure_us=60006.,gain=1.,wait_ms=240,phase_LUT_scope='linear voltage; NOT verified linear phase',
        cases=[{k:v[k] for k in ('tag','digest','source_report_SHA256','source_audit_SHA256')} for v in cases])
    m.save(a.out/'protocol.json',protocol);records=[];arrays={};white=0;started=time.monotonic()
    nominal=a.sequence/'L4_response_20261020/capture/2/phase.bmp'
    with m.LOCK.open('x') as stream:json.dump(dict(pid=os.getpid(),kind='TRAIN_mask_response_comparison',out=str(a.out)),stream)
    try:
        with m.Bench(cfg,a.out) as bench:
            try:
                for case in cases:
                    folder=a.out/case['tag'];folder.mkdir();phase=folder/'phase.bmp';Image.fromarray(m.phase_native(case['phase'])).save(phase);bench.stage(phase);values=[]
                    for i,gray in enumerate(case['inputs']):
                        stem=folder/f'{i:06d}';canonical=stem.with_name(stem.name+'_input.png');bmp=stem.with_suffix('.bmp')
                        Image.fromarray(gray).save(canonical);Image.fromarray(m.native(drive[gray],'amplitude')).save(bmp)
                        raw,meta=bench.capture(bmp);crop=m.orient(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0),cfg['orientation'])
                        rawpath=stem.with_name(stem.name+'_raw.png');croppath=stem.with_suffix('.png');Image.fromarray(raw).save(rawpath,compress_level=1);Image.fromarray(crop).save(croppath,compress_level=1)
                        sat=float((crop==255).mean());white=white+1 if sat>.99 else 0
                        row=dict(tag=case['tag'],index=i,sample_id=str(data['train'][2][i]),PT_SHA256=case['digest'],phase_SHA256=m.sha(phase),canonical_SHA256=m.sha(canonical),BMP_SHA256=m.sha(bmp),raw_SHA256=m.sha(rawpath),crop_SHA256=m.sha(croppath),p99=float(np.percentile(crop,99)),mean=float(crop.mean()),saturation_fraction=sat,frame_metadata=meta)
                        m.save(stem.with_suffix('.json'),row);records.append(row);values.append(crop)
                        m.save(a.out/'progress.json',dict(state='TRAIN_mask_response_comparison',frames=len(records),expected=60,latest=row))
                        assert white<2,'Two consecutive whole-crop white frames; preserve all evidence, no retry'
                    arrays[case['tag']]=np.stack(values);np.save(folder/'ccd.npy',arrays[case['tag']])
            finally:bench.stage(nominal)
        assert len(records)==60 and protected=={p:m.sha(p) for p in protected}
        small={'train':(data['train'][0][:20],data['train'][1][:20],data['train'][2][:20])};results=[]
        for case in cases:
            observed={k:{'train':np.asarray(v['train'][:20]).copy()} for k,v in fields.items()};observed['2']={'train':arrays[case['tag']]}
            metric,prob=m.assess(case['model'],small,observed,'train');np.savez_compressed(a.out/case['tag']/'TRAIN_predictions.npz',ids=small['train'][2],labels=small['train'][1],probabilities=prob)
            results.append(dict(tag=case['tag'],PT_SHA256=case['digest'],TRAIN_correct=metric['correct'],TRAIN_NLL=metric['nll'],array_SHA256=m.sha(a.out/case['tag']/'ccd.npy'),predictions_SHA256=m.sha(a.out/case['tag']/'TRAIN_predictions.npz')))
        m.save(a.out/'report.json',dict(state='complete',response_frames=60,TRAIN_unique=20,TEST_inference=False,SDK_released=True,settings_LUT_original_PT_unchanged=True,new_PT_written=False,nominal_restored=True,diagnostic_only=True,stage_admission=False,records=records,results=results,elapsed_s=time.monotonic()-started,protocol_SHA256=m.sha(a.out/'protocol.json')))
        print(json.dumps(dict(complete=True,TRAIN_only=True,results=results,next_layer_admitted=False)),flush=True)
    except BaseException as error:
        m.save(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),frames=len(records),automatic_retry=False,all_evidence_preserved=True));raise
    finally:
        assert m.read(m.LOCK)['pid']==os.getpid();m.LOCK.unlink()


def audit(a):
    import numpy as np
    from PIL import Image
    m,data,dm,fields,identities,cases=prepare_cases(a.sequence);r=m.read(a.out/'report.json');p=m.read(a.out/'protocol.json')
    assert r['state']=='complete' and r['response_frames']==60 and len(r['records'])==60 and len(r['results'])==3 and r['SDK_released'] and not r['stage_admission'] and not m.LOCK.exists()
    assert r['protocol_SHA256']==m.sha(a.out/'protocol.json') and p['source_SHA256']==m.sha(__file__) and p['protected']=={name:m.sha(name) for name in p['protected']}
    for case,score in zip(cases,r['results']):
        folder=a.out/case['tag'];array=np.load(folder/'ccd.npy');assert array.shape==(20,478,478) and array.dtype==np.uint8 and m.sha(folder/'ccd.npy')==score['array_SHA256'] and score['tag']==case['tag']
        assert np.array_equal(np.asarray(Image.open(folder/'phase.bmp')),m.phase_native(case['phase']))
        for i,gray in enumerate(case['inputs']):
            stem=folder/f'{i:06d}';row=m.read(stem.with_suffix('.json'));assert row['PT_SHA256']==case['digest'] and row['sample_id']==str(data['train'][2][i])
            for key,path in (('phase_SHA256',folder/'phase.bmp'),('canonical_SHA256',stem.with_name(stem.name+'_input.png')),('BMP_SHA256',stem.with_suffix('.bmp')),('raw_SHA256',stem.with_name(stem.name+'_raw.png')),('crop_SHA256',stem.with_suffix('.png'))):assert m.sha(path)==row[key]
            assert np.array_equal(gray,np.asarray(Image.open(stem.with_name(stem.name+'_input.png')))) and np.array_equal(array[i],np.asarray(Image.open(stem.with_suffix('.png'))))
            meta=row['frame_metadata'];assert abs(meta['exposure_us']-60006)<2 and meta['gain']==1 and meta['shape']==[3648,5480] and not any(meta[k] for k in ('flip_h','flip_v','rotate','opposite'))
        observed={k:{'train':np.asarray(v['train'][:20]).copy()} for k,v in fields.items()};observed['2']={'train':array};small={'train':(data['train'][0][:20],data['train'][1][:20],data['train'][2][:20])}
        metric,prob=m.assess(case['model'],small,observed,'train');assert metric['correct']==score['TRAIN_correct']
        with np.load(folder/'TRAIN_predictions.npz') as z:assert np.array_equal(z['ids'],small['train'][2]) and np.max(np.abs(z['probabilities']-prob))<=3e-6
    proof=dict(passed=True,report_SHA256=m.sha(a.out/'report.json'),protocol_SHA256=m.sha(a.out/'protocol.json'),physical_frames=60,TRAIN_unique=20,TEST_inference=False,stage_admission=False,SDK_opened=False,original_graph_replay=True,IDs_phase_input_raw_crop_array_PT_SHA256=True)
    assert not (a.out/'independent_audit.json').exists();m.save(a.out/'independent_audit.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('run','audit','plan'),required=True);parser.add_argument('--sequence',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.mode=='plan':
        m,_,_,_,_,cases=prepare_cases(args.sequence);print(json.dumps(dict(passed=True,TRAIN_only=True,physical_frames=0,exact_reference_canonical_phase=True,candidate_distinct_phase=True,all_measured_upstream_frozen=True,cases=[dict(tag=v['tag'],PT_SHA256=v['digest']) for v in cases])))
    else:run(args) if args.mode=='run' else audit(args)
