"""Bounded offline TRAIN-only operation audit. No SDK, fitting or TEST images."""
import argparse,hashlib,io,json,math,time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import lab_caltech10_20261015 as l
from lab_depth_capture_20261016 import load_model


def correlation(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>0 else None


def views(a):
    return {**{f'rot{k*90}':np.rot90(a,k) for k in range(4)},
            **{f'flip_lr_rot{k*90}':np.rot90(np.fliplr(a),k) for k in range(4)}}


def tests():
    a=np.arange(25).reshape(5,5).astype(float)
    assert abs(correlation(a,a)-1)<1e-12
    assert abs(correlation(a,-a)+1)<1e-12
    assert correlation(np.ones((5,5)),a) is None
    assert np.array_equal(views(a)['rot0'],a) and len(views(a))==8


@torch.inference_mode()
def audit(a):
    tests();assert not a.out.exists();a.out.mkdir(parents=True)
    started=time.perf_counter();torch.set_num_threads(4)
    assets=a.session/'assets';images,labels,ids=l.dataset(assets,'train')
    assert len(ids)==len(labels)==761
    manifest=json.loads((assets/'depth_manifest_20261016.json').read_text())
    rows=[];contracts=[]
    for depth in (2,3,4):
        train=a.session/'full_train_20261016' if depth==3 else a.session/f'depth_L2L4_20261016/L{depth}/train'
        contract=json.loads((train/'contract.json').read_text());report=json.loads((train/'report.json').read_text())
        assert contract['role']=='train' and contract['samples']==761
        assert report['state']=='complete' and report['physical_ccd']==2*761*(depth+1)
        pts=np.float32(contract['corners_TL_TR_BR_BL']);warp=cv2.getPerspectiveTransform(pts,np.float32([[0,0],[477,0],[477,477],[0,477]]))
        contracts.append({'depth':depth,'data_sha256':contract['data_sha256'],'coordinates_sha256':contract['coordinates_sha256'],'response_lut_sha256':contract['response_lut_sha256'],'orientation':contract['camera_orientation'],'corners':pts.tolist()})
        assert contract['data_sha256']==l.sha(assets/'caltech10.npz') and contract['camera_orientation']=='flip_v'
        response=json.loads((a.session/'response621/response_lut.json').read_text());drive=np.asarray(response['drive'],np.uint8)
        assert contract['response_lut_sha256']==l.sha(a.session/'response621/response_lut.json')
        for arm in ('residual','control'):
            model=l.model_for(assets,arm) if depth==3 else load_model(assets,manifest,depth,arm)
            routing=np.load(train/arm/'routing.npy',allow_pickle=False)
            assert routing.shape==(761,4)
            for stage in ['router']+[str(k) for k in range(depth)]:
                folder=train/arm/stage;native_phase=np.asarray(Image.open(folder/'phase.bmp'))
                phase=np.asarray(Image.fromarray(np.flipud(np.fliplr(255-native_phase[92:1108,452:1468]))).resize((478,478),Image.Resampling.NEAREST))
                for index in (0,380,760):
                    receipt=json.loads((folder/f'{index:06d}.json').read_text());assert receipt['index']==index and receipt['sample_id']==str(ids[index])
                    assert receipt['phase_sha256']==l.sha(folder/'phase.bmp')
                    input_path=folder/f'{index:06d}_input.png';assert receipt['canonical_sha256']==l.sha(input_path)
                    gray=np.asarray(Image.open(input_path));assert gray.shape==(478,478)
                    if stage in ('router','0'):amplitude=l.encode(images[index:index+1],None,100)
                    else:
                        previous=np.load(train/arm/str(int(stage)-1)/'ccd.npy',mmap_mode='r')[index:index+1].copy()
                        amplitude=model.nonlinear(torch.from_numpy(previous).float(),model.stages[int(stage)])
                    route=None if stage=='router' else torch.from_numpy(routing[index:index+1]).float()
                    expected_gray,expected_phase=l.encode_stage(model,amplitude,stage,route)
                    assert np.array_equal(gray,expected_gray[0]) and np.array_equal(phase,expected_phase),'Saved encoding differs from exact same-model TRAIN operation'
                    buf=io.BytesIO();Image.fromarray(l.native(drive[gray],'amplitude')).save(buf,format='BMP')
                    assert hashlib.sha256(buf.getvalue()).hexdigest()==receipt['amplitude_sha256']
                    raw=np.asarray(Image.open(folder/f'{index:06d}_raw.png'))
                    reconstructed=np.flipud(cv2.warpPerspective(raw,warp,(478,478),flags=cv2.INTER_AREA,borderMode=cv2.BORDER_CONSTANT,borderValue=0)).copy()
                    observed=np.asarray(Image.open(folder/f'{index:06d}.png'))
                    assert np.array_equal(reconstructed,observed)
                    assert np.array_equal(np.load(folder/'ccd.npy',mmap_mode='r')[index],observed)
                    field=torch.from_numpy(gray.astype(np.float32)/255)*torch.exp(2j*math.pi*torch.from_numpy(phase.astype(np.float32)/256))
                    ideal=model.propagator(torch.nn.functional.pad(field[None],(20,20,20,20))).abs().square()[0,20:498,20:498].numpy()
                    geometry={name:correlation(ideal,v) for name,v in views(observed).items()}
                    best=max(geometry,key=lambda k:geometry[k] if geometry[k] is not None else -2)
                    rows.append({'depth':depth,'arm':arm,'stage':stage,'TRAIN_index':index,'TRAIN_label':int(labels[index]),'saved_encoding_and_raw_crop_exact':True,'identity_ideal_vs_CCD_PCC':geometry['rot0'],'D4_diagnostic_best_view':best,'D4_diagnostic_best_PCC':geometry[best],'D4_PCC':geometry,'observed_mean':float(observed.mean()),'observed_p99':float(np.percentile(observed,99)),'ideal_mean':float(ideal.mean()),'ideal_p99':float(np.percentile(ideal,99))})
                    if time.perf_counter()-started>180:raise TimeoutError('TRAIN-only audit exceeded180s; no retry')
    assert len({v['data_sha256'] for v in contracts})==len({v['coordinates_sha256'] for v in contracts})==len({v['response_lut_sha256'] for v in contracts})==1
    groups=[]
    for depth in (2,3,4):
        for arm in ('residual','control'):
            selected=[v for v in rows if v['depth']==depth and v['arm']==arm]
            for stage in ['router']+[str(k) for k in range(depth)]:
                values=[v for v in selected if v['stage']==stage]
                groups.append({'depth':depth,'arm':arm,'stage':stage,'sampled_TRAIN_operations':len(values),'identity_PCC_median':float(np.median([v['identity_ideal_vs_CCD_PCC'] for v in values])),'best_D4_PCC_median':float(np.median([v['D4_diagnostic_best_PCC'] for v in values])),'diagnostic_best_views':[v['D4_diagnostic_best_view'] for v in values]})
    result={'state':'complete','source_sha256':l.sha(__file__),'TRAIN_only':True,'TEST_arrays_read':0,'SDK_opened':False,'optimizer_steps':0,'saved_operations_checked':len(rows),'selected_TRAIN_indices':[0,380,760],'all_saved_inputs_phases_raw_crops_match':True,'physical_alignment_or_accuracy_acceptance':False,'geometry_transform_applied_to_deployment':False,'contracts':contracts,'groups':groups,'rows':rows,'elapsed_s':time.perf_counter()-started}
    (a.out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','contracts')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true');parser.add_argument('--session',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.selftest:tests();print('Synthetic correlation/view checks passed; no data/devices/models used')
    else:
        assert args.session and args.out
        try:audit(args)
        except BaseException as error:
            if args.out.exists():(args.out/'failure.json').write_text(json.dumps({'error':repr(error),'SDK_opened':False,'optimizer_steps':0,'automatic_retry':False}),encoding='utf-8')
            raise
