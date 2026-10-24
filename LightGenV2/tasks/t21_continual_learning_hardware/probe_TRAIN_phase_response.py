"""Fixed20-frame TRAIN phase-response diagnosis, no calibration or refitting.

Canonical/zero/uniformpi/checkerpi at the SAME two fixed TRAIN inputs. All
individual reads saved. No averaged network frame, EVAL acquisition, changed
exposure/coordinates/LUT/weights, or automatically restarted camera session.
"""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .hardware_bridge import hardware,load_checkpoint,read_source,native_image,crop_native,signal
from .train_native import sha,put


def control_phases(canonical):
    if canonical.shape!=(1016,1016):raise ValueError('Exact native phase aperture required')
    y,x=np.indices((1016,1016));checker=((y//32+x//32)%2)*.5
    return dict(canonical=canonical,zero=np.zeros_like(canonical),uniform_pi=np.full_like(canonical,.5),checker_pi=checker.astype(canonical.dtype))


def normalized(value):return value/max(float(value.mean()),1e-20)


def difference(a,b):
    a=normalized(a).ravel();b=normalized(b).ravel()
    return dict(correlation=float(np.corrcoef(a,b)[0,1]),relative_L2=float(np.linalg.norm(a-b)/max(1e-20,np.linalg.norm(a))))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('package','hardware','capture','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    package=json.loads(a.package.read_text());cfg=json.loads(a.hardware.read_text());report=json.loads((a.capture/'report.json').read_text())
    if package['architecture']!='d2nn' or sha(a.hardware)!=package['hardware_sha256'] or not report['passed'] or not report['SDK_released']:
        raise ValueError('Complete immutable D2NN source and hardware required')
    if report['checkpoint_sha256']!=package['checkpoint_sha256'] or sha(a.capture/'features.npz')!=report['features_sha256']:raise ValueError('Source identity differs')
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    spec=next(v for v in package['datasets'] if v['task']=='clevr' and v['split']=='train')
    if spec['indices']!=list(range(1000)):raise ValueError('Original complete current TRAIN required')
    source=read_source(spec['path'],spec['sha256'],'train',np.array([0,1]))
    if source['source_indices'][1]!=source['source_indices'][0]+1 or source['y'].tolist()!=[1,0]:raise ValueError('Fixed first TRAIN pair differs')
    with np.load(a.capture/'features.npz',allow_pickle=False) as d:
        masks=[(d['tasks']=='clevr')&(d['splits']=='train')&(d['source_indices']==i) for i in source['source_indices']]
        if any(int(v.sum())!=1 for v in masks):raise ValueError('Ambiguous TRAIN source identity')
        rows=[int(np.flatnonzero(v)[0]) for v in masks]
    inputs=[];amplitudes=[];receipts=[];canonical=None
    for j,i in enumerate(rows):
        folder=a.capture/'first'/f'{i:05d}';receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['split']!='train' or receipt['source_index']!=int(source['source_indices'][j]) or sha(folder/'input.bmp')!=receipt['input_BMP_sha256']:
            raise ValueError('Actual displayed TRAIN source differs')
        amp,phase,_=model.payload(torch.from_numpy(source['x'][j])[None].float(),'first',{})
        amplitudes.append(amp);canonical=phase.numpy()
        inputs.append(folder/'input.bmp');receipts.append(receipt)
    phasefile=a.capture/'first/phase.bmp'
    if any(sha(phasefile)!=r['phase_BMP_sha256'] for r in receipts):raise ValueError('Source canonical phase changed')
    image=native_image(np.rint(canonical*256).clip(0,255).astype(np.uint8),True)
    if not np.array_equal(image,np.array(Image.open(phasefile))):raise ValueError('Canonical model phase differs before SDK')
    for key,digest in [('coordinates','coordinates_SHA256'),('phase_lut','phase_LUT_SHA256'),('response_lut','response_lut_SHA256')]:
        if sha(cfg[key])!=cfg[digest]:raise ValueError('Hardware artifact differs before SDK')
    drive=np.array(json.loads(Path(cfg['response_lut']).read_text())['drive'],dtype=np.uint8)
    for amp,inputfile in zip(amplitudes,inputs):
        gray=np.rint(amp[0].numpy()*255).clip(0,255).astype(np.uint8)
        if not np.array_equal(native_image(drive[gray]),np.array(Image.open(inputfile))):raise ValueError('Model and actual TRAIN input BMP differ before SDK')
    phases=control_phases(canonical);simulation={}
    for name,phase in phases.items():
        simulation[name]=[model.propagate_payload(amp,torch.from_numpy(phase)).cpu().numpy()[0] for amp in amplitudes]
    sim_compare={name:[difference(simulation['canonical'][j],simulation[name][j]) for j in range(2)] for name in phases}
    if a.prepare_only:
        print(json.dumps(dict(passed=True,TRAIN_source_indices=source['source_indices'].tolist(),expected_frames=20,
            simulation_phase_comparison=sim_compare,SDK_opened=False,run_written=False)));return
    Bench,LOCK,_,drive,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK is owned, no takeover')
    a.out.mkdir(parents=True,exist_ok=False)
    put(a.out/'protocol.json',dict(package_SHA256=sha(a.package),hardware_SHA256=sha(a.hardware),source_checkpoint_SHA256=package['checkpoint_sha256'],
        source_capture=str(a.capture),source_features_SHA256=report['features_sha256'],TRAIN_source_indices=source['source_indices'].tolist(),source_physical_rows=rows,
        phase_conditions=list(phases),expected_frames=20,exposure_us=cfg['exposure_us'],coordinates_changed=False,LUT_changed=False,weights_changed=False,
        averaged_network_capture=False,old_network_CCD_replaced=False,TEST_used=False,labels_for_pair_identity_only=True,
        simulation_phase_comparison=sim_compare,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__)))
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='fixed20_TRAIN_phase_response'));values={};frames=[];restored=False
    try:
        with Bench(cfg,a.out) as bench:
            try:
                black=a.out/'black.bmp';Image.fromarray(native_image(np.zeros((1016,1016),np.uint8))).save(black)
                current_phase=phasefile
                def snap(name,inputfile):
                    raw,metadata=bench.capture(inputfile);value=crop_native(raw,warp);np.save(a.out/(name+'_raw.npy'),raw);np.save(a.out/(name+'_ccd.npy'),value)
                    s=signal(value);row=dict(name=name,signal=s,metadata=metadata,raw_SHA256=sha(a.out/(name+'_raw.npy')),ccd_SHA256=sha(a.out/(name+'_ccd.npy')),phase_BMP_SHA256=sha(current_phase));frames.append(row)
                    put(a.out/(name+'_receipt.json'),row)
                    put(a.out/'progress.json',dict(frames=len(frames),latest=row,updated=time.time()))
                    if s['saturation_fraction']>.15:raise RuntimeError('Control saturation, preserve frame without exposure adjustment')
                    return value
                bench.stage(phasefile);snap('dark_before',black)
                for name,phase in phases.items():
                    f=a.out/(name+'_phase.bmp');Image.fromarray(native_image(np.rint(phase*256).clip(0,255).astype(np.uint8),True)).save(f)
                    if name=='canonical' and not np.array_equal(np.array(Image.open(f)),np.array(Image.open(phasefile))):raise ValueError('Reconstructed phase not identical to actual source')
                    current_phase=f;bench.stage(f);values[name]=[]
                    for j,inputfile in enumerate(inputs):
                        values[name].append([snap(f'{name}_TRAIN{j}_repeat{k}',inputfile) for k in range(2)])
                current_phase=phasefile;bench.stage(phasefile);values['restored']=[[snap(f'restored_TRAIN{j}',inputfile)] for j,inputfile in enumerate(inputs)]
                snap('dark_after',black)
            finally:bench.stage(phasefile);restored=True
        if len(frames)!=20:raise RuntimeError('Fixed control frame budget differs')
        comparisons={name:[difference(values['canonical'][j][0],values[name][j][0]) for j in range(2)] for name in values}
        repeat={name:[difference(values[name][j][0],values[name][j][1]) for j in range(2)] for name in phases}
        result=dict(passed=True,scope='Completed TRAIN/control diagnostic, not a numerical performance or admission pass',frames=20,
            phase_comparison=comparisons,within_phase_repeat=repeat,simulation_phase_comparison=sim_compare,frame_receipts=frames,
            canonical_phase_restored=restored,all_individual_reads_saved=True,averaged_network_capture=False,SDK_released=True,TEST_used=False,
            hardware_unchanged=True,new_network_frames=0,source_checkpoint_unchanged=True)
        put(a.out/'report.json',result);print(json.dumps(result))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),frames=len(frames),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK owner changed')
        LOCK.unlink()


if __name__=='__main__':main()
