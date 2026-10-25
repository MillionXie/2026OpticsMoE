"""Fixed40-frame TRAIN voltage-code/spatial-frequency response diagnostic.

Not a calibrated phase LUT: voltage codes are NOT asserted radians. No device
LUT, exposure, coordinates, model weights, or network-frame transformation is
changed. A prior sealed20-frame source is required; canonical phase restored.
"""
import argparse,json,os,time,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .hardware_bridge import hardware,native_image,crop_native,signal,load_checkpoint,read_source
from .probe_TRAIN_phase_response import difference
from .train_native import sha,put

LEVELS=(0,32,64,96,128,160,192,224,255)
CELLS=(32,8)


def voltage_grid(cell,level):
    if cell not in CELLS or level not in LEVELS:raise ValueError('Outside fixed TRAIN grid')
    y,x=np.indices((1016,1016));return (((y//cell+x//cell)%2)*level).astype(np.uint8)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('package','hardware','previous_probe','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    old=json.loads((a.previous_probe/'protocol.json').read_text());done=json.loads((a.previous_probe/'report.json').read_text())
    cfg=json.loads(a.hardware.read_text());package=json.loads(a.package.read_text())
    if not done['passed'] or done['frames']!=20 or not done['SDK_released'] or not done['canonical_phase_restored']:raise ValueError('Previous sealed control incomplete')
    if sha(a.package)!=old['package_SHA256'] or sha(a.hardware)!=old['hardware_SHA256'] or sha(package['checkpoint'])!=old['source_checkpoint_SHA256']:
        raise ValueError('Changed source hardware or checkpoint')
    case=Path(old['source_capture']);phasefile=case/'first/phase.bmp';inputs=[]
    for row,source_id in zip(old['source_physical_rows'],old['TRAIN_source_indices']):
        folder=case/'first'/f'{row:05d}';r=json.loads((folder/'receipt.json').read_text())
        if r['split']!='train' or r['source_index']!=source_id or sha(folder/'input.bmp')!=r['input_BMP_sha256'] or sha(phasefile)!=r['phase_BMP_sha256']:
            raise ValueError('Original actual TRAIN control changed')
        inputs.append(folder/'input.bmp')
    model,_=load_checkpoint(package['checkpoint'],package['checkpoint_sha256'])
    spec=next(v for v in package['datasets'] if v['task']=='clevr' and v['split']=='train')
    source=read_source(spec['path'],spec['sha256'],'train',np.array([0,1]))
    if source['source_indices'].tolist()!=old['TRAIN_source_indices']:raise ValueError('Current TRAIN control identity changed')
    amp=[model.payload(torch.from_numpy(x)[None].float(),'first',{})[0] for x in source['x']]
    sim={};zero=[model.propagate_payload(x,torch.zeros((1016,1016))).numpy()[0] for x in amp]
    for cell in CELLS:
        sim[str(cell)]={}
        for level in LEVELS:
            phase=torch.from_numpy(voltage_grid(cell,level).astype(np.float32)/256)
            sim[str(cell)][str(level)]=[difference(zero[j],model.propagate_payload(x,phase).numpy()[0]) for j,x in enumerate(amp)]
    if a.prepare_only:
        print(json.dumps(dict(passed=True,expected_frames=40,TRAIN_source_indices=old['TRAIN_source_indices'],voltage_levels=LEVELS,cell_pixels=CELLS,simulation=sim,SDK_opened=False,run_written=False)));return
    Bench,LOCK,_,_,warp=hardware(cfg)
    if LOCK.exists():raise RuntimeError('SDK owned, no takeover')
    a.out.mkdir(parents=True,exist_ok=False);put(a.out/'protocol.json',dict(previous_probe_report_SHA256=sha(a.previous_probe/'report.json'),
        package_SHA256=sha(a.package),hardware_SHA256=sha(a.hardware),source_checkpoint_SHA256=package['checkpoint_sha256'],
        TRAIN_source_indices=old['TRAIN_source_indices'],expected_frames=40,voltage_levels=LEVELS,cell_pixels=CELLS,
        voltage_codes_not_calibrated_phase=True,simulation=sim,exposure_changed=False,LUT_changed=False,coordinates_changed=False,weights_changed=False,
        averaged_network_capture=False,TEST_used=False,new_network_frames=0,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_SHA256=sha(__file__)))
    put(LOCK,dict(pid=os.getpid(),task='T21',scope='fixed40_TRAIN_voltage_frequency'));frames=[];values={};restored=False
    try:
        with Bench(cfg,a.out) as bench:
            try:
                black=a.out/'black.bmp';Image.fromarray(native_image(np.zeros((1016,1016),np.uint8))).save(black);current_phase=phasefile
                def snap(name,inputfile):
                    raw,metadata=bench.capture(inputfile);v=crop_native(raw,warp);np.save(a.out/(name+'_raw.npy'),raw);np.save(a.out/(name+'_ccd.npy'),v)
                    s=signal(v);row=dict(name=name,signal=s,metadata=metadata,phase_BMP_SHA256=sha(current_phase),raw_SHA256=sha(a.out/(name+'_raw.npy')),ccd_SHA256=sha(a.out/(name+'_ccd.npy')))
                    put(a.out/(name+'_receipt.json'),row);frames.append(row);put(a.out/'progress.json',dict(frames=len(frames),latest=row,updated=time.time()))
                    if s['saturation_fraction']>.15:raise RuntimeError('Control saturation; preserve data without changing exposure')
                    return v
                bench.stage(phasefile);snap('dark_before',black)
                for cell in CELLS:
                    values[str(cell)]={}
                    for level in LEVELS:
                        current_phase=a.out/f'cell{cell}_voltage{level}_phase.bmp';Image.fromarray(native_image(voltage_grid(cell,level),True)).save(current_phase);bench.stage(current_phase)
                        values[str(cell)][str(level)]=[snap(f'cell{cell}_voltage{level}_TRAIN{j}',f) for j,f in enumerate(inputs)]
                current_phase=phasefile;bench.stage(phasefile)
                for j,f in enumerate(inputs):snap(f'restored_TRAIN{j}',f)
                snap('dark_after',black)
            finally:bench.stage(phasefile);restored=True
        if len(frames)!=40:raise RuntimeError('Declared40-frame budget differs')
        comparisons={c:{g:[difference(values[c]['0'][j],v[j]) for j in range(2)] for g,v in data.items()} for c,data in values.items()}
        put(a.out/'report.json',dict(passed=True,scope='Completed fixed TRAIN diagnostic, not calibrated phase or performance admission',
            frames=40,phase_comparison=comparisons,simulation=sim,frame_receipts=frames,canonical_phase_restored=restored,SDK_released=True,
            LUT_unchanged=True,hardware_unchanged=True,TEST_used=False,new_network_frames=0,averaged_network_capture=False))
    except BaseException:put(a.out/'failure.json',dict(traceback=traceback.format_exc(),frames=len(frames),automatic_retry=False));raise
    finally:
        if json.loads(LOCK.read_text())['pid']!=os.getpid():raise RuntimeError('SDK owner changed')
        LOCK.unlink()


if __name__=='__main__':main()
