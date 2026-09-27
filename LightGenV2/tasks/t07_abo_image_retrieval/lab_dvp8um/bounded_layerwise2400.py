"""Whole-dataset layers, persistent hardware, verified physical upstream replay."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parent
OLD=Path('E:/code/guest/2026OpticsMoE/ABO_I2I_Lab_DVP_8um')
sys.path[:0]=[str(ROOT),str(OLD/'lab_dvp8um')]
from standalone.model import OpticalRetrieval
from standalone.bounded_export import stage_active,quantize
import four_image_flow as flow
import full_query_flow as pipeline
from shs_physical2400 import SHSBench
from transformers import AutoProcessor
SHA='04c216209fe822636a73f03e68fdcecdc2008252cf8db33ffc5ed9c68dc5c593'
STAGES=('vision_router','vision_expert','vision_global','language_router','language_expert','language_global')
class EndCurrentLayer(Exception):pass

def save(active,out,stage,ids):
    folder=out/'amplitude'/stage;folder.mkdir(parents=True,exist_ok=True);paths=[]
    for sid,a in zip(ids,quantize(active)):
        p=folder/(sid+'.bmp');Image.fromarray(flow.active_to_native(a)).save(p);paths.append(p)
    return paths,1.

def main():
    weight=ROOT/'assets/best.pt';assert hashlib.sha256(weight.read_bytes()).hexdigest()==SHA
    payload=torch.load(weight,map_location='cpu',weights_only=True)
    model=OpticalRetrieval(payload['metadata']);model.load_state_dict(payload['state_dict'],strict=True);model.eval().requires_grad_(False).to('cuda')
    processor=AutoProcessor.from_pretrained(str(OLD/'assets/processor'),local_files_only=True)
    geometry=json.loads((OLD/'runs/abo_i2i_20260926/shs_geometry.json').read_text(encoding='utf8'))
    flow.BASE_CORNERS=np.asarray(geometry['base_corners_screen_TL_TR_BR_BL'],np.float32)
    pipeline.STAGE_CALIBRATION.clear();pipeline.STAGE_CALIBRATION.update({stage:(r['phase_candidate'],r['camera_orientation']) for stage,r in geometry['stage_calibration'].items()})
    pipeline.stage_active=stage_active
    protocol=json.loads((OLD/'protocol.json').read_text(encoding='utf8'))
    gallery=[r for r in protocol['rows'] if r['split']=='train'];query=[r for r in protocol['rows'] if r['split']=='query'];rows=gallery+query
    assert len(gallery)==1600 and len(query)==800 and len({r['sample_id'] for r in rows})==2400
    out=ROOT/'runs/layerwise_physical2400';out.mkdir(parents=True,exist_ok=True)
    contract={'checkpoint_sha256':SHA,'encoding':'bounded tanh(abs/.5) after fanout before phase; round255a no peak scaling','exposure_us':400,'gain':'Gain_X4','wait_ms':240,'geometry':geometry['base_corners_screen_TL_TR_BR_BL'],'stage_orientation':pipeline.STAGE_CALIBRATION,'ids':[r['sample_id'] for r in rows]}
    cp=out/'contract.json'
    normalized=json.loads(json.dumps(contract))
    if cp.exists():assert json.loads(cp.read_text())==normalized,'Resume contract changed'
    else:cp.write_text(json.dumps(contract,indent=2))
    if (out/'report.json').exists():raise FileExistsError('Completed run must not restart')
    phase_dir=out/'phase';phase_dir.mkdir(exist_ok=True);phase_paths={}
    with torch.no_grad():phases=flow.phase_planes(model)
    for stage,radians in phases.items():
        p=phase_dir/(stage+'.bmp');expected=pipeline.selected_phase(radians,pipeline.STAGE_CALIBRATION[stage][0])
        if p.exists():assert np.array_equal(np.array(Image.open(p)),expected)
        else:Image.fromarray(expected).save(p)
        phase_paths[stage]=p
    counts={s:0 for s in STAGES};started=time.perf_counter();current=[None];warmed=set()
    def read(stage,ids):
        arrays=[]
        for sid in ids:
            folder=out/'ccd'/stage
            receipt=json.loads((folder/(sid+'.json')).read_text())
            assert receipt['phase_sha256']==flow.sha(phase_paths[stage])
            assert receipt['wait_ms']==240 and receipt['exposure']['exposure_us']==400
            arrays.append(np.array(Image.open(folder/(sid+'.png'))))
        return np.stack(arrays)
    def capture(bench,unused,stage,phase_path,active,ids,orientation):
        if STAGES.index(stage)>STAGES.index(current[0]):raise EndCurrentLayer()
        folder=out/'ccd'/stage
        complete=all((folder/(sid+'.png')).is_file() and (folder/(sid+'.json')).is_file() for sid in ids)
        if stage!=current[0] or complete:raw=read(stage,ids);receipt={'reused_same_checkpoint':True}
        else:
            paths,_=save(active,out,stage,ids)
            if stage not in warmed:
                # Short diagnosis found an initial black frame. Retain two
                # warmup receipts separately; never count them as dataset CCDs.
                for rep in range(2):bench.capture('warmup_'+stage,phase_path,[paths[0]],[ids[0]+f'_warmup{rep}'],orientation,save=True)
                warmed.add(stage)
            raw,receipt=bench.capture(stage,phase_path,paths,ids,orientation,save=True)
            for p in paths:p.unlink() # only just-generated transient input, never CCDs
        return torch.from_numpy(raw).to('cuda').float(),receipt,1.
    # Validate the manual bridge end-to-end with ideal CCDs before any SDK opens.
    first=rows[:4]
    images=[flow.picture(OLD/'data'/r['image_path'],model.metadata.get('input_preprocessing','contain_white')) for r in first]
    batch=flow.inputs(processor,images,torch.device('cuda'))
    model.vision.optics.router.measured_ccd=None;model.language.optics.router.measured_ccd=None
    with torch.inference_mode():ideal=flow.snapshot_simulation(model,batch)
    def ideal_capture(bench,output,stage,phase,active,ids,orientation):return ideal[stage].to('cuda').float(),{},1.
    pipeline.capture_stage=ideal_capture
    with torch.inference_mode():replay=pipeline.process_batch(model,processor,None,out,phase_paths,first)
    error=float((replay['descriptor']-ideal['descriptor']).abs().max())
    (out/'bridge_selftest.json').write_text(json.dumps({'maximum_descriptor_error':error,'samples':4,'camera_used':False},indent=2))
    assert error<1e-5,'Ideal six-stage bridge does not match source model'
    pipeline.capture_stage=capture
    chunks=out/'features';chunks.mkdir(exist_ok=True)
    with SHSBench(out,400,240,phase_paths) as bench:
        for stage in STAGES:
            current[0]=stage
            for index in range(0,len(rows),4):
                batch=rows[index:index+4]
                with torch.inference_mode():
                    try:result=pipeline.process_batch(model,processor,bench,out,phase_paths,batch)
                    except EndCurrentLayer:result=None
                if stage==STAGES[-1]:
                    assert result is not None;torch.save(result,chunks/(f'{index:06d}.pt'))
                counts[stage]=index+len(batch)
                progress={'status':'capturing','stage':stage,'stage_completed':counts[stage],'stage_total':2400,'ccd_counts':counts,'total_ccd':sum(counts.values()),'elapsed_seconds':time.perf_counter()-started,'checkpoint_sha256':SHA}
                (out/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    vectors=[];ids=[]
    for index in range(0,len(rows),4):
        result=torch.load(chunks/(f'{index:06d}.pt'),map_location='cpu',weights_only=True);vectors.append(result['descriptor']);ids.extend(result['ids'])
    vectors=torch.cat(vectors);assert ids==[r['sample_id'] for r in rows]
    bank={'ids':ids,'vectors':vectors};torch.save(bank,out/'physical_features.pt')
    predictions,metrics=pipeline.evaluate(vectors[len(gallery):],query,protocol,bank)
    report={'status':'complete','checkpoint_sha256':SHA,'samples':2400,'gallery':1600,'query':800,'ccd_counts':counts,'physical_to_physical':True,'metrics':metrics,'predictions':predictions,'elapsed_seconds':time.perf_counter()-started}
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'progress.json').write_text(json.dumps({'status':'complete','metrics':metrics,'ccd_counts':counts}));print(json.dumps({'status':'complete','metrics':metrics}),flush=True)

if __name__=='__main__':main()
