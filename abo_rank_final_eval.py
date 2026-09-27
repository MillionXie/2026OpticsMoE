"""One sealed-query evaluation after TRAIN/VAL chose the ranking head."""
import hashlib,json
from pathlib import Path
import torch
from torch.nn import functional as F
import layerwise as lab

def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parent
    out=root/'runs/rank_final_sealed';out.mkdir(exist_ok=False)
    report=json.loads((root/'runs/rank_recovery_trainonly/report.json').read_text())
    assert report['status']=='complete' and not report['test_evaluated'] and not report['query_used_for_training_or_selection']
    chosen=torch.load(root/'runs/rank_recovery_trainonly/best_head.pt',map_location='cpu',weights_only=True)
    assert chosen['selection']==report['selection']
    cache=torch.load(root/'runs/rank_cache_trainonly/cache.pt',map_location='cpu',weights_only=True)
    payload=torch.load(root/'assets/best.pt',map_location='cpu',weights_only=True)
    assert cache['source_sha256']==lab.SHA and lab.flow.sha(root/'assets/best.pt')==lab.SHA
    prior=torch.load(root/'runs/readout50_trainonly_aligned/best.pt',map_location='cpu',weights_only=True)
    assert len(cache['features'])==1600
    # A fixed, preselected head is the only candidate ever scored on the sealed query.
    for key,value in chosen['state_dict'].items():prior['state_dict']['readout.projection.'+key]=value
    selected=root/'runs/rank_recovery_trainonly/selected_full.pt';torch.save(prior,selected)
    model=lab.OpticalRetrieval(prior['metadata']).cuda().eval().requires_grad_(False);model.load_state_dict(prior['state_dict'],strict=True)
    import numpy as np
    from PIL import Image
    protocol=json.loads((lab.OLD/'protocol.json').read_text());train=[r for r in protocol['rows'] if r['split']=='train'];query=[r for r in protocol['rows'] if r['split']=='query']
    assert cache['ids']==[r['sample_id'] for r in train] and len(query)==800
    capture=root/'runs/layerwise_physical2400';phases={s:capture/'phase'/(s+'.bmp') for s in lab.STAGES}
    geometry=json.loads((lab.OLD/'runs/abo_i2i_20260926/shs_geometry.json').read_text())
    lab.flow.BASE_CORNERS=np.asarray(geometry['base_corners_screen_TL_TR_BR_BL'],np.float32)
    lab.pipeline.STAGE_CALIBRATION.clear();lab.pipeline.STAGE_CALIBRATION.update({s:(r['phase_candidate'],r['camera_orientation']) for s,r in geometry['stage_calibration'].items()})
    def replay(bench,unused,stage,phase,active,ids,orientation):
        arrays=[]
        for sid in ids:
            folder=capture/'ccd'/stage;r=json.loads((folder/(sid+'.json')).read_text())
            assert r['phase_sha256']==lab.flow.sha(phases[stage]) and r['wait_ms']==240 and r['exposure']['exposure_us']==400
            arrays.append(np.array(Image.open(folder/(sid+'.png'))))
        return torch.from_numpy(np.stack(arrays)).cuda().float(),{},1.
    lab.pipeline.stage_active=lab.stage_active;lab.pipeline.capture_stage=replay
    processor=lab.AutoProcessor.from_pretrained(str(lab.OLD/'assets/processor'),local_files_only=True)
    with torch.no_grad():gallery=F.normalize(model.readout.projection(cache['features'].cuda()),dim=-1).cpu()
    desc=[]
    for i in range(0,len(query),4):
        with torch.inference_mode():v=lab.pipeline.process_batch(model,processor,None,out,phases,query[i:i+4])['descriptor'].cpu()
        desc.append(v)
    vectors=torch.cat([gallery,torch.cat(desc)])
    bank={'ids':[r['sample_id'] for r in train+query],'vectors':vectors}
    predictions,metrics=lab.pipeline.evaluate(vectors[1600:],query,protocol,bank)
    result={'status':'complete','source_sha256':lab.SHA,'best_sha256':lab.flow.sha(selected),'selected_before_query':True,'selection':chosen['selection'],'validation_r1':report['best_validation_r1'],'physical_gallery':1600,'physical_query':800,'same_weight_six_ccd_replay':True,'metrics':metrics,'predictions':predictions}
    (out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)
if __name__=='__main__':main()
