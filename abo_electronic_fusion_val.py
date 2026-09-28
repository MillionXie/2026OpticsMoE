"""TRAIN-only fusion audit for a frozen electronic-only branch and physical cache."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoProcessor

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import OpticalRetrieval
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.io import inputs,picture
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.cli import autocast

PROJECT=Path('/DATA/DATA1/guest3/2026OpticsMoE')
RUNS=PROJECT/'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation'
ROOT=RUNS/'electronic_only_robust_short_20260928'
PROTOCOL=RUNS/'abo200_enrolled_protocol_20260913/protocol.json'
EXECUTION=RUNS/'sixhour_strong_20260928/execution.json'
DATA=PROJECT/'data/abo_similarity10_data'
ASSETS=PROJECT/'.codex_tmp/t07_robust_assets_20260926'

def main():
    torch.set_num_threads(4)
    checkpoint=Path(os.environ.get('ABO_ELECTRONIC_EVAL_CHECKPOINT',str(ROOT/'run/best.pt')))
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    source_sha='f413efa865d6b1b7252d2b3babd0cedfbb4415a362f0818feafa36f2843f292e'
    assert (payload.get('source_checkpoint_sha256',source_sha)==source_sha)
    model=OpticalRetrieval(payload['metadata'])
    model.load_state_dict(payload['state_dict'],strict=True)
    model.eval().requires_grad_(False).cuda()
    model.set_remove_optical(True)
    processor=AutoProcessor.from_pretrained(str(ASSETS/'processor'),local_files_only=True)
    protocol=json.loads(PROTOCOL.read_text())
    rows=[r for r in protocol['rows'] if r['split']=='train']
    assert len(rows)==1600
    cache=torch.load(ROOT/'physical_train1600_selected_vectors.pt',map_location='cpu',weights_only=True)
    assert cache['ids']==[r['sample_id'] for r in rows]
    assert cache['selected_sha256']==source_sha
    physical=F.normalize(cache['vectors'].float(),dim=-1)
    holdout=json.loads(EXECUTION.read_text())['holdout_audit']
    fit_set,val_set=set(holdout['fitting_ids']),set(holdout['validation_ids'])
    fi=torch.tensor([i for i,r in enumerate(rows) if r['sample_id'] in fit_set])
    vi=torch.tensor([i for i,r in enumerate(rows) if r['sample_id'] in val_set])
    assert len(fi)==1200 and len(vi)==400 and not fit_set.intersection(val_set)
    vectors=[]
    with torch.no_grad():
        for start in range(0,1600,16):
            batch_rows=rows[start:start+16]
            images=[picture(DATA/r['image_path'],model.metadata.get('input_preprocessing','contain_white')) for r in batch_rows]
            batch=inputs(processor,images,torch.device('cuda'))
            with autocast(torch.device('cuda')):
                vectors.append(model(batch).float().cpu())
    electronic=F.normalize(torch.cat(vectors),dim=-1)
    labels=[r['product_id'] for r in rows]
    def score(v):
        nearest=(v[vi]@v[fi].T).argmax(1).tolist()
        return sum(labels[int(fi[j])]==labels[int(i)] for i,j in zip(vi,nearest))/400
    grid=[]
    for share in (0,.25,.5,.75,1):
        fused=F.normalize((1-share)*physical+share*electronic,dim=-1)
        grid.append({'electronic_share':share,'validation_r1':score(fused)})
    report={'status':'complete','scope':'1200 TRAIN fit/400 TRAIN holdout; same-image warm-start caveat; no original 800 query',
            'electronic_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'electronic_epoch':payload.get('epoch'),'physical_checkpoint_sha256':cache['selected_sha256'],
            'grid':grid,'best':max(grid,key=lambda x:x['validation_r1']),
            'original_query_read_or_evaluated':False}
    target=Path(os.environ.get('ABO_ELECTRONIC_EVAL_OUTPUT',str(ROOT/'fusion_validation.json')))
    assert not target.exists()
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
