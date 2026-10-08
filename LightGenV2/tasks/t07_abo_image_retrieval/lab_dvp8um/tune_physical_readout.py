"""TRAIN-only linear metric adaptation, folded into the existing final head.

No optical or preceding electronic parameters change. Validation views come
only from the 1600 physical TRAIN/gallery images; 800 queries remain sealed.
Final quality is verified by replaying the six saved linear CCDs, not camera.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=50)
    args = p.parse_args()
    import four_image_flow as flow
    import full_query_flow as pipe
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    protocol = json.loads((flow.PROJECT/'protocol.json').read_text())
    train = [r for r in protocol['rows'] if r['split'] == 'train']
    query = [r for r in protocol['rows'] if r['split'] == 'query']
    rows = train + query
    cache = torch.load(args.capture/'features.pt', map_location='cpu', weights_only=True)
    assert cache['ids'] == [r['sample_id'] for r in rows]
    source = flow.PROJECT/'assets/best.pt'
    assert flow.sha(source) == flow.BEST
    payload = torch.load(source, map_location='cpu', weights_only=True)
    assert isinstance(payload['state_dict']['readout.projection.weight'], torch.Tensor)
    assert payload['state_dict']['readout.projection.weight'].shape[0] == 64
    labels = {s:i for i,s in enumerate(sorted({r['product_id'] for r in train}))}
    fit, val = [], []
    for product in labels:
        indices = [i for i,r in enumerate(train) if r['product_id'] == product]
        indices.sort(key=lambda i: hashlib.sha256(('physical-head42:'+train[i]['sample_id']).encode()).hexdigest())
        assert len(indices) == 8
        fit += indices[:6]; val += indices[6:]
    device = torch.device('cuda')
    vectors = cache['descriptors'].float().to(device)
    fit_t, val_t = torch.tensor(fit,device=device), torch.tensor(val,device=device)
    y = torch.tensor([labels[r['product_id']] for r in train],device=device)
    x = vectors[fit_t]
    positive = y[fit_t,None].eq(y[fit_t][None,:])
    eye = torch.eye(len(fit),device=device,dtype=torch.bool)
    positive = positive & ~eye
    matrix = torch.nn.Parameter(torch.eye(64,device=device))
    identity = torch.eye(64,device=device)
    optimizer = torch.optim.AdamW([matrix], lr=0.003, weight_decay=0)
    def validate():
        z = F.normalize(vectors[:1600] @ matrix.T,dim=-1)
        ranking = (z[val_t] @ z[fit_t].T).argsort(dim=1,descending=True)
        return float(y[fit_t][ranking[:,0]].eq(y[val_t]).float().mean())
    best_score = validate(); best_epoch = 0; best_matrix = matrix.detach().clone()
    history = [dict(epoch=0,validation_r1=best_score)]
    for epoch in range(1,args.epochs+1):
        optimizer.zero_grad()
        z = F.normalize(x @ matrix.T,dim=-1)
        logits = (z @ z.T / 0.1).masked_fill(eye,-1e4)
        logp = logits.log_softmax(dim=1)
        loss = -(logp*positive).sum(1).div(positive.sum(1)).mean() + 0.01*(matrix-identity).square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_([matrix],1); optimizer.step()
        with torch.no_grad(): score = validate()
        entry = dict(epoch=epoch,loss=float(loss.detach()),validation_r1=score)
        history.append(entry); print(json.dumps(entry),flush=True)
        if score > best_score:
            best_score, best_epoch, best_matrix = score, epoch, matrix.detach().clone()
    def folded(m):
        result = dict(payload,state_dict=dict(payload['state_dict']))
        for suffix in ('weight','bias'):
            name = 'readout.projection.'+suffix
            old = payload['state_dict'][name]
            result['state_dict'][name] = (m.cpu().double() @ old.double()).to(old.dtype)
        return result
    torch.save(folded(best_matrix),args.output/'best.pt')
    torch.save(folded(matrix.detach()),args.output/'last.pt')
    contract = dict(fit_ids=[train[i]['sample_id'] for i in fit],validation_ids=[train[i]['sample_id'] for i in val],
                    test_ids=[r['sample_id'] for r in query],epochs=args.epochs,best_epoch=best_epoch,
                    best_validation_r1=best_score,source_sha256=flow.sha(source),
                    changed_parameters=['readout.projection.weight','readout.projection.bias'],
                    added_inference_parameters=0,history=history)
    flow.write(args.output/'selection.json',contract)
    # Replay original saved CCDs through the unchanged backbone and folded head.
    model = flow.OpticalRetrieval(payload['metadata']).to(device).eval().requires_grad_(False)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(flow.PROJECT/'assets/processor'),local_files_only=True)
    geometry = json.loads((args.capture/'run_contract.json').read_text())
    pipe.STAGE_CALIBRATION.update(geometry['stage_calibration'])
    def capture(bench,out,stage,phase,active,ids,orientation):
        images = np.stack([np.asarray(Image.open(args.capture/'ccd'/stage/(s+'.png'))) for s in ids])
        return torch.from_numpy(images).to(device).float(), {}, []
    pipe.capture_stage = capture
    def sim(model,batch):
        count = len(batch['input_ids'])
        return {**{s:torch.zeros(count,478,478) for s in pipe.STAGE_CALIBRATION},'descriptor':torch.zeros(count,64)}
    pipe.snapshot_simulation = sim
    model.load_state_dict(payload['state_dict'],strict=True)
    phase_paths = {stage:args.capture/'phase'/(stage+'.bmp') for stage in pipe.STAGE_CALIBRATION}
    original = pipe.process_batch(model,processor,None,args.output,phase_paths,rows[:4])['descriptor']
    max_error = float((original-cache['descriptors'][:4]).abs().max())
    if max_error > 0.0001:
        raise RuntimeError(f'Original CCD replay differs from capture descriptors: {max_error}')
    model.load_state_dict(folded(best_matrix)['state_dict'],strict=True)
    measured = []
    for start in range(0,len(rows),4):
        result = pipe.process_batch(model,processor,None,args.output,phase_paths,rows[start:start+4])
        measured.append(result['descriptor'])
        if start % 80 == 0: print(json.dumps(dict(replay_completed=start+4,total=len(rows))),flush=True)
    measured = torch.cat(measured)
    expected = F.normalize(cache['descriptors'].float() @ best_matrix.cpu().T,dim=-1)
    folded_error = float((measured-expected).abs().max())
    if folded_error > 0.0002: raise RuntimeError(f'Folded replay mismatch: {folded_error}')
    predictions, metrics = pipe.evaluate(measured[1600:],query,protocol,{'ids':[r['sample_id'] for r in train],'vectors':measured[:1600]})
    torch.save(dict(ids=cache['ids'],descriptors=measured),args.output/'verified_features.pt')
    flow.write(args.output/'report.json',dict(status='complete',metrics=metrics,predictions=predictions,
        source_capture=str(args.capture),best_epoch=best_epoch,validation_r1=best_score,
        fit_count=len(fit),validation_count=len(val),query_count=800,gallery_count=1600,
        physical_gallery=True,original_replay_max_error=max_error,folded_replay_max_error=folded_error,
        test_used_for_selection=False,best_checkpoint_sha256=flow.sha(args.output/'best.pt')))
    print(json.dumps(dict(status='complete',metrics=metrics)),flush=True)


if __name__ == '__main__':
    main()
