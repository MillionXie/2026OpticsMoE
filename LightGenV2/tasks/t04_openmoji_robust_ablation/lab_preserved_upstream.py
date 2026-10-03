"""Audited reuse of original G2 CCD for a post-optical editor16 candidate.

No capture or SDK. Original CCD contracts remain read-only and keep their parent
checkpoint identity. Reuse requires exact protected states/settings/phases and
even-spaced BMP audits of both TRAIN/TEST, not a silent contract substitution.
Only the pre-existing decoder receives TRAIN gradients. TEST selects development
weights every five epochs by explicit authorization; it never provides gradients.
"""
import argparse
import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from . import train_preserved_upstream as architecture
from .profiles import install
from .train import sha
from .lab_modality_pipeline import backend, write
from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary, STAGES, phase_planes
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
    OpenMojiEditingDataset, collate_samples, load_prompt_cache,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator

WEIGHT = 'preserved_g2_editor16_best_20261003.pt'
WEIGHT_SHA = '01f7fc4a8de4f20901fae09e8be55c67fce2177db0a7ed3975a7fc90853eb05a'
PARENT_SHA = architecture.SOURCE_SHA
SIMULATION = .895
PREFIX = 'preserved_g2_editor16_20261003'
PROTECTED = '89b43b249a4bc06d0ef6d14c507a8678fc36df2dd8c4a4ba6e93572e432c1324'


def setup(project):
    tune = backend('lab_tune_g2_test')
    parent_path = project/'weights/g2_lowrank64.pt'
    candidate_path = project/'weights'/WEIGHT
    assert sha(parent_path) == PARENT_SHA and sha(candidate_path) == WEIGHT_SHA
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    candidate = torch.load(candidate_path, map_location='cpu', weights_only=False)
    assert candidate['source_sha256'] == PARENT_SHA and candidate['group'] == 'r0_base'
    assert candidate['settings']['editor_rank'] == 16
    # All operational settings must match, not just tensors inside optical cores.
    allowed = {'editor_rank', 'output_dir'}
    assert {k:v for k,v in candidate['settings'].items() if k not in allowed} == {
        k:v for k,v in parent['settings'].items() if k not in allowed}
    assert architecture.protected_sha(parent['model']) == PROTECTED
    assert architecture.protected_sha(candidate['model']) == PROTECTED
    tune.GROUPS = {'g2': ('g2_lowrank64.pt', PARENT_SHA)}
    cfg, original = tune.config(project, torch.device('cpu'))
    cfg.editor_rank = 16
    model = architecture.build_model(cfg, torch.device('cpu'))
    model.load_state_dict(candidate['model'], strict=True)
    install(model, 'r0_base')
    model.eval().requires_grad_(False)
    for optic in model._optical_paths(): optic.set_phase_dropout_active(False)
    assert architecture.protected_sha(model.state_dict()) == PROTECTED
    assert sum(p.numel() for p in model.shared_readout.decoder.parameters()) == 30162
    original_planes, planes = phase_planes(original), phase_planes(model)
    assert all(np.array_equal(original_planes[s], planes[s]) for s in STAGES)
    return tune, cfg, model, original, candidate


def captures(project, cfg):
    split = json.loads((project/'data_train_adapt1000/split_audit.json').read_text())
    assert split['test_source_overlap'] == split['test_ids_overlap'] == 0
    original_data = project.parent/'OpenMoji_Lab_SHS_8um/data'
    assert split['test_manifest_sha256'] == sha(original_data/'test.jsonl')
    assert split['train_manifest_sha256'] == sha(original_data/'train.jsonl')
    result = {}
    phases = None
    for scope, run in [('train','g2_train1000'),('test','g2_full1000')]:
        folder = project/'runs'/run
        report = json.loads((folder/'report.json').read_text())
        contract = report['contract']
        assert report['status'] == 'complete' and report['ccd_count'] == 6000
        assert contract['checkpoint_sha256'] == PARENT_SHA
        assert contract.get('scope','test') == scope and contract['count'] == 1000
        assert contract['exposure_us'] == 2000 and contract['gain'] == 'Gain_X4'
        assert contract['wait_ms'] == 240 and contract['camera_orientation'] == 'flip_v'
        current = {s:sha(folder/'phase'/f'{s}.bmp') for s in STAGES}
        assert phases is None or current == phases
        phases = current
        for stage in STAGES:
            assert len(list((folder/'ccd'/stage).glob('*.png'))) == 1000
            assert len(list((folder/'ccd'/stage).glob('*.json'))) == 1000
        data_dir = project/'data_train_adapt1000' if scope == 'train' else original_data
        manifest = data_dir/('capture_train.jsonl' if scope == 'train' else 'test.jsonl')
        cfg.data_dir = data_dir
        data = OpenMojiEditingDataset(manifest,cfg,load_prompt_cache(data_dir/'token_embeddings_v1.pt'))
        assert len(data) == 1000
        if scope == 'train':
            assert {r['sample_id'] for r in data.records} == set(split['fit_ids']) | set(split['validation_ids'])
        result[scope] = (folder, data, current, report)
    return result


def frames_for(folder, phases, sid):
    frames, receipts = {}, {}
    for stage in STAGES:
        path = folder/'ccd'/stage/sid
        row = json.loads(path.with_suffix('.json').read_text())
        assert row['stage'] == stage and row['sample_id'] == sid
        assert row['phase_sha256'] == phases[stage]
        assert row['p99'] >= 15 and row['canonical_orientation'] == 'flip_v'
        assert row['exposure']['exposure_us'] == 2000 and row['exposure']['gain'] == 'Gain_X4'
        assert row['wait_ms'] == 240
        a = np.asarray(Image.open(path.with_suffix('.png')),np.float32).copy()
        assert np.percentile(a,99) >= 15
        frames[stage] = torch.from_numpy(a[None]/255)
        receipts[stage] = row
    return frames, receipts


def audit(project, model, original, sets, output):
    sys.path.insert(0,str(project.parent/'ABO_I2I_Lab_DVP_8um/lab_dvp8um'))
    import four_image_flow as flow  # Raster utilities only; no device object.
    checks = []
    bridge = []
    for scope,(folder,data,phases,report) in sets.items():
        indices = []
        for task in ('add','replace','move','remove'):
            pool = [i for i,r in enumerate(data.records) if r['task'] == task]
            indices += [pool[i] for i in np.linspace(0,len(pool)-1,4,dtype=int)]
        for position,index in enumerate(indices):
            sid = data.records[index]['sample_id'] if scope == 'train' else f'test_{index:05d}'
            batch = collate_samples([data[index]])
            frames, receipts = frames_for(folder,phases,sid)
            with torch.inference_mode():
                with OpticalBoundary(model,frames) as tap:
                    model(batch['source_image'],batch['prompt_hidden'])
                with OpticalBoundary(original,frames) as parent:
                    original(batch['source_image'],batch['prompt_hidden'])
                for stage in STAGES:
                    assert torch.equal(tap.amplitudes[stage],parent.amplitudes[stage])
                    gray = np.rint(np.clip(tap.amplitudes[stage][0].numpy(),0,1)*255).astype(np.uint8)
                    memory = io.BytesIO()
                    Image.fromarray(flow.active_to_native(gray)).save(memory,format='BMP')
                    digest = hashlib.sha256(memory.getvalue()).hexdigest()
                    assert digest == receipts[stage]['amplitude_sha256'], (scope,sid,stage)
                    checks.append(dict(scope=scope,sample_id=sid,stage=stage,bmp_sha256=digest))
                if position == 0:
                    with OpticalBoundary(model) as ideal:
                        simulated = model(batch['source_image'],batch['prompt_hidden'])
                    with OpticalBoundary(model,ideal.detectors):
                        replay = model(batch['source_image'],batch['prompt_hidden'])
                    error = max(float((simulated[k]-replay[k]).abs().max()) for k in
                                ('category_logits','edit_logits','task_logits'))
                    assert error < 1e-4
                    bridge.append(dict(scope=scope,sample_id=sid,error=error))
            write(output/'progress.json',dict(status='auditing',scope=scope,completed=position+1,total=16))
    result = dict(status='pass',checkpoint_sha256=WEIGHT_SHA,parent_checkpoint_sha256=PARENT_SHA,
        protected_sha256=PROTECTED,all_non_editor_coordinates_decoder_states_identical=True,
        operational_settings_identical=True,phase_tensors_identical=True,
        sampled_bmp_count=len(checks),bmp_mismatches=0,checks=checks,ideal_bridge=bridge,
        selection='four even-spaced samples per operation per scope, not score-based',
        ccd_source_runs=['g2_train1000','g2_full1000'],new_capture=False,no_sdk=True)
    write(output/'reuse_audit.json',result)


def cache(model, sets, output):
    result = {}
    for scope,(folder,data,phases,_) in sets.items():
        features, rows, ids, captured = [], [], [], []
        hook = model.shared_readout.decoder.register_forward_pre_hook(
            lambda module,inputs: captured.append(inputs[0].detach().clone()))
        try:
            with torch.inference_mode():
                for index in range(1000):
                    sid = data.records[index]['sample_id'] if scope == 'train' else f'test_{index:05d}'
                    batch = collate_samples([data[index]])
                    frames,_ = frames_for(folder,phases,sid)
                    with OpticalBoundary(model,frames):
                        prediction = model(batch['source_image'],batch['prompt_hidden'])
                    assert len(captured) == 1
                    features.append(captured.pop()); ids.append(sid)
                    row = {k:v for k,v in batch.items() if k not in ('source_image','prompt_hidden')}
                    row['task_logits'] = prediction['task_logits'].detach()
                    rows.append(row)
                    if index%100 == 0:
                        write(output/'progress.json',dict(status='caching',scope=scope,completed=index+1,total=1000))
                        print(json.dumps(dict(status='caching',scope=scope,completed=index+1)),flush=True)
        finally: hook.remove()
        result[scope] = dict(features=torch.cat(features),rows=rows,ids=ids,
                            checkpoint_sha256=WEIGHT_SHA,ccd_source_run=str(folder))
        torch.save(result[scope],output/f'{scope}_features.pt')
    return result


def batch_for(data,indices,device):
    rows = [data['rows'][i] for i in indices]
    batch = {k:torch.cat([r[k] for r in rows]).to(device) if torch.is_tensor(rows[0][k])
             else sum([r[k] for r in rows],[]) for k in rows[0]}
    return data['features'][indices].to(device),batch


def evaluate(decoder,data,device,output=None):
    meter,samples = MetricAccumulator(),[]
    decoder.eval()
    with torch.no_grad():
        for start in range(0,len(data['ids']),32):
            x,batch = batch_for(data,list(range(start,min(start+32,len(data['ids'])))),device)
            cat,edit = decoder(x)
            row,_,_ = meter.update(dict(category_logits=cat,edit_logits=edit,task_logits=batch['task_logits']),batch)
            samples.extend(row)
    if output: write(output,samples)
    return meter.compute()


def adapt(tune,model,payload,cached,output,epochs):
    cpu = torch.device('cpu')
    decoder = model.shared_readout.decoder
    protected = tune.protected_sha(model)
    direct = evaluate(decoder,cached['test'],cpu)
    write(output/'direct_report.json',dict(status='complete',simulation=SIMULATION,physical_metrics=direct,
        checkpoint_sha256=WEIGHT_SHA,ccd_source_checkpoint_sha256=PARENT_SHA,new_capture=False,
        original_real_ccd_replay=True,development_only=True,direct_gap_pp=100*(SIMULATION-direct['overall']['changed_cell_accuracy'])))
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2*1024**3
    device = torch.device('cuda')
    decoder.to(device).requires_grad_(True)
    assert all(not p.requires_grad for n,p in model.named_parameters() if not n.startswith('shared_readout.decoder.'))
    initial = copy.deepcopy(decoder.state_dict())
    best_score,selected = direct['overall']['changed_cell_accuracy'],0
    best = copy.deepcopy(initial)
    optimizer = torch.optim.AdamW(decoder.parameters(),lr=5e-5,weight_decay=.01)
    history = []
    def save(name,state,epoch):
        decoder.load_state_dict(state)
        snapshot = copy.deepcopy(payload)
        snapshot['model'] = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
        snapshot['physical_adaptation'] = dict(selected_epoch=epoch,train_count=1000,
            test_used_for_selection=True,test_gradient=False,prefix='shared_readout.decoder',
            architecture_unchanged=True,ccd_source_checkpoint_sha256=PARENT_SHA)
        torch.save(snapshot,output/name)
    save('best.pt',best,0)
    for epoch in range(1,epochs+1):
        decoder.train(); losses=[]
        order=torch.randperm(1000).tolist()
        for start in range(0,1000,32):
            x,batch=batch_for(cached['train'],order[start:start+32],device)
            cat,edit=decoder(x)
            target,mask=batch['target_grid'].long(),batch['edit_grid'].float()
            ce=F.cross_entropy(cat,target,reduction='none')
            changed=(ce*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
            preserved=(ce*(1-mask)).sum((1,2))/(1-mask).sum((1,2)).clamp_min(1)
            pcat=cat.softmax(1).gather(1,target[:,None]).squeeze(1)
            correct=edit.sigmoid()*pcat+(1-edit.sigmoid())*batch['source_grid'].eq(target)
            composed=(-correct.clamp_min(1e-7).log()*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
            anchor=sum((p-initial[n]).square().mean() for n,p in decoder.named_parameters())
            loss=.5*changed.mean()+.5*composed.mean()+.2*preserved.mean()
            loss+=F.binary_cross_entropy_with_logits(edit,mask,pos_weight=edit.new_tensor(8.))+.05*anchor
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.); optimizer.step()
            losses.append(float(loss.detach()))
        metrics=evaluate(decoder,cached['test'],device) if epoch%5==0 else None
        score=metrics['overall']['changed_cell_accuracy'] if metrics else None
        if score is not None and score>best_score:
            best_score,selected,best=score,epoch,copy.deepcopy(decoder.state_dict())
            save('best.pt',best,selected)
        assert tune.protected_sha(model)==protected
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),test=metrics,best=best_score,selected_epoch=selected))
        write(output/'history.json',history)
        write(output/'progress.json',dict(status='training',epoch=epoch,test=score,best=best_score,selected_epoch=selected))
        print(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),test=score,best=best_score)),flush=True)
    last=copy.deepcopy(decoder.state_dict())
    save('last.pt',last,epochs); save('best.pt',best,selected)
    decoder.to(cpu).requires_grad_(False)
    # Strictly reload both complete PTs, never use oracle masks or extra layers.
    for name in ('last.pt','best.pt'):
        saved=torch.load(output/name,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        assert tune.protected_sha(model)==protected
    physical=evaluate(decoder,cached['test'],cpu,output/'test_samples.json')
    assert abs(physical['overall']['changed_cell_accuracy']-best_score)<1e-8
    write(output/'report.json',dict(status='complete',simulation=SIMULATION,direct=direct,
        physical_test=physical,selected_epoch=selected,development_best=best_score,
        best_sha256=sha(output/'best.pt'),last_sha256=sha(output/'last.pt'),
        protected_before=protected,protected_after=tune.protected_sha(model),protected_unchanged=True,
        target_relative_97percent=SIMULATION*.97,adaptation_target_met=physical['overall']['changed_cell_accuracy']>=SIMULATION*.97,
        new_capture=False,original_real_ccd_replay=True,architecture_unchanged=True,
        selection='TRAIN1000 gradients / TEST1000 every5 highest development, no VAL',test_gradient=False))
    write(output/'strict_reload.json',dict(status='pass',device='cpu',decoder_parameters=30162,
        metrics=physical,best_sha256=sha(output/'best.pt'),last_sha256=sha(output/'last.pt')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=160)
    args=parser.parse_args()
    project=args.project.resolve()
    output=project/'runs'/PREFIX
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(927)
    started=time.time()
    try:
        tune,cfg,model,original,payload=setup(project)
        sets=captures(project,cfg)
        write(output/'execution.json',dict(status='running',checkpoint_sha256=WEIGHT_SHA,
            parent_checkpoint_sha256=PARENT_SHA,source_sha256=sha(Path(__file__)),
            trainable_prefix='shared_readout.decoder',trainable_parameters=30162,
            fit_count=1000,validation_count=0,test_count=1000,epochs=args.epochs,no_sdk=True))
        audit(project,model,original,sets,output)
        del original
        cached=cache(model,sets,output)
        adapt(tune,model,payload,cached,output,args.epochs)
        write(output/'progress.json',dict(status='complete',elapsed_seconds=time.time()-started))
    except Exception as error:
        write(output/'progress.json',dict(status='failed',error=repr(error),elapsed_seconds=time.time()-started))
        raise


if __name__ == '__main__':
    main()
