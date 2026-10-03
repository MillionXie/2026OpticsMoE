"""Read-only, no-SDK diagnosis of the catastrophic split-rank deployment gap.

Formal baseline uses all six measured frames. Partial physical prefixes and
ground-truth edit masks are DIAGNOSTICS, never deployable benchmark results.
Compare each simulated detector with its measured frame at the exact captured
physical-prefix input, and re-create the deleted amplitude BMP in memory.
"""
import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch

from .lab_split_rank_tune import prepare
from .lab_split_rank_capture import WEIGHT_SHA
from .train import sha
from LightGenV2.tasks.t04_semantic_interaction.lab_runtime import OpticalBoundary, STAGES
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import (
    OpenMojiEditingDataset, collate_samples, load_prompt_cache,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def score_parts(category, edit, batch):
    generated = category.argmax(1)
    changed = batch['edit_grid'].bool()
    preserved = batch['preserve_grid'].bool()
    target = batch['target_grid'].long()
    predicted_edit = edit.sigmoid().ge(.5)
    values = []
    for i in range(len(generated)):
        m, keep = changed[i], preserved[i]
        values.append({
            'task': batch['task'][i],
            'category_at_true_changed': float(generated[i][m].eq(target[i][m]).float().mean()),
            'edit_recall': float(predicted_edit[i][m].float().mean()),
            'preserved_false_edit': float(predicted_edit[i][keep].float().mean()),
            'changed_edit_probability': float(edit[i].sigmoid()[m].mean()),
            'preserved_edit_probability': float(edit[i].sigmoid()[keep].mean()),
        })
    return values


def summarize(rows):
    result = {}
    for task in ('overall', 'add', 'replace', 'move', 'remove'):
        selected = rows if task == 'overall' else [r for r in rows if r['task'] == task]
        if selected:
            result[task] = {'samples': len(selected), **{
                key: float(np.mean([r[key] for r in selected]))
                for key in selected[0] if key != 'task'
            }}
    return result


def comparison(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    x, y = a.flatten(), b.flatten()
    xc, yc = x-x.mean(), y-y.mean()
    an, bn = a/a.mean().clamp_min(1e-8), b/b.mean().clamp_min(1e-8)
    return {
        'pcc': float((xc*yc).sum()/(xc.norm()*yc.norm()).clamp_min(1e-8)),
        'cosine': float((x*y).sum()/(x.norm()*y.norm()).clamp_min(1e-8)),
        'mean_normalized_rmse': float((an-bn).square().mean().sqrt()),
        'simulation_mean': float(a.mean()), 'measured_mean': float(b.mean()),
        'measured_background_p01': float(torch.quantile(y, .01)),
        'measured_dynamic_p99_p01': float(torch.quantile(y,.99)-torch.quantile(y,.01)),
    }


def candidate_setup(project, candidate):
    """Each comparison replays only the CCD captured with its own upstream."""
    if candidate == 'splitrank48':
        tune, payload = prepare(project)
        return tune, WEIGHT_SHA, 'splitrank48_test1000', 'splitrank48_decoder_train1000_testselected'
    if candidate == 'original_g2':
        from . import lab_tune_g2_test as tune
        tune.GROUPS = {'g2': ('g2_lowrank64.pt',
            '2acc2f58c9d38b78fe329d98af0e884b90834c6a0aa2496cf199830fb10d7194')}
        return tune, tune.GROUPS['g2'][1], 'g2_full1000', 'g2_decoder_testselected'
    from . import lab_modality_pipeline as pipeline
    tune = pipeline.backend('lab_tune_g2_test')
    tune.GROUPS = {'g2': (pipeline.WEIGHT, pipeline.WEIGHT_SHA)}
    from types import SimpleNamespace
    tune.t = SimpleNamespace(build_model=pipeline.factory(project))
    return tune, pipeline.WEIGHT_SHA, pipeline.PREFIX+'_test1000', pipeline.PREFIX+'_decoder_train1000_testselected'


def cached_gate(project, tune, model, output, folder_name, weight_sha, expected_initial):
    folder = project/'runs'/folder_name
    cache = torch.load(folder/'test_features.pt', map_location='cpu', weights_only=False)
    metrics, parts = {}, {}
    initial = {k: v.clone() for k,v in model.state_dict().items()}
    for label, checkpoint in [('initial', None), ('adapted', folder/'best.pt')]:
        if checkpoint:
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
        else:
            model.load_state_dict(initial, strict=True)
        meter, rows = MetricAccumulator(), []
        with torch.inference_mode():
            for start in range(0,len(cache['ids']),32):
                raw = cache['rows'][start:start+32]
                batch = {k: torch.cat([r[k] for r in raw]) if torch.is_tensor(raw[0][k])
                         else sum([r[k] for r in raw], []) for k in raw[0]}
                category, edit = model.shared_readout.decoder(cache['features'][start:start+32])
                meter.update({'category_logits':category,'edit_logits':edit,
                              'task_logits':batch['task_logits']}, batch)
                rows.extend(score_parts(category, edit, batch))
        metrics[label], parts[label] = meter.compute(), summarize(rows)
    expected_adapted = json.loads((folder/'report.json').read_text())['physical_test']['overall']['changed_cell_accuracy']
    assert abs(metrics['initial']['overall']['changed_cell_accuracy']-expected_initial)<1e-8
    assert abs(metrics['adapted']['overall']['changed_cell_accuracy']-expected_adapted)<1e-8
    model.load_state_dict(initial, strict=True)
    write(output/'full_cached_gate.json', {
        'status':'complete', 'test_samples':1000, 'weight_sha256':weight_sha,
        'metrics':metrics, 'parts':parts,
        'category_at_true_changed_is_diagnostic_not_deployable':True,
        'no_parameter_updates':True, 'no_sdk':True,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-per-task', type=int, default=8)
    parser.add_argument('--candidate', choices=('splitrank48', 'original_g2', 'modality55'), default='splitrank48')
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if output.exists():
        raise RuntimeError('Preserve existing diagnostic output; do not overwrite')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(73)
    tune, weight_sha, capture_name, cache_name = candidate_setup(project, args.candidate)
    capture = project/'runs'/capture_name
    formal = json.loads((capture/'report.json').read_text())
    assert formal['status'] == 'complete' and formal['contract']['checkpoint_sha256'] == weight_sha
    cfg, model = tune.config(project, torch.device('cpu'))
    model.eval().requires_grad_(False)
    protected = tune.protected_sha(model)
    write(output/'execution.json', {
        'status':'running', 'weight_sha256':weight_sha, 'protected_before':protected,
        'candidate':args.candidate, 'capture_run':capture_name, 'cache_run':cache_name,
        'alphas':{name:[float(getattr(getattr(model,name),f'block{i}_optical_fusion'))
                        for i in (1,2)] for name in ('language_core','vision_core')},
        'head_parameters':sum(p.numel() for p in model.shared_readout.parameters()),
        'source_sha256':sha(Path(__file__)), 'device':'cpu', 'sdk':False,
        'partial_prefix_is_diagnostic_not_formal_physical_metric':True,
    })
    cached_gate(project, tune, model, output, cache_name, weight_sha,
                formal['physical_metrics']['overall']['changed_cell_accuracy'])
    data = OpenMojiEditingDataset(cfg.test_manifest,cfg,load_prompt_cache(cfg.prompt_cache_path))
    selected = []
    for task in ('add','replace','move','remove'):
        indices = [i for i,r in enumerate(data.records) if r['task'] == task]
        selected += [indices[i] for i in np.linspace(0,len(indices)-1,args.samples_per_task,dtype=int)]
    contract = json.loads((capture/'contract.json').read_text())
    assert contract['checkpoint_sha256'] == weight_sha
    sys.path.insert(0,str(project.parent/'ABO_I2I_Lab_DVP_8um/lab_dvp8um'))
    import four_image_flow as flow  # raster utility only, never instantiate devices
    phase_sha = {s: sha(capture/'phase'/f'{s}.bmp') for s in STAGES}
    prefix_meters = [MetricAccumulator() for _ in range(7)]
    prefix_parts = [[] for _ in range(7)]
    stage_rows, upstream_rows = [], []
    started = time.time()
    for n,index in enumerate(selected):
        sid = f'test_{index:05d}'
        batch = collate_samples([data[index]])
        measured = {}
        for stage in STAGES:
            base = capture/'ccd'/stage/sid
            receipt = json.loads(base.with_suffix('.json').read_text())
            assert receipt['sample_id']==sid and receipt['stage']==stage
            assert receipt['phase_sha256']==phase_sha[stage]
            assert receipt['p99']>=15 and receipt['canonical_orientation']=='flip_v'
            measured[stage] = torch.from_numpy(np.asarray(Image.open(base.with_suffix('.png')),np.float32).copy()[None]/255)
        with torch.inference_mode():
            for count in range(7):
                prefix = {s:measured[s] for s in STAGES[:count]}
                matched_sim = {}
                with OpticalBoundary(model,prefix) as tap:
                    # Each underlying propagator input is the same field that
                    # capture used after replaying all prior real CCD frames.
                    if count == 6:
                        for prop, original in tap.originals:
                            replaced = prop.forward
                            def forward(module, field, replaced=replaced, original=original):
                                stage = STAGES[tap.index]
                                active = model.language_core.optical_branch.core.geometry.active_aperture
                                simulated = original(field)
                                matched_sim[stage] = simulated[:,active.y0:active.y1,active.x0:active.x1].abs().square()
                                return replaced(field)
                            prop.forward = MethodType(forward,prop)
                    result = model(batch['source_image'],batch['prompt_hidden'])
                prefix_meters[count].update(result,batch)
                prefix_parts[count].extend(score_parts(result['category_logits'],result['edit_logits'],batch))
                if count in (0,6):
                    diagnostics = {}
                    for modality in ('language','vision'):
                        core = getattr(model,modality+'_core')
                        diagnostics[modality] = {
                            'latent_rms':float(torch.cat(core.last_latent_groups).square().mean().sqrt()),
                            'fusion':core.last_fusion_diagnostics,
                        }
                    # JSON-compatible tensors only; retain model diagnostics.
                    upstream_rows.append({'sample_id':sid,'prefix':count,'diagnostics':diagnostics})
                if count == 6:
                    for stage in STAGES:
                        amplitude = tap.amplitudes[stage][0].cpu().numpy()
                        gray = np.rint(np.clip(amplitude,0,1)*255).astype(np.uint8)
                        raster = flow.active_to_native(gray)
                        memory = io.BytesIO(); Image.fromarray(raster).save(memory,format='BMP')
                        actual_sha = hashlib.sha256(memory.getvalue()).hexdigest()
                        receipt = json.loads((capture/'ccd'/stage/f'{sid}.json').read_text())
                        stage_rows.append({'sample_id':sid,'stage':stage,
                            'input_bmp_sha_match':actual_sha==receipt['amplitude_sha256'],
                            'amplitude_mean':float(amplitude.mean()),
                            **comparison(matched_sim[stage],measured[stage])})
        write(output/'progress.json',{'status':'diagnosing','samples':n+1,'total':len(selected),
                                     'elapsed_seconds':time.time()-started})
        print(json.dumps({'completed':n+1,'total':len(selected)}),flush=True)
    def convert(value):
        if torch.is_tensor(value): return value.detach().cpu().tolist()
        if isinstance(value,dict): return {k:convert(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [convert(v) for v in value]
        return value
    assert tune.protected_sha(model)==protected
    stage_summary = {}
    for stage in STAGES:
        rows = [r for r in stage_rows if r['stage']==stage]
        stage_summary[stage] = {'samples':len(rows),
            'bmp_mismatches':sum(not r['input_bmp_sha_match'] for r in rows),
            **{k:float(np.mean([r[k] for r in rows])) for k in
               ('pcc','cosine','mean_normalized_rmse','measured_background_p01','measured_dynamic_p99_p01')}}
    write(output/'report.json',convert({
        'status':'complete','weight_sha256':weight_sha,'candidate':args.candidate,'samples':len(selected),
        'formal_full_test':{k:formal[k]['overall'] for k in ('simulation_metrics','physical_metrics')},
        'selected_indices':selected, 'selection':'even spacing within all four operations, not score-based',
        'protected_before':protected,'protected_after':tune.protected_sha(model),
        'prefix_metrics':[m.compute() for m in prefix_meters],
        'prefix_parts':[summarize(p) for p in prefix_parts],
        'prefix_is_diagnostic_not_formal_physical_accuracy':True,
        'matched_input_stage_summary':stage_summary,'stage_rows':stage_rows,
        'upstream_rows':upstream_rows,'no_sdk':True,'no_parameter_updates':True,
    }))
    write(output/'progress.json',{'status':'complete','samples':len(selected)})


if __name__ == '__main__':
    main()
