"""Read-only checkpoint diagnosis: live/EMA masks, parameter motion, causal ablations."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .run import ROOT, TASK, sha256, git, write_json, cpu_state
from .models.generator import StaticGenerator
from .models.injection import expert_planes
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, initialize_student, load_backbone
from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.prepare_caltech101_retrieval import prepare_caltech101_subset
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import seed_everything
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import encode_student_samples
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.retrieval_metrics import evaluate_embeddings


def physical(raw):
    return 2 * torch.pi * torch.sigmoid(raw.float())


def phase_distance(a, b):
    difference = torch.atan2(torch.sin(a-b), torch.cos(a-b))
    flat = difference.flatten(-2)
    piston = torch.atan2(flat.sin().mean(-1), flat.cos().mean(-1))
    centered = torch.atan2(torch.sin(difference-piston[...,None,None]), torch.cos(difference-piston[...,None,None]))
    return {'rms_rad': float(difference.square().mean().sqrt()),
            'per_expert_rms_rad': flat.square().mean(-1).sqrt().tolist(),
            'piston_removed_rms_rad': float(centered.square().mean().sqrt()),
            'per_expert_piston_removed_rms_rad': centered.flatten(-2).square().mean(-1).sqrt().tolist(),
            'max_abs_rad': float(difference.abs().max())}


def strip_parametrizations(state):
    return {name.replace('.parametrizations.raw_phase.original', '.raw_phase'): value for name,value in state.items()}


def motion(initial, current):
    groups = {}
    for key, value in current.items():
        if key not in initial or not value.is_floating_point():
            continue
        if '.expert_layers.' in key and 'raw_phase' in key:
            continue  # injected generated states store the frozen original, not the generated mask
        category = 'router' if '.router.' in key else 'global_phase' if '.global_phase.' in key else 'electronics_and_fusion'
        before = initial[key].float()
        delta = value.float()-before
        row = groups.setdefault(category, {'change_sq':0.,'initial_sq':0.,'n':0,'changed':0})
        row['change_sq'] += float(delta.square().sum())
        row['initial_sq'] += float(before.square().sum())
        row['n'] += value.numel()
        row['changed'] += int(torch.count_nonzero(delta))
    return {key:{'rms_change':(x['change_sq']/x['n'])**.5,
                 'relative_l2_change':(x['change_sq']/max(x['initial_sq'],1e-30))**.5,
                 'element_count':x['n'],'changed_elements':x['changed']} for key,x in groups.items()}


def summary_metrics(result):
    return {key:result.metrics[key] for key in ('top1_retrieval_accuracy','top3_retrieval_accuracy','mrr')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs-root', default=str(TASK/'runs/simulation'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_everything(42)
    settings = load_settings(ROOT/'LightGenV2/tasks/t01_object_retrieval/configs/moe_optical_router_scale_matched_dc20.yaml')
    settings.output_dir = output
    settings.num_workers = 0
    bundle = prepare_caltech101_subset(settings, persist=False)
    device = torch.device('cuda')
    loaded = load_backbone(settings,device)
    replacement, readout = build_student(loaded,settings)
    initialization = initialize_student(settings,replacement,readout)
    planes = expert_planes(replacement)
    anchor = torch.randn(2,4,224,224,generator=torch.Generator().manual_seed(42))*.02
    def set_bank(raw):
        # Evaluation only: deliberately materialize fixed tensors, never train through this copy.
        for plane,value in zip(planes,raw.flatten(0,1)):
            plane.raw_phase.copy_(value.to(device))
    set_bank(anchor)
    original = {'vision':cpu_state(replacement.vision_surrogate),
                'language':cpu_state(replacement.language_surrogate),'readout':cpu_state(readout)}
    def load_body(payload):
        replacement.vision_surrogate.load_state_dict(strip_parametrizations(payload['vision']),strict=True)
        replacement.language_surrogate.load_state_dict(strip_parametrizations(payload['language']),strict=True)
        readout.load_state_dict(payload['readout'],strict=True)
        replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); readout.eval()
        replacement.set_fusion_ablation('none')
    def evaluate(name):
        g = encode_student_samples(loaded,replacement,readout,bundle.gallery_samples,settings)
        q = encode_student_samples(loaded,replacement,readout,bundle.test_samples,settings)
        result = evaluate_embeddings(q,bundle.test_samples,g,bundle.gallery_samples,bundle.class_names,
                                     settings.gallery_aggregation,system_name=name)
        return q,result
    initial_q, initial_eval = evaluate('initial_warmstart_electronics_random_experts')
    report = {'diagnostic_git_sha':git('rev-parse','HEAD'),'initialization':initialization,
              'initial_metrics':summary_metrics(initial_eval), 'ema_initial_weight_after_50_steps':.995**50,
              'methods':{},'source_checkpoints':[],'scope':'read-only diagnosis; no training or checkpoint selection'}
    banks = {}
    outputs = {}
    try:
        for method in ('direct','small_hyper','qwen_frozen','qwen_lora'):
            run = Path(args.runs_root)/f'20260907_static_{method}_pilot_s42'
            method_report = {}
            generator = None
            for variant,filename in (('live','last_checkpoint.pt'),('ema','best_checkpoint.pt')):
                path = run/filename
                payload = torch.load(path,map_location='cpu',weights_only=False)
                report['source_checkpoints'].append({'run_id':run.name,'file':filename,'sha256':sha256(path)})
                load_body(payload)
                raw_without_lora = None
                if method != 'direct':
                    if generator is None:
                        seed_everything(42)
                        generator = StaticGenerator(method,payload['generator_source'],device,seed=42)
                    generator.load_compact_state(payload['generator'])
                    raw = generator().cpu()
                    if method == 'qwen_lora':
                        lora_b = [p for name,p in generator.named_parameters() if name.endswith('lora_b')]
                        norm = float(torch.stack([p.float().square().sum() for p in lora_b]).sum().sqrt())
                        backups = [p.clone() for p in lora_b]
                        for p in lora_b:
                            p.zero_()
                        raw_without_lora = generator().cpu()
                        for p,b in zip(lora_b,backups):
                            p.copy_(b)
                        lora_report = {'b_parameter_norm':norm,
                                       'mask_change_if_lora_disabled':phase_distance(physical(raw),physical(raw_without_lora))}
                else:
                    raw = torch.stack([p.raw_phase.detach().cpu() for p in planes]).reshape_as(anchor)
                set_bank(raw)
                banks[f'{method}_{variant}'] = physical(raw)
                q, result = evaluate(f'{method}_{variant}')
                outputs[f'{method}_{variant}'] = (q,result)
                record = {'metrics':summary_metrics(result), 'phase_motion':phase_distance(physical(raw),physical(anchor)),
                          'phase_dc_power_per_expert':torch.abs(torch.exp(1j*physical(raw)).flatten(-2).mean(-1)).square().tolist(),
                          'nonexpert_parameter_motion':{name:motion(original[name],strip_parametrizations(payload[name])) for name in original},
                          'fusion_diagnostics_last_batch':replacement.fusion_diagnostics()}
                if method == 'qwen_lora':
                    record['lora'] = lora_report
                if variant == 'ema':
                    old = json.loads((run/'final_report.json').read_text())['metrics']
                    if abs(result.metrics['top1_retrieval_accuracy']-old['top1_retrieval_accuracy']) > 1e-7:
                        raise RuntimeError(f'Original EMA metrics did not reproduce for {method}')
                # Paired interventions: freeze the entire trained body and change only expert masks,
                # or separately bypass all optical feature contributions using the existing fusion API.
                if method in ('direct','qwen_lora'):
                    interventions = [('initial_experts',anchor),('flat_pi_experts',torch.zeros_like(raw))]
                    if raw_without_lora is not None:
                        interventions.append(('lora_disabled_same_head',raw_without_lora))
                    record['ablations'] = {}
                    for name,bank in interventions:
                        set_bank(bank)
                        aq, ar = evaluate(f'{method}_{variant}_{name}')
                        record['ablations'][name] = {**summary_metrics(ar),
                            'query_embedding_rms_difference':float((aq-q).square().mean().sqrt()),
                            'changed_top1_predictions':sum(a['predicted_sku_index']!=b['predicted_sku_index'] for a,b in zip(ar.rows,result.rows))}
                    set_bank(raw)
                    replacement.set_fusion_ablation('remove_optical')
                    aq, ar = evaluate(f'{method}_{variant}_remove_optical')
                    record['ablations']['remove_optical'] = {**summary_metrics(ar),
                        'query_embedding_rms_difference':float((aq-q).square().mean().sqrt()),
                        'changed_top1_predictions':sum(a['predicted_sku_index']!=b['predicted_sku_index'] for a,b in zip(ar.rows,result.rows))}
                    replacement.set_fusion_ablation('none')
                method_report[variant] = record
                del payload
                gc.collect()
                print(json.dumps({'method':method,'variant':variant,'metrics':record['metrics'],'motion':record['phase_motion']['rms_rad']}),flush=True)
            report['methods'][method] = method_report
            del generator
            gc.collect(); torch.cuda.empty_cache()
        report['pairwise'] = {}
        for variant in ('live','ema'):
            aq, ar = outputs[f'direct_{variant}']
            dq, dr = outputs[f'qwen_lora_{variant}']
            report['pairwise'][variant] = {'qwen_vs_direct_phase':phase_distance(banks[f'qwen_lora_{variant}'],banks[f'direct_{variant}']),
                'query_embedding_rms_difference':float((aq-dq).square().mean().sqrt()),
                'changed_top1_predictions':sum(a['predicted_sku_index']!=b['predicted_sku_index'] for a,b in zip(ar.rows,dr.rows))}
        write_json(output/'diagnosis.json',report)
        plot_phases(banks,physical(anchor),output)
        print(json.dumps({'initial_metrics':report['initial_metrics'],'pairwise':report['pairwise']}),flush=True)
    finally:
        replacement.close()


def plot_phases(banks,initial,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Shared scale: same physical differences, no per-image contrast normalization.
    fig, axes = plt.subplots(2,4,figsize=(12,6),layout='constrained')
    names = ('direct_live','qwen_lora_live','direct_ema','qwen_lora_ema')
    differences = [banks[n]-initial for n in names]
    limit = max(float(x.abs().quantile(.995)) for x in differences)
    for r,modality in enumerate(('Vision expert 0','Language expert 0')):
        for c,(name,diff) in enumerate(zip(names,differences)):
            im=axes[r,c].imshow(diff[r,0].numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[r,c].set_title(name.replace('_',' '),fontsize=11)
            axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
            if c==0: axes[r,c].set_ylabel(modality)
    fig.colorbar(im,ax=axes,label='Phase change from shared initialization (rad)',shrink=.8)
    fig.suptitle('Learned phase changes: live weights versus EMA (shared scale; 99.5% range)')
    fig.savefig(output/'phase_changes.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(11,5.7),layout='constrained')
    delta=banks['qwen_lora_live']-initial
    limit=float(delta.abs().max())
    for m in range(2):
        for e in range(4):
            im=axes[m,e].imshow(delta[m,e],cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[m,e].set_title(f'{"Vision" if m==0 else "Language"} expert {e}')
            axes[m,e].set_xticks([]);axes[m,e].set_yticks([])
    fig.colorbar(im,ax=axes,label='Phase change (rad)',shrink=.8)
    fig.suptitle('Qwen LoRA: all eight live expert masks, relative to initialization')
    fig.savefig(output/'qwen_all_expert_changes.png',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
