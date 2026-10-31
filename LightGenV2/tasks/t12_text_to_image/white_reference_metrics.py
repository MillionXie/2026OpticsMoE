"""Target-free automatic feasibility scores; not attribute accuracy or TEST results.

CLIP-T uses the complete canonical request for fair scoring across native prompt
styles. DINOv2-S/14 and CLIP-I compare full generated RGB to the white reference:
they are appearance proxies, not pure outline or identity classifiers. ImageReward
uses its official unmodified score() API on a fixed compact evaluation caption,
checked to fit its 35-token window without truncation; it does NOT evaluate every
clause of the long instruction. No unique target, mask, selected seed, or GT crop.
"""
import argparse
from collections import defaultdict
import csv
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
import types

from .white_reference_scene_probe import digest, write, GPU


def reward_loader():
    # Import the unchanged official scoring modules without the package root's
    # unrelated ReFL training API, which imports modern diffusers/PEFT. Metrics
    # have a private transformers4.35 runtime; generator/Qwen environments stay
    # unchanged. No vendor files, model forward, or score formula are patched.
    if 'ImageReward' not in sys.modules:
        package = types.ModuleType('ImageReward')
        package.__path__ = [str(importlib.metadata.distribution('image-reward').locate_file('ImageReward'))]
        sys.modules['ImageReward'] = package
    from ImageReward.utils import load
    return load


def reward_caption(category, condition_id, profile='free_lighting_v2'):
    if profile=='appearance_preserving_room_v1':
        from .appearance_scene_protocol import conditions
        return conditions(category)[int(condition_id)]['reward_caption']
    captions = {
        'lamp': [
            'A blue fabric lampshade and brass base on an oak desk in a study with bookshelves, curtains and a plant.',
            'A red lampshade and matte black base on a stone table in a reading corner with an armchair and bookshelves.'],
        'pillow': [
            'A blue woven cushion with cream piping on a beige sofa in a living room with curtains and a plant.',
            'A terracotta cushion with broad cream stripes on a grey armchair in a reading nook with bookshelves and a plant.']}
    return captions[category][int(condition_id)]


def check_architecture_diagnostic(report, records):
    """Admit the fixed 16-reference/two-scene ablation, never arbitrary subsets."""
    from .scene_depth_probe import checked_config
    graph=report.get('experimental_compression',{})
    if graph.get('schema')!='scene_depth_probe_actual_graph_v1':
        raise ValueError('Missing actual compressed-graph identity')
    config=checked_config(graph.get('profile',{}))
    protocol=report.get('expanded_DEV_protocol',{})
    if (report.get('checkpoint_sha256')!=config['parent_generator_sha256'] or
        protocol.get('contexts')!=32 or protocol.get('full_audited_contexts')!=128 or
        protocol.get('deterministic_scene_subset')!=['0','1'] or
        protocol.get('selection_by_quality') is not False or
        graph.get('TEST_opened') is not False or graph.get('training_performed') is not False):
        raise ValueError('Only the prespecified target-free depth diagnostic is supported')
    refs={r['reference_id'] for r in records}
    if len(records)!=32 or len(refs)!=16 or any(r.get('split')!='dev' for r in records):
        raise ValueError('Require all thirty-two outputs from the sixteen DEV references')
    if {c:sum(r.get('category')==c for r in records) for c in ('lamp','pillow')}!={'lamp':16,'pillow':16}:
        raise ValueError('Architecture cohort must contain both complete categories')
    for ref in refs:
        group=[r for r in records if r['reference_id']==ref]
        if (len(group)!=2 or {r['condition_id'] for r in group}!={'0','1'} or
            len({r['seed'] for r in group})!=1 or len({r['input_sha256'] for r in group})!=1):
            raise ValueError('Changed reference, source seed or prespecified scene pair')
    if any(r.get('experimental_compression')!=graph for r in records):
        raise ValueError('Per-image compressed graph differs from the run identity')


def load_records(run,allow_frozen_test=False):
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    is_test=report.get('scope')=='TEST'
    if is_test and (not allow_frozen_test or not report.get('frozen_TEST_protocol')):
        raise ValueError('Explicit frozen TEST metric scope required')
    if (report.get('complete') is not True or report.get('training_performed') is not False or
            report.get('independent_TEST_opened') is not is_test):
        raise ValueError('Only completed, frozen TRAIN feasibility runs allowed')
    records = json.loads((run/'images.json').read_text(encoding='utf-8'))
    keys = [(r['reference_id'], r['condition_id'], r['seed']) for r in records]
    expanded=report.get('scope')=='DEV_expanded'
    if expanded and not report.get('expanded_DEV_protocol'):raise ValueError('Expanded DEV requires its audited canonical-grid identity')
    architecture=report.get('scope')=='DEV_architecture_diagnostic'
    if architecture:check_architecture_diagnostic(report,records)
    from .scene_metric_cohort import expanded_identity
    expanded_count=expanded_identity(report,records)[0] if expanded else None
    expected = 500 if is_test else (expanded_count if expanded else (32 if architecture or report.get('scope') == 'DEV' else 16))
    if len(records) != report['generated_count'] or len(records) != expected or len(set(keys)) != expected:
        raise ValueError('All prespecified unique outputs required; no best-of-seed selection')
    if report.get('scope') in ['DEV','DEV_expanded','DEV_architecture_diagnostic'] and any(r.get('split') != 'dev' for r in records):
        raise ValueError('DEV comparison contains another split')
    if is_test and any(r.get('split')!='test' for r in records):raise ValueError('TEST comparison contains another split')
    for r in records:
        if (Path(r['generated_PNG']).name != r['generated_PNG'] or
                digest(run/r['generated_PNG']) != r['generated_sha256'] or
                digest(run/(r['reference_id']+'.png')) != r['input_sha256']):
            raise ValueError('Input or output identity changed')
        r = dict(r)
        r.update(run_id=run.name, run_path=str(run), prompt_profile=report.get('prompt_profile', 'directional_v1'))
        yield r


def aggregate(rows):
    result = {}
    for group in sorted({(r['run_id'], r['category']) for r in rows} | {(r['run_id'], 'all') for r in rows}):
        selected = [r for r in rows if r['run_id'] == group[0] and group[1] in ['all', r['category']]]
        scores = {}
        for name in ['CLIP_T', 'DINOv2_I', 'CLIP_I', 'ImageReward_compact']:
            vals = [float(r[name]) for r in selected]
            if not all(math.isfinite(v) for v in vals): raise ValueError('Non-finite metric')
            scores[name] = {'mean': statistics.mean(vals), 'sample_std': statistics.stdev(vals) if len(vals)>1 else 0}
        result[group[0]+'/'+group[1]] = {'n': len(selected), 'scores': scores}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clip-checkpoint', type=Path, required=True)
    p.add_argument('--dino-path', type=Path, required=True)
    p.add_argument('--reward-path', type=Path, required=True)
    p.add_argument('--allow-frozen-test',action='store_true',help='Only after a shared model/protocol freeze; no DEV/TEST mixing')
    args = p.parse_args()
    commit = os.environ.get('T12_SOURCE_COMMIT', '')
    if args.output.exists() or len(commit) != 40 or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise ValueError('New run, published source and authorized GPU6 required')
    args.output.mkdir(parents=True)
    write(args.output/'execution.json', {'argv': sys.argv, 'pid': os.getpid(), 'source_commit': commit,
          'source_sha256': digest(Path(__file__)), 'GPU_UUID': GPU,
          'versions': {n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'image-reward']}})
    try:
        import torch
        import clip
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
        load_reward = reward_loader()
        torch.set_num_threads(4)
        reports=[json.loads((run/'report.json').read_text()) for run in args.runs]
        if args.allow_frozen_test:
            assert all(r.get('scope')=='TEST' for r in reports)
            assert len({r['frozen_TEST_protocol']['manifest_sha256'] for r in reports})==1
        rows = [r for run in args.runs for r in load_records(run,args.allow_frozen_test)]
        if any(r.get('scope')=='DEV_architecture_diagnostic' for r in reports):
            if not all(r.get('scope')=='DEV_architecture_diagnostic' for r in reports):
                raise ValueError('Do not mix depth diagnostics with another evaluation cohort')
            signatures=[{r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'],r['initial_latent_sha256']) for r in rows if r['run_id']==run.name} for run in args.runs]
            if not all(s==signatures[0] for s in signatures):
                raise ValueError('Depth comparisons require the same complete RGB/prompt/noise requests')
        if any(r.get('scope')=='DEV_expanded' for r in reports):
            assert all(r.get('scope')=='DEV_expanded' for r in reports),'Do not mix expanded and original DEV cohorts'
            from .scene_metric_cohort import expanded_identity
            assert len({expanded_identity(r)[1] for r in reports})==1
            signatures=[{r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'],r['initial_latent_sha256']) for r in rows if r['run_id']==run.name} for run in args.runs]
            assert all(s==signatures[0] for s in signatures)
        if args.allow_frozen_test:
            signatures=[]
            for run in args.runs:
                signatures.append({r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'],r['initial_latent_sha256']) for r in rows if r['run_id']==run.name})
            assert all(s==signatures[0] for s in signatures)
        if len({r.name for r in args.runs}) != len(args.runs): raise ValueError('Duplicate run identity')
        assets = {str(args.clip_checkpoint): digest(args.clip_checkpoint)}
        for root in [args.dino_path, args.reward_path]:
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix in ['.bin', '.json', '.safetensors', '.pt']:
                    assets[str(path)] = digest(path)
        write(args.output/'model_assets.json', assets)
        model, preprocess = clip.load(str(args.clip_checkpoint), device='cuda', jit=False)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            for i, row in enumerate(rows):
                root = Path(row['run_path'])
                with Image.open(root/row['generated_PNG']) as image: generated = preprocess(image.convert('RGB')).unsqueeze(0).cuda()
                with Image.open(root/(row['reference_id']+'.png')) as image: reference = preprocess(image.convert('RGB')).unsqueeze(0).cuda()
                # clip.tokenize raises on overflow; no silent truncation.
                tokens = clip.tokenize([row['prompt']], truncate=False).cuda()
                text = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
                image = torch.nn.functional.normalize(model.encode_image(generated).float(), dim=-1)
                ref = torch.nn.functional.normalize(model.encode_image(reference).float(), dim=-1)
                row.update(CLIP_T=(text*image).sum().item(), CLIP_I=(ref*image).sum().item())
                write(args.output/'progress.json', {'stage':'CLIP','done':i+1,'total':len(rows)})
        del model; gc.collect(); torch.cuda.empty_cache()
        processor = AutoImageProcessor.from_pretrained(args.dino_path, local_files_only=True)
        model = AutoModel.from_pretrained(args.dino_path, local_files_only=True).cuda().eval().requires_grad_(False)
        with torch.inference_mode():
            for i, row in enumerate(rows):
                root = Path(row['run_path'])
                with Image.open(root/row['generated_PNG']) as im: generated = im.convert('RGB').copy()
                with Image.open(root/(row['reference_id']+'.png')) as im: reference = im.convert('RGB').copy()
                batch = processor(images=[generated, reference], return_tensors='pt').to('cuda')
                features = torch.nn.functional.normalize(model(**batch).last_hidden_state[:,0].float(), dim=-1)
                row['DINOv2_I'] = (features[0]*features[1]).sum().item()
                write(args.output/'progress.json', {'stage':'DINOv2','done':i+1,'total':len(rows)})
        del model; gc.collect(); torch.cuda.empty_cache()
        model = load_reward(str(args.reward_path/'ImageReward.pt'), device='cuda', med_config=str(args.reward_path/'med_config.json'))
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            for i, row in enumerate(rows):
                caption = row.get('reward_caption') or reward_caption(row['category'], row['condition_id'],row['prompt_profile'])
                n = len(model.blip.tokenizer(caption, truncation=False)['input_ids'])
                if n > 35: raise ValueError(f'ImageReward compact caption would truncate: {n}')
                row.update(reward_evaluation_caption=caption, reward_caption_tokens=n,
                           ImageReward_compact=float(model.score(caption, str(Path(row['run_path'])/row['generated_PNG']))))
                write(args.output/'progress.json', {'stage':'ImageReward','done':i+1,'total':len(rows)})
        del model; gc.collect(); torch.cuda.empty_cache()
        write(args.output/'per_image.json', rows)
        summary = aggregate(rows)
        write(args.output/'summary.json', summary)
        with (args.output/'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle);writer.writerow(['run/category','n','CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact'])
            for name, value in summary.items():writer.writerow([name,value['n']]+[value['scores'][m]['mean'] for m in ['CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact']])
        write(args.output/'report.json', {'complete':True,'n':len(rows),'source_commit':commit,
              'input_whitelist':['generated RGB','white reference RGB','canonical full prompt','fixed compact evaluation caption'],
              'unique_GT_or_masks_used':False,'independent_TEST_opened':args.allow_frozen_test,'training_performed':False,
              'automatic_scores_are_not_attribute_pass_rates':True,'quality_accepted':False,
              'metric_protocol':{'CLIP_T':'OpenAI ViT-B/32 cosine, untruncated canonical request, full generated image',
                'DINOv2_I':'facebook/dinov2-small CLS cosine, full reference and generated image; differs from original DreamBench DINOv1-S/16; not pure outline',
                'CLIP_I':'OpenAI ViT-B/32 full-image cosine, auxiliary appearance metric',
                'ImageReward_compact':'Official score() with fixed <=35-token caption. Auxiliary preference; does not score every full-prompt clause.'},
              'limitations':[f"{len({r['reference_id'] for r in rows})} reference views; "+('frozen TEST draws are clustered by product/context, not independent products or attribute accuracy.' if args.allow_frozen_test else 'TRAIN/DEV diagnosis only, not independent test or automatic attribute accuracy.'),
                'Generation resolution/protocol must be checked in each source report before any ranking; historical runs may differ.',
                'Appearance similarity can reward an unchanged white-background output that ignores text; never use it alone.',
                'Reward captions omit peripheral details and lighting; original prompts remain preserved and fully scored by CLIP-T.',
                'FID not computed: no matched eligible scene reference distribution; white input images are not a valid scene GT pool.']})
    except Exception:
        write(args.output/'failure.json', {'traceback':traceback.format_exc()});raise


if __name__ == '__main__': main()
