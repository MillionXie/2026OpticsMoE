"""Complete paired TEST500 three-metric evaluation; no ImageReward acceptance.

Original CLIP/DINO model calls and preprocessing are retained. Results are
development selection, never an attribute-accuracy or independent-test claim.
"""
import argparse,gc,json,math,os,statistics,sys
from pathlib import Path
from .scene_lab_protocol import put,sha

METRICS=('CLIP_T','DINOv2_I','CLIP_I')
GROUPS=('all','lamp','pillow')
GPU='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'
FIXED_SIM='aa2d277bd743d4dd831ec33852629dbf3001e772b55a8ac436a0d2a557a06fd8'
FIXED_METRICS='3dfecc66b0ad604547cddbc22feee546be8bf2a542fea68d7110d735027fec24'


def quality_gate(sim,exp,tolerance=.04):
    if tolerance!=.04:raise ValueError('This explicitly approved protocol is4 percent')
    rows=[]
    for group in GROUPS:
        for metric in METRICS:
            s,e=float(sim[group][metric]),float(exp[group][metric])
            if not math.isfinite(s)or not math.isfinite(e)or s<=1e-6:raise ValueError('Finite nonzero positive SIM reference required')
            ratio=e/s;drop=(s-e)/s
            rows.append(dict(group=group,metric=metric,SIM=s,EXP=e,EXP_div_SIM=ratio,
                signed_relative_drop=drop,absolute_relative_difference=abs(drop),
                within_4percent=abs(drop)<=tolerance+1e-12,target_96_to_100percent=-1e-12<=drop<=tolerance+1e-12,
                naturally_above_SIM=e>s+1e-12))
    return dict(all_nine_within_4percent=all(r['within_4percent']for r in rows),
        all_nine_in_target_range=all(r['target_96_to_100percent']for r in rows),
        natural_improvement_preserved=any(r['naturally_above_SIM']for r in rows),
        no_score_clipping_or_bad_PT_selection=True,relative_difference_limit=.04,rows=rows,
        ImageReward_in_acceptance=False,TEST_selection_development_only=True,independent_generalization_claim=False)


def paired_reports(sim,exp):
    if (sim.get('complete')is not True or exp.get('complete')is not True
        or sim.get('scope')!='TEST_selection'or exp.get('scope')!='TEST_selection'
        or sim.get('generated_count')!=500 or exp.get('generated_count')!=500
        or sim.get('actual_CCD_replay')is True or exp.get('actual_CCD_replay')is not True
        or exp.get('physical_CCD_stage_calls')!=3000
        or sim.get('checkpoint_sha256')!=exp.get('checkpoint_sha256')
        or not sim.get('checkpoint_sha256')
        or sim.get('TEST_selection_protocol')!=exp.get('TEST_selection_protocol')):
        raise ValueError('Complete same-PT paired SIM/actual500 required; never use old-PT SIM')


def fixed_reference_reports(sim,exp):
    if (sim.get('complete')is not True or exp.get('complete')is not True
        or sim.get('scope')!='TEST_selection'or exp.get('scope')!='TEST_selection'
        or sim.get('generated_count')!=500 or exp.get('generated_count')!=500
        or sim.get('checkpoint_sha256')!=FIXED_SIM or sim.get('actual_CCD_replay')is True
        or exp.get('actual_CCD_replay')is not True or exp.get('physical_CCD_stage_calls')!=3000
        or exp.get('parent_physical_generator_SHA256')!=FIXED_SIM or exp.get('adapted_existing_head')is not True
        or exp.get('fixed_SIM_reference_SHA256')!=FIXED_SIM
        or sim.get('TEST_selection_protocol')!=exp.get('TEST_selection_protocol')):
        raise ValueError('Explicit unchanged original SIM and complete adapted physical500 required')


def fixed_reference_gate(sim,exp):
    rows=[]
    for g in GROUPS:
        for m in METRICS:
            s,e=float(sim[g][m]),float(exp[g][m])
            if not math.isfinite(s)or not math.isfinite(e)or s<=1e-6:raise ValueError('Finite fixed reference required')
            d=(s-e)/s;rows.append(dict(group=g,metric=m,SIM=s,EXP=e,signed_relative_drop=d,
                absolute_relative_difference=abs(d),naturally_above_SIM=e>s+1e-12,within_5percent=abs(d)<=.05+1e-12))
    overall=[r for r in rows if r['group']=='all'];above=sum(r['naturally_above_SIM']for r in overall)
    return dict(rows=rows,comparison='Fixed original pre-adaptation SIM, NOT same-checkpoint comparison',
        overall_items_above_SIM=above,overall_all_within_5percent=all(r['within_5percent']for r in overall),
        overall_at_most_one_above_and_within_5percent=above<=1 and all(r['within_5percent']for r in overall),
        all_nine_within_5percent=all(r['within_5percent']for r in rows),natural_improvement_preserved=any(r['naturally_above_SIM']for r in rows),
        scores_unchanged=True,TEST_selection_development_only=True,independent_generalization_claim=False)


def aggregate(rows):
    output={}
    for run in sorted({r['run_id']for r in rows}):
        for group in GROUPS:
            selected=[r for r in rows if r['run_id']==run and(group=='all'or r['category']==group)]
            if len(selected)!=(500 if group=='all'else 250):raise ValueError('All500/250 category outputs required')
            output[run+'/'+group]=dict(n=len(selected),scores={m:dict(mean=statistics.mean(float(r[m])for r in selected),
                sample_std=statistics.stdev(float(r[m])for r in selected))for m in METRICS})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in('sim','exp','out','clip-checkpoint','dino-path'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--fixed-reference',action='store_true');p.add_argument('--fixed-reference-metrics',type=Path)
    p.add_argument('--adapted-checkpoint',type=Path);p.add_argument('--adapted-parent',type=Path)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('Only explicit published shared GPU1 evaluation')
    sim,exp=[json.loads((f/'report.json').read_text())for f in(a.sim,a.exp)]
    if a.fixed_reference:
        fixed_reference_reports(sim,exp)
        if None in(a.fixed_reference_metrics,a.adapted_checkpoint,a.adapted_parent)or sha(a.fixed_reference_metrics)!=FIXED_METRICS:
            raise ValueError('Explicit immutable original per-image metrics and both checkpoint paths required')
        from .scene_lab_middle8_runtime import load_head
        cr=json.loads((a.adapted_checkpoint/'report.json').read_text())
        _,child=load_head(a.adapted_checkpoint/'best_checkpoint.pt',cr,exp['lab_middle8_binding'],parent_checkpoint=a.adapted_parent)
        if(cr['best_sha256']!=exp['checkpoint_sha256']or cr.get('fixed_SIM_reference_SHA256')!=FIXED_SIM
           or child['lab_adaptation'].get('paired_consistency_weight')!=1.):raise ValueError('Wrong approved TRAIN calibration lineage')
    else:paired_reports(sim,exp)
    if a.sim.name==a.exp.name:raise ValueError('Distinct run identities required')
    from .white_reference_metrics import load_records
    records=[list(load_records(f))for f in(a.sim,a.exp)]
    signature=lambda rows:{r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'],r['initial_latent_sha256'])for r in rows}
    if signature(records[0])!=signature(records[1]):raise ValueError('RGB/prompt/seed/noise grid changed')
    if a.fixed_reference:
        old=json.loads(a.fixed_reference_metrics.read_text());lookup={r['sample_id']:r for r in old if r['checkpoint_sha256']==FIXED_SIM}
        if len(lookup)!=500:raise ValueError('All500 original SIM metric records required')
        for row in records[0]:
            prior=lookup[row['sample_id']]
            if any(row[k]!=prior[k]for k in ('input_sha256','prompt','seed','initial_latent_sha256','generated_sha256')):raise ValueError('Fixed original image identity changed')
            row.update({m:prior[m]for m in METRICS})
    rows=records[0]+records[1];scored=records[1]if a.fixed_reference else rows;a.out.mkdir(parents=True,exist_ok=False)
    import torch,clip
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25,0)
    assets={str(a.clip_checkpoint):sha(a.clip_checkpoint)}
    for f in sorted(a.dino_path.iterdir()):
        if f.is_file()and f.suffix in('.json','.safetensors','.bin'):assets[str(f)]=sha(f)
    put(a.out/'model_assets.json',assets)
    put(a.out/'protocol.json',dict(source_commit=os.environ['T12_SOURCE_COMMIT'],source_SHA256=sha(__file__),
        same_checkpoint_SHA256=None if a.fixed_reference else sim['checkpoint_sha256'],TEST=500,metrics=METRICS,groups=GROUPS,
        threshold=.05 if a.fixed_reference else .04,fixed_original_SIM_reference=a.fixed_reference,
        fixed_SIM_metrics_SHA256=FIXED_METRICS if a.fixed_reference else None,
        ImageReward_in_acceptance=False,TEST_gradients=False,SDK=False,independent_generalization_claim=False))
    def progress(stage,n):(a.out/'progress.json').write_text(json.dumps(dict(stage=stage,completed=n,total=len(scored))))
    model,preprocess=clip.load(str(a.clip_checkpoint),device='cuda',jit=False);model.eval().requires_grad_(False)
    with torch.inference_mode():
        for i,row in enumerate(scored):
            root=Path(row['run_path'])
            with Image.open(root/row['generated_PNG'])as im:g=preprocess(im.convert('RGB'))[None].cuda()
            with Image.open(root/(row['reference_id']+'.png'))as im:r=preprocess(im.convert('RGB'))[None].cuda()
            tokens=clip.tokenize([row['prompt']],truncate=False).cuda()
            t=torch.nn.functional.normalize(model.encode_text(tokens).float(),dim=-1)
            g=torch.nn.functional.normalize(model.encode_image(g).float(),dim=-1);r=torch.nn.functional.normalize(model.encode_image(r).float(),dim=-1)
            row.update(CLIP_T=(t*g).sum().item(),CLIP_I=(r*g).sum().item());progress('CLIP',i+1)
    del model;gc.collect();torch.cuda.empty_cache()
    processor=AutoImageProcessor.from_pretrained(a.dino_path,local_files_only=True)
    model=AutoModel.from_pretrained(a.dino_path,local_files_only=True).cuda().eval().requires_grad_(False)
    with torch.inference_mode():
        for i,row in enumerate(scored):
            root=Path(row['run_path'])
            with Image.open(root/row['generated_PNG'])as im:g=im.convert('RGB').copy()
            with Image.open(root/(row['reference_id']+'.png'))as im:r=im.convert('RGB').copy()
            inputs=processor(images=[g,r],return_tensors='pt').to('cuda')
            f=torch.nn.functional.normalize(model(**inputs).last_hidden_state[:,0].float(),dim=-1)
            row['DINOv2_I']=(f[0]*f[1]).sum().item();progress('DINOv2',i+1)
    del model;gc.collect();torch.cuda.empty_cache()
    summary=aggregate(rows)
    means=lambda run:{group:{m:summary[run.name+'/'+group]['scores'][m]['mean']for m in METRICS}for group in GROUPS}
    gate=(fixed_reference_gate if a.fixed_reference else quality_gate)(means(a.sim),means(a.exp));put(a.out/'per_image.json',rows);put(a.out/'summary.json',summary);put(a.out/'quality_gate.json',gate)
    put(a.out/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],TEST=500,
        same_checkpoint_SHA256=None if a.fixed_reference else sim['checkpoint_sha256'],SIM=str(a.sim),EXP=str(a.exp),SDK=False,TEST_gradients=False,
        fixed_original_SIM_reference=a.fixed_reference,SIM_checkpoint_SHA256=sim['checkpoint_sha256'],EXP_checkpoint_SHA256=exp['checkpoint_sha256'],
        ImageReward_in_acceptance=False,all_nine_within_4percent=gate.get('all_nine_within_4percent'),
        all_nine_in_target_range=gate.get('all_nine_in_target_range'),natural_improvement_preserved=gate['natural_improvement_preserved'],
        overall_at_most_one_above_and_within_5percent=gate.get('overall_at_most_one_above_and_within_5percent'),
        development_only=True,visual_review_still_required=True,independent_generalization_claim=False))


if __name__=='__main__':main()
