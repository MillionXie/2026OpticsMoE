"""Complete paired DEV128 F9 SIM/actual-CCD quality comparison on shared GPU1.

Same original metric functions/models as white_reference_metrics, no altered
score or reference. Means are development proxies, not attribute/test accuracy.
"""
import argparse,json,math,os,statistics,gc,sys
from pathlib import Path
from .scene_lab_protocol import sha,put
from .scene_lab_native import F9

GPU='GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d'
METRICS=('CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact')


def quality_gate(sim,exp,tolerance=.05):
    rows=[]
    for group in ('all','lamp','pillow'):
        for metric in METRICS:
            s=float(sim[group][metric]);e=float(exp[group][metric])
            if not math.isfinite(s)or not math.isfinite(e):raise ValueError('Nonfinite paired quality')
            delta=s-e;near=abs(s)<1e-6;drop=None if near else delta/abs(s)
            rows.append(dict(group=group,metric=metric,SIM=s,EXP=e,absolute_difference=delta,relative_drop=drop,
                near_zero_reference=near,passed=delta<=1e-6 if near else drop<=tolerance))
    return dict(all_passed=all(r['passed']for r in rows),relative_drop_limit=tolerance,rows=rows,development_only=True,independent_TEST=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sim','exp','out','clip-checkpoint','dino-path','reward-path'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--extra',type=Path,nargs='*',default=[]);p.add_argument('--matched-sim',type=Path);p.add_argument('--source-commit',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(a.source_commit)!=40:raise ValueError('Shared GPU1 and published source required')
    import torch,clip,numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModel
    from .white_reference_metrics import load_records,aggregate,reward_loader
    runs=[a.sim,a.exp]+a.extra+([a.matched_sim]if a.matched_sim else[])
    if len({r.name for r in runs})!=len(runs):raise ValueError('Run names must be distinct')
    reports=[json.loads((run/'report.json').read_text())for run in runs]
    if reports[0]['checkpoint_sha256']!=F9 or reports[0].get('actual_CCD_replay'):raise ValueError('Primary reference must be the original frozen F9 SIM')
    if not reports[1].get('actual_CCD_replay'):raise ValueError('Primary EXP must replay all actual CCDs')
    if reports[1]['checkpoint_sha256']!=F9 and (not a.matched_sim or reports[-1]['checkpoint_sha256']!=reports[1]['checkpoint_sha256'] or reports[-1].get('actual_CCD_replay')):
        raise ValueError('Adapted heads also require a same-weight SIM comparison; original F9 reference is never replaced')
    if any(r['scope']!='DEV_expanded'or r['generated_count']!=128 or r.get('independent_TEST_opened') for r in reports):raise ValueError('Complete DEV128 only')
    rows=[r for run in runs for r in load_records(run)]
    signatures=[{r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'],r['initial_latent_sha256'])for r in rows if r['run_id']==run.name}for run in runs]
    if any(x!=signatures[0]for x in signatures):raise ValueError('Not the same RGB/prompt/seed/noise requests')
    if len({json.dumps(r['expanded_DEV_protocol'],sort_keys=True)for r in reports})!=1:raise ValueError('Different DEV manifests')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25,0)
    put(a.out/'protocol.json',dict(source_commit=a.source_commit,source_SHA256=sha(__file__),DEV=128,metrics=METRICS,groups=['all','lamp','pillow'],
        threshold=.05,primary_reference_F9=F9,no_TEST_or_gradients=True,original_metric_implementation_SHA256=sha(Path(__file__).with_name('white_reference_metrics.py'))))
    assets={str(a.clip_checkpoint):sha(a.clip_checkpoint)}
    for folder in (a.dino_path,a.reward_path):
        for file in sorted(folder.iterdir()):
            if file.is_file()and file.suffix in ('.pt','.bin','.json','.safetensors'):assets[str(file)]=sha(file)
    put(a.out/'model_assets.json',assets)
    def progress(stage,count):(a.out/'progress.json').write_text(json.dumps(dict(stage=stage,completed=count,total=len(rows))))
    model,preprocess=clip.load(str(a.clip_checkpoint),device='cuda',jit=False);model.eval().requires_grad_(False)
    with torch.inference_mode():
        for index,row in enumerate(rows):
            root=Path(row['run_path'])
            with Image.open(root/row['generated_PNG'])as im:generated=preprocess(im.convert('RGB')).unsqueeze(0).cuda()
            with Image.open(root/(row['reference_id']+'.png'))as im:reference=preprocess(im.convert('RGB')).unsqueeze(0).cuda()
            tokens=clip.tokenize([row['prompt']],truncate=False).cuda()
            text=torch.nn.functional.normalize(model.encode_text(tokens).float(),dim=-1);image=torch.nn.functional.normalize(model.encode_image(generated).float(),dim=-1);ref=torch.nn.functional.normalize(model.encode_image(reference).float(),dim=-1)
            row.update(CLIP_T=(text*image).sum().item(),CLIP_I=(ref*image).sum().item());progress('CLIP',index+1)
    del model;gc.collect();torch.cuda.empty_cache()
    processor=AutoImageProcessor.from_pretrained(a.dino_path,local_files_only=True);model=AutoModel.from_pretrained(a.dino_path,local_files_only=True).cuda().eval().requires_grad_(False)
    with torch.inference_mode():
        for index,row in enumerate(rows):
            root=Path(row['run_path'])
            with Image.open(root/row['generated_PNG'])as im:generated=im.convert('RGB').copy()
            with Image.open(root/(row['reference_id']+'.png'))as im:reference=im.convert('RGB').copy()
            batch=processor(images=[generated,reference],return_tensors='pt').to('cuda');features=torch.nn.functional.normalize(model(**batch).last_hidden_state[:,0].float(),dim=-1)
            row['DINOv2_I']=(features[0]*features[1]).sum().item();progress('DINOv2',index+1)
    del model;gc.collect();torch.cuda.empty_cache();model=reward_loader()(str(a.reward_path/'ImageReward.pt'),device='cuda',med_config=str(a.reward_path/'med_config.json'));model.eval().requires_grad_(False)
    with torch.inference_mode():
        for index,row in enumerate(rows):
            caption=row['reward_caption'];n=len(model.blip.tokenizer(caption,truncation=False)['input_ids'])
            if n>35:raise ValueError('Compact reward caption would truncate')
            row.update(ImageReward_compact=float(model.score(caption,str(Path(row['run_path'])/row['generated_PNG']))),reward_evaluation_caption=caption,reward_caption_tokens=n);progress('ImageReward',index+1)
    del model;gc.collect();torch.cuda.empty_cache();summary=aggregate(rows)
    put(a.out/'per_image.json',rows);put(a.out/'summary.json',summary)
    def means(run):return {group:{m:summary[run.name+'/'+group]['scores'][m]['mean']for m in METRICS}for group in ('all','lamp','pillow')}
    gate=quality_gate(means(a.sim),means(a.exp));put(a.out/'quality_gate.json',gate)
    matched=quality_gate(means(a.matched_sim),means(a.exp))if a.matched_sim else gate
    put(a.out/'matched_weight_quality_gate.json',matched)
    paired=[];sim_rows={r['sample_id']:r for r in rows if r['run_id']==a.sim.name}
    import importlib.util
    perceptual=None
    if importlib.util.find_spec('lpips')is not None:
        import lpips
        perceptual=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False)
    for row in [r for r in rows if r['run_id']==a.exp.name]:
        other=sim_rows[row['sample_id']]
        with Image.open(Path(other['run_path'])/other['generated_PNG'])as im:s=np.asarray(im.convert('RGB'),dtype=np.float64)/255
        with Image.open(Path(row['run_path'])/row['generated_PNG'])as im:e=np.asarray(im.convert('RGB'),dtype=np.float64)/255
        mse=float(((s-e)**2).mean());item=dict(sample_id=row['sample_id'],MAE=float(abs(s-e).mean()),PSNR_dB=None if mse==0 else -10*math.log10(mse),identical=mse==0)
        if perceptual is not None:
            with torch.inference_mode():
                x=torch.from_numpy(s.copy()).permute(2,0,1)[None].cuda().float()*2-1;y=torch.from_numpy(e.copy()).permute(2,0,1)[None].cuda().float()*2-1
                item['LPIPS_Alex']=float(perceptual(x,y).item())
        paired.append(item)
    put(a.out/'paired_image_change.json',dict(rows=paired,LPIPS_computed=perceptual is not None,LPIPS_note='Official AlexNet LPIPS, RGB [-1,1]; paired change, not attribute success'if perceptual is not None else'lpips package unavailable; not computed, never substitute PSNR for a perceptual success rate'))
    put(a.out/'report.json',dict(complete=True,source_commit=a.source_commit,DEV=128,runs=[str(r)for r in runs],all_twelve_proxy_gates_passed=gate['all_passed'],
        threshold=.05,matched_weight_gate_passed=matched['all_passed'],both_reference_gates_passed=gate['all_passed']and matched['all_passed'],primary_SIM_original_F9=F9,TEST_used=False,training_performed=False,development_metrics_only=True,
        automatic_scores_are_not_attribute_accuracy=True,visual_review_still_required=True,metric_assets={str(a.clip_checkpoint):sha(a.clip_checkpoint),str(a.reward_path/'ImageReward.pt'):sha(a.reward_path/'ImageReward.pt')}))
    print(json.dumps(dict(complete=True,DEV=128,all_twelve_proxy_gates_passed=gate['all_passed'])),flush=True)


if __name__=='__main__':main()
