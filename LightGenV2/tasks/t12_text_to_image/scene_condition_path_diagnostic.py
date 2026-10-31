"""CPU-only condition-path diagnosis after a completed reviewed paired fit.

Feature distances are not image adherence, quality, or independent TEST scores.
No target pixels/latents are used and no model parameters are updated.
"""
import argparse,hashlib,json,os,statistics
from collections import defaultdict
from pathlib import Path
from .scene_instruction_data import sha,write


def centered_stats(pred,wanted):
    import torch
    p=pred.float()-pred.float().mean(0,keepdim=True);t=wanted.float()-wanted.float().mean(0,keepdim=True)
    pp=p.square().sum().item();tt=t.square().sum().item();pt=(p*t).sum().item()
    if tt<=1e-12:raise ValueError('Distinct native conditions required')
    return dict(variation_ratio=(pp/tt)**.5,direction_cosine=pt/max((pp*tt)**.5,1e-12),centered_relative_MSE=(p-t).square().sum().item()/tt,native_variation_RMS=(tt/t.numel())**.5)


def diagnose(a):
    import torch
    from .qwen3_scene_conditioning import QwenSDXLConditions
    from .scene_instruction_balanced_data import assert_admission_current
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or a.output.exists():raise ValueError('New CPU-only diagnostic required')
    torch.set_num_threads(4);assert_admission_current(a.data)
    dr=json.loads((a.data/'report.json').read_text());cr=json.loads((a.cache/'report.json').read_text());mr=json.loads((a.checkpoint/'report.json').read_text())
    if not(dr['eligible_for_training']and cr['complete']and mr['complete']):raise ValueError('Completed reviewed data, cache and actual fit required')
    if sha(a.data/'pairs.json')!=dr['pairs_sha256']or sha(a.cache/'paired_cache.pt')!=cr['cache_sha256']or sha(a.checkpoint/'best_checkpoint.pt')!=mr['best_sha256']:raise ValueError('Asset identity changed')
    payload=torch.load(a.cache/'paired_cache.pt',map_location='cpu',weights_only=False);state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if state['variant']!='qwen'or state['step']<=0 or not(state['pair_sha256']==payload['pair_sha256']==dr['pairs_sha256'])or state['cache_sha256']!=cr['cache_sha256']:raise ValueError('Unrelated data/model/cache')
    head=QwenSDXLConditions(joint_image_context=state.get('joint_image_context',False)).float().eval();head.load_state_dict(state['head'],strict=True);head.requires_grad_(False)
    groups=defaultdict(dict)
    for row in payload['records']:
        if row['split']=='train':groups['train',row['reference_id']][row['condition_id']]=row
    for row in json.loads((a.data/'inference_inputs.json').read_text()):
        src=a.data/row['reference'];assert sha(src)==row['reference_sha256'];key=hashlib.sha256(src.read_bytes()+b'\0'+row['prompt'].encode()+f"\0Qwen{state['Qwen_input_side']}".encode()).hexdigest()
        groups['dev',row['reference_id']][row['condition_id']]=dict(row,input_key=key)
    if len(groups)!=60 or any(len(v)!=3 for v in groups.values()):raise ValueError('All44 TRAIN and16 DEV reference triplets required')
    captions=payload['captions'];names=sorted(captions);records=[];nearest=[]
    with torch.inference_mode():
        for (split,rid),items in sorted(groups.items()):
            predictions=[];targets=[]
            for row in sorted(items.values(),key=lambda r:r['condition_id']):
                feature=payload['features'][row['input_key']];pred=head(feature['hidden'].unsqueeze(0).float(),feature['visual'].unsqueeze(0).float());target=captions[row['native_caption']]
                predictions.append(pred);targets.append(target)
                distances=[]
                for name in names:
                    candidate=captions[name];distance=0.
                    for field in ['text','pooled']:
                        wanted=candidate[field].float();distance+=(pred[field][0]-wanted).square().mean().item()/max(wanted.square().mean().item(),1e-6)
                    distances.append(distance)
                chosen=names[min(range(len(names)),key=lambda i:distances[i])];nearest.append(dict(split=split,reference_id=rid,condition_id=row['condition_id'],correct_native_caption=chosen==row['native_caption']))
            rec=dict(split=split,reference_id=rid)
            for field in ['text','pooled']:rec[field]=centered_stats(torch.cat([p[field]for p in predictions]),torch.stack([p[field]for p in targets]))
            rec['image_condition_max_change']=max((p['image']-predictions[0]['image']).abs().max().item()for p in predictions);records.append(rec)
    summary={}
    for split in ['train','dev']:
        rr=[r for r in records if r['split']==split];nn=[r for r in nearest if r['split']==split]
        summary[split]=dict(references=len(rr),native_caption_nearest_correct=sum(r['correct_native_caption']for r in nn),native_caption_nearest_total=len(nn),text_variation_ratio_median=statistics.median(r['text']['variation_ratio']for r in rr),pooled_variation_ratio_median=statistics.median(r['pooled']['variation_ratio']for r in rr),text_centered_relative_MSE_median=statistics.median(r['text']['centered_relative_MSE']for r in rr),pooled_centered_relative_MSE_median=statistics.median(r['pooled']['centered_relative_MSE']for r in rr),image_change_max=max(r['image_condition_max_change']for r in rr))
    a.output.mkdir();write(a.output/'reference_triplets.json',records);write(a.output/'nearest_native_caption_diagnostic.json',nearest)
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ.get('T12_SOURCE_COMMIT'),checkpoint_sha256=mr['best_sha256'],cache_sha256=cr['cache_sha256'],pairs_sha256=dr['pairs_sha256'],summary=summary,CPU_only=True,target_pixels_or_latents_used=False,model_parameters_updated=False,TEST_opened=False,feature_distances_are_not_image_adherence_or_quality=True))
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','cache','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
    diagnose(p.parse_args())


if __name__=='__main__':main()
