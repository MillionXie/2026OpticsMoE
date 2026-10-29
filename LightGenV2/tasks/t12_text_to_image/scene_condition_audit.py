"""CPU-only TRAIN condition-fit audit, not a generated-image quality metric."""
import argparse
import json
import os
from pathlib import Path
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions
from .white_reference_scene_probe import digest,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--models',nargs='+',type=Path,required=True)
    p.add_argument('--include-last',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--native-caption-diagnosis',action='store_true',help='Closed TRAIN caption-prototype diagnostic, never image instruction accuracy')
    a=p.parse_args();assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    ready=json.loads((a.cache/'report.json').read_text());assert ready['complete'] and digest(a.cache/'paired_cache.pt')==ready['cache_sha256']
    cache=torch.load(a.cache/'paired_cache.pt',map_location='cpu',weights_only=False,mmap=True)
    rows={r['input_key']:r for r in cache['records'] if r['split']=='train'}
    targets=cache['TRAIN_electronic_conditions'];assert set(targets)==set(rows)
    assert not set(rows)&{r['input_key'] for r in cache['records'] if r['split']=='dev'}
    torch.set_num_threads(2);a.output.mkdir(parents=True);result={};per=[]
    captions=sorted({r['native_caption'] for r in rows.values()})
    vectors=torch.stack([cache['captions'][c]['pooled'].float() for c in captions]) if a.native_caption_diagnosis else None
    if vectors is not None:
        center=vectors.mean(0);vectors=torch.nn.functional.normalize(vectors-center,dim=-1)
    def nearest_caption(value):
        return captions[(torch.nn.functional.normalize(value.float().reshape(-1)-center,dim=0)@vectors.T).argmax().item()]
    for folder in a.models:
        report=json.loads((folder/'report.json').read_text());assert report['complete']
        assert digest(folder/'best_checkpoint.pt')==report['best_sha256']
        for kind in (['best','last'] if a.include_last else ['best']):
            path=folder/(kind+'_checkpoint.pt');state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['pair_sha256']==cache['pair_sha256'] and state['optical_frontend']['checkpoint_sha256']==cache['optical_frontend']['checkpoint_sha256']
            if kind=='last':assert state['step']==report['actual_paired_updates']
            head=QwenSDXLConditions(joint_image_context=state.get('joint_image_context',False)).eval().requires_grad_(False);head.load_state_dict(state['head'],strict=True)
            values=[]
            with torch.inference_mode():
                for key,row in rows.items():
                    f=cache['features'][key];actual=head(f['hidden'].unsqueeze(0).float(),f['visual'].unsqueeze(0).float())
                    scores={n:((actual[n][0]-targets[key][n].float()).square().mean()/targets[key][n].float().square().mean().clamp_min(1e-6)).item() for n in ['image','text','pooled']}
                    if a.native_caption_diagnosis:
                        guessed=nearest_caption(actual['pooled'][0]);teacher_guess=nearest_caption(targets[key]['pooled'])
                        scores.update(native_caption_condition_retrieval_match=guessed==row['native_caption'],
                            electronic_teacher_caption_condition_retrieval_match=teacher_guess==row['native_caption'],
                            requested_caption=row['native_caption'],retrieved_caption=guessed,electronic_teacher_retrieved_caption=teacher_guess)
                    values.append(scores);per.append({'model':folder.name+'/'+kind,'input_key':key,'category':row['category'],**scores})
            result[folder.name+'/'+kind]={'n':len(values),'step':state['step'],'checkpoint_sha256':digest(path),
                'mean_normalized_MSE':{n:sum(v[n] for v in values)/len(values) for n in ['image','text','pooled']}}
            if a.native_caption_diagnosis:
                result[folder.name+'/'+kind]['closed_TRAIN_caption_condition_diagnostic']={
                    'caption_prototypes':len(captions),'LG_retrieval_fraction':sum(v['native_caption_condition_retrieval_match'] for v in values)/len(values),
                    'electronic_teacher_retrieval_fraction':sum(v['electronic_teacher_caption_condition_retrieval_match'] for v in values)/len(values),
                    'not_generated_image_instruction_accuracy':True}
            write(a.output/'progress.json',{'completed':folder.name+'/'+kind})
    write(a.output/'per_input.json',per);write(a.output/'summary.json',result)
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],'source_sha256':digest(Path(__file__)),
        'CPU_float32_TRAIN_only':True,'n_TRAIN_inputs':len(rows),'GPU_processes_started':0,
        'not_generation_quality_or_DEV_generalization':True,'cache_sha256':ready['cache_sha256'],'quality_accepted':False})


if __name__=='__main__':main()
