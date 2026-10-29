"""Numerical score replay and an independent, unblinded VQA calibration.

The calibration tests an evaluator, not model quality or independent accuracy.
No unique target is read. A VQA failure must not become an automatic GT filter.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
from .white_reference_scene_probe import digest,write,GPU

NAMES=['CLIP_T','DINOv2_I','CLIP_I','ImageReward_compact']


def numeric_audit(metric_runs):
    shared=defaultdict(list);replayed=[];assets=[]
    for run in metric_runs:
        if not json.loads((run/'report.json').read_text())['complete']:raise ValueError('Incomplete scoring')
        rows=json.loads((run/'per_image.json').read_text());summary=json.loads((run/'summary.json').read_text())
        seen=set()
        for row in rows:
            key=(row['run_id'],row['generated_PNG'])
            if key in seen:raise ValueError('Duplicate output in one evaluation')
            seen.add(key)
            if not all(math.isfinite(row[n]) for n in NAMES):raise ValueError('Nonfinite score')
            shared[key].append(row)
        maximum=0.
        for group,stored in summary.items():
            rid,category=group.rsplit('/',1)
            selected=[r for r in rows if r['run_id']==rid and category in ['all',r['category']]]
            if len(selected)!=stored['n']:raise ValueError('Denominator mismatch')
            for name in NAMES:
                values=[r[name] for r in selected]
                mean=math.fsum(values)/len(values)
                sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/(len(values)-1)) if len(values)>1 else 0.
                maximum=max(maximum,abs(mean-stored['scores'][name]['mean']),abs(sd-stored['scores'][name]['sample_std']))
        if maximum>1e-12:raise ValueError('Aggregate replay differs')
        replayed.append({'run':str(run),'n':len(rows),'aggregate_maximum_absolute_error':maximum,
          'per_image_sha256':digest(run/'per_image.json'),'summary_sha256':digest(run/'summary.json')})
        assets.append(json.loads((run/'model_assets.json').read_text()))
    differences=[];repeats=0
    for key,rows in shared.items():
        if len(rows)<2:continue
        repeats+=1
        for other in rows[1:]:
            for field in ['generated_sha256','input_sha256','prompt','reward_evaluation_caption']:
                if rows[0][field]!=other[field]:raise ValueError('Repeated request identity changed')
            differences.extend(abs(rows[0][n]-other[n]) for n in NAMES)
    if repeats==0:raise ValueError('No independent scoring repeat')
    common=set.intersection(*(set(a) for a in assets))
    if not common or any(len({a[k] for a in assets})!=1 for k in common):raise ValueError('Metric assets changed')
    error=max(differences)
    if error>1e-5:raise ValueError('Repeated forward scores differ')
    return {'complete':True,'numerical_replay':replayed,'repeated_output_count':repeats,
      'repeated_forward_maximum_absolute_error':error,'common_model_assets_verified':len(common),
      'attribute_accuracy_validated':False,'interpretation':'Arithmetic and repeated forward values verified, not instruction correctness.'}


def calibration_cases(root):
    """Cases visually reviewed before running BLIP; uncertain examples excluded."""
    base=root/'white_reference_structural_canny_sweep_20261029'
    specs=[('canny045',0,0,1043,'blue'),('canny045',1,0,1042,'blue'),('canny045',1,0,1043,'blue'),
      ('canny045',0,1,1042,'red'),('canny025',0,1,1043,'black'),('canny025',1,0,1042,'white'),
      ('canny025',0,0,1043,'blue'),('canny025',1,0,1043,'blue'),
      ('canny025',2,0,1042,'blue'),('canny045',2,0,1042,'blue'),
      ('canny025',3,0,1042,'white'),('canny045',3,0,1042,'white')]
    aliases={'blue':['blue','navy','dark blue'],'red':['red','orange','brown','terracotta'],
      'black':['black'],'white':['white','cream','beige','white and blue','blue and white']}
    out=[]
    for variant,ref,c,seed,color in specs:
        image=base/variant/f'reference_{ref:02d}_condition{c}_seed{seed}.png'
        question=('What color is the outside of the lampshade?' if ref<2 else 'What is the main background color of the front cushion fabric?')
        out.append({'image':str(image),'question':question,'accepted_answers':aliases[color],
          'expected_color':color,'requested_color_matches_visible':not (color=='black' or (ref==1 and c==0 and color=='white') or (ref==3 and color=='white')),
          'group':'component_color','image_sha256':digest(image)})
    for variant,ref,c,seed,answer in [('canny025',3,0,1042,'no'),('canny045',3,0,1042,'no'),
        ('canny025',2,1,1042,'yes'),('canny025',3,1,1042,'yes')]:
        image=base/variant/f'reference_{ref:02d}_condition{c}_seed{seed}.png'
        out.append({'image':str(image),'question':'Does the front cushion have broad cream stripes?',
          'accepted_answers':[answer],'group':'broad_stripes','image_sha256':digest(image)})
    return out


def run_vqa(root,model_path,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('GPU6 only')
    import torch
    from PIL import Image
    from transformers import BlipProcessor,BlipForQuestionAnswering
    torch.set_num_threads(4)
    cases=calibration_cases(root);write(output/'calibration_labels_before_inference.json',cases)
    processor=BlipProcessor.from_pretrained(model_path,local_files_only=True)
    model=BlipForQuestionAnswering.from_pretrained(model_path,local_files_only=True).float().cuda().eval().requires_grad_(False)
    records=[]
    for case in cases:
        with Image.open(case['image']) as im:inputs=processor(images=im.convert('RGB'),text=case['question'],return_tensors='pt').to('cuda')
        with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=8,do_sample=False,num_beams=1)
        answer=processor.decode(tokens[0],skip_special_tokens=True).lower().strip().rstrip('.')
        records.append({**case,'answer':answer,'calibration_correct':answer in case['accepted_answers']})
        write(output/'per_question.json',records)
    by_group={g:{'n':sum(r['group']==g for r in records),'correct':sum(r['group']==g and r['calibration_correct'] for r in records)} for g in {r['group'] for r in records}}
    correct=sum(r['calibration_correct'] for r in records)
    return {'complete':True,'stage':'evaluator_calibration_only','n':len(records),'correct':correct,
      'agreement_with_predeclared_visual_labels':correct/len(records),'groups':by_group,
      'validated_for_automatic_training_label_acceptance':False,
      'limitation':'Small unblinded TRAIN controls; no automatic training-label filter or paper attribute rate validated.',
      'model_sha256':digest(model_path/'model.safetensors'),'independent_TEST_opened':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metric-runs',nargs='+',type=Path);p.add_argument('--vqa-model',type=Path);p.add_argument('--root',type=Path)
    a=p.parse_args();commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40:raise ValueError('New run/published source required')
    a.output.mkdir(parents=True);write(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':commit,'source_sha256':digest(Path(__file__))})
    try:
        report=run_vqa(a.root,a.vqa_model,a.output) if a.vqa_model else numeric_audit(a.metric_runs)
        write(a.output/'report.json',report)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
