"""Sequential, one-GPU data/cache/smoke/train/export workflow for T12 960/96.

Requires a visually reviewed screen. Training and all postprocessing are pinned
to the published source environment. No process belonging to another task is stopped.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .qwen_vl_turbo_data import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('screen','data','data-run','parent-data','parent','qwen','sd','config','smoke-output','output','clip-repo','clip-weights'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reviewed-profile',type=Path)
    p.add_argument('--resume-before-training',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'workflow_progress.json').exists():
        if not a.resume_before_training or (a.output/'execution.json').exists() or (a.output/'best_checkpoint.pt').exists():raise ValueError('Existing training workflow; do not restart')
        old=json.loads((a.output/'workflow_progress.json').read_text())
        if old['stage'] not in ('cache','prepare','annotate'):raise ValueError('Resume restricted to pre-training failure')
        write_json(a.output/'workflow_before_caption_correction.json',old)
        failure=a.output/'workflow_failure.json'
        if failure.exists():
            destination=a.output/'workflow_failure_before_caption_correction.json'
            if destination.exists():raise ValueError('Repeated resume requires inspection')
            failure.rename(destination)
    cfg=json.loads(a.config.read_text(encoding='utf-8'));gpu=cfg['gpu_uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Unauthorized GPU')
    review=json.loads((a.screen/'visual_review.json').read_text())
    if not review['passed'] or review['screen_profile_sha256']!=sha256(a.screen/'screen_profile.json'):raise ValueError('Screen review mismatch')
    started=time.time();children=[];stage='initializing';commands=[]
    module='LightGenV2.tasks.t12_text_to_image.'
    def run(label, tail):
        nonlocal stage
        stage=label;cmd=[sys.executable,'-m',*tail];commands.append({'stage':label,'argv':cmd})
        with (a.output/(label+'.log')).open('ab') as log:
            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children.append(child.pid)
            write_json(a.output/'workflow_progress.json',{'stage':stage,'pid':child.pid,'own_child_pids':children,'started_unix':started,'commands':commands,'source_commit':os.environ.get('T12_SOURCE_COMMIT')})
            result=child.wait()
        if result:raise RuntimeError(f'{label} failed, exit {result}; inspect its log')
    try:
        if a.reviewed_profile:
            checked=json.loads((a.screen/'caption_review.json').read_text())
            if not checked['passed'] or checked['reviewed_profile_sha256']!=sha256(a.reviewed_profile):raise ValueError('Caption review identity differs')
            profile=a.reviewed_profile
        else:
            run('annotate',[module+'qwen_vl_expanded_data','--stage','annotate','--output',str(a.screen),'--qwen',str(a.qwen),'--gpu-uuid',gpu])
            profile=a.screen/'annotated_profile.json'
        run('prepare',[module+'qwen_vl_curated','--stage','prepare','--profile',str(profile),'--data',str(a.data),'--parent-data',str(a.parent_data),'--output',str(a.data_run)])
        audit=json.loads((a.data_run/'data_audit.json').read_text())
        if audit['pairs']!={'train':960,'val':96} or not audit['role_disjoint'] or audit['evaluation_label']!='development_test':raise ValueError('Approved data budget/identity differs')
        run('cache',[module+'qwen_vl_curated','--stage','cache','--profile',str(profile),'--data',str(a.data),'--parent-data',str(a.parent_data),'--output',str(a.data_run),'--qwen',str(a.qwen),'--sd',str(a.sd),'--gpu-uuid',gpu])
        smoke={**cfg,'smoke':True,'train_steps':4,'batch_size':4,'lr_warmup_steps':0,'eval_every':4}
        smoke.pop('expected_train_pairs');smoke.pop('expected_evaluation_pairs')
        smoke_profile=a.data_run/'smoke_config.json';write_json(smoke_profile,smoke)
        def training(config,out):return [module+'qwen_vl_spatial_run','--config',str(config),'--parent',str(a.parent),'--data-run',str(a.data_run),'--qwen',str(a.qwen),'--sd',str(a.sd),'--output',str(out)]
        run('smoke',training(smoke_profile,a.smoke_output))
        passed=json.loads((a.smoke_output/'report.json').read_text())
        if not passed['complete'] or passed['best_step']<=0:raise ValueError('Real backward smoke did not complete')
        run('train',training(a.config,a.output))
        report=json.loads((a.output/'report.json').read_text())
        if not report['complete'] or report['epochs_completed']!=100 or report['train_optimization_pairs']!=960 or report['best_val']['count']!=96:raise ValueError('Full run incomplete')
        out=a.output/'overview'
        run('comparison',[module+'reports.qwen_vl_spatial_20261027.summarize','--run',str(a.output),'--output',str(out)])
        run('export',[module+'reports.qwen_vl_data_review_20261027.export_val','--run',str(a.output)])
        run('prompts',[module+'reports.qwen_vl_data_review_20261027.prompt_review','--run',str(a.output),'--output',str(out/'prompt_review')])
        run('semantics',[module+'reports.qwen_vl_data_review_20261027.semantic_metrics','--run',str(a.output),'--data-run',str(a.data_run),'--clip-repo',str(a.clip_repo),'--clip-weights',str(a.clip_weights),'--gpu-uuid',gpu,'--output',str(out/'semantic_metrics')])
        # End with a real RGB+text Qwen pass. Manifest contains no target/mask.
        records=[];seen=set()
        for i,r in enumerate(report['best_val']['images']):
            source=r['sample_id'].split(':')[2].rsplit('-',1)[0];key=(source,r['mode'])
            cell=(r['category'],r['mode'])
            if key in seen or sum(x['category']==cell[0] and x['mode']==cell[1] for x in records)>=2:continue
            seen.add(key);records.append({'image':str((a.output/'images/val_best/reference'/f'sample_{i:04d}.png').resolve()),
                'prompt':r['prompt'],'sample_id':r['sample_id'],'category':r['category'],'mode':r['mode']})
        manifest=a.output/'raw_inputs.json';write_json(manifest,records)
        raw=a.output/'raw_input_check'
        run('raw_inference',[module+'qwen_vl_spatial_infer','--checkpoint',str(a.output/'best_checkpoint.pt'),'--qwen',str(a.qwen),'--sd',str(a.sd),'--manifest',str(manifest),'--output',str(raw)])
        result=json.loads((raw/'report.json').read_text());indices={r['sample_id']:i for i,r in enumerate(report['best_val']['images'])}
        import numpy as np
        from PIL import Image
        differences=[]
        for record in result['records']:
            index=indices[record['sample_id']]
            original=np.asarray(Image.open(a.output/'images/val_best/generated'/f'sample_{index:04d}.png')).astype(float)
            actual=np.asarray(Image.open(record['output'])).astype(float)
            differences.append({'sample_id':record['sample_id'],'mean_abs_8bit':float(abs(actual-original).mean()),'max_abs_8bit':float(abs(actual-original).max())})
        write_json(out/'raw_identity.json',{'target_or_mask_input':result['target_or_mask_input'],'differences':differences,'exact_png_match':all(x['max_abs_8bit']==0 for x in differences)})
        write_json(a.output/'workflow_report.json',{'complete':True,'checkpoint_sha256':report['checkpoint_sha256'],'best_step':report['best_step'],'evaluation_label':'development_test','evaluation_used_for_selection':True,'independent_test_claim':False,'elapsed_seconds':time.time()-started,'commands':commands,'quality_accepted':False,'note':'Training/export completion does not by itself certify visual quality; review all prompt-labelled tasks before optical replacement.'})
        stage='complete'
    except Exception:
        write_json(a.output/'workflow_failure.json',{'stage':stage,'traceback':traceback.format_exc(),'commands':commands});raise
    finally:
        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().splitlines()
        active=[int(x.strip()) for x in raw if x.strip().isdigit() and int(x.strip()) in children]
        write_json(a.output/'gpu_release_receipt.json',{'own_child_pids':children,'own_remaining_gpu_pids':active,'gpu_released':not active,'other_processes_untouched':True})
        write_json(a.output/'workflow_progress.json',{'stage':stage,'own_child_pids':children,'finished_unix':time.time(),'elapsed_seconds':time.time()-started})


if __name__=='__main__':main()
