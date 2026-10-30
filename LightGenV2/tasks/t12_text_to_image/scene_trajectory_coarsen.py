"""Coarsen saved, accepted electronic-teacher endpoints without re-generation.

The teacher remains the original full electronic model. Saved 30->12 endpoints
can be regrouped into fewer intervals, with explicitly nonuniform original-step
boundaries. This does NOT make the unaccepted LG student a new teacher.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import traceback
import torch
from .scene_tail_distill import collapsed_epsilon,interval_edges,complete_state,load_cache
from .white_reference_scene_probe import digest,write


def coarsen(saved,steps):
    previous=saved['student_steps'];edges=interval_edges(previous,steps);sigmas=saved['sigmas']
    if len(sigmas)!=previous+1 or not all(x>y for x,y in zip(sigmas,sigmas[1:])) or sigmas[-1]!=0:
        raise ValueError('Complete descending endpoint grid required')
    groups=defaultdict(dict)
    for row in saved['records']:
        if row['split']!='train' or row['interval'] in groups[row['input_key']]:raise ValueError('TRAIN-only unique intervals required')
        groups[row['input_key']][row['interval']]=row
    output=[];maximum=0.
    for key,items in groups.items():
        if set(items)!=set(range(previous)):raise ValueError('Missing teacher endpoints')
        for i in range(previous-1):
            item=items[i];end=item['x'].float()+item['target_epsilon'].float()*(sigmas[i+1]-sigmas[i])
            observed=items[i+1]['x'].float();error=float((end-observed).abs().max());maximum=max(maximum,error)
            if not torch.allclose(end,observed,atol=2e-5,rtol=2e-5):raise ValueError('Adjacent saved teacher states do not match')
        last=items[previous-1];final=last['x'].float()+last['target_epsilon'].float()*(sigmas[-1]-sigmas[-2])
        for i,(start,stop) in enumerate(zip(edges,edges[1:])):
            x=items[start]['x'];end=items[stop]['x'] if stop<previous else final
            effective=collapsed_epsilon(x,end,sigmas[start],sigmas[stop]);assert torch.isfinite(effective).all()
            output.append(dict(split='train',input_key=key,interval=i,x=x.clone(),target_epsilon=effective,
                sigma=sigmas[start],timestep=saved['timesteps'][start]))
    return dict(saved,records=output,student_steps=steps,timesteps=[saved['timesteps'][i] for i in edges[:-1]],
        sigmas=[sigmas[i] for i in edges]),edges,maximum


def run(a):
    state,teacher=complete_state(a.teacher);cache,cache_report=load_cache(a.teacher_cache)
    assert not state.get('optical_frontend') and not state.get('sampling_contract')
    assert state['pair_sha256']==cache['pair_sha256'] and not cache.get('optical_frontend')
    report=json.loads((a.source/'report.json').read_text());assert report['complete'] and report['teacher_is_full_electronic_Qwen']
    assert report['teacher_steps']==30 and report['teacher_CFG']==7.5 and not report['DEV_or_TEST_targets_created']
    assert report['teacher_sha256']==teacher['best_sha256'] and digest(a.source/'TRAIN_trajectories.pt')==report['trajectory_sha256']
    saved=torch.load(a.source/'TRAIN_trajectories.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['teacher_sha256']==teacher['best_sha256'] and saved['teacher_cache_sha256']==cache_report['cache_sha256']
    assert saved['pair_sha256']==cache['pair_sha256']
    train={r['input_key'] for r in cache['records'] if r['split']=='train'}
    assert {r['input_key'] for r in saved['records']}==train
    result,edges,error=coarsen(saved,a.steps)
    # Verify the proposed custom grid against the unchanged official scheduler.
    from diffusers import EulerDiscreteScheduler
    scheduler=EulerDiscreteScheduler.from_pretrained(a.model_path,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(timesteps=result['timesteps'])
    torch.testing.assert_close(scheduler.sigmas,torch.tensor(result['sigmas']),atol=1e-5,rtol=1e-5)
    torch.save(result,a.output/'TRAIN_trajectories.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],
        trajectory_sha256=digest(a.output/'TRAIN_trajectories.pt'),parent_trajectory_sha256=report['trajectory_sha256'],
        TRAIN_contexts=len(train),TRAIN_intervals=len(result['records']),teacher_sha256=teacher['best_sha256'],
        teacher_steps=30,teacher_CFG=7.5,student_steps=a.steps,student_CFG=1.,
        student_CFG_metadata='Target endpoints independent of student CFG; effective training CFG must be recorded separately',
        teacher_is_full_electronic_Qwen=True,parent_saved_intervals=saved['student_steps'],regrouped_parent_edges=edges,
        original_teacher_step_edges=[interval_edges(30,saved['student_steps'])[i] for i in edges],
        adjacent_endpoint_reconstruction_max_abs_error=error,teacher_GPU_forward_repeated=False,
        CPU_only=True,synthetic_scene_GT_used_for_trajectory_targets=False,DEV_or_TEST_targets_created=False,
        quality_or_speed_accepted=False,TEST_opened=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','teacher','teacher-cache','model-path','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--steps',type=int,default=8);a=p.parse_args()
    assert not a.output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(2)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__))))
    try:run(a)
    except Exception:write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
