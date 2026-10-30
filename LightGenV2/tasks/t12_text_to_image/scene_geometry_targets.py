"""TRAIN-only electronic trajectory targets on a geometry-selected sigma grid."""
import argparse,json,os,sys
from pathlib import Path
import torch
from .scene_path_geometry import optimize_edges
from .scene_tail_distill import load_cache,collapsed_epsilon
from .white_reference_scene_probe import digest,write


def regrid(saved,edges):
    if len(edges)not in[7,9]or edges[0]!=0 or edges[-1]!=30 or not all(a<b for a,b in zip(edges,edges[1:])):
        raise ValueError('Six/eight ordered intervals of the original30-step path required')
    if max(b-a for a,b in zip(edges,edges[1:]))>8:raise ValueError('Unreviewed long interval')
    sigmas=saved['sigmas'];records=[]
    if len(sigmas)!=31 or sigmas[-1]!=0 or not all(a>b for a,b in zip(sigmas,sigmas[1:])):raise ValueError('Invalid sigma grid')
    for key,x in saved['paths'].items():
        if x.shape!=(31,4,64,64)or not torch.isfinite(x).all():raise ValueError('Invalid teacher path')
        for i,(start,stop)in enumerate(zip(edges,edges[1:])):
            wanted=collapsed_epsilon(x[start],x[stop],sigmas[start],sigmas[stop])
            records.append(dict(split='train',input_key=key,interval=i,x=x[start].clone(),target_epsilon=wanted,sigma=sigmas[start],timestep=saved['timesteps'][start]))
    return records


def run(a):
    report=json.loads((a.source/'report.json').read_text());path=a.source/'TRAIN_full_teacher_paths.pt'
    assert report['complete']and report['teacher_is_full_electronic_Qwen']and not report['DEV_or_TEST_targets_created']
    assert digest(path)==report['full_TRAIN_teacher_paths_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True);cache,cr=load_cache(a.student_cache)
    train={r['input_key']for r in cache['records']if r['split']=='train'};dev={r['input_key']for r in cache['records']if r['split']=='dev'}
    assert saved['all_TRAIN']and len(saved['paths'])==192 and set(saved['paths'])<=train and not set(saved['paths'])&dev
    assert saved['teacher_sha256']=='662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76'
    geometry=json.loads((a.geometry/'report.json').read_text());costfile=a.geometry/'pairwise_TRAIN_geometry_costs.json';costs=json.loads(costfile.read_text())
    assert geometry['complete']and geometry['CPU_only']and not geometry['DEV_or_TEST_forwarded']and geometry['TRAIN_contexts']==192
    assert geometry['teacher_path_sha256']==digest(path)and geometry['teacher_sha256']==saved['teacher_sha256']
    selected=optimize_edges(costs,a.steps);assert selected['edges']==geometry['suggestions'][str(a.steps)]['edges']
    edges=selected['edges'];records=regrid(saved,edges);grid=[saved['timesteps'][i]for i in edges[:-1]];sigmas=[saved['sigmas'][i]for i in edges]
    from diffusers import EulerDiscreteScheduler
    scheduler=EulerDiscreteScheduler.from_pretrained(a.model_path,subfolder='scheduler',local_files_only=True);scheduler.set_timesteps(timesteps=grid)
    torch.testing.assert_close(scheduler.sigmas,torch.tensor(sigmas),atol=1e-5,rtol=1e-5)
    out=dict(records=records,pair_sha256=saved['pair_sha256'],teacher_sha256=saved['teacher_sha256'],teacher_cache_sha256=saved['teacher_cache_sha256'],
        teacher_steps=30,student_steps=a.steps,timesteps=grid,sigmas=sigmas,noise_index=saved['noise_index'])
    torch.save(out,a.output/'TRAIN_trajectories.pt')
    write(a.output/'report.json',dict(complete=True,source_commit=os.environ['T12_SOURCE_COMMIT'],trajectory_sha256=digest(a.output/'TRAIN_trajectories.pt'),
        TRAIN_contexts=192,TRAIN_intervals=len(records),student_steps=a.steps,student_CFG=7.5,teacher_steps=30,teacher_CFG=7.5,
        teacher_sha256=saved['teacher_sha256'],teacher_is_full_electronic_Qwen=True,full_teacher_path_sha256=digest(path),
        geometry_report_sha256=digest(a.geometry/'report.json'),geometry_costs_sha256=digest(costfile),original_teacher_step_edges=edges,
        CPU_only=True,teacher_GPU_forward_repeated=False,DEV_or_TEST_targets_created=False,TEST_opened=False,quality_or_speed_accepted=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','geometry','student-cache','model-path','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,choices=[6,8],default=6);a=p.parse_args()
    assert not a.output.exists()and os.environ.get('CUDA_VISIBLE_DEVICES')==''and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    a.output.mkdir(parents=True);torch.set_num_threads(2)
    write(a.output/'execution.json',dict(argv=sys.argv,pid=os.getpid(),source_commit=os.environ['T12_SOURCE_COMMIT'],source_sha256=digest(Path(__file__)),GPU_allocated=False))
    try:run(a)
    except BaseException:
        import traceback
        write(a.output/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
