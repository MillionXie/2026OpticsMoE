"""TRAIN-only electronic-condition targets for the same frozen optical cache.

No additional deployed network. Teacher receives original RGB/text features,
not generated scene targets. DEV/TEST never produce these KD targets.
"""
import argparse,json,os,shutil,sys,traceback
from pathlib import Path
import torch
from .qwen3_scene_conditioning import QwenSDXLConditions
from .white_reference_scene_probe import GPU,digest,write


def make(a):
    caches=[];reports=[]
    for folder in [a.teacher_cache,a.student_cache]:
        report=json.loads((folder/'report.json').read_text());assert report['complete']
        assert digest(folder/'paired_cache.pt')==report['cache_sha256']
        reports.append(report);caches.append(torch.load(folder/'paired_cache.pt',map_location='cpu',weights_only=False,mmap=True))
    teacher,student=caches
    assert teacher['pair_sha256']==student['pair_sha256'] and student.get('optical_frontend')
    assert not teacher.get('optical_frontend')
    assert [r['sample_id'] for r in teacher['records']]==[r['sample_id'] for r in student['records']]
    ready=json.loads((a.teacher/'report.json').read_text());assert ready['complete']
    assert digest(a.teacher/'best_checkpoint.pt')==ready['best_sha256']
    state=torch.load(a.teacher/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    assert state['variant']=='qwen' and state['pair_sha256']==student['pair_sha256'] and not state.get('optical_frontend')
    head=QwenSDXLConditions().cuda().eval().requires_grad_(False);head.load_state_dict(state['head'],strict=True)
    targets={}
    for row in teacher['records']:
        if row['split']!='train':continue
        key=row['input_key']
        if key in targets:continue
        f=teacher['features'][key]
        with torch.inference_mode():out=head(f['hidden'].unsqueeze(0).cuda().float(),f['visual'].unsqueeze(0).cuda().float())
        targets[key]={n:v[0].cpu().bfloat16() for n,v in out.items()}
        write(a.output/'progress.json',{'stage':'TRAIN_only_electronic_conditions','unique':len(targets)})
    dev={r['input_key'] for r in teacher['records'] if r['split']=='dev'}
    assert not set(targets)&dev
    student['TRAIN_electronic_conditions']=targets;student['condition_teacher_sha256']=ready['best_sha256']
    torch.save(student,a.output/'paired_cache.pt')
    for n in ['model_assets.json','DEV_frontend_identity_only.pt']:shutil.copyfile(a.student_cache/n,a.output/n)
    report=dict(reports[1],cache_sha256=digest(a.output/'paired_cache.pt'),source_commit=os.environ['T12_SOURCE_COMMIT'],
        condition_teacher_sha256=ready['best_sha256'],TRAIN_only_condition_KD_count=len(targets),
        parent_optical_cache_sha256=reports[1]['cache_sha256'],DEV_condition_KD_targets_created=False,
        extra_deployed_parameters=0,TEST_opened=False)
    write(a.output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['teacher','teacher-cache','student-cache','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('New run/published source/GPU6 required')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
    write(a.output/'execution.json',{'argv':sys.argv,'source_commit':os.environ['T12_SOURCE_COMMIT'],'source_sha256':digest(Path(__file__)),'pid':os.getpid()})
    try:make(a)
    except Exception:write(a.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
