"""Bounded one-GPU teacher -> optical student -> development comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--test-development',action='store_true')
    p.add_argument('--use-spare-memory',action='store_true')
    p.add_argument('--memory-gib',type=float,default=3.)
    a=p.parse_args();root=Path(__file__).resolve().parent
    assert a.gpu=='GPU-1b963983-7909-af6e-0528-f0f0661ab549'
    a.out.mkdir(parents=True,exist_ok=False)
    save=lambda name,value:(a.out/name).write_text(json.dumps(value,indent=2))
    now=lambda:datetime.now(timezone.utc).isoformat()
    save('metadata.json',dict(command=sys.argv,pid=os.getpid(),gpu_uuid=a.gpu,time=now(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpu_limit=1,use_spare_memory=a.use_spare_memory,memory_limit_gib=a.memory_gib,
        scope='test-selected development, optical inference unchanged'))
    def execute(stage,cmd):
        from gpu_budget import preflight
        policy=preflight(a.gpu,a.use_spare_memory,a.memory_gib)
        save(stage+'.gpu_policy.json',policy)
        cmd+=['--memory-gib',str(a.memory_gib)]
        if a.use_spare_memory:cmd+=['--use-spare-memory']
        save('status.json',dict(state=stage,time=now()))
        with (a.out/(stage+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(stage+'.process.json',dict(pid=proc.pid,command=cmd,gpu_uuid=a.gpu))
            assert proc.wait()==0,stage
    try:
        teacher_cmd=[sys.executable,'-u',str(root/'train_teacher.py'),'--data',str(a.data),'--out',str(a.out/'teacher')]
        if a.test_development:teacher_cmd+=['--profile','bn32','--epochs','80','--test-development']
        execute('teacher',teacher_cmd)
        teacher=json.loads((a.out/'teacher/result.json').read_text())
        if not teacher['eligible_for_distillation']:
            save('status.json',dict(state='teacher_below_test_development_gate' if a.test_development else
                'teacher_below_validation_gate',result=teacher,time=now(),gpu_released=True))
            return
        student_cmd=[sys.executable,'-u',str(root/'regularized_continuation.py'),'--phase','train',
            '--profile','distill','--teacher',str(a.out/'teacher/best_checkpoint.pt'),
            '--data',str(a.data),'--checkpoint',str(root/'runs/simulation/mango_rho03_L6_capture_smooth_20261010/smooth/moe_L6_seed17/best_checkpoint.pt'),
            '--out',str(a.out/'distill')]
        if a.test_development:student_cmd+=['--test-development','--parent-sweep',str(root/'runs/simulation/mango_rho03_L6_test_selected_development_20261010')]
        if a.use_spare_memory:student_cmd+=['--microbatch','2']
        execute('student',student_cmd)
        if a.test_development:
            # Selected and last states were already evaluated during training. No re-inference.
            result=json.loads((a.out/'distill/result.json').read_text())
            save('development_results.json',dict(student=result,teacher=teacher,test_selected=True))
        else:execute('development_evaluation',[sys.executable,'-u',str(root/'regularized_continuation.py'),'--phase','evaluate',
            '--profiles','distill','--data',str(a.data),'--candidates',str(a.out),
            '--parent-sweep',str(root/'runs/simulation/mango_rho03_L6_test_selected_development_20261010'),
            '--out',str(a.out/'development_evaluation')])
        save('status.json',dict(state='complete',time=now(),gpu_released=True))
    except BaseException as error:
        save('status.json',dict(state='failed',error=repr(error),time=now()))
        raise


if __name__=='__main__':main()
