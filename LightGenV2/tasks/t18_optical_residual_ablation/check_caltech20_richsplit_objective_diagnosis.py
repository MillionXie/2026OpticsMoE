"""Independent exact-self/authorized-parent guard proof; zero real forwards."""
import argparse,os
from pathlib import Path
import caltech20_richsplit_objective_diagnosis as m

def check(out):
    assert out==m.CPU and not out.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='';m.identity();m.parents();m.claim(out.with_name(out.name+'.claim.json'),'objective_diagnosis_exact_resource_CPU_proof');parent={200:m.fresh.b.GPU[4],300:m.fresh.b.GPU[5]};valid=[dict(pid=100,gpu_uuid=m.GPU)]+[dict(pid=p,gpu_uuid=g) for p,g in parent.items()];assert m.guard(valid,100,parent);invalid=[[],valid[1:],valid+[dict(pid=400,gpu_uuid=m.GPU)],valid+[dict(pid=400,gpu_uuid=m.fresh.b.GPU[0])],[dict(pid=101,gpu_uuid=m.GPU)]+valid[1:],[dict(pid=100,gpu_uuid=m.fresh.b.FORBIDDEN)]+valid[1:],[dict(pid=100,gpu_uuid=m.GPU),dict(pid=201,gpu_uuid=m.fresh.b.GPU[4]),valid[2]]];rejected=0
    for jobs in invalid:
        try:m.guard(jobs,100,parent)
        except AssertionError:rejected+=1
        else:raise AssertionError('Bad resource case accepted')
    assert rejected==7;out.mkdir();m.s.save(out/'checks.json',dict(m.identity(),passed=True,accepted_exact_self_and_two_authorized_parents=True,rejected_resource_counterexamples=rejected,real_model_forwards=0,optimizer_updates=0,TEST_images_read=0),exclusive=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.out)
