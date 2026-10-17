"""Independent resource truth table; no data, models or optimizer."""
import os
import caltech20_original_deep_resource as m
def main():
    out=m.CPU;assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not out.exists();m.claim(out.with_name(out.name+'.claim.json'),'exact_authorized_PID_GPU_resource_table');g=list(m.m.GPU.values());forbidden=m.m.frozen.b.FORBIDDEN
    def live(rows):return dict(total=len(rows),jobs=[dict(pid=p,gpu_uuid=u) for p,u in rows],counts={u:sum(v==u for _,v in rows) for u in g+[forbidden]})
    known={(11,g[0]),(12,g[1]),(13,g[2]),(14,g[3])}
    for rows in [[],[(11,g[0])],sorted(known)]:m.guard(live(rows),known)
    rejected=0
    for rows,allowed in [([(99,g[0])],known),([(11,g[0]),(12,g[0])],known|{(12,g[0])}),([(11,forbidden)],known|{(11,forbidden)}),([(11,g[1])],known),([(11,g[0]),(12,g[1]),(13,g[2]),(14,g[3]),(15,g[0])],known|{(15,g[0])})]:
        try:m.guard(live(rows),allowed)
        except AssertionError:rejected+=1
    assert rejected==5;out.mkdir();m.m.s.save(out/'checks.json',dict(m.identity(),passed=True,accepted_truth_cases=3,rejected_truth_cases=rejected,model_objects=0,forward_calls=0,optimizer_steps=0,TEST_images_read=0),exclusive=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        out=m.CPU;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():m.m.s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
