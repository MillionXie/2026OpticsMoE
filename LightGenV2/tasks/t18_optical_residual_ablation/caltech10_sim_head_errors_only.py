"""Same8554 head; correct optical TRAIN examples get preservation KL only."""
import argparse,inspect,os,sys,textwrap
from pathlib import Path
import caltech10_sim_head_preserve as parent
s=parent.s;HERE=parent.HERE;ROOT=parent.ROOT;resources=parent.resources;prior=parent.prior
FILES=('caltech10_sim_head_errors_only.py','audit_caltech10_sim_head_errors_only.py','caltech10_sim_head_errors_only_profile.json','CALTECH10_SIM_HEAD_ERRORS_ONLY_PROTOCOL_20261016.md')
BASE_SOURCES=parent.sources();P=s.read(HERE/'caltech10_sim_head_errors_only_profile.json');RUN=P['run'];CPU=HERE/'runs/smoke'/P['cpu_run']
def sources():
    v=dict(BASE_SOURCES)
    for n in FILES:v[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return v
parent.P=P;parent.RUN=RUN;parent.CPU=CPU;parent.sources=sources;parent.__file__=__file__
# Loss and graph are unchanged functions; the explicit fixed profile changes
# correct_CE_weight .05->0, and continues trained E12 with declared new Adam state.
code=textwrap.dedent(inspect.getsource(parent.cpu));assert code.count("ck['epoch']==3")==1;code=code.replace("ck['epoch']==3","ck['epoch']==12");exec(compile(code,__file__+':cpu','exec'),parent.__dict__)
code=textwrap.dedent(inspect.getsource(parent.run_case));changes={"head_epoch=3,phase_epoch=0":"head_epoch=12,phase_epoch=0","best_epoch=3":"best_epoch=12","epoch=phase+3":"epoch=phase+12","actual_head_epochs=phase+3":"actual_head_epochs=phase+12","str(HERE/'audit_caltech10_sim_head_preserve.py')":"str(HERE/'audit_caltech10_sim_head_errors_only.py')"}
for old,new in changes.items():assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
anchor="control=P['controls'][a.case];baseline=";assert code.count(anchor)==1;code=code.replace(anchor,"known.update({q.name:q for q in Path(P['known_tests_root'][a.case]).iterdir() if (q/'result.json').exists()});"+anchor);exec(compile(code,__file__+':run_case','exec'),parent.__dict__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['cpu','run_case','campaign'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--case');p.add_argument('--gpu');a=p.parse_args()
    try:getattr(parent,a.phase)(a)
    except BaseException as error:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_suffix('.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(error),automatic_retry=False),exclusive=True)
        raise
