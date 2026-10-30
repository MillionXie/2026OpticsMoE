"""Optional TRAIN-only grid comparison using the same complete-rollout budget.

The warm compact checkpoint is never used directly with an untrained new grid.
Its sampling grid is changed only inside a new TRAIN run supervised by the same
stored electronic30 trajectories. This adds no deployment parameter or feedback
to Qwen. Existing fixed-grid training entry points remain untouched.
"""
import argparse,copy,json,sys
from pathlib import Path


def training_grid(geometry,teacher,steps):
    if (not geometry['complete'] or not geometry['CPU_only'] or geometry['DEV_or_TEST_forwarded']
        or geometry['TRAIN_contexts']!=192 or geometry['teacher_sha256']!=teacher['teacher_sha256']
        or geometry['teacher_path_sha256']!=teacher['full_TRAIN_teacher_paths_sha256']):
        raise ValueError('Geometry must be bound to the same192 TRAIN electronic paths')
    edges=geometry['suggestions'][str(steps)]['edges'];times=geometry['teacher_timesteps']
    if (steps not in(6,8) or len(edges)!=steps+1 or edges[0]!=0 or edges[-1]!=30 or len(times)!=30
        or not all(isinstance(x,int)and not isinstance(x,bool)for x in edges)
        or not all(0<b-a<=8 for a,b in zip(edges,edges[1:]))
        or not all(a>b for a,b in zip(times,times[1:]))):
        raise ValueError('Ordered six/eight intervals on the original30 grid required')
    return [times[index]for index in edges[:-1]],edges


def with_training_grid(state,grid,geometry_sha):
    sampling=state['sampling_contract'];assert sampling['steps']==len(grid)and sampling['CFG']==7.5
    result=dict(state);result['sampling_contract']=copy.deepcopy(sampling)
    result['sampling_contract']['timesteps']=list(grid)
    result['TRAIN_grid_recovery']=dict(geometry_report_sha256=geometry_sha,
        original_sampling_contract=copy.deepcopy(sampling),new_grid_is_for_training_not_zero_shot_use=True)
    return result


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--geometry-analysis',type=Path,required=True)
    option,remaining=p.parse_known_args()
    from . import scene_cached_rollout_recovery as base
    from .white_reference_scene_probe import digest,write
    old_state,old_run=base.complete_state,base.run
    geometry_path=option.geometry_analysis/'report.json';geometry=json.loads(geometry_path.read_text())
    def run(a):
        report=json.loads((a.teacher_paths/'report.json').read_text())
        # The path analysis records its original timetable in the suggestions;
        # load the authoritative saved path metadata, not a grid copied from DEV.
        import torch
        saved=torch.load(a.teacher_paths/'TRAIN_full_teacher_paths.pt',map_location='cpu',weights_only=False,mmap=True)
        checked=dict(geometry,teacher_timesteps=saved['timesteps'])
        original,parent=old_state(a.student);grid,edges=training_grid(checked,report,original['sampling_contract']['steps'])
        assert digest(a.teacher_paths/'TRAIN_full_teacher_paths.pt')==geometry['teacher_path_sha256']
        assert grid!=original['sampling_contract']['timesteps'],'Use the unchanged trainer for an identical grid'
        changed=with_training_grid(original,grid,digest(geometry_path))
        def state(folder):
            if Path(folder)!=a.student:raise ValueError('Unexpected additional checkpoint load')
            return changed,parent
        base.complete_state=state
        write(a.output/'TRAIN_geometry_grid_contract.json',dict(complete=True,
            wrapper_source_sha256=digest(Path(__file__)),base_source_sha256=digest(Path(base.__file__)),
            geometry_report_sha256=digest(geometry_path),warm_parent_sha256=parent['best_sha256'],
            original_sampling_contract=original['sampling_contract'],training_sampling_contract=changed['sampling_contract'],
            original_teacher_edges=edges,changes_only_student_sampling_grid=True,not_zero_shot_inference=True,
            extra_deployment_parameters=0,DEV_or_TEST_used=False))
        try:old_run(a)
        finally:base.complete_state=old_state
        r=json.loads((a.output/'report.json').read_text());r.update(TRAIN_geometry_grid_recovery=True,
            geometry_report_sha256=digest(geometry_path),wrapper_source_sha256=digest(Path(__file__)))
        write(a.output/'report.json',r)
    base.run=run;sys.argv=[sys.argv[0],*remaining]
    try:base.main()
    finally:base.run=old_run;base.complete_state=old_state


if __name__=='__main__':main()
