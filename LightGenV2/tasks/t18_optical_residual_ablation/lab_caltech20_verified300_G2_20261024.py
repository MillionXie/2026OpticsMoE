"""One fresh G2 after a fully audited actual300 E2, finite final-head budget."""
import argparse,traceback,shutil
from pathlib import Path
import lab_caltech20_actual_suffix_continue_20261024 as previous
base=previous.base


def validate(p):
    assert p['schema']=='TUCam_verified_actual300_G2_time_budget_v1'
    assert p['actual_E2_target']==300 and p['final_target']==340 and 300<=p['expected_parent_correct']<=411
    assert p['TRAIN']==1600 and p['TEST']==411 and p['VALID']==0 and not p['TEST_gradient']
    assert p['deadline']=='2026-10-24T08:00:00+08:00' and not p['scheduler'] and not p['automatic_retry']
    assert p['capture_family']=='old282' and p['E2_phase_SHA256']=='376cc3ee847f2a6502784301817958fd5ec86fde099fd6c5a860c8444f5a0bf7'
    assert p['minimum_remaining_seconds']==2300 and p['final_fit_seconds']==300 and p['final_fit_epochs']==60
    assert p['budget_basis']==dict(measured_2011_capture_seconds=1164,audit_allowance_seconds=300,fit_seconds=300,native_replay_allowance_seconds=180,safety_seconds=356)
    assert sum(p['budget_basis'].values())==2300


def run(a):
    np,torch,m,loop=base.numerical_modules();p=previous.load_policy(a.profile);validate(p)
    for name,digest in p['source_SHA256'].items():assert m.sha(base.HERE/name)==digest
    bp=base.HERE/p['base_profile'];assert m.sha(bp)==p['base_profile_SHA256'];c=m.read(bp);base.validate_config(c)
    assert previous.admission.seconds_left(p)>=p['minimum_remaining_seconds'] and not a.out.exists() and not m.LOCK.exists()
    a.out.mkdir();torch.set_num_threads(2);torch.manual_seed(73)
    m.save(a.out/'protocol.json',dict(source_git=m.git_head(),source_SHA256=m.sha(__file__),policy=p,profile_SHA256=m.sha(a.profile),new_inference_layers=0,TEST_gradient=False,TEST_development=True,scheduler=False))
    model,data,dm,fields,paths,pt,reference=previous.admit(p,c,a.out)
    verified=m.read(pt.parent/'report.json');assert previous.admission.can_start_G2(verified,300)
    # The previous 45-minute dispatcher did not start G2. This is a fresh,
    # explicitly budgeted handoff, NOT a retry of any optical acquisition.
    stopped=m.read(base.ROOT/p['completed_controller'].replace('.return.json','/report.json'))
    assert stopped['state']=='actual_E2_passed_deadline_boundary' and stopped['G2_not_started'] and stopped['strict_original_CPU']
    assert not (base.ROOT/p['completed_controller'].replace('.return.json','/s_3')).exists()
    assert previous.admission.seconds_left(p)>=p['minimum_remaining_seconds']
    assert sum(v.numel()for n,v in model.named_parameters()if n in m.allowed_keys(model,5))==5150
    import lab_caltech20_tucam_capture_20261023 as capture
    folder=a.out/'s_3';folder.mkdir();hw=base.ROOT/c['hardware_by_stage']['3']['path']
    assert m.sha(hw)==c['hardware_by_stage']['3']['SHA256'] and shutil.disk_usage(base.ROOT).free>=120*1024**3
    fields=capture.acquire(model,data,dm,fields,paths,hw,'3',folder/'capture',pt,interstage_fit=True);paths['3']=folder/'capture'
    recipe=dict(name='verified300_full_physical_original5150_finite_first_fit',epochs=p['final_fit_epochs'],seconds=p['final_fit_seconds'],lr_scale=1,TRAIN_class_emphasis=.75)
    model,pt,result=base.fit_suffix(model,data,dm,fields,paths,pt,reference,c,recipe,folder/'fit',340)
    m.save(a.out/'report.json',dict(state='complete',full_physical=True,actual_E2_verified_correct=verified['selected_correct'],full_physical_correct=result['selected_correct'],target_met=result['selected_correct']>=340,
        strict_original_CPU=True,TEST_development=True,SDK_released=not m.LOCK.exists(),PT_SHA256=m.sha(pt),new_inference_layers=0,prefix_SHA256=base.identities(m,paths)))


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--profile',type=Path,required=True);q.add_argument('--out',type=Path,required=True);a=q.parse_args()
    try:run(a)
    except BaseException as e:
        if a.out.exists():
            _,_,m,_=base.numerical_modules();m.save(a.out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,all_data_preserved=True))
        raise
