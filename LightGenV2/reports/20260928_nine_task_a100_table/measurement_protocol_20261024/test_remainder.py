"""Pure CPU contracts: resume never remeasures qualified points or permits foreign clients."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('remaining_controller',Path(__file__).with_name('run_temporal_remainder.py'))
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def flatten(resume):
    return [(m,c,r) for _,models,caps,repeats in controller.segments(resume) for m in models for c in caps for r in repeats]


def test_approved_resume_exactly_sixteen_missing_points():
    original=set(flatten(False));missing=set(flatten(True))
    completed={('qwen',50,r) for r in (1,2,3)}|{('glm',100,r) for r in (1,2)}
    assert len(original)==21 and len(missing)==16
    assert len(flatten(True))==16 and not completed&missing
    assert missing|completed==original
    assert ('glm',100,3) in missing


def test_loading_only_own_worker_and_private_mps_allowed():
    result=controller.classify_processes([['10','gpu','30'],['11','gpu','4000']],{10},{11},11,'gpu')
    assert result==dict(foreign_gpu_processes=[],foreign_mps_clients=[])


def test_same_user_foreign_context_is_not_allowed_by_owner_name():
    row=['12','gpu','598']
    result=controller.classify_processes([['10','gpu','30'],row],{10},{11},11,'gpu')
    assert result['foreign_gpu_processes']==[row]


def test_foreign_mps_client_and_wrong_uuid_rejected():
    row=['11','different_gpu','4000']
    result=controller.classify_processes([row],{10},{11,12},11,'gpu')
    assert result==dict(foreign_gpu_processes=[row],foreign_mps_clients=[12])
