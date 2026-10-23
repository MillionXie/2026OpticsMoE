import json
import pytest
from LightGenV2.tasks.t19_resnet20_baselines.queue_remaining import read_plan, worker_command
from LightGenV2.tasks.t19_resnet20_baselines.train_easy100 import TASK


def test_small_text_plan_dispatches_four_explicit_trainable_text_jobs(tmp_path):
    plan=read_plan(TASK/'configs/resnet18_frozen_smalltext15m_single30_20261023.json')
    assert plan['full_model_parameters_strict_upper_bound']==15000000
    assert plan['gpu_indices_requested_first']==[1,2,6]
    assert plan['max_physical_gpus']==6
    assert plan['max_jobs_per_gpu']==3
    assert plan['preserve_running_jobs']==['lgvq_temporal','lgvq_spatial']
    for task in plan['tasks']:
        encoder=plan['text_encoders'].get(task,'minilm')
        tag=plan['job_tags'].get(task,'')
        name=task+'_formal_s42_suite'+('_'+tag if tag else '')
        command=worker_command(dict(task=task,phase='formal',name=name,text_encoder=encoder),
                               tmp_path,'GPU-6',plan['backbone'])
        if task in plan['text_encoders']:
            assert command[-2:]==['--text-encoder','byte_gru']
            assert name.endswith('byte_gru15m')
        else:
            assert '--text-encoder' not in command
        assert command[command.index('--epochs')+1]=='30'


@pytest.mark.parametrize('overrides', [{'job_tags':{'lsp':'../escape'}},
    {'text_encoders':{'lsp':'byte_gru'}},{'text_encoders':{'openmoji':'unknown'}}])
def test_rejects_invalid_or_misapplied_profile_overrides(tmp_path,overrides):
    path=tmp_path/'plan.json';path.write_text(json.dumps(dict(tasks=['lsp','openmoji'],**overrides)))
    with pytest.raises(ValueError):read_plan(path)


def test_expanded_resource_flags_reach_worker(tmp_path):
    command=worker_command(dict(task='abo_conditional_edit',phase='formal',name='example',
        text_encoder='byte_gru',max_physical_gpus=6,max_jobs_per_gpu=3),tmp_path,'GPU-test','resnet18_imagenet_frozen')
    assert command[command.index('--max-physical-gpus')+1]=='6'
    assert command[command.index('--max-jobs-per-gpu')+1]=='3'
