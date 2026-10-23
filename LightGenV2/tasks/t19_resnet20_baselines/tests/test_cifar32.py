import torch
import pytest

from LightGenV2.tasks.t19_resnet20_baselines.models import (
    BasicBlock, TASK_IDS, build_model, parameter_counts, tokenize_text,
)
from LightGenV2.tasks.t19_resnet20_baselines.queue_remaining import worker_command, read_plan


@pytest.mark.parametrize('task',TASK_IDS)
def test_resnet32_preserves_task_heads_and_output_contracts(task):
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        torch.manual_seed(42)
        small=build_model(task)
        large=build_model(task,backbone='resnet32').eval()
        counts=parameter_counts(large)
        assert counts['visual_encoder']==463504
        assert counts['total']-parameter_counts(small)['total']==194432
        assert counts['task_heads']==parameter_counts(small)['task_heads']
        assert [len(getattr(large.encoder,f'stage{i}')) for i in [1,2,3]]==[5,5,5]
        assert sum(isinstance(m,BasicBlock) for m in large.encoder.modules())==15
        image=torch.randn(1,3,32,32)
        if task.startswith('lgvq_'):
            output=large(image[:,None].repeat(1,4,1,1,1))['quality_logits']
            assert output.shape==(1,5)
        elif task in {'openmoji','abo_conditional_edit'}:
            output=large(image,tokenize_text(['Change color.']))
            output=output['category_logits'] if isinstance(output,dict) else output
        else:output=large(image)
        output.square().mean().backward()
        assert large.encoder.stem[0].weight.grad.isfinite().all()
    finally:torch.set_num_threads(old_threads)


def test_all_nine_jobs_dispatch_to_correct_trainer_and_backbone(tmp_path):
    from LightGenV2.tasks.t19_resnet20_baselines.train_easy100 import TASK
    plan=read_plan(TASK/'configs/resnet32_single30_20261023.json')
    assert set(plan['tasks'])==set(TASK_IDS) and plan['selection']=='fixed_last_epoch'
    for task in TASK_IDS:
        for phase in ['smoke','formal']:
            job=dict(task=task,phase=phase,name='example')
            command=worker_command(job,tmp_path,'GPU-test','resnet32')
            assert command[command.index('--backbone')+1]=='resnet32'
            assert command[command.index('--epochs')+1]==('1' if phase=='smoke' else '30')
            module=command[2]
            assert module.endswith('train_easy100' if task in {'abo_image_to_text','abo_text_to_image'} else 'train_tasks')
