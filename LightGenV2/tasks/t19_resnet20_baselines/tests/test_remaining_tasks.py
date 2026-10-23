from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines.models import build_model
from LightGenV2.tasks.t19_resnet20_baselines.other_data import ProductBatches,RGBEditorPairs,fixed_tokens
from LightGenV2.tasks.t19_resnet20_baselines.task_problems import Problem


def test_product_sampler_visits_all_train_views_once_with_no_false_negatives():
    rows=[{'product_id':str(sku)} for sku in range(20) for _ in range(8)]
    sampler=ProductBatches(rows)
    batches=list(sampler)
    assert len(batches)==5 and sorted(i for b in batches for i in b)==list(range(160))
    for batch in batches:
        assert len(batch)==32
        assert min(Counter(rows[i]['product_id'] for i in batch).values())>=2
    sampler.set_epoch(1)
    assert batches!=list(sampler)


def test_instruction_batching_and_editor_target_isolation():
    assert fixed_tokens('short').shape==fixed_tokens('a longer instruction').shape==(514,)
    class Source:
        def __len__(self):return 1
        def __getitem__(self,i):
            return {'reference':torch.zeros(3,16,16),'target':torch.ones(3,16,16),
                    'prompt':'move lamp','sample_id':'a','mode':'joint','qwen_text':torch.ones(2048),
                    'object_union':torch.ones(1,16,16)}
    row=RGBEditorPairs(Source())[0]
    assert set(row)=={'image','target','tokens','sample_id','mode'}
    assert row['target'].min()==1


@pytest.mark.parametrize('task',['abo_image_to_image','lsp','salicon','openmoji','abo_conditional_edit'])
def test_all_remaining_objectives_have_finite_real_gradients(task):
    torch.set_num_threads(1)
    p=Problem.__new__(Problem);p.task=task
    p.settings=SimpleNamespace(heatmap_loss_weight=1.,coordinate_loss_weight=.1,
        kl_weight=1.,cc_weight=.5,sim_weight=.25,nss_weight=.1,map_kd_weight=0.,
        changed_cell_weight=12.,foreground_cell_weight=3.,category_loss_weight=1.,
        edit_loss_weight=1.25,preservation_loss_weight=.15,task_loss_weight=.1,
        ccd_loss_weight=0.,router_balance_weight=0.)
    count=4
    batch={'image':torch.randn(count,3,32,32),'sample_id':['a','b','c','d'],
           'sku':['x','x','y','y'],'tokens':torch.stack([fixed_tokens('edit')]*count),
           'heatmaps':torch.rand(count,14,56,56),'visible':torch.ones(count,14,dtype=torch.bool),
           'keypoints':torch.full((count,14,2),112.),'density':torch.full((count,1,224,224),1/224**2),
           'fixation':torch.ones(count,1,224,224),'target_grid':torch.ones(count,6,6,dtype=torch.long),
           'edit_grid':torch.ones(count,6,6),'preserve_grid':torch.zeros(count,6,6),
           'task_index':torch.zeros(count,dtype=torch.long),'target':torch.rand(count,3,32,32)}
    model=build_model(task)
    loss=p._other_loss(model,batch,'cpu')
    assert torch.isfinite(loss)
    loss.backward()
    gradients=[x.grad for x in model.parameters() if x.grad is not None]
    assert gradients and all(torch.isfinite(x).all() for x in gradients)
    assert sum(x.abs().sum().item() for x in gradients)>0
