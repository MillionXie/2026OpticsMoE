import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from LightGenV2.tasks.t19_resnet20_baselines.retrieval_subepoch import train_prefix,validate_plan
from LightGenV2.tasks.t19_resnet20_baselines.retrieval import cross_modal_loss


def test_fixed_budgets_not_a_desired_score_search():
    plan=json.loads((Path(__file__).parents[1]/'configs/retrieval_subepoch_20261024.json').read_text())
    validate_plan(plan)
    other=copy.deepcopy(plan);other['optimizer_updates']=[60,110]
    with pytest.raises(ValueError):validate_plan(other)


def test_prefix_updates_exactly_and_consumes_remaining_sampler_without_training():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.projection=nn.Linear(4,4);self.text=nn.Embedding(2,4)
        def encode_text(self,tokens):return F.normalize(self.text(tokens),dim=-1)
    torch.manual_seed(42);model=Tiny();reference=copy.deepcopy(model)
    cache=dict(features=torch.randn(8,4),skus=['a','b']*4);tokens=torch.arange(2);skus=['a','b']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    opt_ref=torch.optim.AdamW(reference.parameters(),lr=.001,weight_decay=.0001)
    with patch('LightGenV2.tasks.t19_resnet20_baselines.retrieval_subepoch.base.shutdown_loader') as shutdown:
        result=train_prefix(model,optimizer,cache,tokens,skus,2,DataLoader(range(8),batch_size=2))
        shutdown.assert_called_once()
    for start in (0,2):
        opt_ref.zero_grad();z=F.normalize(reference.projection(cache['features'][start:start+2]),dim=-1)
        loss=cross_modal_loss(z,reference.encode_text(tokens),cache['skus'][start:start+2],skus,direction='image_to_text')
        loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);opt_ref.step()
    for actual,expected in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert result['optimizer_updates']==2 and result['optimized_sample_indices']==[0,1,2,3]
    assert result['complete_sampler_indices']==list(range(8))
    assert {int(s['step']) for s in optimizer.state.values()}=={2}
