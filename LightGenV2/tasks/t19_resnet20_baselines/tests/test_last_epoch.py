import json
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines import evaluate_last as module


def test_rejects_incomplete_or_wrong_epoch_checkpoint():
    manifest=dict(status='complete',epochs=30,task='lsp')
    with pytest.raises(ValueError,match='final authorized epoch'):
        module.check_last_epoch(dict(epoch=25,manifest=manifest),manifest)
    with pytest.raises(ValueError,match='completed'):
        module.check_last_epoch(dict(epoch=30,manifest=manifest),dict(manifest,status='running'))


def test_reloads_last_even_when_best_has_a_larger_score(tmp_path,monkeypatch):
    manifest=dict(status='complete',epochs=30,task='lsp',data_identity={'fixed':True},
                  git_commit='original',selection='historical test best')
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'result.json').write_text('{"historical": true}')
    model=torch.nn.Linear(1,1,bias=False)
    torch.save(dict(epoch=30,manifest=manifest,model={'weight':torch.tensor([[2.]])}),
               tmp_path/'last_checkpoint.pt')
    torch.save(dict(epoch=25,manifest=manifest,model={'weight':torch.tensor([[9.]])}),
               tmp_path/'best_checkpoint.pt')
    monkeypatch.setattr(module,'build_model',lambda task:model)
    monkeypatch.setattr(module,'git',lambda *args:'evaluation')
    monkeypatch.setattr(module,'Problem',lambda task:SimpleNamespace(identity={'fixed':True},
        final_test=None,test=[1],batch_size=1,
        evaluate=lambda m,loader,device:({'score':m.weight.item()},[{'value':m.weight.item()}])))
    result=module.evaluate_one(tmp_path)
    assert result['metrics']['score']==2.
    assert result['selected_epoch']==30
    assert result['selection_uses_test'] is False
    assert json.loads((tmp_path/'result.json').read_text())=={'historical':True}
