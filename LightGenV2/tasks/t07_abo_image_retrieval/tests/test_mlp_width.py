import copy
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import Residual
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import expand_electronic_mlp


def source_payload():
    torch.manual_seed(42);state={}
    for mode in ('vision','language'):
        for index in (0,1):
            state.update({f'{mode}.blocks.{index}.'+k:v for k,v in Residual(mode=='vision').state_dict().items()})
    state['vision.optics.router.raw_router_phase']=torch.randn(224,224)
    return dict(metadata={'fusion_alpha_min':.4001},state_dict=state)


@pytest.mark.parametrize('vision',[True,False])
def test_wider_mlp_preserves_eval_function_and_only_expands_electronics(vision):
    payload=source_payload();snapshot=copy.deepcopy(payload);wide=expand_electronic_mlp(payload,768)
    mode='vision' if vision else 'language';prefix=mode+'.blocks.0.'
    old,new=Residual(vision),Residual(vision,mlp_width=768)
    old.load_state_dict({k[len(prefix):]:v for k,v in payload['state_dict'].items() if k.startswith(prefix)})
    new.load_state_dict({k[len(prefix):]:v for k,v in wide['state_dict'].items() if k.startswith(prefix)})
    old.eval();new.eval();x=torch.randn(2,196 if vision else 77,192)
    with torch.no_grad():a,b=old(x),new(x)
    torch.testing.assert_close(a,b,rtol=2e-6,atol=2e-6)
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==147840
    assert len(list(old.modules()))==len(list(new.modules()))
    assert expand_electronic_mlp(wide,768) is wide
    for n,v in payload['state_dict'].items():
        assert torch.equal(v,snapshot['state_dict'][n])
        if not any(n.endswith('mlp.'+suffix) for suffix in ('0.weight','0.bias','3.weight')):
            assert wide['state_dict'][n] is v
    # Same architecture but dropout masks differ in training, intentionally not
    # claimed as training-equivalent or a pure parameter-count-only ablation.
    new.train();torch.manual_seed(8);new(x).square().mean().backward()
    assert new.mlp[0].weight.grad is not None and new.mlp[3].weight.grad is not None


def test_width_conversion_rejects_invalid_and_shrinking_contracts():
    p=source_payload()
    for w in (True,0,512,768.,'768'):
        with pytest.raises(ValueError):expand_electronic_mlp(p,w)
    with pytest.raises(ValueError):expand_electronic_mlp(expand_electronic_mlp(p,768),384)
    with pytest.raises(ValueError):Residual(True,mlp_width=512)


def test_wider_profile_preserves_other_training_configuration():
    from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config
    base=overlay_config({},'domain_distill_joint_restart');cfg=overlay_config({},'domain_distill_joint_mlp768')
    assert cfg.pop('electronic_mlp_width')==768
    for c in (base,cfg):c.pop('protocol')
    assert base==cfg
