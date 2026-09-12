import copy
import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import Residual
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import expand_electronic_context, overlay_config


def test_context13_preserves_initial_function_and_enables_outer_kernel_gradients():
    torch.manual_seed(42); torch.set_num_threads(2)
    state={}
    for mode in ('vision','language'):
        for i in (0,1):
            state.update({f'{mode}.blocks.{i}.'+k:v for k,v in Residual(mode=='vision').state_dict().items()})
    state['vision.optics.router.raw_router_phase']=torch.randn(224,224)
    payload=dict(metadata={'fusion_alpha_min':.4001},state_dict=state)
    snapshot=copy.deepcopy(payload)
    expanded=expand_electronic_context(payload,{'vision':13,'language':5})
    added=sum(t.numel() for t in expanded['state_dict'].values())-sum(t.numel() for t in state.values())
    assert added==61440
    prefix='vision.blocks.0.'
    old,new=Residual(True).eval(),Residual(True,13).eval()
    for module,p in ((old,payload),(new,expanded)):
        module.load_state_dict({k[len(prefix):]:v for k,v in p['state_dict'].items() if k.startswith(prefix)})
    x=torch.randn(2,196,192)
    torch.testing.assert_close(old(x),new(x),rtol=2e-6,atol=2e-6)
    assert len(list(old.modules()))==len(list(new.modules()))
    new(x).square().mean().backward()
    gradient=new.token_depthwise.weight.grad
    assert torch.isfinite(gradient).all()
    outside=gradient.clone();outside[:,:,5:8,5:8]=0
    assert outside.abs().sum()>0
    changed={k for k,v in expanded['state_dict'].items() if v is not state[k]}
    assert changed=={f'vision.blocks.{i}.token_depthwise.weight' for i in (0,1)}
    for k,v in state.items():assert torch.equal(v,snapshot['state_dict'][k])
    again=expand_electronic_context(expanded,{'vision':13,'language':5})
    for k,v in again['state_dict'].items():assert v is expanded['state_dict'][k]


@pytest.mark.parametrize('kernel',[True,13.,'13',9,15])
def test_context13_rejects_invalid_model_contract(kernel):
    with pytest.raises(ValueError):Residual(True,kernel)


def test_language13_not_permitted():
    with pytest.raises(ValueError):Residual(False,13)
    p=dict(metadata={},state_dict={})
    with pytest.raises(ValueError):expand_electronic_context(p,{'vision':15,'language':5})


def test_vision13_profile_changes_only_existing_visual_kernel_contract():
    base=overlay_config({},'domain_distill_joint_restart')
    candidate=overlay_config({},'domain_distill_joint_vision13')
    assert candidate.pop('electronic_context_kernels')=={'vision':13,'language':5}
    for cfg in (base,candidate):cfg.pop('protocol')
    assert base==candidate
