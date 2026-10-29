import copy
import unittest
import torch
from torch import nn
from ..qwen_vl_hybrid import OpticalMacro,MacroStageBlock,IdentityBlock
from ..qwen_vl_postfree import QuietFusion
from ..scene_macro_expansion import expand,contract,validate_state,placement
from ..scene_frontend_rebind import binding_metadata


def tiny():
    root=nn.Module();root.model=nn.Module();root.model.visual=nn.Module();root.model.language_model=nn.Module()
    root.model.visual.blocks=nn.ModuleList([nn.Linear(4,4).bfloat16() if i<6 else IdentityBlock() for i in range(24)])
    root.model.language_model.layers=nn.ModuleList([nn.Linear(4,4).bfloat16() if i<4 else IdentityBlock() for i in range(28)])
    macros=nn.ModuleDict({m:OpticalMacro(m,4,1,2) for m in ['vision','language']});root.model.hybrid=macros
    for macro in macros.values():
        for i,f in enumerate(macro.fusions):
            q=QuietFusion(.65,.6,.9,1e-6);q.load_state_dict(f.state_dict());macro.fusions[i]=q
    for blocks,starts,macro,vision in [(root.model.visual.blocks,[6,12],macros['vision'],True),
            (root.model.language_model.layers,[4,14],macros['language'],False)]:
        for stage,index in enumerate(starts):blocks[index]=MacroStageBlock(macro,stage,vision)
    root.requires_grad_(False).eval();return root


def forwards(root,v,l):
    values=[]
    for i,layer in enumerate(root.model.visual.blocks):
        v=layer(v)
        if i in [11,17,23]:values.append(v)
    for i,layer in enumerate(root.model.language_model.layers):
        l=layer(l)
        if i in [13,27]:values.append(l)
    return values


class MacroExpansion(unittest.TestCase):
    def test_physical_two_pair_warmstart_preserves_BF16_native_taps_and_router(self):
        torch.manual_seed(1042);root=tiny();v=torch.randn(256,4).bfloat16();l=torch.randn(1,7,4).bfloat16()
        with torch.no_grad():before=forwards(root,v,l)
        old={'macros':copy.deepcopy(root.model.hybrid.state_dict())};native=[id(p) for n,p in root.named_parameters() if '.hybrid.' not in n]
        macros,extra=expand(root);macros.eval();self.assertGreater(extra,0)
        with torch.no_grad():after=forwards(root,v,l)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,after)))
        self.assertEqual(native,[id(p) for n,p in root.named_parameters() if '.hybrid.' not in n])
        self.assertEqual([i for i,b in enumerate(root.model.visual.blocks) if isinstance(b,MacroStageBlock)],[6,12,14,16])
        self.assertEqual([i for i,b in enumerate(root.model.language_model.layers) if isinstance(b,MacroStageBlock)],[4,14,18,22])
        self.assertEqual(sum(isinstance(b,nn.Linear) for b in root.model.visual.blocks),6)
        self.assertEqual(sum(isinstance(b,nn.Linear) for b in root.model.language_model.layers),4)
        for macro in macros.values():
            self.assertEqual(macro.router_calls,1);self.assertIsInstance(macro.paths[1].core.router,nn.Identity)
            self.assertEqual(macro.diagnostics()['optical_cycles_per_forward'],5)
            self.assertTrue(all(.6<=float(f.alpha.detach())<=.9 for f in macro.fusions))
        state={'config':{'pair_expansion':contract()},'macros':macros.state_dict()};validate_state(state,old)
        bad=dict(state,macros=dict(state['macros'],unexpected_router=torch.zeros(1)))
        with self.assertRaises(ValueError):validate_state(bad,old)
        macros.requires_grad_(True);opt=torch.optim.AdamW(macros.parameters(),lr=.001)
        for _ in range(4):
            opt.zero_grad();a=forwards(root,v,l);loss=sum((x.float()*torch.randn_like(x.float())).mean() for x in a)
            loss.backward();opt.step()
        for macro in macros.values():
            gradients=[p.grad for n,p in macro.paths[1].named_parameters() if 'raw_phase' in n and p.grad is not None]
            self.assertGreater(sum(float(x.abs().sum()) for x in gradients),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in macros.parameters() if p.grad is not None))

    def test_binding_requires_approved_cycle_contract_and_frozen_native_TF(self):
        state={'optical_frontend':{'checkpoint_sha256':'old'}}
        report=dict(complete=True,parent_frontend_sha256='old',pair_expansion=contract(),extra_deployment_parameters=20,
            head_UNet_VAE_retained_Qwen_TF_frozen=True,TEST_opened=False,DEV_or_TEST_forward_performed=False)
        bound=binding_metadata(state,report,'new');self.assertEqual(bound['pair_expansion']['optical_cycles'],10)
        for change in [dict(pair_expansion=dict(contract(),optical_cycles=6)),dict(extra_deployment_parameters=0),
                dict(head_UNet_VAE_retained_Qwen_TF_frozen=False),dict(DEV_or_TEST_forward_performed=True)]:
            with self.assertRaises(ValueError):binding_metadata(state,dict(report,**change),'new')


if __name__=='__main__':torch.set_num_threads(2);unittest.main()
