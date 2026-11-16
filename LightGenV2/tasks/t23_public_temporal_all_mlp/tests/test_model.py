import ast
from pathlib import Path
import tempfile
import unittest

import torch

from LightGen2_MLP import model as public
from LightGen2 import model as parent
from ..checkpoints import inherit_parent,checkpoint,export_checkpoint


class MLPGlobalTests(unittest.TestCase):
    def test_global_mixes_complete_token_vectors_and_preserves_leading_axes(self):
        module=public.GlobalMLP(public.ModelConfig(),49,192)
        value=torch.randn(2,16,4,49,192,requires_grad=True)
        result=module(value)
        self.assertEqual(result.shape,value.shape)
        result.square().mean().backward()
        self.assertGreater(float(module.fc1.weight.grad.abs().sum()),0)

    def test_parent_components_are_inherited_and_global_tensors_are_new(self):
        original=parent.LightGen2(parent.ModelConfig(alpha_eval=.5))
        with torch.no_grad():
            original.vision_frame_position.zero_()
            original.language_sequence_position.zero_()
        model=inherit_parent(original)
        for name,value in original.state_dict().items():
            if not name.startswith(('vision_global.','language_global.')):
                self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        forbidden=('raw_phase','transfer','propagation')
        self.assertFalse(any(any(word in name for word in forbidden) for name in model.state_dict()))
        tree=ast.parse(Path(public.__file__).read_text('utf-8'))
        self.assertFalse(any(isinstance(node,ast.Attribute) and node.attr=='fft' for node in ast.walk(tree)))

    def test_shared_mlp_is_invariant_to_video_order(self):
        module=public.GlobalMLP(public.ModelConfig(),42,96).eval()
        value=torch.randn(1,16,42,192)
        permutation=torch.randperm(16)
        with torch.inference_mode():
            expected=module(value)[:,permutation]
            actual=module(value[:,permutation])
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

    def test_single_checkpoint_restores_original_position_table(self):
        model=public.LightGen2(public.ModelConfig())
        with torch.no_grad():
            model.vision_frame_position.zero_()
            model.language_sequence_position.zero_()
        table=torch.randn(16,1024).to(torch.bfloat16)
        model.vision_frontend.prepare_position_table(table)
        payload=checkpoint(model,dict(frontend={'position_table':table},metadata={}))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'LightGen2.pt'
            torch.save(payload,path)
            output=Path(directory)/'separate_project/weights/LightGen2.pt'
            export_checkpoint(path,output)
            loaded,_=public.load_lightgen2(output)
        for name,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,loaded.state_dict()[name]),name)


if __name__=='__main__':unittest.main()
