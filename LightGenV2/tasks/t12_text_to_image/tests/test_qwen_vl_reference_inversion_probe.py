import unittest
import torch
from LightGenV2.tasks.t12_text_to_image.qwen_vl_reference_inversion_probe import native_vision_grid,InverseReadout


class ReferenceProbeTests(unittest.TestCase):
    def test_native_tile_order_is_restored_to_actual_image_coordinates(self):
        values=[]
        for y in range(8):
            for x in range(8):
                for dy in range(2):
                    for dx in range(2):values.append((2*y+dy)*16+2*x+dx)
        actual=native_vision_grid(torch.tensor(values)[None,:,None])
        torch.testing.assert_close(actual[0,:,:,0],torch.arange(256).reshape(16,16))

    def test_both_readouts_produce_identical_latent_shape_with_finite_gradients(self):
        for grid in (8,16):
            model=InverseReadout(12,grid);value=model(torch.randn(2,grid**2,12))
            self.assertEqual(tuple(value.shape),(2,4,32,32));value.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
