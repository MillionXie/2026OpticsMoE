import unittest

import torch

from LightGenV2.tasks.t12_text_to_image.qwen_vl_spatial import SpatialInputConv
from LightGenV2.tasks.t12_text_to_image.qwen_vl_tail import build_tail_student, resize_tensor


class TailTests(unittest.TestCase):
    def test_gated_halves_and_skip_halves_are_preserved(self):
        value = torch.arange(16).reshape(8, 2)
        gated = resize_tensor(value, (4, 2), 'block.ff.net.0.proj.weight')
        self.assertTrue(torch.equal(gated, value[[0, 1, 4, 5]]))
        weight = torch.arange(24).reshape(2, 12)
        sliced = resize_tensor(weight, (2, 6), 'up_blocks.0.resnets.0.conv1.weight', concat=(8, 4))
        self.assertTrue(torch.equal(sliced, weight[:, [0, 1, 2, 3, 8, 9]]))

    def test_pruned_unet_retains_20_channel_input_and_native_output(self):
        from diffusers import UNet2DConditionModel
        teacher = UNet2DConditionModel(sample_size=32, in_channels=4, out_channels=4,
              block_out_channels=(32, 64, 64, 64), layers_per_block=2,
              cross_attention_dim=32, attention_head_dim=(1, 1, 1, 1),
              down_block_types=('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'),
              up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'))
        teacher.conv_in = SpatialInputConv(teacher.conv_in, 16)
        teacher.register_to_config(in_channels=20)
        student, profile = build_tail_student(teacher, (32, 32, 32, 32), (1, 1, 1, 1))
        self.assertLess(profile['student_parameters'], profile['teacher_parameters'])
        with torch.no_grad():
            out = student(torch.randn(1, 20, 32, 32), 999, encoder_hidden_states=torch.randn(1, 77, 32)).sample
        self.assertEqual(tuple(out.shape), (1, 4, 32, 32))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
