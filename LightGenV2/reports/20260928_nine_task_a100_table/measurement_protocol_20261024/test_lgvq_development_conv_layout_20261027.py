import ast
import unittest
import lgvq_development_conv_layout_20261027 as layout

SOURCE='''def forward(self,hidden_states):
    target_dtype=self.proj.weight.dtype
    hidden_states=hidden_states.view(-1,self.in_channels,self.temporal_patch_size,self.patch_size,self.patch_size)
    hidden_states=self.proj(hidden_states.to(dtype=target_dtype)).view(-1,self.embed_dim)
    return hidden_states
'''


class LayoutTests(unittest.TestCase):
    def test_pinned_geometry_is_actual_frozen_Qwen3_patch16(self):
        self.assertEqual(layout.PINNED_KERNEL,(2,16,16))

    def test_original_convolution_and_dtype_AST_unchanged_except_layout_keyword(self):
        expected=SOURCE.replace('to(dtype=target_dtype)','to(dtype=target_dtype,memory_format=torch.channels_last_3d)')
        self.assertEqual(ast.dump(layout.layout_AST(SOURCE)),ast.dump(ast.parse(expected)))

    def test_drift_fails_closed(self):
        with self.assertRaises(ValueError):layout.layout_AST(SOURCE.replace('dtype=target_dtype','dtype=torch.float32'))
        with self.assertRaises(ValueError):layout.layout_AST(SOURCE+'\nx=hidden_states.to(dtype=target_dtype)')

    def test_import_does_not_load_model_or_torch(self):
        import sys
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
