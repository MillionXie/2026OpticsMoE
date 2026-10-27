import ast
import sys
import unittest
import numpy as np
from lgvq_development_patch_conv2d_20261027 import conv2d_AST

SOURCE='''def forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(-1,self.in_channels,self.temporal_patch_size,self.patch_size,self.patch_size)
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1,self.embed_dim)
    return hidden_states
'''


class Conv2dTests(unittest.TestCase):
    def test_only_complete_patch_call_changes(self):
        old=ast.parse(SOURCE);new=conv2d_AST(SOURCE)
        for i in [0,1,3]:self.assertEqual(ast.dump(old.body[0].body[i]),ast.dump(new.body[0].body[i]))
        text=ast.unparse(new)
        self.assertEqual(text.count('torch.nn.functional.conv2d'),1)
        self.assertIn('hidden_states.to(dtype=target_dtype).reshape',text)
        self.assertIn('self.proj.bias, stride=',text)
        self.assertNotIn('linear',text)

    def test_missing_duplicate_call_rejected(self):
        with self.assertRaises(ValueError):conv2d_AST(SOURCE.replace('dtype=target_dtype','dtype=other'))
        with self.assertRaises(ValueError):conv2d_AST(SOURCE+SOURCE.replace('forward','other'))

    def test_full_temporal_height_fold_preserves_products_and_order(self):
        x=np.arange(2*3*2*4*4,dtype=np.int64).reshape(2,3,2,4,4)%13
        w=np.arange(5*3*2*4*4,dtype=np.int64).reshape(5,3,2,4,4)%11
        y=x.reshape(2,3,8,4);v=w.reshape(5,3,8,4);bias=np.arange(5,dtype=np.int64)
        for n in range(2):
            for o in range(5):
                products3=[x[n,c,t,h,k]*w[o,c,t,h,k] for c in range(3) for t in range(2) for h in range(4) for k in range(4)]
                products2=[y[n,c,h,k]*v[o,c,h,k] for c in range(3) for h in range(8) for k in range(4)]
                self.assertEqual(products3,products2)
                self.assertEqual(sum(products3)+bias[o],sum(products2)+bias[o])

    def test_no_torch_model_or_backend_mutation_on_import(self):
        self.assertNotIn('torch',sys.modules)

    def test_channel_temporal_fold_preserves_product_order(self):
        x=np.arange(2*3*2*4*4,dtype=np.int64).reshape(2,3,2,4,4)%13
        w=np.arange(5*3*2*4*4,dtype=np.int64).reshape(5,3,2,4,4)%11
        y=x.reshape(2,6,4,4);v=w.reshape(5,6,4,4)
        for n in range(2):
            for o in range(5):
                self.assertEqual([x[n,c,t,h,k]*w[o,c,t,h,k] for c in range(3) for t in range(2) for h in range(4) for k in range(4)],
                    [y[n,c,h,k]*v[o,c,h,k] for c in range(6) for h in range(4) for k in range(4)])

    def test_channel_fold_AST_only_changes_complete_patch_geometry(self):
        text=ast.unparse(conv2d_AST(SOURCE,channels_fold=True))
        self.assertIn('reshape(-1, self.in_channels * self.temporal_patch_size, self.patch_size, self.patch_size)',text)
        self.assertIn('self.proj.bias, stride=(self.patch_size, self.patch_size)',text)
        self.assertEqual(text.count('torch.nn.functional.conv2d'),1)
        self.assertNotIn('linear',text)


if __name__=='__main__':unittest.main()
