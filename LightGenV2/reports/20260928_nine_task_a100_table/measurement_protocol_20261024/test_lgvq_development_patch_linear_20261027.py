import ast
import sys
import unittest
import numpy as np
from lgvq_development_patch_linear_20261027 import linear_AST

SOURCE='''def forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
    return hidden_states
'''


class PatchLinearTests(unittest.TestCase):
    def test_only_one_call_changes_and_original_cast_and_view_remain(self):
        old=ast.parse(SOURCE);new=linear_AST(SOURCE)
        self.assertEqual(ast.dump(old.body[0].body[0]),ast.dump(new.body[0].body[0]))
        self.assertEqual(ast.dump(old.body[0].body[1]),ast.dump(new.body[0].body[1]))
        self.assertEqual(ast.dump(old.body[0].body[-1]),ast.dump(new.body[0].body[-1]))
        text=ast.unparse(new)
        self.assertIn('self.proj.weight.view(self.embed_dim, -1), self.proj.bias',text)
        self.assertIn('hidden_states.to(dtype=target_dtype).reshape',text)
        self.assertEqual(text.count('torch.nn.functional.linear'),1)

    def test_missing_or_duplicate_original_call_rejected(self):
        with self.assertRaises(ValueError):linear_AST(SOURCE.replace('self.proj(hidden_states.to(dtype=target_dtype))','self.proj(hidden_states)'))
        with self.assertRaises(ValueError):linear_AST(SOURCE+SOURCE.replace('forward','other'))

    def test_complete_patch_correlation_equals_flattened_matrix_in_same_order(self):
        # Integer-valued arithmetic proof of the exact Conv3d indexing, not a
        # floating-point tolerance waiver or CPU model execution.
        x=np.arange(4*3*2*4*4,dtype=np.int64).reshape(4,3,2,4,4)%13
        w=np.arange(7*3*2*4*4,dtype=np.int64).reshape(7,3,2,4,4)%11
        b=np.arange(7,dtype=np.int64)
        actual=x.reshape(4,-1)@w.reshape(7,-1).T+b
        expected=np.empty((4,7),dtype=np.int64)
        for n in range(4):
            for o in range(7):
                expected[n,o]=b[o]+sum(x[n,c,t,h,k]*w[o,c,t,h,k]
                    for c in range(3) for t in range(2) for h in range(4) for k in range(4))
        self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_import_does_not_import_torch(self):
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
