import unittest
from types import SimpleNamespace
from ..scene_depth_recovery import parent_key,expand_lora


class DepthNamespaceTests(unittest.TestCase):
    def test_deep_indices_rebind_to_original_weights(self):
        name='up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_k.lora_A.scene_pair.weight'
        self.assertIn('transformer_blocks.8.',parent_key(name,8))
        self.assertEqual(parent_key(name,10),name)

    def test_shallow_and_image_projection_names_stay_unchanged(self):
        for name in ('up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k.lora_B.scene_pair.weight',
                     'encoder_hid_proj.image_projection_layers.0.layers.0.0.to_q.lora_A.scene_pair.weight'):
            self.assertEqual(parent_key(name,8),name)

    def test_removed_parent_entries_remain_inert_and_parent_not_mutated(self):
        shape=(2,3);prefix='mid_block.attentions.0.transformer_blocks.'
        parent={prefix+str(i)+'.attn2.to_q.lora_A.weight':SimpleNamespace(shape=shape,value=i) for i in range(10)}
        update=SimpleNamespace(shape=shape,value=99)
        result=expand_lora(parent,{prefix+'6.attn2.to_q.lora_A.weight':update},8)
        self.assertIs(result[prefix+'8.attn2.to_q.lora_A.weight'],update)
        self.assertEqual(parent[prefix+'8.attn2.to_q.lora_A.weight'].value,8)
        self.assertIs(result[prefix+'2.attn2.to_q.lora_A.weight'],parent[prefix+'2.attn2.to_q.lora_A.weight'])
        self.assertEqual(set(result),set(parent))

    def test_unexpected_keys_or_shapes_are_rejected(self):
        with self.assertRaises(ValueError):expand_lora({}, {'unknown':SimpleNamespace(shape=(1,))},8)
        with self.assertRaises(ValueError):expand_lora({'x':SimpleNamespace(shape=(2,))},{'x':SimpleNamespace(shape=(1,))},8)
        with self.assertRaises(ValueError):parent_key('mid_block.attentions.0.transformer_blocks.8.attn2.to_q.weight',8)


if __name__=='__main__':unittest.main()
