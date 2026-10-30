import copy
import unittest
from LightGenV2.tasks.t12_text_to_image.scene_reference_strength import bind


class ReferenceStrength(unittest.TestCase):
    def state(self):
        return dict(head={'a':[1,2]},unet_lora={'b':[3]},source_commit='old',
            optical_frontend=dict(checkpoint_sha256='f6',pair_expansion=None),
            sampling_contract=dict(kind='progressive_guided_Euler_CFGpreserved_v1',
                steps=12,teacher_steps=30,CFG=7.5,IP_scale=.25,timesteps=[9,4,1]))

    def test_training_protocol_and_all_weights_are_preserved(self):
        s=self.state();before=copy.deepcopy(s);new=bind(s,.5,'parent','new')
        self.assertEqual(s,before);self.assertIs(s['head'],new['head'])
        self.assertIs(s['unet_lora'],new['unet_lora'])
        self.assertEqual(new['training_sampling_contract'],before['sampling_contract'])
        self.assertEqual(new['sampling_contract'],dict(before['sampling_contract'],IP_scale=.5))
        self.assertFalse(new['inference_reference_strength_ablation']['training_performed'])

    def test_invalid_or_stacked_or_two_pair_ablation_is_rejected(self):
        for scale in [0,-.2,1.01,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):bind(self.state(),scale,'p','s')
        with self.assertRaises(ValueError):bind(bind(self.state(),.5,'p','s'),.6,'p','s')
        s=self.state();s['optical_frontend']['pair_expansion']={'pairs':2}
        with self.assertRaises(ValueError):bind(s,.5,'p','s')


if __name__=='__main__':unittest.main()
