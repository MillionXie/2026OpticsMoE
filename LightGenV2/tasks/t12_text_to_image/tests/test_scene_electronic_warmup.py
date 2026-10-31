import copy
import unittest
from ..scene_electronic_warmup import checked_target


class ElectronicWarmup(unittest.TestCase):
    def payload(self):
        return dict(optical_frontend={'checkpoint_sha256':'f'*64},condition_teacher_sha256='a'*64,
                    records=[dict(split='train',input_key='t'),dict(split='dev',input_key='d')],
                    TRAIN_electronic_conditions={'t':dict(text=1,image=2,pooled=3)})
    def test_default_preserves_historical_native_target(self):
        self.assertEqual(checked_target({}, {}, None, 'native'),'native_encoders')
    def test_exact_bound_training_teacher(self):
        p=self.payload();before=copy.deepcopy(p)
        self.assertEqual(checked_target({'condition_warmup_target':'bound_electronic'},p,'a'*64,'qwen'),'bound_electronic')
        self.assertEqual(before,p)
    def test_wrong_teacher_and_leaked_or_missing_targets_rejected(self):
        cfg={'condition_warmup_target':'bound_electronic'}
        for change in ['teacher','leak','missing','overlap','branch']:
            p=self.payload();c=dict(cfg)
            if change=='teacher':p['condition_teacher_sha256']='b'*64
            if change=='leak':p['TRAIN_electronic_conditions']['d']=dict(text=1,image=2,pooled=3)
            if change=='missing':del p['TRAIN_electronic_conditions']['t']['pooled']
            if change=='overlap':p['records'][1]['input_key']='t'
            if change=='branch':c['joint_image_context']=True
            with self.assertRaises(ValueError):checked_target(c,p,'a'*64,'qwen')
    def test_invalid_modes(self):
        with self.assertRaises(ValueError):checked_target({'condition_warmup_target':'other'}, {},None,'qwen')
        for variant,parent in [('native','a'*64),('qwen',None)]:
            with self.assertRaises(ValueError):checked_target({'condition_warmup_target':'bound_electronic'},self.payload(),parent,variant)


if __name__=='__main__':unittest.main()
