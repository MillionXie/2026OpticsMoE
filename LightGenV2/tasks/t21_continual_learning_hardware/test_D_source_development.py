import json,unittest,tempfile
from pathlib import Path
import torch
from .refine_LG_D_source_development import validate,bands,save_development_checkpoint


class DSourceDevelopmentTests(unittest.TestCase):
    def test_exact_development_scope_never_clean_S1(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_D_source_development_20261026.json').read_text())
        validate(cfg);self.assertEqual(bands(cfg)['speech_binary'],(761,871))
        self.assertFalse(cfg['valid_for_formal_test_table']);self.assertEqual(cfg['independent_test_samples'],0)
        for key,value in [('SDK',True),('maximum_steps',10000),('all_TEST_sources_authorized',False)]:
            with self.assertRaises(ValueError):validate(dict(cfg,**{key:value}))
    def test_bands_keep_LG_at_least_frozen_D2NN(self):
        cfg=json.loads(Path(__file__).with_name('config_LG_D_source_development_20261026.json').read_text())
        self.assertEqual(bands(cfg)['physical_binary_raw'][0],735)
        cfg['development_bands']['clevr']=[0,1000]
        with self.assertRaises(ValueError):bands(cfg)
    def test_saved_student_cannot_be_mistaken_for_clean_S1(self):
        class Toy(torch.nn.Module):
            def __init__(self):super().__init__();self.shared_head=torch.nn.Linear(4,10,bias=False)
        model=Toy();weight=torch.randn(10,4)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'student.pt';save_development_checkpoint(model,dict(architecture='moe',stage=4),weight,path,'parent')
            cp=torch.load(path,map_location='cpu',weights_only=False)
            self.assertFalse(cp['valid_for_formal_test_table']);self.assertTrue(cp['development_only'])
            self.assertTrue(cp['teacher_uses_TEST_derived_gradients']);self.assertTrue(cp['provisional_not_hardware_admitted'])
            torch.testing.assert_close(cp['model']['shared_head.weight'],weight)


if __name__=='__main__':unittest.main()
