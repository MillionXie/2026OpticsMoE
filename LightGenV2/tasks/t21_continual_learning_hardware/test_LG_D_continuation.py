import json,unittest
from pathlib import Path
from .continue_LG_D_optics import validate,configure_D
from .model import build_model


class DTests(unittest.TestCase):
    def test_finite_profile_does_not_authorize_TEST_gradients(self):
        c=json.loads(Path(__file__).with_name('config_LG_D_dependent_C_20261025.json').read_text());validate(c)
        for key,value in [('TEST_gradients',True),('epochs',100),('memory_records_per_old_task',1000)]:
            with self.assertRaises(ValueError):validate(dict(c,**{key:value}))

    def test_only_last_four_experts_and_original_shared_parameters_train(self):
        model=build_model('moe')
        for v in model.parameters():v.requires_grad_(False)
        actual=configure_D(model);self.assertEqual(len(actual),7)
        for i,v in enumerate(model.first_phase):self.assertEqual(v.requires_grad,i in model.active_indices[12:16].tolist())


if __name__=='__main__':unittest.main()
