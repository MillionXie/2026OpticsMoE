import json,unittest
from pathlib import Path
from .continue_LG_D_optics import validate,configure_D,native_C_retention
import torch
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

    def test_native_C_retention_only_constrains_TRAIN_tensors(self):
        c=json.loads(Path(__file__).with_name('config_LG_D_native_C_retention_20261025.json').read_text());validate(c)
        self.assertEqual(c['epochs'],4);self.assertFalse(c['TEST_gradients'])
        z=torch.tensor([[2.,0.],[0.,2.]],requires_grad=True);q=torch.full((2,16),1/16,requires_grad=True);features=torch.ones(2,784,requires_grad=True)
        teacher={'logits':z.detach().clone(),'route_power':q.detach().clone(),'ccd_features':features.detach().clone()}
        loss=native_C_retention(dict(logits=z,route_power=q,ccd_features=features),teacher,torch.tensor([0,1]),c)
        self.assertAlmostEqual(float(loss),0.,places=5);loss.backward();self.assertIsNotNone(z.grad)
        bad=dict(teacher,logits=torch.tensor([[0.,2.],[2.,0.]]));loss=native_C_retention(dict(logits=z,route_power=q,ccd_features=features),bad,torch.tensor([0,1]),c)
        self.assertAlmostEqual(float(loss),0.,places=5)


if __name__=='__main__':unittest.main()
