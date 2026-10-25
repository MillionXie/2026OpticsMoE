import inspect,unittest,math,json
from pathlib import Path
import torch
from . import routing_diversity as m


class DiversityTests(unittest.TestCase):
    def test_global_balance_can_coexist_with_sensor_specialization(self):
        q=torch.tensor([[.49,.49,.01,.01],[.01,.01,.49,.49]],dtype=torch.float64,requires_grad=True);balance,js=m.balance_and_diversity(q,torch.tensor([0,1]),torch.arange(4))
        self.assertLess(float(balance),1e-12);self.assertGreater(float(js),.5);self.assertLess(float(js),math.log(2));(1.5*balance-2*js).backward();self.assertTrue(torch.isfinite(q.grad).all())

    def test_identical_domains_and_collapsed_expert_not_rewarded(self):
        d=torch.tensor([0,1]);ix=torch.arange(4);_,js=m.balance_and_diversity(torch.ones(2,4)/4,d,ix);self.assertEqual(float(js),0.)
        balance,js=m.balance_and_diversity(torch.tensor([[1.,0.,0.,0.]]*2),d,ix);self.assertGreater(float(balance),.7);self.assertEqual(float(js),0.)

    def test_slot_permutation_and_loss_only_contract(self):
        q=torch.tensor([[.4,.3,.2,.1],[.1,.2,.3,.4]],dtype=torch.float64);d=torch.tensor([0,1]);a=m.balance_and_diversity(q,d,torch.arange(4));b=m.balance_and_diversity(q[:,[3,0,2,1]],d,torch.arange(4))
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=1e-14)
        self.assertNotIn('nn.Module',inspect.getsource(m));c=json.loads(Path(__file__).with_name('config_static_four_routing_diversity_20261026.json').read_text());self.assertTrue(c['paired_TRAIN_batches']);self.assertFalse(c['recalibrate_gains']);self.assertFalse(c['TEST_teacher']);self.assertEqual(c['epochs'],25)


if __name__=='__main__':unittest.main()
