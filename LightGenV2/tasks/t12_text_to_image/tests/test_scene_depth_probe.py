import unittest
from LightGenV2.tasks.t12_text_to_image.scene_depth_probe import keep_indices,checked_config,prune_unet


class PlanTests(unittest.TestCase):
    def test_noop_preserves_all_indices(self):
        self.assertEqual(keep_indices(10,10),tuple(range(10)))

    def test_compression_preserves_order_endpoints_and_exact_budget(self):
        for target in (6,8,9):
            ids=keep_indices(10,target)
            self.assertEqual(len(ids),target)
            self.assertEqual(ids,tuple(sorted(set(ids))))
            self.assertEqual((ids[0],ids[-1]),(0,9))

    def test_invalid_budgets_are_rejected(self):
        for args in ((10,1),(10,11),(10,True),(10,8.0)):
            with self.assertRaises(ValueError):keep_indices(*args)

    def test_unbound_parent_rejected(self):
        with self.assertRaises(ValueError):checked_config(dict(schema='scene_depth_probe_v1'))

    def test_structural_pruning_removes_registered_IP_parameters_and_keeps_gradients(self):
        try:
            import torch
            from torch import nn
        except (ImportError,OSError) as error:
            self.skipTest('Real torch structural test must pass on server: '+type(error).__name__)
        class Group(nn.Module):
            def __init__(self,n):
                super().__init__()
                self.transformer_blocks=nn.ModuleList([nn.Linear(2,2,bias=False) for _ in range(n)])
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.groups=nn.ModuleList([Group(2) for _ in range(5)]+[Group(10) for _ in range(6)])
        net=Net();original=net.groups[5].transformer_blocks[9]
        before=sum(p.numel() for p in net.parameters())
        proof=prune_unet(net,8)
        self.assertEqual(proof['actual_blocks'],58)
        self.assertEqual(proof['removed_parameters'],12*4)
        self.assertEqual(sum(p.numel() for p in net.parameters()),before-48)
        self.assertIs(net.groups[5].transformer_blocks[-1],original)
        net.groups[5].transformer_blocks[-1](torch.ones(1,2)).sum().backward()
        self.assertTrue(torch.isfinite(original.weight.grad).all())


if __name__=='__main__':unittest.main()
