import unittest
import torch
from ..scene_path_geometry import path_costs


class GeometryNumerics(unittest.TestCase):
    def test_Gram_cost_equals_direct_interpolation_on_nonlinear_path(self):
        sigma=[float(30-i)for i in range(31)];base=torch.linspace(-1,1,4*64*64).reshape(4,64,64)
        trajectory=torch.stack([base*s+(s/30)**2 for s in sigma])
        actual=path_costs([trajectory],sigma)
        for start,stop in [(0,5),(12,20),(24,30)]:
            expected=0.
            for k in range(start+1,stop):
                lam=(sigma[k]-sigma[start])/(sigma[stop]-sigma[start]);wanted=(1-lam)*trajectory[start].double()+lam*trajectory[stop].double()
                expected+=float((trajectory[k].double()-wanted).square().mean()/trajectory[k].double().square().mean().clamp_min(1.))
            self.assertAlmostEqual(actual[start][stop],expected,places=9)

    def test_sigma_linear_path_has_negligible_geometry_cost(self):
        sigma=[float(30-i)for i in range(31)];trajectory=torch.stack([torch.full((4,64,64),s+1.)for s in sigma])
        self.assertLess(max(max(row)for row in path_costs([trajectory],sigma)),1e-11)


if __name__=='__main__':unittest.main()
