import unittest
from ..scene_path_geometry import optimize_edges


class GeometryGrid(unittest.TestCase):
    def test_fixed_endpoints_and_interval_limit(self):
        costs=[[float((b-a)**2) if b>a else 0. for b in range(31)]for a in range(31)]
        result=optimize_edges(costs,6)
        self.assertEqual(result['edges'],[0,5,10,15,20,25,30])
        self.assertEqual(result['TRAIN_geometry_cost'],150.)

    def test_does_not_assume_uniformity_when_known_curvature_differs(self):
        wanted=[0,2,6,12,18,24,30];costs=[[1000. for _ in range(31)]for _ in range(31)]
        for a,b in zip(wanted,wanted[1:]):costs[a][b]=0.
        self.assertEqual(optimize_edges(costs,6)['edges'],wanted)

    def test_unknown_teacher_or_nonfinite_cost_rejected(self):
        with self.assertRaises(ValueError):optimize_edges([[0.]*13 for _ in range(13)],6)
        costs=[[float('nan')]*31 for _ in range(31)]
        with self.assertRaises(ValueError):optimize_edges(costs,6)


if __name__=='__main__':unittest.main()
