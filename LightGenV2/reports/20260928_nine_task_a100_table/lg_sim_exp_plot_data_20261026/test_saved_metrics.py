"""Pure CPU contracts for archived-ranking completion; no formal data writes."""
import ast,unittest
from pathlib import Path
import numpy as np

source=Path(__file__).with_name('collect_archived_data.py')
tree=ast.parse(source.read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='full_rankings')
scope={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
rank=scope['full_rankings']

class SavedRankingContracts(unittest.TestCase):
    def test_multi_positive_full_ap_is_not_reciprocal_rank(self):
        _,first,ap,ap10,positive=rank([[1,0]],[[1,0],[.8,.2],[0,1]],['A'],['B','A','A'])
        self.assertEqual(first.tolist(),[2]);self.assertEqual(positive,[[2,3]])
        self.assertAlmostEqual(float(ap[0]),(.5+2/3)/2)
        self.assertNotAlmostEqual(float(ap[0]),1/first[0]);self.assertAlmostEqual(float(ap10[0]),float(ap[0]))
    def test_single_positive_ap_equals_reciprocal_rank(self):
        _,first,ap,_,_=rank([[1,0]],[[1,0],[.8,.2],[0,1]],['A'],['B','A','C'])
        self.assertAlmostEqual(float(ap[0]),1/first[0])
    def test_cosine_is_invariant_to_positive_scale(self):
        q=np.array([[1,.1]],np.float32);c=np.array([[1,1],[2,0],[0,2]],np.float32)
        a=rank(q,c,['A'],['A','B','A']);b=rank(q*9,c*np.array([[2],[4],[8]]),['A'],['A','B','A'])
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_allclose(a[2],b[2])
    def test_equal_scores_keep_original_candidate_order(self):
        order,first,_,_,_=rank([[1,0]],[[1,0],[1,0]],['A'],['B','A'])
        self.assertEqual(order.tolist(),[[0,1]]);self.assertEqual(first.tolist(),[2])
    def test_no_positive_is_not_silently_filled(self):
        with self.assertRaises(AssertionError):rank([[1,0]],[[1,0]],['A'],['B'])

if __name__=='__main__':unittest.main()
