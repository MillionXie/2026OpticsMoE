import unittest
from types import SimpleNamespace
from LightGenV2.tasks.t07_abo_image_retrieval.analysis.audit_for_meeting import bootstrap


class MeetingAuditTests(unittest.TestCase):
    def test_bootstrap_resamples_products_and_pairs_models(self):
        samples=[];rows=[]
        for c in range(10):
            for p in range(4):
                for v in range(12):
                    sid=f'{c}-{p}-{v}'
                    samples.append(SimpleNamespace(sample_id=sid,product_id=f'{c}-{p}',category_id=c))
                    rows.append(dict(sample_id=sid,top1_relevant=p<2))
        result=bootstrap(samples,dict(optical=rows,identical=rows),repeats=200)
        self.assertEqual(result['optical']['hit1'],.5)
        self.assertEqual(result['identical']['paired_gap_minus_optical_95'],[0.,0.])
        interval=result['optical']['conditional_product_bootstrap_95']
        self.assertLess(interval[0],.5);self.assertGreater(interval[1],.5)
