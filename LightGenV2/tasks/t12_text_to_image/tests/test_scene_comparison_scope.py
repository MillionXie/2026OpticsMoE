import unittest
from LightGenV2.tasks.t12_text_to_image.reports.scene_generation_20261029.plot_LG_recovery import cohort_scope


class SceneComparisonScope(unittest.TestCase):
    def test_development_does_not_accept_TEST_or_partial_frozen_cohort(self):
        report={'complete':True,'scope':'DEV_expanded','independent_TEST_opened':False}
        self.assertEqual(cohort_scope([report,report],128),('DEV',None))
        with self.assertRaises(ValueError):cohort_scope([report,report],500)
        with self.assertRaises(ValueError):cohort_scope([report,report],500,True)

    def test_frozen_comparison_requires_identical_valid_freeze(self):
        report={'complete':True,'scope':'TEST','independent_TEST_opened':True,
                'frozen_TEST_protocol':{'manifest_sha256':'a'*64}}
        self.assertEqual(cohort_scope([report,report],500,True),('frozen TEST','a'*64))
        with self.assertRaises(ValueError):cohort_scope([report,dict(report,frozen_TEST_protocol={'manifest_sha256':'b'*64})],500,True)
        with self.assertRaises(ValueError):cohort_scope([report,report],128,True)
        with self.assertRaises(ValueError):cohort_scope([report,report],500)

    def test_unfinished_and_missing_identity_are_rejected(self):
        report={'complete':True,'scope':'TEST','independent_TEST_opened':True}
        for other in [dict(report,complete=False),report,dict(report,frozen_TEST_protocol={'manifest_sha256':'z'*64})]:
            with self.assertRaises(ValueError):cohort_scope([other,other],500,True)


if __name__=='__main__':unittest.main()
