"""CPU-only presentation accounting tests; no models, data or CUDA loaded."""
import math
import sys
import unittest

from temporal_static_values_20261027 import CONTROL_KEYS,PHASE_NAMES,detailed_appendix


class AppendixAccountingTests(unittest.TestCase):
    def setUp(self):
        phases=[k for k in PHASE_NAMES if k!='scheduling']
        self.compute={k:(i+1)/10 for i,k in enumerate(phases)}
        self.overhead={k:0.01 for k in phases+list(CONTROL_KEYS)}
        self.evidence=[dict(fields=1,fine=dict(work={k:7*v for k,v in self.compute.items()},
            compute=self.compute,overhead=self.overhead,control_breakdown=[]))]
        old=math.fsum(self.overhead.values())
        self.detail=dict(rows=[[PHASE_NAMES['scheduling'],None,old]])
        self.total=math.fsum(self.compute.values())+old
        self.result=detailed_appendix(self.evidence,self.detail,self.total)

    def test_every_original_key_assigned_once(self):
        keys=[k for r in self.result['phase_accounting'] for k in r['keys']]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertEqual(set(keys),(set(PHASE_NAMES)-{'scheduling'})|set(CONTROL_KEYS))

    def test_equivalent_not_cumulative_work(self):
        rows=self.result['rows'][:-1]
        self.assertAlmostEqual(math.fsum(r[2] for r in rows),self.total)
        first=rows[0][2];keys=['tokenizer','frontend','vision_entry']
        self.assertAlmostEqual(first,sum(self.compute[k]+self.overhead[k] for k in keys))
        self.assertNotAlmostEqual(first,7*sum(self.compute[k] for k in keys))

    def test_one_control_row_no_duplicate_subtotals(self):
        self.assertEqual(len(self.result['rows']),18)
        control=[r for r in self.result['rows'] if r[0]=='调度与同步']
        self.assertEqual(len(control),1)
        self.assertAlmostEqual(control[0][2],sum(self.overhead[k] for k in CONTROL_KEYS))

    def test_language_field_adapter_after_merger(self):
        rows=self.result['rows'];labels=[r[1] for r in rows]
        self.assertLess(labels.index('Merger与多模态序列准备'),labels.index('语言Adapter'))
        self.assertEqual(rows[labels.index('语言Adapter')][0],'语言光电处理')
        fusion=[r for r in self.result['phase_accounting'] if r['module']=='多模态融合']
        self.assertEqual(fusion[0]['keys'],['merger'])

    def test_preprocessing_contains_text_and_vision(self):
        self.assertIn('文本Embedding',self.result['rows'][0][1])
        self.assertIn('视觉Adapter',self.result['rows'][0][1])
        self.assertEqual(self.result['phase_accounting'][0]['keys'],['tokenizer','frontend','vision_entry'])

    def test_module_totals_close(self):
        self.assertAlmostEqual(math.fsum(r[1] for r in self.result['module_rows']),self.total)
        self.assertAlmostEqual(sum(r['control_ms16'] for r in self.result['phase_accounting']),self.detail['rows'][0][2])

    def test_no_tensor_framework_import(self):
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
