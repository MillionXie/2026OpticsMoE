import unittest
from pathlib import Path
from types import SimpleNamespace as N
from first_tf_supplement_20261025 import block_candidate,original_command,integrate
from run_first_tf_seven_20261025 import templates
class Tests(unittest.TestCase):
    def test_plan_only_seven(self):
        self.assertEqual([x[0] for x in templates()],['02','03','04','05','06','07','08'])
    def test_vision(self):
        block=type('Qwen3VLVisionBlock',(),{})()
        self.assertIs(block_candidate(N(model=N(visual=N(blocks=[block]))),'vision')[0],block)
        self.assertIsNone(block_candidate(N(layers=[block]),'language'))
    def test_text(self):
        block=type('Qwen3VLTextDecoderLayer',(),{})()
        self.assertIs(block_candidate(N(layers=[block]),'language')[0],block)
    def test_t12_original_gpu_entry(self):
        c=original_command({'command':['/x/measure_t12_cost.py','--output','old','--repeat','1','--deadline','1']},Path('/new'),3,9)
        self.assertTrue(c[0].endswith('gpu_checked_cost_entry_20261025.py'));self.assertEqual(c[1:3],['--task','t12'])
        self.assertEqual(c[c.index('--repeat')+1],'3');self.assertEqual(c[c.index('--deadline')+1],'9')
    def test_energy_omits_gaps(self):
        p=[dict(host_monotonic_s=i*.1,watts=100) for i in range(11)]
        v=integrate([(.15,.25),(.7,.8)],p)
        self.assertAlmostEqual(v['wall_s'],.2);self.assertAlmostEqual(v['system_j'],87.64)
    def test_stale_power_rejected(self):
        with self.assertRaises(AssertionError):integrate([(.01,.4)],[dict(host_monotonic_s=0,watts=100),dict(host_monotonic_s=1,watts=100)])
if __name__=='__main__':unittest.main()
