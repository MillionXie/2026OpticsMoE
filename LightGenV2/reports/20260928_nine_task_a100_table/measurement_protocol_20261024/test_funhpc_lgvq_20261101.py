"""No-Torch CPU plan/timing contracts."""
import unittest
import run_funhpc_lgvq_20261101 as q

class Contract(unittest.TestCase):
 def test_human_order_and_full_plan(self):
  p=q.plan();self.assertEqual(p[:5],[('deepseek',50,1),('kimi',50,1),('LG2',100,1),('deepseek',100,1),('kimi',100,1)])
  self.assertEqual(len(p),15);self.assertEqual(len(set(p)),15)
  self.assertEqual(len([x for x in p if x[0]=='LG2']),3)
 def records(self):
  return [dict(sample_id=str(i),preprocess_start=i,prepare_end=i+.1,model_call_start=i+.2,first_block_start=i+.25,gpu_output=i+.35,call_return=i+.4) for i in range(512)]
 def test_all511_gaps_charged(self):
  r=q.timing(self.records());self.assertAlmostEqual(r['continuous_wall_s'],511.4);self.assertAlmostEqual(r['components']['internal_gap_s'],511*.6);self.assertAlmostEqual(r['time16_ms'],511.4/32*1000);self.assertFalse(r['energy_measured'])
 def test_overlap_rejected(self):
  r=self.records();r[1]['preprocess_start']=.3
  with self.assertRaises(ValueError):q.timing(r)
 def test_duplicate_rejected(self):
  r=self.records();r[-1]['sample_id']='0'
  with self.assertRaises(ValueError):q.timing(r)
if __name__=='__main__':unittest.main()
