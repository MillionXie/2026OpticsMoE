"""CPU-only timing contract; never load a model or initialize CUDA."""
import unittest
from run_qwen_rental_demo_20261030 import timing
def calls():
 return [dict(sample_id=str(i),preprocess_start=i,prepare_end=i+.1,model_call_start=i+.2,
  first_block_start=i+.25,gpu_output=i+.35,call_return=i+.4,legacy_host_ms=100) for i in range(512)]
class Contract(unittest.TestCase):
 def test_full_window_keeps_all_gaps_and_units(self):
  v=timing(calls());self.assertAlmostEqual(v['file_to_CPU_ms16'],511.4/32*1000)
  self.assertAlmostEqual(v['first_TF_to_score_work_ms16'],1600)
  self.assertFalse(v['matched_LG_compute_scope']);self.assertFalse(v['energy_measured'])
 def test_duplicate_rejected(self):
  r=calls();r[-1]['sample_id']='0'
  with self.assertRaises(ValueError):timing(r)
 def test_overlapping_inputs_rejected(self):
  r=calls();r[1]['preprocess_start']=.3
  with self.assertRaises(ValueError):timing(r)
if __name__=='__main__':unittest.main()
