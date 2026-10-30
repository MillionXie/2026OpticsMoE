"""Pure CPU DAG/rate contracts, no Torch, video decoding or GPU."""
import unittest,math
from recalculate_lg_front1886_20261030 import optical_s,calculate,nodes_for
def fixture():
 stages=['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual','merger','adapter_language','decoder']
 stages += [f'optics{i}' for i in range(1,7)]+[f'oeo{i}' for i in range(1,7)]
 report=dict(fields=1,raw_phases=[dict(field=0,phase=k,charged_compute_s=.001) for k in stages])
 patch=[dict(field=0,slot=i,worker_thread=i,sample_id=str(i),resize_cpu_s=.001,normalization_cpu_s=.002,patch_cpu_s=.001) for i in range(16)]
 raw=dict(patch=patch,tokenizer=[dict(thread_cpu_s=.001)])
 front={str(i):dict(decode_backend_cpu_s=.01,RGB_cpu_s=.002,crop_metadata_cpu_s=.001,quality_cpu_s=.005) for i in range(16)}
 return report,raw,front
class Contracts(unittest.TestCase):
 def test_only_SLM_period_changes(self):
  self.assertAlmostEqual(optical_s(1886),1/1886+.000300+.0000307)
  self.assertAlmostEqual(optical_s(1400)-optical_s(1886),1/1400-1/1886)
  with self.assertRaises(ValueError):optical_s(0)
 def test_added_CPU_work_increases_time_and_closes(self):
  r,w,f=fixture();a=calculate(r,w,1886);b=calculate(r,w,1886,f)
  self.assertGreater(b['ms16'],a['ms16']);self.assertAlmostEqual(sum(b['stage_critical_contribution_ms16'].values()),b['ms16'])
 def test_no_extra_online_CPU_resources(self):
  r,w,f=fixture();a=nodes_for(r,w,1886);b=nodes_for(r,w,1886,f)
  self.assertEqual({n['resource'] for n in a},{n['resource'] for n in b})
  self.assertEqual(sum(n['phase']=='front_decode_RGB_crop' for n in b),16)
  self.assertEqual(sum(n['phase']=='quality_pixel_preparation' for n in b),16)
 def test_faster_refresh_not_slower_for_fixed_fixture(self):
  r,w,f=fixture();self.assertLess(calculate(r,w,1886,f)['ms16'],calculate(r,w,1400,f)['ms16'])
if __name__=='__main__':unittest.main()
