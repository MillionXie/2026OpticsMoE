import ast,importlib.util,unittest
from pathlib import Path
P=Path(__file__).with_name('rental_corrected_serial_20261026.py')
spec=importlib.util.spec_from_file_location('rental_corrected',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Contract(unittest.TestCase):
 def test_union_nested_no_double_count(self):self.assertAlmostEqual(m.union_ms([dict(start=1,end=3),dict(start=2,end=2.5),dict(start=4,end=5)]),3000)
 def test_corrected_serial_definition(self):
  old=m.N;m.N=2
  try:
   rows=[dict(CPU_start=0,H2D_end=1,GPU_ready_host=2,tokenizer_intervals=[dict(start=.1,end=.2)]),dict(CPU_start=2.1,H2D_end=3,GPU_ready_host=4,tokenizer_intervals=[])]
   corrected,removed,full=m.account(rows);self.assertAlmostEqual(corrected,2200);self.assertAlmostEqual(removed,1800);self.assertAlmostEqual(full,4000)
  finally:m.N=old
 def test_original_compute_and_one_round(self):
  s=P.read_text();self.assertIn('mos,logits=compute(inputs)',s);self.assertIn("plan=['kimi','deepseek']",s);self.assertIn("no_prefetch=True",s);self.assertNotIn('torch.cuda.CUDAGraph',s);self.assertIn('props.multi_processor_count==54',s)
if __name__=='__main__':unittest.main()
