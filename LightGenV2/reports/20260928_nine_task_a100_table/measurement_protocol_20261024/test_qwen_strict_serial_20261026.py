import ast
from pathlib import Path
import unittest
from qwen_strict_serial_20261026 import assert_serial


class StrictSerial(unittest.TestCase):
    def rows(self):
        return [dict(index=i,source_index=i,CPU_start=i*10,CPU_end=i*10+1,H2D_end=i*10+2,
            host_start=i*10+3,host_end=i*10+4,CPU_result_return=i*10+5) for i in range(2)]
    def test_serial(self):assert_serial(self.rows())
    def test_forbid_CPU_lookahead(self):
        rows=self.rows();rows[1]['CPU_start']=4
        with self.assertRaises(AssertionError):assert_serial(rows)
    def test_forbid_unreturned_result(self):
        rows=self.rows();rows[0]['CPU_result_return']=3
        with self.assertRaises(AssertionError):assert_serial(rows)
    def test_native_single_video_loop(self):
        source=Path(__file__).with_name('qwen_strict_serial_20261026.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn('native.prepare_batch([rows[index]]',source)
        self.assertIn('class Timer(BoundaryTimer)',source)
        self.assertNotIn('inputs.append(',source)
        self.assertNotIn('steady_report(',source)


if __name__=='__main__':unittest.main()
