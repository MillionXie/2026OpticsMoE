"""CPU-only preservation checks; never imports models, reads assets or uses GPU."""
import ast
import hashlib
import math
import statistics
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'LightGenV2/scripts/profile_latest_qwen_changed_a100.py'
EXPECTED='298956ccd4a792fedf8fbaad6448676bff067270d8b92e1a68dd66804e7fb244'


class HistoricalProfilerTests(unittest.TestCase):
    def test_original_server_source_preserved(self):
        # Git may restore LF text as CRLF on Windows; no semantic edits permitted.
        data=SOURCE.read_bytes().replace(b'\r\n',b'\n')
        self.assertEqual(hashlib.sha256(data).hexdigest(),EXPECTED)
        compile(data,str(SOURCE),'exec')

    def helpers(self):
        tree=ast.parse(SOURCE.read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('percentile','summarize')]
        self.assertEqual(len(nodes),2)
        namespace={'math':math,'statistics':statistics}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
        return namespace

    def test_statistics_on_small_cpu_fixture(self):
        helpers=self.helpers()
        self.assertEqual(helpers['percentile']([3,1,2],.5),2)
        stats=helpers['summarize']([1,2,3])
        self.assertEqual(stats['mean'],2)
        self.assertEqual(stats['minimum'],1)
        self.assertEqual(stats['maximum'],3)

    def test_empty_sequence_refused(self):
        with self.assertRaises(ValueError):
            self.helpers()['summarize']([])


if __name__=='__main__':
    unittest.main()
