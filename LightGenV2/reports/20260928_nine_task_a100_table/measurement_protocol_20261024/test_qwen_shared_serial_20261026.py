import ast
from pathlib import Path
import unittest
import tempfile
import json
import hashlib
from unittest.mock import patch
from qwen_shared_serial_20261026 import validate_shared_snapshot
import qwen_shared_serial_20261026 as entry


class SharedSerial(unittest.TestCase):
    def test_other_application_observed_not_rejected(self):
        row=dict(clients=[123],foreign_gpu_processes=[['456','GPU','12000']])
        self.assertEqual(validate_shared_snapshot(row,123),row['foreign_gpu_processes'])
    def test_private_MPS_client_still_rejected(self):
        with self.assertRaises(RuntimeError):validate_shared_snapshot(dict(clients=[123,456],foreign_gpu_processes=[]),123)
    def test_native_strict_worker_reused(self):
        source=Path(__file__).with_name('qwen_shared_serial_20261026.py').read_text(encoding='utf-8');ast.parse(source)
        self.assertIn('strict.worker(args)',source)
        self.assertNotIn('prepare_batch(',source)
        self.assertNotIn('while idle<12',source)
    def test_own_daemon_cleanup_does_not_wait_other_GPU_users(self):
        source=Path(__file__).with_name('qwen_shared_serial_20261026.py').read_text(encoding='utf-8')
        self.assertNotIn('engine.daemon(False)',source)
        self.assertIn('not_uncontended_baseline_performance=True',source)
    def test_worker_guard_retains_cross_process_query_lock(self):
        source=Path(entry.__file__).read_text(encoding='utf-8')
        tree=ast.parse(source)
        guard=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SharedGuard')
        init=next(n for n in guard.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        calls=[n.func.id for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('serialize_mps_commands',calls)
    def test_missing_only_retention_checks_all_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);point=r/'qwen_repeat01';point.mkdir()
            write=lambda p,v:p.write_text(json.dumps(v),encoding='utf8')
            write(r/'suite_status.json',dict(status='stopped',mps_released=True,pid=999999999,completed=['qwen_repeat01']))
            write(point/'status.json',dict(status='complete'))
            write(point/'qwen_timing.json',dict(video_calls=2048,no_prefetch=True,no_prepared_input_reuse=True,records=[]))
            write(point/'identity.json',dict(nonexclusive_authorized=True))
            for i in range(4):(point/f'raw{i}.json').write_text('{}')
            files=[dict(path=p.name,bytes=p.stat().st_size,sha256=entry.base.sha(p)) for p in point.iterdir()]
            manifest=r/'retained.json';write(manifest,dict(source=str(point),verified_files=files))
            with patch.object(entry.strict,'assert_serial'):
                self.assertEqual(entry.retained_plan(r,manifest,entry.base.sha(manifest))[0],(2,3))
                (point/'raw0.json').write_text('changed')
                with self.assertRaises(AssertionError):entry.retained_plan(r,manifest,entry.base.sha(manifest))
    def test_bad_retained_manifest_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);m=r/'manifest.json';m.write_text('{}')
            (r/'suite_status.json').write_text(json.dumps(dict(status='stopped',mps_released=True,pid=999999999)))
            with self.assertRaises(AssertionError):entry.retained_plan(r,m,'0'*64)


if __name__=='__main__':unittest.main()
