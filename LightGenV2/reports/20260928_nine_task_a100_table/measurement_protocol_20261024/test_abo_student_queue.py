"""CPU-only queue control contracts; no private assets, Torch, CUDA or SSH."""
import ast
import hashlib
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
# Windows has no fcntl; extract only pure functions, never emulate locking.
tree=ast.parse((HERE/'run_abo_student_contract.py').read_text(encoding='utf8'))
scope={'HERE':HERE,'REPO':HERE.parents[3],'TASK':HERE.parents[3]/'LightGenV2/tasks/t08_abo_image_text_retrieval',
    '__file__':str(HERE/'run_abo_student_contract.py'),'Path':Path,'hashlib':hashlib}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
    and n.name in ('predecessor_action','check_result','source_hashes')],type_ignores=[]),'pure_queue_checks','exec'),scope)


class QueueContracts(unittest.TestCase):
    def test_waits_unchanged_occupied_predecessor(self):
        self.assertEqual(scope['predecessor_action']({'status':'waiting_foreign_process'}),'wait')

    def test_admits_only_complete_new_round(self):
        self.assertEqual(scope['predecessor_action']({'status':'complete','completed':['qwen_repeat03']}),'admit')

    def test_rejects_fabricated_predecessor_complete(self):
        with self.assertRaises(ValueError):scope['predecessor_action']({'status':'complete','completed':[]})

    def test_never_bypasses_stopped_round(self):
        with self.assertRaises(RuntimeError):scope['predecessor_action']({'status':'stopped'})

    def result(self):
        return dict(status='passed',no_gpu=True,mode='real_title_original_full_student_CPU_bitwise',
            checks=[dict(title_index=i,bitwise_equal=True) for i in (0,99)],
            identity=dict(body_sha256='cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'))

    def test_real_checks_required(self):
        scope['check_result'](self.result())
        result=self.result();result['checks'][1]['bitwise_equal']=False
        with self.assertRaises(ValueError):scope['check_result'](result)

    def test_wrong_body_not_accepted(self):
        result=self.result();result['identity']['body_sha256']='other'
        with self.assertRaises(ValueError):scope['check_result'](result)

    def test_no_torch_or_mps_import_at_queue_admission(self):
        imports=[n.module or '' for n in tree.body if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(n.startswith(('torch','transformers')) for n in imports))
        self.assertNotIn('daemon(True)',ast.unparse(tree))

    def test_all_frozen_source_paths_exist_before_launch(self):
        hashes=scope['source_hashes']()
        self.assertGreaterEqual(len(hashes),14)
        self.assertTrue(all(len(value)==64 for value in hashes.values()))


if __name__=='__main__':unittest.main()
