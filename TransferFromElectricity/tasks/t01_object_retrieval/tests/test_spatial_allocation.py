import json
from pathlib import Path
import tempfile
import unittest

from TransferFromElectricity.tasks.t01_object_retrieval.allocate_spatial_dataset import reserved_gpus


class AllocationTests(unittest.TestCase):
    def test_completed_child_does_not_release_gpu_while_suite_is_pending(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);path=root/'trial_caltech_execution.json'
            path.write_text(json.dumps({'complete':False,'records':[{'status':'complete','gpu_uuid':'GPU-one'}]}))
            self.assertEqual(reserved_gpus(root,'trial'),{'GPU-one'})
            path.write_text(json.dumps({'complete':True,'records':[{'status':'complete','gpu_uuid':'GPU-one'}]}))
            self.assertEqual(reserved_gpus(root,'trial'),set())

    def test_partial_json_write_blocks_allocation_until_readable(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'trial_cifar_execution.json').write_text('{')
            self.assertIsNone(reserved_gpus(root,'trial'))


if __name__=='__main__':unittest.main()
