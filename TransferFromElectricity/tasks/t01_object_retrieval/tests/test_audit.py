import unittest
from unittest.mock import patch
from TransferFromElectricity.tasks.t01_object_retrieval.launch_rtx import validate_device
from TransferFromElectricity.tasks.t01_object_retrieval.source_audit import audit_sources


class AuditTests(unittest.TestCase):
    def test_rejects_ordinal_and_a100(self):
        inventory = [['GPU-rtx', ' NVIDIA GeForce RTX 3090'], ['GPU-a100', ' NVIDIA A100-PCIE-40GB']]
        self.assertEqual(validate_device('GPU-rtx', inventory), 'NVIDIA GeForce RTX 3090')
        for device in ['4', 'GPU-a100', 'GPU-missing']:
            with self.assertRaises(ValueError):
                validate_device(device, inventory)

    def test_source_comparison_rejects_training_change(self):
        path = 'TransferFromElectricity/tasks/t01_object_retrieval/train_staged.py'
        with patch('subprocess.check_output', side_effect=[f'100644 blob a\t{path}', f'100644 blob b\t{path}']):
            with self.assertRaises(ValueError):
                audit_sources(['first', 'second'])


if __name__ == '__main__':
    unittest.main()
