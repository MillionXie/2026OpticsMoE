"""Admission-only tests: no model, data, CUDA, or formal proof generation."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import gpu_numerical_equivalence_20261025 as validation
from run_gpu_checked_missing_20261025 import previous_finished


class Admission(unittest.TestCase):
    def packet(self):
        return dict(status='passed',proof_kind=validation.KIND,no_gpu=False,cuda_initialized=True,gpu_uuid=validation.GPU,
            mps_percent=100,visible_multiprocessors=108,power_limit_w=250,pid=123,
            separate_validation_process_not_a_timed_repeat=True,first_forward_unix=10,last_forward_unix=20,sources={})

    def events(self):
        return [dict(unix=t,clients=[123],servers=[456],processes=[['456',validation.GPU]],
            platform=dict(uuid=validation.GPU,name='NVIDIA A100-PCIE-40GB',power_limit_w=250)) for t in (9,21)]

    def verify(self,packet=None,events=None):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'process_audit.json').write_text(json.dumps(self.events() if events is None else events))
            value=self.packet() if packet is None else packet
            value['process_audit_sha256']=validation.sha(root/'process_audit.json')
            return validation.platform_proof(root/'contract.json',value,False)

    def test_full_guard_accepts_only_actual_gpu_label(self):self.assertEqual(self.verify()['proof_kind'],validation.KIND)

    def test_cpu_label_not_accepted_as_gpu(self):
        with self.assertRaises(ValueError):self.verify(dict(self.packet(),no_gpu=True))

    def test_failed_proof_rejected(self):
        with self.assertRaises(ValueError):self.verify(dict(self.packet(),status='failed'))

    def test_wrong_uuid_rejected(self):
        with self.assertRaises(ValueError):self.verify(dict(self.packet(),gpu_uuid='another GPU'))

    def test_missing_forward_coverage_rejected(self):
        with self.assertRaises(ValueError):self.verify(events=self.events()[:1])

    def test_foreign_client_rejected(self):
        events=self.events();events[0]['clients'].append(999)
        with self.assertRaises(ValueError):self.verify(events=events)

    def test_foreign_process_rejected(self):
        events=self.events();events[0]['processes'].append(['999',validation.GPU])
        with self.assertRaises(ValueError):self.verify(events=events)

    def test_client_disappearance_rejected(self):
        events=self.events();events[1]['clients']=[]
        with self.assertRaises(ValueError):self.verify(events=events)

    def test_wrong_cap_rejected(self):
        events=self.events();events[0]['platform']['power_limit_w']=400
        with self.assertRaises(ValueError):self.verify(events=events)

    def test_ours_must_reuse_its_original_proof(self):
        with self.assertRaises(ValueError):validation.t12_proof(Path('unused'),'ours')

    def test_previous_missing_is_pending(self):
        with tempfile.TemporaryDirectory() as temp:self.assertFalse(previous_finished(Path(temp)))

    def test_live_predecessor_not_treated_as_released(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'suite_status.json').write_text(json.dumps(dict(status='complete',pid=123,mps_released=True)))
            with patch.object(Path,'exists',return_value=True):self.assertFalse(previous_finished(root))


if __name__=='__main__':unittest.main()
