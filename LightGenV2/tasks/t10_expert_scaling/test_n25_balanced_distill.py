import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .n25_balanced_distill import (SOURCE_SHA, GPU_UUID, validate, config_for_seed,
    assert_shared_gpu_policy, bind_source_environment)
from .expert_tail_schedule import sha


class N25BalancedTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parent / 'configs/kather_n25_balanced_distill_20261028.json').read_text())

    def test_optical_identity_frozen_budget_and_own_source(self):
        c = self.config()
        validate(c)
        for key, value in [('model_experts', 49), ('expected_parameter_count', 11078432),
                           ('rounds', 32), ('fixed_lambda_kd', .5), ('seeds', [17, 27]),
                           ('automatic_extension', True), ('automatic_additional_seeds', True)]:
            with self.assertRaises(AssertionError):
                validate(dict(c, **{key: value}))
        bad = copy.deepcopy(c)
        bad['source_runs']['17']['sha256'] = '2ee3bc9b738d1ad1bdb785d01288a0dbdf541955b7186570afa640451956bad2'
        with self.assertRaises(AssertionError):
            validate(bad)

    def test_reference_and_parent_are_n25_source_bound(self):
        c = self.config()
        c['_profile'] = 'profile.json'
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, T10_FULL_EXPERT_SOURCE_COMMIT='a' * 40):
            root = Path(folder)
            v = config_for_seed(c, 17, root, prepared=False)
            self.assertEqual(v['sources']['student']['sha256'], SOURCE_SHA)
            self.assertEqual(v['sources']['student']['run'], c['source_runs']['17']['run'])
            self.assertEqual(v['parent_checkpoint_sha256'], SOURCE_SHA)
            self.assertEqual(v['model_experts'], 25)
            self.assertEqual(Path(v['parent_cache']), root / 'parent_s17')
            (root / 'reference_s17').mkdir()
            (root / 'parent_s17').mkdir()
            receipt = root / 'parent_s17/receipt.json'
            receipt.write_text('{}')
            preparation = dict(seed=17, source_checkpoint_sha256=SOURCE_SHA, source_commit='a' * 40,
                parent_receipt_sha256=sha(receipt), predictions_sha256='prediction')
            target = root / 'reference_s17/preparation_receipt.json'
            target.write_text(json.dumps(preparation))
            prepared = config_for_seed(c, 17, root)
            self.assertEqual(prepared['sources']['student']['run'], str(root / 'reference_s17'))
            self.assertEqual(prepared['incumbent_predictions_sha256'], 'prediction')
            preparation['source_checkpoint_sha256'] = 'foreign'
            target.write_text(json.dumps(preparation))
            with self.assertRaises(AssertionError):
                config_for_seed(c, 17, root)

    def test_shared_gpu_allows_foreign_but_refuses_capacity_or_duplicate(self):
        c = self.config()
        root = Path('/runs/n25_pilot')
        foreign = [dict(pid=100, gpu_uuid=GPU_UUID, command='python train_yolo.py')]
        assert_shared_gpu_policy(c, GPU_UUID, 6000, foreign, root)
        assert_shared_gpu_policy(c, GPU_UUID, 2000, foreign, root, require_capacity=False)
        with self.assertRaises(AssertionError):
            assert_shared_gpu_policy(c, GPU_UUID, 5999, foreign, root)
        with self.assertRaises(AssertionError):
            assert_shared_gpu_policy(c, 'wrong-gpu', 20000, foreign, root)
        for command in ['python -m LightGenV2.tasks.t10_expert_scaling.n25_balanced_distill --mode worker',
                        'python worker.py --out /runs/n25_pilot/student_s17']:
            with self.assertRaises(AssertionError):
                assert_shared_gpu_policy(c, GPU_UUID, 20000, [dict(pid=101, gpu_uuid=GPU_UUID, command=command)], root)

    def test_source_environment_cannot_mix_published_versions(self):
        keys = ['T10_FULL_EXPERT_SOURCE_COMMIT', 'T10_SEED_FOLLOWUP_SOURCE_COMMIT',
                'T10_REHEAT_SOURCE_COMMIT', 'T10_DISTILL_SOURCE_COMMIT']
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with patch.dict(os.environ, dict(clean, T10_FULL_EXPERT_SOURCE_COMMIT='a' * 40), clear=True):
            self.assertEqual(bind_source_environment(), 'a' * 40)
            self.assertTrue(all(os.environ[k] == 'a' * 40 for k in keys))
        with patch.dict(os.environ, dict(clean, T10_FULL_EXPERT_SOURCE_COMMIT='a' * 40,
                                        T10_REHEAT_SOURCE_COMMIT='b' * 40), clear=True):
            with self.assertRaises(AssertionError):
                bind_source_environment()


if __name__ == '__main__':
    unittest.main()
