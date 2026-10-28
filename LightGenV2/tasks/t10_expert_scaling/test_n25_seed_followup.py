import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .n25_seed_followup import (SOURCE_SHAS, PILOT_COMMIT, PILOT_PT, GPU_UUIDS,
    validate, pilot_evidence_record, config_for_seed, Auxiliary, assert_shared_gpu_policy,
    pilot_release_ready, is_original_pilot_command, bind_source_environment)


class N25SeedFollowupTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).parent / 'configs/kather_n25_seed_balanced_followup_20261028.json').read_text())

    def evidence(self):
        selected = dict(origin='adaptation', round=4, state_field='live_model', test_correct=599,
            selected=True, model_tensor_sha256='tensor', predictions_sha256='prediction',
            checkpoint_sha256='last', predictions='test_round4_live_model.npz')
        return dict(passed=True, phase='round4', seed=17, experts=25, top_k=25, parameter_count=5634176,
            source_commit=PILOT_COMMIT, all_manifest_SHA_verified=True, source20_Git_object_SHA_verified=True,
            complete_teacher_logits_softmax_and_parent8views_verified=True,
            calibration_frozen_lambda_actual_ratio_and_restored_Adam_verified=True,
            actual_sampling_source512_each_class64_and_7union2TEST_coverage_verified=True,
            stored_fixed_TEST_752_argmax_confusion_NLL_pairing_verified=True,
            preparation=dict(source_checkpoint_sha256=PILOT_PT, source_test_fit_correct=598, fixed_lambda=.3203496288685833),
            student=dict(baseline_test_fit_correct=598, selected_test_fit_correct=599, completed_rounds=4,
                cpu_best=dict(parameter_count=5634176, selected_state=selected, model_tensor_sha256='tensor',
                    prediction_sha256='prediction', checkpoint_sha256='best',
                    export_origin=dict(state_field='live_model', source_checkpoint_sha256='last')),
                checks=[dict(round=4, field='live_model', test_fit_correct=599, selected_at_check=True,
                    model_tensor_sha256='tensor', source_last_checkpoint_sha256='last')]))

    def test_finite_architecture_and_each_own_seed_source(self):
        c = self.config()
        validate(c)
        for key, value in [('model_experts', 49), ('expected_parameter_count', 11078432),
            ('seeds', [17, 27]), ('rounds', 32), ('fixed_lambda_kd', .5),
            ('requires_pilot_evidence', False), ('automatic_extension', True)]:
            with self.assertRaises(AssertionError):
                validate(dict(c, **{key: value}))
        with patch.dict(os.environ, T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT='a' * 40):
            for seed in [27, 37]:
                value = config_for_seed(c, seed, Path('/followup'), prepared=False)
                self.assertEqual(value['seed'], seed)
                self.assertEqual(value['sources']['student']['sha256'], SOURCE_SHAS[str(seed)])
                self.assertNotEqual(value['parent_checkpoint_sha256'], PILOT_PT)
                self.assertIn(f's{seed}', value['parent_cache'])
                self.assertEqual(value['recipe_gpu']['source_balanced'], 3 if seed == 27 else 1)

    def test_only_locked_genuine_fixed_check_gain_admits_recipe(self):
        c, audit = self.config(), self.evidence()
        record = pilot_evidence_record(audit, c)
        self.assertEqual(record['test_fit_correct'], 599)
        self.assertEqual(record['baseline_test_fit_correct'], 598)
        for mutation in ['fallback', 'NLL_only', 'wrong_tensor', 'wrong_field', 'nonfixed', 'incomplete_history', 'wrong_architecture']:
            bad = copy.deepcopy(audit)
            if mutation == 'fallback':
                bad['student']['cpu_best']['selected_state']['origin'] = 'incumbent'
            elif mutation == 'NLL_only':
                bad['student']['selected_test_fit_correct'] = 598
                bad['student']['cpu_best']['selected_state']['test_correct'] = 598
            elif mutation == 'wrong_tensor':
                bad['student']['cpu_best']['model_tensor_sha256'] = 'foreign'
            elif mutation == 'wrong_field':
                bad['student']['cpu_best']['export_origin']['state_field'] = 'model'
            elif mutation == 'nonfixed':
                bad['student']['cpu_best']['selected_state']['round'] = 5
            elif mutation == 'incomplete_history':
                bad['student']['completed_rounds'] = 3
            else:
                bad['experts'] = 49
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                pilot_evidence_record(bad, c)

    def test_actual_seed_controls_D4_and_calibration_batch_selection(self):
        labels = np.concatenate([np.repeat(np.arange(8), 437), np.repeat(np.arange(8), 94)])
        teacher = np.eye(8)[labels] * .9 + .1 / 8
        parent = teacher.copy()
        parent[::2] = np.eye(8)[(labels[::2] + 1) % 8] * .9 + .1 / 8
        codes, batches = [], []
        for seed in [27, 37]:
            auxiliary = object.__new__(Auxiliary)
            auxiliary.c, auxiliary.y = {'seed': seed}, labels
            auxiliary.teacher, auxiliary.parent = teacher[None], parent[None]
            auxiliary.calibration_view0 = False
            auxiliary.start_round(1)
            np.testing.assert_array_equal(auxiliary.codes, np.random.default_rng(seed + 1000033).integers(0, 8, 4248))
            codes.append(auxiliary.codes)
            selected = auxiliary.calibration_batches()
            self.assertEqual(len(set(np.concatenate(selected))), 32)
            for batch in selected:
                self.assertEqual(int((batch >= 3496).sum()), 8)
                np.testing.assert_array_equal(np.bincount(labels[batch], minlength=8), np.full(8, 2))
            batches.append(selected[0])
        self.assertFalse(np.array_equal(codes[0], codes[1]))
        self.assertFalse(np.array_equal(batches[0], batches[1]))

    def test_shared_gpu_and_complete_pilot_release_are_both_required(self):
        c = self.config()
        foreign = dict(uuid=GPU_UUIDS['1'], free_memory_mib=6000,
            processes=[dict(pid=100, command='python train_yolo.py')])
        assert_shared_gpu_policy(c, 37, foreign, Path('/followup'))
        with self.assertRaises(AssertionError):
            assert_shared_gpu_policy(c, 37, dict(foreign, free_memory_mib=5999), Path('/followup'))
        active = dict(foreign, processes=[dict(pid=101, command='python -m LightGenV2.tasks.t10_expert_scaling.n25_balanced_distill')])
        with self.assertRaises(AssertionError):
            assert_shared_gpu_policy(c, 37, active, Path('/followup'))
        complete = dict(state='complete', running={}, failures={})
        self.assertTrue(pilot_release_ready(complete, {'state': 'complete'}, {'state': 'complete'}, {'children_reaped': True}, [], []))
        self.assertFalse(pilot_release_ready({'state': 'running'}, {'state': 'complete'}, {'state': 'complete'}, {'children_reaped': True}, [], []))
        self.assertFalse(pilot_release_ready(complete, {'state': 'complete'}, {'state': 'complete'}, {'children_reaped': True}, [101], []))
        self.assertFalse(pilot_release_ready(complete, {'state': 'complete'}, {'state': 'complete'}, {'children_reaped': True}, [], [101]))
        with self.assertRaises(RuntimeError):
            pilot_release_ready({'state': 'failed'}, {'state': 'complete'}, {'state': 'complete'}, {'children_reaped': True}, [], [])

    def test_new_published_source_environment_cannot_mix_with_pilot(self):
        keys = ['T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT', 'T10_SEED_FOLLOWUP_SOURCE_COMMIT', 'T10_REHEAT_SOURCE_COMMIT', 'T10_DISTILL_SOURCE_COMMIT']
        clean = {key: value for key, value in os.environ.items() if key not in keys}
        with patch.dict(os.environ, dict(clean, T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT='b' * 40), clear=True):
            self.assertEqual(bind_source_environment(), 'b' * 40)
            self.assertTrue(all(os.environ[key] == 'b' * 40 for key in keys))
        with patch.dict(os.environ, dict(clean, T10_N25_SEED_FOLLOWUP_SOURCE_COMMIT='b' * 40, T10_REHEAT_SOURCE_COMMIT=PILOT_COMMIT), clear=True):
            with self.assertRaises(AssertionError):
                bind_source_environment()

    def test_pilot_PID_identity_handles_relative_launch_and_foreign_reuse(self):
        root = Path('/server/repo/LightGenV2/tasks/t10_expert_scaling/runs/simulation/kather_n25_balanced_distill_20261028')
        module = 'python -m LightGenV2.tasks.t10_expert_scaling.n25_balanced_distill --out '
        self.assertTrue(is_original_pilot_command(module + str(root), root))
        self.assertTrue(is_original_pilot_command(module + 'LightGenV2/tasks/t10_expert_scaling/runs/simulation/' + root.name, root))
        self.assertFalse(is_original_pilot_command('python train_yolo.py', root))
        self.assertFalse(is_original_pilot_command(module + '/some_other_run', root))


if __name__ == '__main__':
    unittest.main()
