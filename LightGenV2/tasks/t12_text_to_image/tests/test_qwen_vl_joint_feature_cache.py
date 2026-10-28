import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from ..qwen_vl_joint_feature_cache import (
    CACHE_SCHEMA, ORDER, frontend_input_fingerprint, input_fingerprint,
    joint_adapter_digest, validate_frozen_joint_cache, _target_fingerprint, sha256,
    frozen_frontend_checkpoint_sha256)


class FrozenJointCacheTests(unittest.TestCase):
    def test_frontend_identity_survives_frozen_head_continuation(self):
        self.assertEqual(frozen_frontend_checkpoint_sha256({'config': {}}, 'joint_teacher'), 'joint_teacher')
        continued = {'config': {'freeze_joint_qwen': True,
                               'frozen_joint_teacher_checkpoint_sha256': 'joint_teacher'}}
        self.assertEqual(frozen_frontend_checkpoint_sha256(continued, 'new_head_best'), 'joint_teacher')
        continued['config']['frozen_joint_teacher_checkpoint_sha256'] = None
        with self.assertRaises(ValueError):
            frozen_frontend_checkpoint_sha256(continued, 'new_head_best')

    def test_forward_fingerprint_excludes_supervision_but_binds_rgb_and_prompt(self):
        payload = {'reference': torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
                   'target': torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
                   'rows': [{'prompt': 'Make the wall blue.', 'teacher_caption': 'blue target', 'noise_seed': 1}],
                   'frontend_input_order': ORDER}
        initial = frontend_input_fingerprint(payload, 0)
        payload['target'] += 255
        payload['rows'][0].update(teacher_caption='different target', noise_seed=99, target={'product_id': 'secret'})
        self.assertEqual(frontend_input_fingerprint(payload, 0), initial)
        payload['reference'][0, 0, 0, 0] += 1
        self.assertNotEqual(frontend_input_fingerprint(payload, 0), initial)
        payload['reference'].zero_(); payload['rows'][0]['prompt'] = 'Make the wall red.'
        self.assertNotEqual(frontend_input_fingerprint(payload, 0), initial)
        with self.assertRaises(ValueError):
            input_fingerprint(payload['reference'][0], 'Make the wall blue.', 'image_first')
        with self.assertRaises(ValueError):
            input_fingerprint(payload['reference'][0].float(), 'Make the wall blue.')

    def test_joint_digest_binds_both_modalities_dtype_names_and_exact_values(self):
        state = {'vision.lora_A': torch.arange(6.).reshape(2, 3),
                 'language.lora_B': torch.ones(3, 2, dtype=torch.bfloat16)}
        original = joint_adapter_digest(state)
        self.assertEqual(joint_adapter_digest(dict(reversed(list(state.items())))), original)
        altered = {k: v.clone() for k, v in state.items()}
        altered['vision.lora_A'][0, 0] += .1
        self.assertNotEqual(joint_adapter_digest(altered), original)
        altered = {k: v.clone() for k, v in state.items()}
        altered['language.lora_B'][0, 0] += .125
        self.assertNotEqual(joint_adapter_digest(altered), original)
        self.assertNotEqual(joint_adapter_digest({k: v.float() for k, v in state.items()}), original)
        self.assertNotEqual(joint_adapter_digest({'renamed': state['vision.lora_A'], 'language.lora_B': state['language.lora_B']}), original)
        with self.assertRaises(ValueError): joint_adapter_digest({})
        with self.assertRaises(ValueError): joint_adapter_digest({'bad': torch.tensor(float('nan'))})

    def test_stale_parent_and_adapters_are_rejected_before_any_cache_is_used(self):
        adapters = {'vision.lora_A': torch.ones(2, 3), 'language.lora_A': torch.ones(2, 3)}
        state = {'schema': 'qwen_spatial_joint_lora_v1', 'qwen_joint_lora': adapters}
        payload = {'frozen_joint_teacher': {'schema': CACHE_SCHEMA, 'parent_checkpoint_sha256': 'selected',
                   'joint_adapter_sha256': joint_adapter_digest(adapters)}}
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            validate_frozen_joint_cache(payload, state, 'other')
        state['qwen_joint_lora']['language.lora_A'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'adapters differ'):
            validate_frozen_joint_cache(payload, state, 'selected')
        with self.assertRaisesRegex(ValueError, 'joint-adapted teacher'):
            validate_frozen_joint_cache(payload, {'schema': 'public_Qwen'}, 'selected')

    def test_validator_binds_input_and_supervision_separately_and_train_only_clip(self):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        mask_path = Path(temporary.name)/'mask.png'; Image.new('L', (256, 256), 255).save(mask_path)
        mask_sha = sha256(mask_path)
        adapters = {'vision.lora_A': torch.ones(2, 3), 'language.lora_A': torch.ones(2, 3)}
        state = {'schema': 'qwen_spatial_joint_lora_v1', 'qwen_joint_lora': adapters,
                 'config': {'native_input_order': ORDER}, 'visual_layout': {'native_input_order': ORDER}}
        rows = []
        for split in ('train', 'val'):
            def product(role):
                return {'product_id': split+role, 'family_id': split+role+'-family',
                        'rgba_sha256': split+role+'-pixels'}
            rows.append({'sample_id': split, 'split': split, 'category': 'lamp', 'mode': 'object',
                         'render_protocol': 'controlled_room_edit_v1', 'prompt': 'Use a cylindrical lamp.',
                         'edit_mask_path': str(mask_path), 'edit_mask_sha256': mask_sha,
                         'source': product('source'), 'target': product('target')})
        payload = {'rows': rows, 'reference': torch.zeros(2, 3, 256, 256, dtype=torch.uint8),
                   'target': torch.ones(2, 3, 256, 256, dtype=torch.uint8), 'frontend_input_order': ORDER,
                   'hidden': [torch.ones(80, 2048, dtype=torch.bfloat16) for _ in rows],
                   'train_clip_teacher': {0: torch.zeros(77, 1024, dtype=torch.bfloat16)},
                   'qwen_checks': {'vision_blocks': 24, 'language_blocks': 28, 'all_qwen_frozen': True}}
        meta = {'schema': CACHE_SCHEMA, 'parent_checkpoint_sha256': 'selected',
                'joint_adapter_sha256': joint_adapter_digest(adapters), 'frontend_input_order': ORDER,
                'all_qwen_frozen': True, 'feature_scope': 'final_fused_head_training_only',
                'LG_intermediate_cache': False, 'independent_TEST_opened': False,
                'sample_ids': ['train', 'val'], 'raw_hidden_recompute_passed': True,
                'raw_hidden_recompute_cells': 6,
                'input_fingerprints': [frontend_input_fingerprint(payload, i) for i in range(2)],
                'target_fingerprints': [_target_fingerprint(v) for v in payload['target']],
                'edit_mask_sha256': [mask_sha, mask_sha]}
        payload['frozen_joint_teacher'] = meta
        # Small synthetic budget exercises the production validation path;
        # the CLI always enforces the real 960/96 before constructing a cache.
        with patch('LightGenV2.tasks.t12_text_to_image.qwen_vl_joint_feature_cache.COUNTS', {'train': 1, 'val': 1}):
            self.assertIs(validate_frozen_joint_cache(payload, state, 'selected'), meta)
            payload['reference'][0, 0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, 'actual RGB'):
                validate_frozen_joint_cache(payload, state, 'selected')
            payload['reference'].zero_(); payload['target'][0, 0, 0, 0] = 2
            with self.assertRaisesRegex(ValueError, 'supervision changed'):
                validate_frozen_joint_cache(payload, state, 'selected')
            payload['target'].fill_(1); payload['train_clip_teacher'][1] = payload['train_clip_teacher'][0]
            with self.assertRaisesRegex(ValueError, 'TRAIN only'):
                validate_frozen_joint_cache(payload, state, 'selected')
            del payload['train_clip_teacher'][1]; mask_path.write_bytes(b'changed supervision')
            with self.assertRaisesRegex(ValueError, 'edit-mask supervision changed'):
                validate_frozen_joint_cache(payload, state, 'selected')


if __name__ == '__main__':
    unittest.main()
