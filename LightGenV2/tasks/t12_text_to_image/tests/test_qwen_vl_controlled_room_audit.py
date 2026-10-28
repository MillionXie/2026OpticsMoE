import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from ..qwen_vl_controlled_room_audit import (
    exposed_wall_roi, family_prototypes, family_result,
    preservation_result, room_background, wall_palette_result,
    reviewed_dev_rows, COUNTS, PROTOCOL, write_json, sha256)


class ControlledRoomAuditTests(unittest.TestCase):
    def test_family_prototypes_use_generic_families_and_conservative_ties(self):
        families, texts = family_prototypes('lamp')
        self.assertEqual(families, ['cylindrical', 'tapered'])
        self.assertTrue(all('shade' in value for value in texts))
        self.assertTrue(family_result([.8, .2], 'lamp', 'cylindrical')['strict_top1'])
        self.assertFalse(family_result([.2, .8], 'lamp', 'cylindrical')['strict_top1'])
        tied = family_result([.5, .5], 'lamp', 'cylindrical')
        self.assertFalse(tied['strict_top1']); self.assertIsNone(tied['predicted_family'])
        self.assertEqual(family_prototypes('pillow')[0], ['striped', 'geometric'])
        with self.assertRaises(ValueError): family_result([.3, .4], 'pillow', 'brand-SKU')
        with self.assertRaises(ValueError): family_result([float('nan'), .4], 'lamp', 'tapered')

    def test_palette_classifier_uses_exact_light_field_and_ignores_excluded_pixels(self):
        blank = np.zeros((256, 256), dtype=np.float32)
        roi = exposed_wall_roi(blank, blank)
        image = np.asarray(room_background('pale_blue')).copy()
        self.assertTrue(wall_palette_result(image, roi, 'pale_blue')['strict_top1'])
        self.assertEqual(wall_palette_result(image, roi, 'pale_blue')['expected_distance_RGB_01'], 0)
        image[~roi] = (255, 0, 0)
        actual = wall_palette_result(image, roi, 'pale_blue')
        self.assertTrue(actual['strict_top1'])
        self.assertEqual(actual['predicted_palette'], 'pale_blue')
        self.assertFalse(wall_palette_result(image, roi, 'sage_green')['strict_top1'])

    def test_wall_roi_excludes_window_table_and_both_product_supports(self):
        source = np.zeros((256, 256), dtype=np.float32)
        target = source.copy(); source[70:90, 100:120] = 1; target[90:110, 120:140] = .02
        roi = exposed_wall_roi(source, target)
        self.assertFalse(roi[80, 110]); self.assertFalse(roi[100, 130])
        self.assertFalse(roi[40, 30]); self.assertFalse(roi[220, 200])
        self.assertTrue(roi[20, 220])
        with self.assertRaises(ValueError): exposed_wall_roi(np.ones_like(source), target)
        with self.assertRaises(ValueError): wall_palette_result(np.zeros((256, 256, 3), dtype=np.uint8), np.zeros_like(roi), 'light_gray')

    def test_retained_region_excludes_edit_and_rejects_missing_changed_pixels(self):
        reference = np.zeros((256, 256, 3), dtype=np.uint8)
        target = reference.copy(); target[30:60, 30:60] = 255
        edit = np.zeros((256, 256), dtype=np.uint8); edit[30:60, 30:60] = 255
        result = preservation_result(target, reference, target, edit)
        self.assertEqual(result['L1_RGB_01'], 0); self.assertEqual(result['PSNR_dB'], 120)
        prediction = target.copy(); prediction[0, 0] = 255
        result = preservation_result(prediction, reference, target, edit)
        self.assertGreater(result['L1_RGB_01'], 0); self.assertLess(result['PSNR_dB'], 120)
        missing = edit.copy(); missing[30, 30] = 0
        with self.assertRaisesRegex(ValueError, 'missing'):
            preservation_result(target, reference, target, missing)
        extra = edit.copy(); extra[0, 0] = 255
        with self.assertRaisesRegex(ValueError, 'exact changed-pixel'):
            preservation_result(target, reference, target, extra)

    def test_native_resolution_and_expected_palette_are_enforced(self):
        roi = np.ones((256, 256), dtype=bool)
        with self.assertRaises(ValueError): wall_palette_result(np.zeros((128, 128, 3), dtype=np.uint8), roi, 'light_gray')
        with self.assertRaises(ValueError): wall_palette_result(np.zeros((256, 256, 3), dtype=np.uint8), roi, 'unknown')

    def test_finalized_dev_identity_and_prompt_guards_reject_wrong_cohorts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); data = root/'data'; review = root/'review'
            data.mkdir(); review.mkdir(); rows = []; records = []
            for category in ('lamp', 'pillow'):
                for mode in ('background', 'object', 'joint'):
                    for number in range(16):
                        row = {'sample_id': f'val:{category}:{mode}:{number}', 'split': 'val',
                               'render_protocol': PROTOCOL, 'category': category, 'mode': mode,
                               'prompt': f'Edit {category} {mode}.',
                               'requested_family': None if mode == 'background' else family_prototypes(category)[0][0]}
                        rows.append(row)
                        records.append({**row, 'object_crop_box': [1, 2, 32, 35]})
            write_json(data/'annotated_pairs.json', rows)
            write_json(data/'data_audit.json', {'protocol': PROTOCOL, 'test_opened': False,
                'role_disjoint': True, 'expected_counts': COUNTS})
            write_json(data/'report.json', {'complete': True, 'pair_sha256': sha256(data/'annotated_pairs.json')})
            report = {'complete': True, 'cohort': 'development', 'evaluation_label': 'development_test',
                      'independent_test_opened': False, 'metrics': {'images': records}}
            write_json(review/'report.json', report)
            # Small metadata fixture isolates this audit's identity checks;
            # production still calls the real full TRAIN/DEV role validator.
            with patch('LightGenV2.tasks.t12_text_to_image.qwen_vl_controlled_room_audit.validate_roles',
                       return_value={'pairs': COUNTS}):
                self.assertEqual(len(reviewed_dev_rows(data, review)[0]), 96)
                report['independent_test_opened'] = True
                write_json(review/'report.json', report)
                with self.assertRaisesRegex(ValueError, 'finalized raw DEV96'):
                    reviewed_dev_rows(data, review)
                report['independent_test_opened'] = False; records[0]['prompt'] = 'Different instruction.'
                write_json(review/'report.json', report)
                with self.assertRaisesRegex(ValueError, 'prompts'):
                    reviewed_dev_rows(data, review)


if __name__ == '__main__':
    unittest.main()
