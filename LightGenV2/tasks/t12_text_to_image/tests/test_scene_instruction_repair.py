import tempfile
from pathlib import Path
import json
import unittest
from ..scene_instruction_repair import form_teacher_prompt,prepare,FORM_PROFILE
from ..scene_instruction_data import condition,sha,write
from ..scene_generation_data import request_seed


class SourceGroundedRepair(unittest.TestCase):
    def test_form_hint_is_construction_only_and_noise_is_input_derived(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);plan=p/'plan';plan.mkdir();(plan/'ref.png').write_bytes(b'white input')
            source=dict(condition('lamp','D'),reference_id='r',reference='ref.png',reference_sha256=sha(plan/'ref.png'),source_product_id='p',source_RGBA_sha256='rgba',split='train')
            write(plan/'requests.json',[source]);write(plan/'report.json',dict(schema='scene_balanced_visible_anchor_plan_v1',request_sha256=sha(plan/'requests.json')))
            case=dict(reference_id='r',reference_sha256=source['reference_sha256'],anchor_code='D',source_form_hint='articulated_desk')
            spec=p/'spec.json';write(spec,dict(form_labels_from_white_input_only=True,cases=[case]));out=p/'repair';prepare(plan,spec,out)
            rows=json.loads((out/'requests.json').read_text());self.assertEqual(len(rows),2)
            for index,row in enumerate(rows):
                self.assertEqual(row['seed'],request_seed(source['reference_sha256'],index));self.assertIsNone(row['reused_reviewed_label'])
                self.assertEqual(row['prompt_profile'],FORM_PROFILE);self.assertIn('adjustable desk lamp',row['prompt'])
                self.assertNotIn('adjustable',row['public_training_prompt']);self.assertEqual(row['public_training_prompt'],condition('lamp','D')['prompt'])
                self.assertNotIn('target',row)
            self.assertFalse(json.loads((out/'report.json').read_text())['eligible_for_training'])
            write(spec,dict(form_labels_from_white_input_only=True,teacher_image_guidance=1.2,cases=[case]));prepare(plan,spec,p/'lower_guidance')
            lowered=json.loads((p/'lower_guidance/requests.json').read_text());self.assertTrue(all(r['teacher_image_guidance']==1.2 for r in lowered));self.assertEqual([r['seed']for r in lowered],[r['seed']for r in rows])
            case['source_form_hint']='rectangular_cushion';write(spec,dict(form_labels_from_white_input_only=True,cases=[case]))
            with self.assertRaises(ValueError):prepare(plan,spec,p/'wrong_form')
    def test_teacher_wording_is_short_and_has_no_reference_or_identity(self):
        prompt=form_teacher_prompt('lamp','D','articulated_desk');self.assertEqual(prompt.count('.'),2)
        self.assertIn('bedside table',prompt);self.assertNotIn('reference',prompt);self.assertNotIn('SKU',prompt)
    def test_wide_support_is_explicitly_versioned_and_public_prompt_is_unchanged(self):
        original=form_teacher_prompt('pillow','S','rectangular_cushion')
        wide=form_teacher_prompt('pillow','S','rectangular_cushion','wide_support_v2')
        self.assertIn('seat of a two-seat sofa',wide);self.assertIn('whole two-seat sofa',wide)
        self.assertNotEqual(original,wide);self.assertEqual(wide.count('.'),2)
        self.assertNotIn('rectangular',condition('pillow','S')['prompt'])
        self.assertIn('wide solid cylindrical body',form_teacher_prompt('lamp','W','solid_cylinder_lamp','wide_support_v2'))
        self.assertIn('solid rectangular column body',form_teacher_prompt('lamp','D','rectangular_column_lamp','wide_support_v2'))
        self.assertIn('separate loose rectangular throw pillow',form_teacher_prompt('pillow','S','rectangular_cushion','loose_object_v3'))
        with self.assertRaises(ValueError):form_teacher_prompt('lamp','W','column_lamp','loose_object_v3')
        with self.assertRaises(ValueError):form_teacher_prompt('pillow','S','rectangular_cushion','silent_change')


if __name__=='__main__':unittest.main()
