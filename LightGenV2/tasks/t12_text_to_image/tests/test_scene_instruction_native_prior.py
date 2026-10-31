import tempfile
import unittest
from pathlib import Path
from PIL import Image
from ..scene_instruction_native_prior import checked_requests,descriptive_equivalent
from ..scene_instruction_data import SAFE,sha,write
from ..scene_instruction_balanced import scene_first_condition


class NativeFeasibility(unittest.TestCase):
    def test_native_caption_changes_format_only_without_extra_entities(self):
        for category,codes in [('lamp','BDW'),('pillow','SAD')]:
            for code in codes:
                row=scene_first_condition(category,code);original=row['prompt'];caption=descriptive_equivalent(row)
                self.assertEqual(row['prompt'],original);self.assertTrue(caption.startswith('An indoor photograph'))
                self.assertNotIn('reference',caption);self.assertNotIn('red',caption);self.assertNotIn('sunlight',caption)
                self.assertIn({'B':'bookshelves','D':'bed','W':'window','S':'sofa','A':'armchair'}[code],caption)
    def test_full_TRAIN_cohort_only_and_no_target_inputs(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);rows=[]
            for i in range(8):
                filename=f'{i}.png';Image.new('RGB',(4,4),(i*20,80,90)).save(p/filename)
                for code in 'BDW':
                    rows.append(dict(scene_first_condition('lamp',code),sample_id=str(i)+code,reference_id=str(i),reference=filename,
                        reference_sha256=sha(p/filename),seed=17,split='train',target='never_a_generator_input'))
            write(p/'requests.json',rows);report=dict(complete=True,schema='scene_balanced_scene_first_pilot_plan_v1',request_sha256=sha(p/'requests.json'));write(p/'report.json',report)
            actual=checked_requests(p);self.assertEqual(len(actual),24);self.assertTrue(all(set(r)==set(SAFE)for r in actual))
            rows[0]['split']='test';write(p/'requests.json',rows);report['request_sha256']=sha(p/'requests.json');write(p/'report.json',report)
            with self.assertRaises(ValueError):checked_requests(p)


if __name__=='__main__':unittest.main()
