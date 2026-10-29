import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_paired_train import checked_pairs
from LightGenV2.tasks.t12_text_to_image.scene_paired_render import inference_inputs
from LightGenV2.tasks.t12_text_to_image.white_reference_scene_probe import digest


class PairedGuardsTest(unittest.TestCase):
    def test_render_inputs_have_no_target_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            row={k:'input' for k in ['sample_id','reference','prompt','seed','reference_id','reference_sha256',
                                     'category','condition_id','native_caption','reward_caption','prompt_profile']}
            path=root/'inference_dev_inputs.json';path.write_text(json.dumps([row]))
            self.assertEqual(inference_inputs(root),[row])
            row['target']='not_allowed.png';path.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):inference_inputs(root)

    def test_paired_loader_rejects_duplicate_subject_across_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'reference.png').write_bytes(b'reference');(root/'target.png').write_bytes(b'target')
            row={'source_product_id':'one','reference':'reference.png','target':'target.png',
                 'reference_sha256':digest(root/'reference.png'),'target_sha256':digest(root/'target.png'),
                 'source_RGBA_sha256':'same_original'}
            rows=[dict(row,split='train'),dict(row,split='dev')]
            (root/'pairs.json').write_text(json.dumps(rows))
            (root/'report.json').write_text(json.dumps({'complete':True,'role_disjoint':True,'TEST_opened':False}))
            (root/'visual_review.json').write_text(json.dumps({'approved_for_training':True,'pairs_sha256':digest(root/'pairs.json')}))
            with self.assertRaises(ValueError):checked_pairs(SimpleNamespace(data=root))

    def test_test_records_never_enter_training_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'pairs.json').write_text(json.dumps([{'split':'test'}]))
            (root/'report.json').write_text(json.dumps({'complete':True,'role_disjoint':True,'TEST_opened':False}))
            (root/'visual_review.json').write_text(json.dumps({'approved_for_training':True,'pairs_sha256':digest(root/'pairs.json')}))
            with self.assertRaises(AssertionError):checked_pairs(SimpleNamespace(data=root))


if __name__=='__main__':unittest.main()
