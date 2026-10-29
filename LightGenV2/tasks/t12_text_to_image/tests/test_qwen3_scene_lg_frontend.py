import json,tempfile,unittest
from pathlib import Path
from LightGenV2.tasks.t12_text_to_image.qwen3_scene_lg_frontend import input_rows
from LightGenV2.tasks.t12_text_to_image.white_reference_scene_probe import digest


class LGSceneInputTest(unittest.TestCase):
    def test_features_deduplicate_seeds_and_keep_dev_without_target(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            row={'sample_id':'one','reference':'r.png','reference_sha256':'train','prompt':'Generate scene. Keep shape.','split':'train','target':'never_open.png'}
            (p/'pairs.json').write_text(json.dumps([row,dict(row,sample_id='two')]))
            (p/'inference_dev_inputs.json').write_text(json.dumps([dict(row,sample_id='dev',reference_sha256='dev')]))
            (p/'report.json').write_text(json.dumps({'complete':True,'role_disjoint':True,'TEST_opened':False,'pairs_sha256':digest(p/'pairs.json')}))
            rows=input_rows(p);self.assertEqual(len(rows),2)
            self.assertEqual([r['split'] for r in rows],['train','dev'])
            self.assertTrue(all('target' not in r for r in rows))

    def test_shared_rgb_across_roles_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);row={'sample_id':'one','reference':'r.png','reference_sha256':'same','prompt':'Generate. Keep.','split':'train'}
            (p/'pairs.json').write_text(json.dumps([row]));(p/'inference_dev_inputs.json').write_text(json.dumps([dict(row,sample_id='dev')]))
            (p/'report.json').write_text(json.dumps({'complete':True,'role_disjoint':True,'TEST_opened':False,'pairs_sha256':digest(p/'pairs.json')}))
            with self.assertRaises(ValueError):input_rows(p)


if __name__=='__main__':unittest.main()
