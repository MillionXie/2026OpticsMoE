import json
from pathlib import Path
import tempfile
import unittest
from LightGenV2.tasks.t12_text_to_image.appearance_scene_protocol import SHORT_PROFILE,conditions
from LightGenV2.tasks.t12_text_to_image.scene_development_grid import SAFE,expand,checked_grid
from LightGenV2.tasks.t12_text_to_image.white_reference_scene_probe import digest


class DevelopmentGrid(unittest.TestCase):
    def fixture(self):
        rows=[]
        for category in ['lamp','pillow']:
            for product in range(4):
                for view in range(2):
                    rid=f'{category}-P{product}-{view:02d}'
                    for scene in conditions(category,SHORT_PROFILE)[:2]:
                        row={k:scene[k] for k in ['prompt','condition_id','category','native_caption','reward_caption','prompt_profile']}
                        row.update(sample_id=rid+'-scene'+scene['condition_id']+'-draw2',reference='references/'+rid+'.png',
                            reference_id=rid,reference_sha256=rid+'pixels',seed=product*2+view)
                        rows.append(row)
        return rows

    def test_all_eight_scenes_preserve_original_rows_and_noise(self):
        rows=self.fixture();grid=expand(rows);self.assertEqual(len(grid),128)
        self.assertEqual([r for r in grid if r['condition_id'] in ['0','1']],rows)
        self.assertTrue(all(set(r)==SAFE for r in grid))
        for rid in {r['reference_id'] for r in grid}:
            selected=[r for r in grid if r['reference_id']==rid]
            self.assertEqual({r['condition_id'] for r in selected},set(map(str,range(8))))
            self.assertEqual(len({r['seed'] for r in selected}),1)

    def test_changed_original_noise_or_target_fields_fail(self):
        rows=self.fixture();rows[1]['seed']+=1
        with self.assertRaises(ValueError):expand(rows)
        rows=self.fixture();rows[0]['target']='forbidden.png'
        with self.assertRaises(ValueError):expand(rows)

    def test_grid_must_match_original_DEV_manifest_and_training_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data';grid=root/'grid';data.mkdir();grid.mkdir();rows=self.fixture()
            (data/'inference_dev_inputs.json').write_text(json.dumps(rows));path=grid/'inference_inputs.json';path.write_text(json.dumps(expand(rows)))
            report=dict(complete=True,schema='scene_expanded_development_v1',TEST_opened=False,
                source_DEV_request_sha256=digest(data/'inference_dev_inputs.json'),pairs_sha256='paired',request_sha256=digest(path))
            (grid/'report.json').write_text(json.dumps(report));state=dict(pair_sha256='paired')
            self.assertEqual(len(checked_grid(grid,data,state,rows)[0]),128)
            with self.assertRaises(ValueError):checked_grid(grid,data,dict(pair_sha256='other-data'),rows)
            changed=expand(rows);changed[0]['reference_id']='test-subject';path.write_text(json.dumps(changed));report['request_sha256']=digest(path)
            (grid/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):checked_grid(grid,data,state,rows)

    def test_metrics_require_all_128_and_no_TRAIN_rows(self):
        from LightGenV2.tasks.t12_text_to_image.white_reference_metrics import load_records
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'generated.png').write_bytes(b'generated')
            rows=expand(self.fixture())
            for row in rows:
                path=root/(row['reference_id']+'.png');path.write_bytes(b'original DEV reference')
                row.update(split='dev',generated_PNG='generated.png',generated_sha256=digest(root/'generated.png'),input_sha256=digest(path))
            report=dict(complete=True,training_performed=False,independent_TEST_opened=False,scope='DEV_expanded',generated_count=128,
                expanded_DEV_protocol=dict(manifest_sha256='canonical-grid'))
            (root/'report.json').write_text(json.dumps(report));(root/'images.json').write_text(json.dumps(rows))
            self.assertEqual(len(list(load_records(root))),128)
            rows[0]['split']='train';(root/'images.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):list(load_records(root))
            rows[0]['split']='dev';(root/'images.json').write_text(json.dumps(rows[:-1]))
            with self.assertRaises(ValueError):list(load_records(root))


if __name__=='__main__':unittest.main()
