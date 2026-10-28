import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ..white_reference_scene_probe import conditions, native_caption, digest
from ..white_reference_metrics import aggregate, load_records, reward_caption


class TargetFreeContractTests(unittest.TestCase):
    def test_old_and_new_light_profiles_separate(self):
        for category in ['lamp','pillow']:
            for old,new in zip(conditions(category),conditions(category,'free_lighting_v2')):
                self.assertIn('light_side',old['requested_attributes'])
                self.assertNotIn('light_side',new['requested_attributes'])
                self.assertNotIn('light',new['prompt'].lower())
                self.assertGreaterEqual(len(new['requested_attributes']['background_elements']),3)
                self.assertTrue(new['prompt'].startswith('Generate'))
                caption=native_caption(category,new,'explicit_caption','free_lighting_v2')
                self.assertNotIn('light',caption.lower())
                self.assertIn('Retain the reference',caption)
            self.assertIn('light_side',conditions(category)[0]['requested_attributes'])

    def test_compact_reward_keeps_major_object_attributes(self):
        self.assertIn('brass base',reward_caption('lamp','0'))
        self.assertIn('broad cream stripes',reward_caption('pillow','1'))
        self.assertNotIn('light',reward_caption('pillow','1'))

    def test_complete_dataset_hash_binding_no_seed_selection(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for i in range(4):
                ref=f'reference_{i:02d}';(root/(ref+'.png')).write_bytes(b'reference')
                for condition in range(2):
                    for seed in [1042,1043]:
                        name=f'{ref}_{condition}_{seed}.png';(root/name).write_bytes(b'output')
                        rows.append({'reference_id':ref,'condition_id':str(condition),'seed':seed,'generated_PNG':name,
                          'generated_sha256':digest(root/name),'input_sha256':digest(root/(ref+'.png'))})
            report={'complete':True,'training_performed':False,'independent_TEST_opened':False,'generated_count':16}
            (root/'report.json').write_text(json.dumps(report));(root/'images.json').write_text(json.dumps(rows))
            self.assertEqual(len(list(load_records(root))),16)
            (root/rows[0]['generated_PNG']).write_bytes(b'changed')
            with self.assertRaises(ValueError):list(load_records(root))
            (root/rows[0]['generated_PNG']).write_bytes(b'output');(root/'images.json').write_text(json.dumps(rows[:8]))
            with self.assertRaises(ValueError):list(load_records(root))

    def test_per_category_summary_and_nonfinite_rejected(self):
        def row(category,v):return dict(run_id='run',category=category,CLIP_T=v,DINOv2_I=v,CLIP_I=v,ImageReward_compact=v)
        summary=aggregate([row('lamp',1),row('pillow',3)])
        self.assertEqual(summary['run/all']['scores']['CLIP_T']['mean'],2)
        self.assertEqual(summary['run/lamp']['n'],1)
        with self.assertRaises(ValueError):aggregate([row('lamp',float('nan'))])


if __name__=='__main__':unittest.main()
