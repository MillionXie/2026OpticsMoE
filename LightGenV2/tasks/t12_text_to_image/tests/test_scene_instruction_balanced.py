import tempfile
from pathlib import Path
import json
import unittest
from PIL import Image
from ..scene_instruction_data import sha,write,PROFILE
from ..scene_instruction_balanced import make_plan,choose_sources,checked_prior_candidates,TEACHER_RECIPE,scene_first_condition,bedside_condition


class BalancedCounterfactuals(unittest.TestCase):
    def test_explicit_bedside_support_is_short_and_has_no_reference_directions(self):
        row=bedside_condition();self.assertEqual(row['prompt'].count('.'),2)
        self.assertLessEqual(len(row['prompt'].split()),24);self.assertIn('bedside table',row['prompt'])
        self.assertNotIn('reference',row['prompt']);self.assertIn('visible bed beside the table',row['required_visible_facts'])
    def test_scene_first_keeps_entities_two_sentences_without_reference_directions(self):
        for category,codes in [('lamp','BDW'),('pillow','SAD')]:
            for code in codes:
                row=scene_first_condition(category,code)
                self.assertEqual(row['prompt'].count('.'),2)
                self.assertLessEqual(len(row['prompt'].split()),24)
                self.assertNotIn('reference',row['prompt'].lower())
                self.assertEqual(row['prompt'],row['native_caption'])
                self.assertIn({'B':'bookshelves','D':'bed','W':'window','S':'sofa','A':'armchair'}[code],row['prompt'])
    def test_every_reference_gets_all_scenes_at_shared_source_seeds(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);data=root/'data';data.mkdir();(data/'references').mkdir();rows=[]
            for i in range(44):
                ref=f'references/{i}.png';Image.new('RGB',(16,16),(i*4,80,120)).save(data/ref)
                rows.append(dict(sample_id=str(i),split='train',reference=ref,reference_sha256=sha(data/ref),source_product_id=str(i),reference_id=str(i),source_RGBA_sha256='r'+str(i),category='lamp'if i<20 else'pillow',prompt='old',seed=1))
            write(data/'pairs.json',rows);write(data/'report.json',dict(complete=True,prompt_profile=PROFILE,TEST_opened=False,pairs_sha256=sha(data/'pairs.json')))
            output=root/'plan';make_plan(data,output);requests=json.loads((output/'requests.json').read_text());self.assertEqual(len(requests),264);self.assertEqual(sum(r['pilot']for r in requests),48)
            for rid in {r['reference_id']for r in requests}:
                group=[r for r in requests if r['reference_id']==rid]
                self.assertEqual(len({r['prompt']for r in group}),3)
                for seed_index in [0,1]:self.assertEqual(len({r['seed']for r in group if r['seed_index']==seed_index}),1)
            self.assertFalse(json.loads((output/'report.json').read_text())['eligible_for_training'])
    def test_view_choice_uses_input_pixels_not_generated_output(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);Image.new('RGB',(16,16),'white').save(root/'empty.png');Image.new('RGB',(16,16),'black').save(root/'visible.png')
            rows=[dict(split='train',source_product_id='same',reference_id=k,reference=k+'.png',reference_sha256=sha(root/(k+'.png')))for k in ['empty','visible']]
            selected,audit=choose_sources(rows,root);self.assertEqual(selected[0]['reference_id'],'visible')
            self.assertIn('input_only_visible_pixel_scores',audit[0])

    def test_expand_copies_pilot_evidence_and_rejects_changed_inputs_or_recipe(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);(root/'references').mkdir();(root/'targets').mkdir()
            Image.new('RGB',(4,4),'white').save(root/'references/ref.png')
            Image.new('RGB',(4,4),'gray').save(root/'targets/image.png')
            rows=[dict(sample_id=str(i),pilot=True,split='train',prompt='public'+str(i),seed=i,
                reference='references/ref.png',reference_sha256=sha(root/'references/ref.png'))for i in range(48)]
            records=[dict(r,target='targets/image.png',target_sha256=sha(root/'targets/image.png'),QA_accepted=False)for r in rows]
            assets=dict(official_revision='frozen',files={});write(root/'model_assets.json',assets);write(root/'candidates.json',records)
            report=dict(complete=True,pilot_only=True,eligible_for_training=False,DEV_or_TEST_forwarded=False,
                schema='scene_balanced_visible_anchor_candidates_v1',plan_sha256='plan',candidate_manifest_sha256=sha(root/'candidates.json'),teacher_recipe=TEACHER_RECIPE)
            write(root/'report.json',report)
            self.assertEqual(len(checked_prior_candidates(root,dict(request_sha256='plan'),rows,assets)),48)
            self.assertFalse(any(r['QA_accepted']for r in records))
            changed=[dict(r)for r in rows];changed[0]['seed']=999
            with self.assertRaises(ValueError):checked_prior_candidates(root,dict(request_sha256='plan'),changed,assets)
            altered=dict(report,teacher_recipe=dict(TEACHER_RECIPE,steps=30));write(root/'report.json',altered)
            with self.assertRaises(ValueError):checked_prior_candidates(root,dict(request_sha256='plan'),rows,assets)
            write(root/'report.json',report);(root/'targets/image.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):checked_prior_candidates(root,dict(request_sha256='plan'),rows,assets)


if __name__=='__main__':unittest.main()
