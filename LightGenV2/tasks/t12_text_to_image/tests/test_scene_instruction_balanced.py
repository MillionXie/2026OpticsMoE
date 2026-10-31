import tempfile
from pathlib import Path
import json
import unittest
from PIL import Image
from ..scene_instruction_data import sha,write,PROFILE
from ..scene_instruction_balanced import make_plan,choose_sources


class BalancedCounterfactuals(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
