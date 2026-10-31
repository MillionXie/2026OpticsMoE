import tempfile
from pathlib import Path
import unittest
from ..scene_instruction_data import sha,write,PROFILE
from ..scene_instruction_delivery import checked_run,checked_teacher_syntax
from ..scene_instruction_balanced import scene_first_condition,SCENE_FIRST_PROFILE,TEACHER_RECIPE


class InstructionDelivery(unittest.TestCase):
    def fixture(self,path):
        (path/'one.png').write_bytes(b'pixels');rows=[]
        for i in range(48):rows.append(dict(sample_id=str(i),reference_sha256='reference',prompt='Two sentences. Specific facts.',seed=7,category='lamp',condition_id='0',generated_PNG='one.png',generated_sha256=sha(path/'one.png')))
        write(path/'images.json',rows);write(path/'report.json',dict(complete=True,prompt_profile=PROFILE,independent_TEST_opened=False,resolution=512))
        return rows
    def test_all48_sha_bound_outputs_required(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);rows=self.fixture(p);self.assertEqual(len(checked_run(p,rows)[1]),48)
            (p/'one.png').write_bytes(b'other_pixels')
            with self.assertRaises(ValueError):checked_run(p,rows)
    def test_prompt_swap_or_easy_subset_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);rows=self.fixture(p);actual=[dict(r)for r in rows];actual[0]['prompt']='Different task.';write(p/'images.json',actual)
            with self.assertRaises(ValueError):checked_run(p,rows)
            write(p/'images.json',rows[:-1])
            with self.assertRaises(ValueError):checked_run(p,rows)
    def test_teacher_comparison_preserves_full_same_reference_seed_cohort(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);before=root/'old';after=root/'new';before.mkdir();after.mkdir();old=[];new=[]
            for path in [before,after]:
                (path/'one.png').write_bytes(b'pixels');write(path/'model_assets.json',dict(frozen='same'))
            for i in range(24):
                for seed_index in [0,1]:
                    row=dict(sample_id=f'input{i}-balanced-B-draw{seed_index}',split='train',reference_id=str(i),reference='one.png',reference_sha256=sha(before/'one.png'),
                        category='lamp',seed_index=seed_index,seed=seed_index,required_visible_facts=['lamp','bookshelves'],prompt='Old two sentences. Bookshelves.',target='one.png',target_sha256=sha(before/'one.png'))
                    old.append(row)
                    if seed_index==0:new.append(dict(row,sample_id=row['sample_id']+'-sceneFirst',previous_sample_id=row['sample_id'],previous_prompt=row['prompt'],prompt=scene_first_condition('lamp','B')['prompt']))
            write(before/'candidates.json',old);write(after/'candidates.json',new)
            write(before/'report.json',dict(complete=True,candidate_manifest_sha256=sha(before/'candidates.json'),teacher_recipe=TEACHER_RECIPE))
            write(after/'report.json',dict(complete=True,prompt_profile=SCENE_FIRST_PROFILE,candidate_manifest_sha256=sha(after/'candidates.json'),teacher_recipe=TEACHER_RECIPE))
            self.assertEqual(len(checked_teacher_syntax(before,after)),24)
            new[0]['seed']=999;write(after/'candidates.json',new);write(after/'report.json',dict(complete=True,prompt_profile=SCENE_FIRST_PROFILE,candidate_manifest_sha256=sha(after/'candidates.json'),teacher_recipe=TEACHER_RECIPE))
            with self.assertRaises(ValueError):checked_teacher_syntax(before,after)


if __name__=='__main__':unittest.main()
