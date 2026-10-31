import tempfile
from pathlib import Path
import unittest
from ..scene_instruction_data import sha,write,PROFILE
from ..scene_instruction_delivery import checked_run


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


if __name__=='__main__':unittest.main()
