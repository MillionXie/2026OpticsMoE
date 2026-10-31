import json,tempfile,unittest
from pathlib import Path
from ..scene_initial_alignment import checked_alignment
from ..scene_instruction_data import sha,write


class InitialHeadIdentity(unittest.TestCase):
    def test_original_head_requires_exact_unpaired_identity(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);(p/'best_checkpoint.pt').write_bytes(b'original');digest=sha(p/'best_checkpoint.pt')
            write(p/'alignment_report.json',dict(complete=True,best_sha256=digest,scene_GT_used=False))
            state=dict(schema='fresh_qwen3_sdxl_conditions_v1',step=1200,not_yet_diffusion_finetuned=True)
            self.assertTrue(checked_alignment(p,state,digest)['original_unpaired_alignment'])
            with self.assertRaises(ValueError):checked_alignment(p,dict(state,step=0),digest)
            with self.assertRaises(ValueError):checked_alignment(p,state,'0'*64)
            with self.assertRaises(FileNotFoundError):checked_alignment(p,state)
    def test_default_still_uses_existing_adaptation_manifest(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);(p/'best_checkpoint.pt').write_bytes(b'adapted');write(p/'adaptation_report.json',dict(complete=True,best_sha256=sha(p/'best_checkpoint.pt')))
            self.assertFalse(checked_alignment(p,{},None)['original_unpaired_alignment'])
            (p/'best_checkpoint.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):checked_alignment(p,{},None)


if __name__=='__main__':unittest.main()
