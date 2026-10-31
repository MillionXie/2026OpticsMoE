import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from LightGenV2.tasks.t12_text_to_image.scene_test_PT_selection import checkpoint_payload_identity,snapshots
from LightGenV2.tasks.t12_text_to_image.scene_instruction_data import sha


class CheckpointPayloadIdentity(unittest.TestCase):
    def checkpoint(self,path,root,metadata=b'epoch2 exact metadata',tensor=b'float32 exact tensor bytes'):
        with zipfile.ZipFile(path,'w')as z:
            z.writestr(root+'/data.pkl',metadata);z.writestr(root+'/data/0',tensor)
            z.writestr(root+'/version',b'3\n');z.writestr(root+'/byteorder',b'little')

    def test_container_root_changes_but_all_payload_bytes_bound(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);a=p/'best.pt';b=p/'last.pt';self.checkpoint(a,'best');self.checkpoint(b,'last')
            self.assertNotEqual(sha(a),sha(b));self.assertEqual(checkpoint_payload_identity(a),checkpoint_payload_identity(b))
            self.checkpoint(b,'last',tensor=b'one changed tensor byte')
            self.assertNotEqual(checkpoint_payload_identity(a),checkpoint_payload_identity(b))
            self.checkpoint(b,'last',metadata=b'epoch1 different metadata')
            self.assertNotEqual(checkpoint_payload_identity(a),checkpoint_payload_identity(b))

    def test_snapshots_retain_original_files_and_only_exact_aliases_skip(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp);train=p/'train';train.mkdir();best=train/'best_checkpoint.pt';last=train/'last_checkpoint.pt'
            self.checkpoint(best,'best_checkpoint');self.checkpoint(last,'last_checkpoint');original=[sha(best),sha(last)]
            (train/'report.json').write_text(json.dumps(dict(complete=True,best_sha256=original[0],final_TEST_PT_selection_pending=True)))
            candidates=snapshots(train,p/'bank');self.assertEqual(len(candidates),1)
            report=json.loads((p/'bank/report.json').read_text());self.assertEqual(len(report['identical_complete_state_aliases']),1)
            self.assertEqual(report['identical_complete_state_aliases'][0]['source_file_sha256'],original[1])
            self.assertEqual([sha(best),sha(last)],original)

    def test_non_checkpoint_archive_uses_literal_identity(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'file.pt'
            with zipfile.ZipFile(p,'w')as z:z.writestr('not-a-torch-file',b'data')
            self.assertEqual(checkpoint_payload_identity(p),dict(method='literal_file_sha256',sha256=sha(p)))


if __name__=='__main__':unittest.main()
