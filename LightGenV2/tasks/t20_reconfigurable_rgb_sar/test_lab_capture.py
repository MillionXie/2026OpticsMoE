import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .build_lab_package import sha, export_pilot_subsets
from .lab_capture import sdk_owner, validate_handoff, read_entries


class LabCaptureTests(unittest.TestCase):
    def test_small_pilot_is_bitwise_same_and_not_formal_data_replacement(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);path=folder/'train_inputs.npz'
            source=dict(amplitudes=np.random.default_rng(73).random((1000,2,2),dtype=np.float32),
                labels=np.tile(np.repeat(np.arange(10),50),2),domains=np.repeat([0,1],500),
                ids=np.asarray([f'TRAIN-{i}' for i in range(1000)]),source_indices=np.arange(1000))
            np.savez_compressed(path,**source)
            manifest=dict(files=[dict(name=path.name,sha256=sha(path))],
                datasets=dict(train=dict(ids=source['ids'].tolist(),identity_sha256='trainid',file=path.name)),
                cases=dict(LG=dict(TRAIN_indices=list(range(1000)),checkpoint_sha256='LG'),
                    D2NN=dict(TRAIN_indices=list(range(500)),checkpoint_sha256='D2NN')))
            (folder/'manifest.json').write_text(json.dumps(manifest))
            export_pilot_subsets(folder)
            for case in ('LG','D2NN'):
                rows=read_entries(folder,manifest,case,True)
                self.assertEqual(len(rows),24)
                for row in rows:
                    np.testing.assert_array_equal(row['amplitude'],source['amplitudes'][row['local_index']])
            # The full source need not have arrived for the pilot, but formal
            # acquisition must still demand the original whole file and SHA.
            path.unlink()
            self.assertEqual(len(read_entries(folder,manifest,'LG',True)),24)
            with self.assertRaises(FileNotFoundError):read_entries(folder,manifest,'LG',False)
            proof=json.loads((folder/'pilot_inputs.json').read_text());proof['cases']['LG']['ids'][0]='TEST-0'
            (folder/'pilot_inputs.json').write_text(json.dumps(proof))
            with self.assertRaises(ValueError):read_entries(folder,manifest,'LG',True)

    def test_idle_failure_is_not_completed_handoff(self):
        for row in ({}, {'SDK_released': True}, {'T21_capture_complete':False,'SDK_released':True,'next_task':'T20'}):
            with self.assertRaises(ValueError):
                validate_handoff(row)

    def test_complete_expected_D_only(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'report.json'
            r=dict(passed=True,pilot=False,SDK_released=True,full_physical_prefix=True,
                checkpoint_sha256='D_pt',stages=['router','first','final'],records=5900,frames=17700)
            p.write_text(json.dumps(r))
            handoff=dict(T21_capture_complete=True,SDK_released=True,next_task='T20',report=str(p),
                report_sha256=sha(p),checkpoint_sha256='D_pt')
            validate_handoff(handoff)
            for key, value in (('frames',17699),('pilot',True),('records',1000),('SDK_released',False)):
                p.write_text(json.dumps(dict(r,**{key:value})))
                with self.assertRaises(ValueError):
                    validate_handoff(dict(handoff,report_sha256=sha(p)))

    def test_exclusive_lock_preserves_other_owner(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'lock.json'
            with sdk_owner(p, Path(name)):
                self.assertEqual(json.loads(p.read_text())['pid'],os.getpid())
                before=p.read_bytes()
                with self.assertRaises(FileExistsError):
                    with sdk_owner(p, Path(name)):
                        self.fail('Second SDK owner admitted')
                self.assertEqual(before,p.read_bytes())
            self.assertFalse(p.exists())

    def test_exception_releases_only_owned_lock(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'lock.json'
            with self.assertRaisesRegex(RuntimeError,'training'):
                with sdk_owner(p, Path(name)):
                    raise RuntimeError('training')
            self.assertFalse(p.exists())


if __name__=='__main__':
    unittest.main()
