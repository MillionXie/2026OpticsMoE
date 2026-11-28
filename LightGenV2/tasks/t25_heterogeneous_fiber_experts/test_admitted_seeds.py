"""Receipt-only checks; no GPU access or worker launches."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import run_admitted_seeds as dispatch
from .train import configuration


SOURCE = '3' * 40


class Admissions(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.profile = dispatch.read(dispatch.PROFILE)
        self.sources = {path.relative_to(dispatch.TASK.parents[2]).as_posix():dispatch.sha(path)
                        for path in (dispatch.TASK/'train.py', dispatch.TASK/'model.py')}

    def populate(self, seed):
        for dataset in self.profile['datasets']:
            for arm in self.profile['arms']:
                folder = self.root/f'{dataset}_s{seed}_{arm}'
                folder.mkdir()
                cfg = configuration(dataset, seed, arm, self.profile['epochs'])
                dispatch.write(folder/'config.json', cfg)
                identity = dict(gpu_uuid=cfg['assigned_gpu_uuid'], source_commit=SOURCE, arm=arm,
                                initial_state_sha256='a'*64, fiber_core_counts=cfg['fiber_core_counts'])
                dispatch.write(folder/'metadata.json', dict(identity, smoke=True, device='cuda', config=cfg,
                    seed=seed, dataset=cfg['dataset'], pid=123, source_sha256=self.sources,
                    protocol_sha256=cfg['protocol_sha256'], config_file_sha256=dispatch.sha(folder/'config.json')))
                dispatch.write(folder/'admission.json', dict(identity, domain='original TRAIN only',
                    labels=list(range(8)), ids=[f'{dataset}-{n}' for n in range(8)], shared_router_bitwise=True,
                    diagnostic_logits_bitwise=True, diffraction_control_logits_bitwise=(arm=='diffraction'), loss=1.5,
                    gradient_l2={k:1 for k in ('r','e','g','w')}, update_max_abs={k:.1 for k in ('r','e','g','w')}))
                self.rehash(folder)
                dispatch.write(folder/'status.json', {'state':'complete','smoke':True,'pid':123})

    def rehash(self, folder):
        dispatch.write(folder/'manifest.json',{n:dispatch.sha(folder/n) for n in ('config.json','metadata.json','admission.json')})

    def ready(self, seed):
        with patch.object(dispatch,'smoke_process_alive',return_value=False):
            return dispatch.admit_seed(self.root,self.profile,seed,SOURCE)

    def test_one_seed_can_run_while_other_seed_is_incomplete(self):
        self.populate(53)
        self.assertIsNone(self.ready(37))
        self.assertEqual(len(self.ready(53)['smoke_manifest_sha256']),8)
        self.populate(37)
        self.assertNotEqual(self.ready(53)['gpu_uuid'],self.ready(37)['gpu_uuid'])
        with patch.object(dispatch,'smoke_process_alive',return_value=True):
            self.assertIsNone(dispatch.admit_seed(self.root,self.profile,37,SOURCE))

    def test_corrupt_sha_and_wrong_device_after_rehash_are_rejected(self):
        self.populate(37)
        directory = self.root/'blood_s37_mixed'
        path=directory/'admission.json'
        admission=dispatch.read(path)
        admission['gpu_uuid']=self.profile['gpu_uuids']['53']
        dispatch.write(path,admission)
        with self.assertRaisesRegex(ValueError,'bad SHA'):
            self.ready(37)
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError,'smoke identity'):
            self.ready(37)

    def test_rehashed_mismatched_pair_initialization_rejected(self):
        self.populate(53)
        d=self.root/'kather_s53_mixed'
        for name in ('admission.json','metadata.json'):
            row=dispatch.read(d/name)
            row['initial_state_sha256']='b'*64
            dispatch.write(d/name,row)
        self.rehash(d)
        with self.assertRaisesRegex(ValueError,'initialization differs'):
            self.ready(53)

    def test_global_smoke_failure_precedes_any_formal_launch(self):
        formal=self.root/'formal'
        formal.mkdir()
        dispatch.write(self.root/'cohort_status.json', {'state':'failed','reason':'fixture invalid'})
        with patch.object(dispatch,'run_seed') as invoke, self.assertRaisesRegex(RuntimeError,'fixture invalid'):
            dispatch.dispatch(formal,self.root,self.profile,SOURCE,'8'*40)
        invoke.assert_not_called()
        result=dispatch.read(formal/'cohort_status.json')
        self.assertEqual((result['stage'],result['state'],result['source_commit'],result['scheduler_commit']),
                         ('train','failed',SOURCE,'8'*40))
