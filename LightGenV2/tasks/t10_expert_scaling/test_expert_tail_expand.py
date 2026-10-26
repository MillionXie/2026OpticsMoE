"""Pure CPU checks for lossless scheduler adoption and unchanged training budget."""
import copy
import unittest
from .expert_tail_expand import validate_partition, launch_command


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.jobs=[dict(experts=n,top_k=k,seed=s,source=f'source/{n}/{k}/{s}',
                        out=f'run/{n}/{k}/{s}',best_sha256=f'{n}-{k}-{s}')
                   for n,k in [(4,4),(16,16),(25,25),(49,24),(49,49)] for s in [17,27,37]]
        self.snapshot=dict(pending=self.jobs[2:],complete=[],running={'11':self.jobs[0],'12':self.jobs[1]})

    def test_partition_accepts_all15_once(self):
        validate_partition(self.snapshot,self.jobs)

    def test_duplicate_or_missing_task_rejected(self):
        for change in ('duplicate','missing'):
            s=copy.deepcopy(self.snapshot)
            if change=='duplicate':s['pending'].append(s['pending'][0])
            else:s['pending'].pop()
            with self.assertRaises(ValueError):validate_partition(s,self.jobs)

    def test_source_or_output_substitution_rejected(self):
        for field in ('source','out','best_sha256'):
            s=copy.deepcopy(self.snapshot);s['pending'][0][field]='other'
            with self.assertRaises(ValueError):validate_partition(s,self.jobs)

    def test_command_keeps_budget_and_test_deferral(self):
        cfg=dict(extra_epochs=30,lr=.0005,microbatch=2,margin_weight=.2,capture_weight=.005)
        cmd=launch_command(self.jobs[0],cfg)
        for flag,value in [('--epochs','30'),('--lr','0.0005'),('--batch','2'),
                           ('--margin-weight','0.2'),('--capture-weight','0.005')]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertIn('--defer-test',cmd)
        self.assertIn('--route-audit',cmd)


if __name__=='__main__':unittest.main()
