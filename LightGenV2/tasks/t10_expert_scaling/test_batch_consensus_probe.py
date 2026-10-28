import copy
import unittest

from .batch_consensus_probe import admission, select_banks


class Policies(unittest.TestCase):
    def cases(self):
        row=dict(repair_samples=10,repair_mean_delta=.00004,source_correct_guard=40,
            boundary_guard_samples=6,guard_new_wrong=0,boundary_guard_new_wrong=0,
            guard_delta_q10=.00001,boundary_guard_delta_q10=.00001,all_mean_margin_delta=.00001,nll_delta=-.00001)
        control=copy.deepcopy(row);control['repair_mean_delta']=.00001
        case=dict(actual_update_relative_change=.2,repeat_update_relative_error=0,
            small={v:{o:copy.deepcopy(control) for o in ['train','test']} for v in ['clean','d4']},
            large={v:{o:copy.deepcopy(row) for o in ['train','test']} for v in ['clean','d4']})
        return [copy.deepcopy(case),copy.deepcopy(case)]

    def test_uniform_banks_and_nested_controls(self):
        import numpy as np
        y=np.concatenate([np.repeat(np.arange(8),437),np.repeat(np.arange(8),94)])
        controls,banks,held=select_banks(y)
        self.assertEqual(len(set(held.tolist()+banks[0].tolist()+banks[1].tolist())),512)
        for ix in banks:
            self.assertTrue(np.array_equal(np.bincount(y[ix[ix<3496]],minlength=8),np.repeat(8,8)))
            self.assertTrue(np.array_equal(np.bincount(y[ix[ix>=3496]],minlength=8),np.repeat(8,8)))
        for control,bank in zip(controls,banks):
            self.assertTrue(set(control).issubset(bank));self.assertEqual(sum(control>=3496),8)

    def test_positive_requires_both_trials(self):
        cases=self.cases();self.assertTrue(admission(cases,0,0)['admitted'])
        cases[1]['large']['clean']['test']['repair_mean_delta']=0
        self.assertFalse(admission(cases,0,0)['admitted'])

    def test_guard_failure_and_repeat_error(self):
        cases=self.cases();cases[0]['large']['d4']['train']['guard_new_wrong']=1
        self.assertFalse(admission(cases,0,0)['admitted'])
        self.assertFalse(admission(self.cases(),1e-3,0)['admitted'])

    def test_shortage_inconclusive_not_pass(self):
        cases=self.cases();cases[1]['large']['clean']['test']['repair_samples']=7
        self.assertEqual(admission(cases,0,0)['state'],'inconclusive')
        self.assertFalse(admission(cases,0,0)['admitted'])


if __name__=='__main__':unittest.main()
