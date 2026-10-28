import copy
import unittest

from .optimizer_state_probe import admission, audit_adam_state, optimizer_from_state, state_fingerprint


class PolicyTests(unittest.TestCase):
    def config(self):return dict(minimum_repair_per_origin=8, minimum_boundary_guard_pooled=8)

    def cases(self):
        def row(gain):
            return dict(repair_samples=12, repair_mean_delta=gain, source_correct_guard=100, guard_new_wrong=0,
                boundary_guard_samples=4, boundary_guard_new_wrong=0, guard_delta_q10=-.0001,
                boundary_guard_delta_q10=-.0001, all_mean_margin_delta=.0001, nll_delta=-.0001)
        return [dict(fresh={view: {origin: row(.0001) for origin in ['train', 'test']} for view in ['clean', 'd4']},
            retained={view: {origin: row(.0003) for origin in ['train', 'test']} for view in ['clean', 'd4']}) for _ in range(2)]

    def test_must_gain_both_fit_batches_and_origins_above_fresh_and_noise(self):
        self.assertTrue(admission(self.cases(), 0, 0, self.config())['admitted'])
        cases = self.cases(); cases[1]['retained']['clean']['test']['repair_mean_delta'] = .000105
        self.assertFalse(admission(cases, 0, 0, self.config())['admitted'])
        self.assertFalse(admission(self.cases(), .0001, 0, self.config())['admitted'])
        self.assertFalse(admission(self.cases(), float('nan'), 0, self.config())['admitted'])

    def test_same_guard_and_TRAIN_NLL_rules_not_relaxed_for_retained(self):
        for key, value in [('guard_new_wrong', 1), ('guard_delta_q10', -.0002),
                           ('boundary_guard_delta_q10', -.0002), ('nll_delta', .0002), ('all_mean_margin_delta', -.0002)]:
            cases = self.cases(); cases[0]['retained']['d4']['train'][key] = value
            self.assertFalse(admission(cases, 0, 0, self.config())['admitted'], key)

    def test_insufficient_groups_remain_inconclusive(self):
        cases = self.cases(); cases[1]['retained']['clean']['train']['repair_samples'] = 7
        self.assertEqual(admission(cases, 0, 0, self.config())['state'], 'inconclusive')
        cases = self.cases(); cases[0]['retained']['d4']['test']['boundary_guard_samples'] = 3
        self.assertEqual(admission(cases, 0, 0, self.config())['state'], 'inconclusive')

    def test_Adam_receipt_rejects_wrong_group_shape_moment_and_step(self):
        import numpy as np
        names = ['expert_phases.0', 'expert_phases.1', 'global_phases.0', 'global_phases.1', 'router_phase']
        shapes = {name: [2] for name in names}
        state = dict(param_groups=[dict(name=name, params=keys, lr=lr, lr_scale=scale, weight_decay=0., amsgrad=False)
            for name, keys, lr, scale in [('expert', [0, 1], 3e-5, 1.), ('global', [2, 3], 1.5e-5, .5), ('router', [4], 3e-6, .1)]],
            state={key: dict(step=1024., exp_avg=np.zeros(2, dtype=np.float32), exp_avg_sq=np.ones(2, dtype=np.float32)) for key in range(5)})
        self.assertEqual(len(audit_adam_state(state, shapes, 1024)['parameter_states']), 5)
        bad = copy.deepcopy(state); bad['state'][0]['step'] = 1023.
        with self.assertRaises(AssertionError):audit_adam_state(bad, shapes, 1024)
        bad = copy.deepcopy(state); bad['state'][0]['exp_avg'] = np.zeros(3)
        with self.assertRaises(AssertionError):audit_adam_state(bad, shapes, 1024)
        bad = copy.deepcopy(state); bad['state'][1]['exp_avg_sq'][0] = -1
        with self.assertRaises(AssertionError):audit_adam_state(bad, shapes, 1024)
        bad = copy.deepcopy(state); bad['param_groups'][0]['name'] = 'global'
        with self.assertRaises(AssertionError):audit_adam_state(bad, shapes, 1024)

    def test_real_CPU_Adam_restore_1024_to_1025_fresh_to1_no_cross_batch_carry(self):
        try:import torch
        except (ImportError, OSError) as error:self.skipTest('Local Torch unavailable: ' + str(error).splitlines()[0])

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expert_phases = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.2, -.1])) for _ in range(2)])
                self.global_phases = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.1, -.2])) for _ in range(2)])
                self.router_phase = torch.nn.Parameter(torch.tensor([.3, -.4]))

        model = Tiny()
        opt = torch.optim.AdamW([
            dict(params=list(model.expert_phases), name='expert', lr=3e-5, lr_scale=1.),
            dict(params=list(model.global_phases), name='global', lr=1.5e-5, lr_scale=.5),
            dict(params=[model.router_phase], name='router', lr=3e-6, lr_scale=.1)],
            weight_decay=0., betas=(.8, .99), eps=2e-8, amsgrad=True, foreach=False, maximize=False)
        def step(optimizer):
            optimizer.zero_grad(); loss = sum(p.square().sum() for p in model.parameters()); loss.backward(); optimizer.step()
        for _ in range(1024):step(opt)
        source = copy.deepcopy(opt.state_dict()); weights = copy.deepcopy(model.state_dict()); identity = state_fingerprint(source)
        assert all(int(value['step']) == 1024 for value in source['state'].values())
        for _ in range(2):
            for fresh in [True, False]:
                model.load_state_dict(weights); candidate = optimizer_from_state(model, source, fresh=fresh)
                self.assertEqual(candidate.state_dict()['param_groups'], source['param_groups'])
                if fresh:self.assertFalse(candidate.state)
                else:self.assertTrue(all(int(v['step']) == 1024 for v in candidate.state.values()))
                step(candidate)
                self.assertTrue(all(int(v['step']) == (1 if fresh else 1025) for v in candidate.state.values()))
                receipt = audit_adam_state(candidate.state_dict(), {name: list(p.shape) for name, p in model.named_parameters()}, 1 if fresh else 1025)
                self.assertEqual(receipt['expected_step'], 1 if fresh else 1025)
                self.assertEqual(state_fingerprint(source), identity)


if __name__ == '__main__':unittest.main()
