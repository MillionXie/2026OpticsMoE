import unittest

from .view_conflict_probe import admission, select_indices, summarize


class PolicyTests(unittest.TestCase):
    def config(self):
        return dict(minimum_repair_per_origin=8, minimum_boundary_guard_pooled=8)

    def cases(self):
        def row(gain):
            return dict(repair_samples=12, repair_mean_delta=gain, source_correct_guard=100,
                guard_new_wrong=0, boundary_guard_samples=4, boundary_guard_new_wrong=0,
                guard_delta_q10=-.0001, boundary_guard_delta_q10=-.0001,
                all_mean_margin_delta=.0001, nll_delta=-.0001)
        return [dict(d4={view: {origin: row(.0001) for origin in ['train', 'test']} for view in ['clean', 'd4']},
            clean={view: {origin: row(.0003) for origin in ['train', 'test']} for view in ['clean', 'd4']}) for _ in range(2)]

    def test_requires_both_batches_and_each_origin_repair(self):
        cases = self.cases(); self.assertTrue(admission(cases, 0, 0, self.config())['admitted'])
        cases[1]['clean']['clean']['train']['repair_mean_delta'] = 0
        self.assertFalse(admission(cases, 0, 0, self.config())['admitted'])
        cases = self.cases(); cases[0]['clean']['clean']['test']['repair_samples'] = 7
        self.assertEqual(admission(cases, 0, 0, self.config())['state'], 'inconclusive')

    def test_repeat_noise_and_guard_harm_reject(self):
        self.assertFalse(admission(self.cases(), .0001, 0, self.config())['admitted'])
        self.assertFalse(admission(self.cases(), float('nan'), 0, self.config())['admitted'])
        cases = self.cases(); cases[0]['clean']['d4']['train']['guard_new_wrong'] = 1
        self.assertFalse(admission(cases, 0, 0, self.config())['admitted'])
        cases = self.cases(); cases[1]['clean']['clean']['train']['boundary_guard_delta_q10'] = -.0002
        self.assertFalse(admission(cases, 0, 0, self.config())['admitted'])

    def test_pool_boundary_threshold_and_both_origins_nonempty(self):
        cases = self.cases(); cases[0]['clean']['clean']['train']['boundary_guard_samples'] = 3
        self.assertEqual(admission(cases, 0, 0, self.config())['state'], 'inconclusive')
        cases = self.cases(); cases[0]['clean']['clean']['train']['boundary_guard_samples'] = 0
        cases[0]['clean']['clean']['test']['boundary_guard_samples'] = 8
        self.assertEqual(admission(cases, 0, 0, self.config())['state'], 'inconclusive')

    def test_uniform_selection_has_no_model_based_enrichment_or_overlap(self):
        import numpy as np
        labels = np.concatenate([np.repeat(np.arange(8), 437), np.repeat(np.arange(8), 94)])
        fit, held = select_indices(labels)
        fit2, held2 = select_indices(labels)
        self.assertTrue(np.array_equal(held, held2)); self.assertTrue(np.array_equal(fit[0], fit2[0]))
        for rows, count in [(held, 16), (fit[0], 1), (fit[1], 1)]:
            for cls in range(8):
                self.assertEqual(int(((labels[rows] == cls) & (rows < 3496)).sum()), count)
                self.assertEqual(int(((labels[rows] == cls) & (rows >= 3496)).sum()), count)
        self.assertEqual(len(set(held.tolist() + fit[0].tolist() + fit[1].tolist())), 288)

    def test_source_groups_stay_fixed_when_candidate_repairs_or_harms(self):
        import numpy as np
        source = np.array([[.4, .6, 0, 0, 0, 0, 0, 0], [.51, .49, 0, 0, 0, 0, 0, 0],
                           [.4, .6, 0, 0, 0, 0, 0, 0], [.51, .49, 0, 0, 0, 0, 0, 0]])
        candidate = source.copy(); candidate[0, :2] = [.6, .4]; candidate[1, :2] = [.49, .51]
        teacher = np.tile(np.array([.8, .2, 0, 0, 0, 0, 0, 0]), (4, 1))
        row = summarize(source, candidate, teacher, np.zeros(4, dtype=int), np.array(['train', 'train', 'test', 'test']))
        self.assertEqual(row['train']['repair_samples'], 1); self.assertEqual(row['train']['repair_corrected'], 1)
        self.assertEqual(row['train']['guard_new_wrong'], 1); self.assertEqual(row['train']['boundary_guard_new_wrong'], 1)
        self.assertEqual(row['test']['repair_samples'], 1)


if __name__ == '__main__':unittest.main()
