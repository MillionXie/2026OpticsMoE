"""Dependency-free scope guards; numerical gradient test runs on lab CPU."""
import ast
import json
from pathlib import Path
import unittest

TASK = Path(__file__).resolve().parents[1]
SOURCE = TASK/'lab_caltech20_dvp_conditioned_head_20261020.py'


class ConditionedHeadContract(unittest.TestCase):
    def test_objective_cannot_take_test_inputs(self):
        tree = ast.parse(SOURCE.read_text())
        objective = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'objective')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(objective) if isinstance(n, ast.Name)))
        self.assertEqual([a.arg for a in objective.args.args],
                         ['theta','bias','x','labels','weights','teacher','keep','origin','cfg'])

    def test_only_original_head_is_assigned(self):
        source = SOURCE.read_text()
        self.assertIn("state.update({'linear.weight':w, 'linear.bias':b})", source)
        self.assertIn("if not k.startswith('linear.')", source)
        self.assertIn('core.assess(model, data, fields', source)
        self.assertNotIn('Camera.open', source)
        self.assertNotIn('.cuda(', source)
        self.assertIn("dm['source_cache_SHA256']", source)

    def test_finite_budget_and_separate_targets(self):
        cfg = json.loads((TASK/'lab_caltech20_DVP_conditioned_head_profile_20261020.json').read_text())
        self.assertEqual(sum(cfg['original_TRAIN_counts']), 3141)
        self.assertLessEqual(cfg['maximum_fit_seconds'], 120)
        self.assertLessEqual(cfg['maximum_steps'], 240)
        self.assertEqual(cfg['targets'], {'4':314,'6':322})
        self.assertEqual(cfg['final_targets'], {'4':342,'6':350})

    def test_pair_uses_full_TRAIN_and_existing_direction(self):
        text=SOURCE.read_text()
        self.assertIn('pair_batch(rng,near,pools,hard,len(tr[\'y\']))', text)
        self.assertIn('TRAIN_spectrum_geometry, pair_direction', text)
        self.assertIn('two_original_rows_antisymmetric_other18_fixed=True', text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_pair_head_profile_20261020.json').read_text())
        self.assertEqual(cfg['objective'], 'huber_pair')
        self.assertLessEqual(cfg['max_pairs'],20)
        self.assertLessEqual(cfg['fit_seconds'],120)

    def test_cache_reuse_has_source_state_data_and_native_proofs(self):
        text=SOURCE.read_text()
        for token in ("sp['feature_graph_SHA256']==graph",'nonhead_equal(old,current)',
                      "sp['prefix_SHA256']==identities", "sp['data_SHA256']==dm['sha256']",
                      "sa['report_SHA256']==core.sha(source/'report.json')", "error<=3e-6"):
            self.assertIn(token,text)
        self.assertIn("sa['protocol_SHA256']==core.sha(source/'protocol.json')",text)
        self.assertIn("pa['protocol_SHA256']==core.sha(parent/'protocol.json')",text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_cached_head_profile_20261020.json').read_text())
        old=json.loads((TASK/'lab_caltech20_DVP_head_local_profile_20261019.json').read_text())
        self.assertEqual({k:cfg[k] for k in old if k!='scope'}, {k:v for k,v in old.items() if k!='scope'})
        self.assertEqual(cfg['final_targets'], {'4':342,'6':350})

    def test_shrinkage_uses_TRAIN_statistics_not_test_regularization_grid(self):
        tree=ast.parse(SOURCE.read_text())
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shrink_fit')
        text=ast.unparse(method)
        self.assertIn('shrunk_readout(x, y, counts)',text)
        self.assertIn('F.cross_entropy(train_logits * scale.exp() + prior, y',text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_shrinkage_head_profile_20261020.json').read_text())
        self.assertEqual(sum(cfg['original_TRAIN_counts']),3141)
        self.assertEqual(cfg['temperature_steps'],25)
        self.assertEqual(cfg['final_targets'],{'4':342,'6':350})

    def test_alpha_initialization_requires_TRAIN_evidence_and_true_parent_fallback(self):
        text=SOURCE.read_text()
        for token in ("evidence['TEST_inference'] is False", "a.depth==4 and a.prefix in (4,5) and not a.feature_run",
                      "alpha_key==f'fusion_logits.{a.prefix-2}'",
                      "parent_fallback=alpha_init is not None and name=='best' and selected==0",
                      "reference=baseline if parent_fallback", "all_nonhead_frozen=alpha_init is None",
                      "core.assess(model,data,fields,'train')[0] if parent_fallback"):
            self.assertIn(token,text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_E2_alpha_TRAINcapacity_profile_20261020.json').read_text())
        self.assertEqual(cfg['TRAIN_supported_alpha_initialization'],.4001001)
        self.assertEqual(len(cfg['TRAIN_capacity_evidence_SHA256']),64)
        self.assertEqual(cfg['temperature_steps'],25)

    def test_final_alpha_is_one_TRAIN_supported_existing_coordinate(self):
        cfg=json.loads((TASK/'lab_caltech20_DVP_final_alpha_TRAINcapacity_profile_20261020.json').read_text())
        self.assertEqual(cfg['TRAIN_supported_alpha_key'],'fusion_logits.3')
        self.assertEqual(cfg['TRAIN_supported_alpha_initialization'],.4001001)
        self.assertEqual(cfg['temperature_steps'],25)
        self.assertEqual(len(cfg['TRAIN_capacity_evidence_SHA256']),64)
        self.assertEqual(cfg['final_targets']['4'],342)
        text=SOURCE.read_text()
        self.assertIn("assert key in ('fusion_logits.2','fusion_logits.3')",text)
        self.assertIn("state[alpha_key].fill_(alpha_init)",text)

    def test_compound_admission_cannot_claim_new_training_or_rewrite_components(self):
        tree=ast.parse(SOURCE.read_text());method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admit_chain')
        text=ast.unparse(method)
        self.assertIn("for candidate_role in ('best', 'last')",text)
        self.assertIn("sr['selected_correct'] > pr['selected_correct']",text)
        self.assertIn("admission_training_updates=0",text)
        self.assertIn("no_gain_over_source_claimed=True",text)
        self.assertNotIn('backward',text)
        self.assertNotIn('optimizer', [n.attr for n in ast.walk(method) if isinstance(n,ast.Attribute)])
        self.assertIn("source_prediction_maxabs=error",text)

    def test_parallel_CPU_admission_is_complete_L4_only_and_exact_L6_owner(self):
        tree=ast.parse(SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cpu_fit_admission')
        text=ast.unparse(node)
        for token in ("a.depth == 4 and a.prefix == 5", "a.sequence / 'L6'", "'independent_capture_audit.json'",
                      "r['TRAIN'] == 1600", "r['TEST'] == 411", "'--sequence'", "'--depth'", "'capture'"):
            self.assertIn(token,text)
        self.assertNotIn('Camera.open',text)
        self.assertNotIn('unlink',text)
        source=SOURCE.read_text();self.assertIn("core.save(a.out/'CPU_capture_admission.json'",source)

    def test_terminal10_exact_forward_gate_and_TRAIN_only_loss(self):
        tree=ast.parse(SOURCE.read_text());loss=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal10_objective')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(loss) if isinstance(n,ast.Name)))
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal10_fit');text=ast.unparse(method)
        self.assertIn("trainable = {'electronic.3.local.weight', 'fusion_logits.3'}",text)
        self.assertIn('sum((v.numel() for v in params)) == 10',text)
        self.assertLess(text.index("max(cache_errors) <= 3e-06"),text.index('torch.optim.AdamW'))
        self.assertIn('np.array_equal(p.argmax(1), expected.argmax(1))',text)
        self.assertIn('used == 1600',text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_terminal10_profile_20261020.json').read_text())
        self.assertEqual(sum(cfg['original_TRAIN_counts']),3141)
        self.assertEqual((cfg['epochs'],cfg['fit_seconds'],cfg['batch']),(60,600,8))


if __name__ == '__main__':
    unittest.main()
