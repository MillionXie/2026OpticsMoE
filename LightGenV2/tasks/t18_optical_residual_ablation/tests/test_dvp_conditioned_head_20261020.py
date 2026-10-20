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
        self.assertIn("assert key in ('fusion_logits.2','fusion_logits.3','fusion_logits.5')",text)
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
        self.assertIn('terminal_coordinates(a.depth)',text)
        self.assertIn('trainable = {conv_key, alpha_key}',text)
        self.assertIn('sum((v.numel() for v in params)) == 10',text)
        self.assertLess(text.index("max(cache_errors) <= 3e-06"),text.index('torch.optim.AdamW'))
        self.assertIn('np.array_equal(p.argmax(1), expected.argmax(1))',text)
        self.assertIn('used == 1600',text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_terminal10_profile_20261020.json').read_text())
        self.assertEqual(sum(cfg['original_TRAIN_counts']),3141)
        self.assertEqual((cfg['epochs'],cfg['fit_seconds'],cfg['batch']),(60,600,8))

    def test_joint_native_adapter_reuses_fixed_published_coefficients_and_original_graph(self):
        cfg=json.loads((TASK/'lab_caltech20_DVP_native5150_profile_20261020.json').read_text())
        old=json.loads((TASK/'lab_caltech20_L4_native_TRAINprior_profile_20261019.json').read_text())
        for key in ('epochs','fit_seconds','batch','conv_lr','alpha_lr','head_coordinate_lr','weight_decay','label_smoothing','temperature','teacher_KL','terminal_anchor','head_coordinate_anchor','per_row_head_radius','original_TRAIN_counts','original_TRAIN_cache_SHA256'):
            self.assertEqual(cfg[key],old[key])
        self.assertEqual((len(cfg['original_TRAIN_counts']),sum(cfg['original_TRAIN_counts'])),(20,3141))
        tree=ast.parse(SOURCE.read_text());loss=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_joint_objective')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(loss) if isinstance(n,ast.Name)))
        text=SOURCE.read_text();self.assertIn('from lab_caltech20_native_terminal_20261019 import native_features,folded',text)
        self.assertIn('head_coordinates_folded_into_original_5140=True',text)
        self.assertIn('sum(v.numel() for v in params)==5150',text)
        self.assertIn("joint==(a.mode=='native5150-fit')",text)

    def test_binary_row_requires_TRAIN_evidence_and_temperature_correct_teacher(self):
        cfg=json.loads((TASK/'lab_caltech20_DVP_binary_head_profile_20261020.json').read_text())
        self.assertEqual(cfg['objective'],'binary_row')
        self.assertEqual(cfg['minimum_TRAIN_accuracy'],.8)
        self.assertEqual(cfg['temperature'],2.)
        self.assertEqual((cfg['maximum_passes'],cfg['maximum_fit_seconds']),(3,120))
        text=SOURCE.read_text()
        self.assertIn("tr['metric']['accuracy']>=cfg['minimum_TRAIN_accuracy']",text)
        self.assertIn("(z0[:,label]-lse)/cfg['temperature']",text)
        self.assertIn('binary_row_direction(u,u0,torch.from_numpy(xx[ix])',text)

    def test_L6_G1_teacher_reuses_published_TRAIN_only_recipe_with_shorter_budget(self):
        cfg=json.loads((TASK/'lab_caltech20_DVP_L6_G1_TRAINteacher_profile_20261020.json').read_text())
        old=json.loads((TASK/'lab_caltech20_DVP_G1_TRAINteacher_profile_20261020.json').read_text())
        for key in ('phase_lr','electronic_lr','alpha_lr','head_lr','label_smoothing','confusion_emphasis','teacher_weight','teacher_temperature','teacher_simulated_newest_layers','minimum_epochs','evaluate_every'):
            self.assertEqual(cfg['overrides'][key],old['overrides'][key])
        self.assertEqual(cfg['overrides']['target_correct'],322)
        self.assertEqual(cfg['overrides']['middle_fit'],{'epochs':24,'seconds':600})
        self.assertEqual(len(cfg['TRAIN_teacher_evidence_SHA256']),64)


    def test_L4_renewed_anchor_is_one_config_change_not_new_graph(self):
        old=json.loads((TASK/'lab_caltech20_DVP_conditioned_head_profile_20261020.json').read_text())
        cfg=json.loads((TASK/'lab_caltech20_DVP_L4_head_anchor_profile_20261020.json').read_text())
        self.assertEqual({k:v for k,v in cfg.items() if k not in ('parameter_anchor','scope')},
                         {k:v for k,v in old.items() if k not in ('parameter_anchor','scope')})
        self.assertEqual(cfg['parameter_anchor'],.02)
        self.assertEqual(cfg['final_targets']['4'],342)
        self.assertIn('no camera/SDK/GPU',cfg['scope'])


    def test_TRAIN_teacher_uses_saved_native_TRAIN_only_and_original_head(self):
        tree=ast.parse(SOURCE.read_text())
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='teacher_readout_fit')
        text=ast.unparse(method)
        self.assertIn("session / 'train_reference.npz'",text)
        self.assertNotIn('test_reference.npz',text)
        self.assertIn("gate['checks']['train']['original_reference_maxabs'] == 0",text)
        self.assertIn("np.array_equal(z['ids'], data['train'][2])",text)
        self.assertIn('ridge_logit_readout(x, target_logits)',text)
        self.assertIn('nonhead_equal(original, state)',text)
        self.assertIn('core.assess(model, data, fields',text)
        objective=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='teacher_blend_objective')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(objective) if isinstance(n,ast.Name)))
        cfg=json.loads((TASK/'lab_caltech20_DVP_L4_TRAINteacher_readout_profile_20261020.json').read_text())
        old=json.loads((TASK/'lab_caltech20_L4_residual_teacher_profile_20261019.json').read_text())
        for key in ('teacher_PT_SHA256','blend_steps','blend_lr','correct_TRAIN_parent_KL'):
            self.assertEqual(cfg[key],old[key])
        self.assertEqual(len(cfg['teacher_TRAIN_reference_SHA256']),64)


    def test_bias_only_freezes_original_weight_and_has_no_TEST_gradient(self):
        tree=ast.parse(SOURCE.read_text())
        objective=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bias_objective')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(objective) if isinstance(n,ast.Name)))
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bias_fit')
        text=ast.unparse(method)
        self.assertIn("allowed_keys=['linear.bias']",text)
        self.assertIn("if k != 'linear.bias'",text)
        self.assertIn('original_head_weight_frozen=True',text)
        self.assertIn('gradient_parameters=20',text)
        cfg=json.loads((TASK/'lab_caltech20_DVP_L4_bias_only_profile_20261020.json').read_text())
        self.assertEqual((cfg['maximum_steps'],cfg['maximum_fit_seconds']),(120,30))
        self.assertEqual(sum(cfg['original_TRAIN_counts']),3141)


    def test_dropout_views_are_TRAIN_only_and_never_enter_inference(self):
        tree=ast.parse(SOURCE.read_text())
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='TRAIN_feature_views')
        self.assertFalse(any('test' in n.id.lower() for n in ast.walk(method) if isinstance(n,ast.Name)))
        text=ast.unparse(method)
        for token in ('labels.repeat(repeats + 1)','teacher.repeat(repeats + 1, 1)',
                      'inference_dropout=False','new_independent_samples=0','torch.Generator'):
            self.assertIn(token,text)
        self.assertIn('base_TRAIN_records=len(z)',text)
        self.assertIn('statistical_independence_not_asserted=True',text)
        self.assertNotIn('independent_TRAIN',text)
        fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit')
        gate=next(n for n in ast.walk(fit) if isinstance(n,ast.If) and 'TRAIN_feature_dropout_repeats' in ast.unparse(n.test))
        predicate=compile(ast.Expression(next(n for n in gate.body if isinstance(n,ast.Assert)).test),'<dropout_guard>','eval')
        from types import SimpleNamespace
        for depth,prefix,size,allowed in ((4,5,1600,True),(4,4,1600,False),(6,5,1600,False),(4,5,1200,False)):
            self.assertEqual(eval(predicate,{'a':SimpleNamespace(depth=depth,prefix=prefix),'labels':[0]*size}),allowed)
        old=json.loads((TASK/'lab_caltech20_DVP_conditioned_head_profile_20261020.json').read_text())
        cfg=json.loads((TASK/'lab_caltech20_DVP_L4_TRAINfeature_dropout_profile_20261020.json').read_text())
        self.assertEqual({k:v for k,v in cfg.items() if not k.startswith('TRAIN_feature_dropout_') and k!='scope'},
                         {k:v for k,v in old.items() if k!='scope'})
        self.assertEqual((cfg['TRAIN_feature_dropout_probability'],cfg['TRAIN_feature_dropout_repeats'],cfg['TRAIN_feature_dropout_seed']),(.1,3,73))


    def test_final_native_adapter_uses_exact_depth_tail_and_complete_prefix(self):
        tree=ast.parse(SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal_coordinates')
        namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<terminal_coordinates>','exec'),namespace)
        fn=namespace['terminal_coordinates']
        self.assertEqual(fn(4),(3,'electronic.3.local.weight','fusion_logits.3'))
        self.assertEqual(fn(6),(5,'electronic.5.local.weight','fusion_logits.5'))
        for invalid in (2,3,5,7):
            with self.assertRaises(AssertionError):fn(invalid)
        method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='terminal10_fit');text=ast.unparse(method)
        self.assertIn('a.prefix == a.depth + 1',text)
        self.assertIn('model.electronic[last]',text)
        self.assertIn("{'4': 342, '6': 350}",text)
        self.assertIn('depth=a.depth, prefix=a.prefix',text)
        old=json.loads((TASK/'lab_caltech20_DVP_native5150_profile_20261020.json').read_text())
        cfg=json.loads((TASK/'lab_caltech20_DVP_L6_native5150_profile_20261020.json').read_text())
        self.assertEqual({k:v for k,v in cfg.items() if k not in ('scope','targets','final_targets')},
                         {k:v for k,v in old.items() if k!='scope'})
        self.assertEqual(cfg['final_targets']['6'],350)

    def test_L6_alpha_initialization_only_after_complete_prefix7(self):
        tree=ast.parse(SOURCE.read_text());method=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shrink_fit')
        gate=next(n for n in ast.walk(method) if isinstance(n,ast.Assert) and 'a.depth == 6' in ast.unparse(n.test))
        expression=compile(ast.Expression(gate.test),'<alpha_admission>','eval')
        from types import SimpleNamespace
        for depth,prefix,feature,allowed in ((4,4,None,True),(4,5,None,True),(6,7,None,True),(6,6,None,False),(6,5,None,False),(6,7,'old_bank',False)):
            self.assertEqual(eval(expression,{'a':SimpleNamespace(depth=depth,prefix=prefix,feature_run=feature)}),allowed)
        text=ast.unparse(method)
        for token in ("evidence['TEST_inference'] is False","evidence['TRAIN'] == 1600",'parent_fallback'):
            self.assertIn(token,text)


if __name__ == '__main__':
    unittest.main()
