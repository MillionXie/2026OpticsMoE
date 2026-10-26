"""Controller gates only: no remote, model, CUDA, daemon or signal actions."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_lgvq_async_20261026 as controller


class ControllerTests(unittest.TestCase):
    def test_existing_suite_detector_does_not_match_worker_or_other_task(self):
        self.assertTrue(controller.suite_command(['python',str(controller.ENTRY),'--phase','suite']))
        self.assertFalse(controller.suite_command(['python',str(controller.ENTRY),'--phase','worker']))
        self.assertFalse(controller.suite_command(['python','another_task.py','--phase','suite']))
    def predecessor(self):
        plan=[(m,c,r) for m in ('qwen','glm','clip','yolo') for c in (100,50) for r in (1,2,3)]
        names=[f'{m}_cap{c}_repeat{r:02d}' for m,c,r in plan]
        return dict(status='complete',plan=plan[4:],completed=names[4:],retained=[dict(name=n) for n in names[:4]])
    def test_all_original_24_names_required_not_only_count(self):
        state=self.predecessor();self.assertTrue(controller.predecessor_complete(state))
        state['completed'][0]='unrelated'
        self.assertFalse(controller.predecessor_complete(state))
        state=self.predecessor();state['status']='running'
        self.assertFalse(controller.predecessor_complete(state))
    def test_duplicate_or_incomplete_predecessor_rejected(self):
        for key in ['plan','completed','retained']:
            state=self.predecessor();state[key].append(state[key][0])
            self.assertFalse(controller.predecessor_complete(state))
    def test_canonical_paths_and_finite_deadline(self):
        args=SimpleNamespace(phase='suite',root=controller.SIM/'a100_lgvq_async_20261026_01',
                             after_suite=controller.SIM/'a100_lab_corrected_energy_20261026_06',deadline=1100.)
        with patch.object(controller.time,'time',return_value=1000.):controller.validate_args(args)
        args.deadline=float('inf')
        with self.assertRaises(ValueError):controller.validate_args(args)
        args.deadline=1100.;args.root=Path('C:/unrelated')
        with self.assertRaises(ValueError):controller.validate_args(args)
    def test_pending_software_or_fake_physical_measurement_never_accepted(self):
        report=dict(status='complete',qualified_for_final_hybrid_table=True,video_calls=2048,fields=128,
                    all_predictions_bitwise_equal=True,physical_hardware_time_measured=False,
                    fixed_optics_projection=dict(status='COUNTERFACTUAL_FIXED_OPTICS_PROJECTION'))
        self.assertTrue(controller.accepted_report(report))
        for key,value in [('status','actual_software_complete_projection_pending'),('physical_hardware_time_measured',True),
                          ('all_predictions_bitwise_equal',False),('video_calls',224)]:
            bad=copy.deepcopy(report);bad[key]=value;self.assertFalse(controller.accepted_report(bad))
    def test_exact_owned_worker_identity_only(self):
        leaf=controller.SIM/'a100_lgvq_async_20261026_01/lg_repeat01'
        argv=['python',str(controller.ENTRY),'--phase','worker','--output',str(leaf)]
        self.assertTrue(controller.child_identity_matches(argv,10,20,20,leaf,10))
        self.assertFalse(controller.child_identity_matches(argv,11,20,20,leaf,10))
        self.assertFalse(controller.child_identity_matches(argv,10,21,20,leaf,10))
        wrong=argv[:-1]+[str(leaf)+'_other']
        self.assertFalse(controller.child_identity_matches(wrong,10,20,20,leaf,10))
    def test_import_has_no_model_or_GPU_actions(self):
        tree=ast.parse(Path(controller.__file__).read_text(encoding='utf-8'))
        imports=[a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('torch',imports)
        self.assertFalse(any(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name) and n.value.func.id in ('suite','worker') for n in tree.body))


if __name__=='__main__':unittest.main()
