import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from LightGenV2.tasks.t12_text_to_image.scene_paired_render import frozen_test_protocol,inference_inputs
from LightGenV2.tasks.t12_text_to_image.white_reference_metrics import load_records
from LightGenV2.tasks.t12_text_to_image.white_reference_scene_probe import digest


class FrozenSceneTEST(unittest.TestCase):
    def fixture(self,root):
        (root/'inference_inputs.json').write_text('[]')
        profile='two_sentence_furnished_room_v3'
        (root/'report.json').write_text(json.dumps({'complete':True,'role_disjoint':True,'prompt_profile':profile,
            'request_sha256':digest(root/'inference_inputs.json'),'current_pairs_sha256':'pairs'}))
        model=dict(checkpoint_sha256='weights',sampling_steps=30,IP_adapter_scale=.25,resolution=512,Qwen_input_side=256,
            variant='qwen',joint_image_context=False,sampling_contract=None,optical_frontend_sha256=None)
        frozen={'schema':'scene_frozen_evaluation_v1','development_selection_finished':True,'TEST_used_for_model_selection':False,
            'frozen_before_first_TEST_prediction':True,'prompt_profile':profile,'request_sha256':digest(root/'inference_inputs.json'),'models':{'qwen':model}}
        path=root/'freeze.json';path.write_text(json.dumps(frozen))
        a=SimpleNamespace(data=root,frozen_evaluation=path,frozen_model_key='qwen',steps=30,ip_scale=.25,measure_software_latency=False)
        state=dict(pair_sha256='pairs',resolution=512,Qwen_input_side=256,variant='qwen')
        return a,state,{'best_sha256':'weights'},frozen

    def test_protocol_requires_freeze_and_exact_model_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            a,state,ready,frozen=self.fixture(Path(folder));self.assertEqual(frozen_test_protocol(a,state,ready)['model_key'],'qwen')
            a.frozen_evaluation=None
            with self.assertRaises(ValueError):frozen_test_protocol(a,state,ready)
            a.frozen_evaluation=Path(folder)/'freeze.json';a.ip_scale=.4
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)
            a.ip_scale=.25
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,{'best_sha256':'another_weight'})

    def test_changed_requests_and_target_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a,state,ready,frozen=self.fixture(root)
            (root/'inference_inputs.json').write_text('[{}]')
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)
            safe={k:'input' for k in ['sample_id','reference','prompt','seed','reference_id','reference_sha256','category',
                'condition_id','native_caption','reward_caption','prompt_profile']}
            safe['target']='forbidden.png';(root/'inference_inputs.json').write_text(json.dumps([safe]))
            with self.assertRaises(ValueError):inference_inputs(root,'test')

    def test_natural_prompt_TEST_revision_keeps_frozen_source_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a,state,ready,frozen=self.fixture(root)
            report=json.loads((root/'report.json').read_text())
            report.update(schema='scene_natural_prompt_TEST_v1',prompt_profile='two_sentence_natural_room_v4',
                original_training_prompt_profile='two_sentence_furnished_room_v3',source_TEST_request_sha256='a'*64)
            (root/'report.json').write_text(json.dumps(report))
            frozen.update(prompt_profile=report['prompt_profile'],source_TEST_request_sha256='a'*64)
            a.frozen_evaluation.write_text(json.dumps(frozen))
            self.assertEqual(frozen_test_protocol(a,state,ready)['model_key'],'qwen')
            frozen['source_TEST_request_sha256']='b'*64;a.frozen_evaluation.write_text(json.dumps(frozen))
            with self.assertRaises(AssertionError):frozen_test_protocol(a,state,ready)

    def test_TEST_metrics_need_explicit_scope_and_all_draws(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'ref.png').write_bytes(b'reference');(root/'generated.png').write_bytes(b'generation')
            report={'complete':True,'training_performed':False,'independent_TEST_opened':True,'scope':'TEST','generated_count':500,
                'frozen_TEST_protocol':{'manifest_sha256':'freeze'}}
            (root/'report.json').write_text(json.dumps(report))
            rows=[dict(sample_id=str(i),reference_id='ref',condition_id='0',seed=i,split='test',generated_PNG='generated.png',
                generated_sha256=digest(root/'generated.png'),input_sha256=digest(root/'ref.png')) for i in range(500)]
            (root/'images.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):list(load_records(root))
            self.assertEqual(len(list(load_records(root,True))),500)
            (root/'images.json').write_text(json.dumps(rows[:-1]))
            with self.assertRaises(ValueError):list(load_records(root,True))


if __name__=='__main__':unittest.main()
