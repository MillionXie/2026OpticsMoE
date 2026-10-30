import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ..scene_compression_delivery import verified_models,degradation,ROLES,METRICS
from ..white_reference_scene_probe import digest


def fixture(root):
    specs=[]
    for role in ROLES:
        run=root/role;run.mkdir();scores=root/(role+'_scores');scores.mkdir();rows=[]
        for index in range(128):
            ref=run/(f'ref{index//8}.png');ref.write_bytes(b'fixed reference')
            image=run/(str(index)+'.png');image.write_bytes(role.encode()+str(index).encode())
            rows.append(dict(sample_id=str(index),reference_id=f'ref{index//8}',condition_id=str(index%8),
                seed=index//8,initial_latent_sha256='fixed noise',category='lamp'if index<64 else'pillow',
                input_sha256=digest(ref),split='dev',prompt='Generate a scene. Include a plant.',
                generated_PNG=image.name,generated_sha256=digest(image)))
        report=dict(complete=True,scope='DEV_expanded',training_performed=False,independent_TEST_opened=False,
            generated_count=128,resolution=512,prompt_profile='two_sentence_natural_room_v4',
            expanded_DEV_protocol={'identity':'same'},compression_training_performed=role=='light_lg')
        (run/'report.json').write_text(json.dumps(report));(run/'images.json').write_text(json.dumps(rows))
        (scores/'report.json').write_text(json.dumps(dict(complete=True,metric_protocol={'fixed':'v1'})))
        (scores/'per_image.json').write_text(json.dumps([dict(r,run_id=run.name,**{m:.5 for m in METRICS}) for r in rows]))
        specs.append(dict(role=role,label=role,run=str(run),metrics=str(scores),training_note='fixture'))
    return {'models':specs}


class CompressionDeliveryAudit(unittest.TestCase):
    def test_complete_same_requests_are_accepted(self):
        with TemporaryDirectory()as temp:
            manifest=fixture(Path(temp));models=verified_models(manifest)
            self.assertEqual([m['role']for m in models],list(ROLES))
            self.assertEqual(sum(len(m['images'])for m in models),640)

    def test_score_bound_to_other_pixels_is_rejected(self):
        with TemporaryDirectory()as temp:
            manifest=fixture(Path(temp));p=Path(manifest['models'][0]['metrics'])/'per_image.json'
            rows=json.loads(p.read_text());rows[0]['generated_sha256']='other pixels';p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'displayed'):verified_models(manifest)

    def test_changed_prompt_and_partial_comparisons_are_rejected(self):
        with TemporaryDirectory()as temp:
            manifest=fixture(Path(temp));p=Path(manifest['models'][1]['run'])/'images.json'
            rows=json.loads(p.read_text());rows[0]['prompt']='Other request';p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'differ'):verified_models(manifest)
            with self.assertRaisesRegex(ValueError,'five'):verified_models({'models':manifest['models'][:-1]})

    def test_negative_reward_drop_has_correct_sign(self):
        self.assertAlmostEqual(degradation(-2.2,-2.),10.)
        self.assertAlmostEqual(degradation(-1.8,-2.),-10.)
        self.assertIsNone(degradation(0.,0.))


if __name__=='__main__':unittest.main()
