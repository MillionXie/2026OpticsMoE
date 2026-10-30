import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ..appearance_scene_protocol import SHORT_PROFILE,conditions
from ..scene_development_grid import expand,SAFE
from ..scene_natural_prompt import revise,PROFILE
from ..scene_omnigen_peer import matched_natural_inputs
from ..white_reference_scene_probe import digest


def fixture(root):
    data=root/'data';revision=root/'revision';data.mkdir();revision.mkdir();rows=[]
    for item in range(8):
        for view in range(2):
            rid=f'product{item}-view{view}';ref=data/(rid+'.png');ref.write_bytes(rid.encode())
            for c in conditions('lamp' if item<4 else 'pillow',SHORT_PROFILE)[:2]:
                row=dict(c,reference=ref.name,reference_sha256=digest(ref),reference_id=rid,
                    sample_id=rid+'-scene'+c['condition_id']+'-draw2',seed=1042+item*2+view)
                rows.append({k:v for k,v in row.items() if k in SAFE})
    source=data/'inference_dev_inputs.json';source.write_text(json.dumps(rows))
    (data/'report.json').write_text(json.dumps(dict(complete=True,role_disjoint=True,TEST_opened=False,pairs_sha256='pair')))
    request=revision/'inference_inputs.json';request.write_text(json.dumps([revise(r) for r in expand(rows)]))
    (revision/'report.json').write_text(json.dumps(dict(complete=True,schema='scene_natural_prompt_DEV_v1',
        TEST_opened=False,reference_RGB_still_input=True,source_DEV_request_sha256=digest(source),
        pairs_sha256='pair',request_sha256=digest(request))))
    return data,revision


class OmniGenMatchedProtocol(unittest.TestCase):
    def test_exact_all128_public_prompts_and_counterfactual_noise(self):
        with TemporaryDirectory() as temp:
            data,revision=fixture(Path(temp));rows,protocol=matched_natural_inputs(data,revision)
            self.assertEqual(len(rows),128);self.assertEqual(protocol['prompt_profile'],PROFILE)
            self.assertEqual(len({r['reference_id'] for r in rows}),16)
            for rid in {r['reference_id'] for r in rows}:
                self.assertEqual(len({r['seed'] for r in rows if r['reference_id']==rid}),1)
            self.assertTrue(all(r['prompt'].count('.')==2 and 'reference' not in r['prompt'] for r in rows))

    def test_changed_reference_pixels_are_rejected(self):
        with TemporaryDirectory() as temp:
            data,revision=fixture(Path(temp));next(data.glob('*.png')).write_bytes(b'other image')
            with self.assertRaises(ValueError):matched_natural_inputs(data,revision)

    def test_teacher_target_inference_key_is_rejected(self):
        with TemporaryDirectory() as temp:
            data,revision=fixture(Path(temp));p=data/'inference_dev_inputs.json';rows=json.loads(p.read_text())
            rows[0]['target']='forbidden.png';p.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):matched_natural_inputs(data,revision)

    def test_changed_prompt_cannot_pass_by_rehashing(self):
        with TemporaryDirectory() as temp:
            data,revision=fixture(Path(temp));p=revision/'inference_inputs.json';rows=json.loads(p.read_text())
            rows[0]['prompt']='Generate an easier scene.';p.write_text(json.dumps(rows))
            q=revision/'report.json';report=json.loads(q.read_text());report['request_sha256']=digest(p);q.write_text(json.dumps(report))
            with self.assertRaises(ValueError):matched_natural_inputs(data,revision)


if __name__=='__main__':unittest.main()
