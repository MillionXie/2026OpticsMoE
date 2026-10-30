import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from ..white_reference_metrics import check_architecture_diagnostic,load_records
from ..white_reference_scene_probe import digest


def fixture():
    config=dict(schema='scene_depth_probe_v1',vision_keep=6,language_keep=4,UNet_deep_keep=8,
        parent_generator_sha256='f9fbdb7fd36d1c862686616a624e7c18f4e9149cef61b0704bbf97767de6eaec',
        parent_frontend_sha256='6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e')
    graph=dict(schema='scene_depth_probe_actual_graph_v1',profile=config,training_performed=False,TEST_opened=False)
    report=dict(scope='DEV_architecture_diagnostic',complete=True,generated_count=32,training_performed=False,
        independent_TEST_opened=False,checkpoint_sha256=config['parent_generator_sha256'],experimental_compression=graph,
        expanded_DEV_protocol=dict(contexts=32,full_audited_contexts=128,deterministic_scene_subset=['0','1'],selection_by_quality=False))
    rows=[dict(reference_id='r'+str(i),condition_id=str(c),seed=1042+i,input_sha256='same-reference-'+str(i),
        category='lamp'if i<8 else'pillow',split='dev',experimental_compression=graph)for i in range(16)for c in range(2)]
    return report,rows


class DepthMetricScope(unittest.TestCase):
    def test_complete_balanced_fixed_cohort_is_supported(self):
        report,rows=fixture();check_architecture_diagnostic(report,rows)

    def test_equal_count_cherry_picked_scene_or_noise_is_rejected(self):
        for key,value in [('condition_id','7'),('seed',88),('input_sha256','different')]:
            report,rows=fixture();rows[0][key]=value
            with self.assertRaises(ValueError):check_architecture_diagnostic(report,rows)

    def test_partial_or_unknown_graph_is_not_relabelled_as_valid(self):
        report,rows=fixture()
        with self.assertRaises(ValueError):check_architecture_diagnostic(report,rows[:-1])
        report['experimental_compression']=dict(report['experimental_compression'],schema='unknown')
        with self.assertRaises(ValueError):check_architecture_diagnostic(report,rows)

    def test_real_loader_checks_all_file_hashes_after_scope_validation(self):
        report,rows=fixture()
        with TemporaryDirectory()as temp:
            root=Path(temp)
            for i,row in enumerate(rows):
                ref=root/(row['reference_id']+'.png');ref.write_bytes(b'white input')
                out=root/(str(i)+'.png');out.write_bytes(b'generated output')
                row.update(generated_PNG=out.name,generated_sha256=digest(out),input_sha256=digest(ref))
            (root/'report.json').write_text(json.dumps(report));(root/'images.json').write_text(json.dumps(rows))
            self.assertEqual(len(list(load_records(root))),32)
            (root/'0.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):list(load_records(root))


if __name__=='__main__':unittest.main()
