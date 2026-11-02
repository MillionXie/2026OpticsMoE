import unittest
from pathlib import Path
import tempfile
from ..scene_portable_inference import inside,checked_manifest,subset,GENERATOR,FRONTEND,PAIRS,REQUESTS


class PortableTests(unittest.TestCase):
    def test_paths_cannot_escape_package(self):
        with tempfile.TemporaryDirectory()as name:
            root=Path(name)
            for invalid in ['../weights','data/../../secret',str(root.resolve())]:
                with self.assertRaises(ValueError):inside(root,invalid)
            self.assertEqual(inside(root,'models/Qwen'),root.resolve()/'models/Qwen')

    def test_weight_and_sampling_identity_are_not_interchangeable(self):
        v=dict(schema='t12_portable_scene_assets_v1',complete=True,generator_sha256=GENERATOR,
            frontend_sha256=FRONTEND,pairs_sha256=PAIRS,requests_sha256=REQUESTS,TRAIN=624,TEST=500,
            UNet_blocks=70,steps=8,cached_Qwen_features_included=False)
        checked_manifest(v)
        for key,value in [('generator_sha256','0'*64),('steps',12),('UNet_blocks',64),('cached_Qwen_features_included',True)]:
            with self.assertRaises(ValueError):checked_manifest(dict(v,**{key:value}))

    def test_smoke_subset_does_not_relabel_full_cohort(self):
        rows=[dict(sample_id=str(i))for i in range(500)];metadata=dict(contexts=500)
        selected,record=subset(rows,metadata,1)
        self.assertEqual(selected,[rows[0]]);self.assertEqual(len(rows),500)
        self.assertEqual(record['portable_subset_count'],1);self.assertEqual(metadata,dict(contexts=500))
        for n in [0,501,True]:
            with self.assertRaises(ValueError):subset(rows,metadata,n)


if __name__=='__main__':unittest.main()
