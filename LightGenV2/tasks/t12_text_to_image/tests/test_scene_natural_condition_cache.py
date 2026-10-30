import hashlib
from pathlib import Path
import tempfile
import unittest
from LightGenV2.tasks.t12_text_to_image.appearance_scene_protocol import conditions,SHORT_PROFILE
from LightGenV2.tasks.t12_text_to_image.scene_natural_condition_cache import requests,trajectory_keys


class NaturalTRAINCache(unittest.TestCase):
    def test_only_actual_TRAIN_input_fingerprint_and_whitelisted_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'ref.png').write_bytes(b'input pixels')
            rows=[]
            for category in ['lamp','pillow']:
                rows.append(dict(conditions(category,SHORT_PROFILE)[0],split='train',reference='ref.png',
                    reference_id=category,reference_sha256=hashlib.sha256(b'input pixels').hexdigest(),
                    target='forbidden.png',target_latent='forbidden.pt',source_product_id='SKU'))
            rows.append(dict(rows[0],split='dev',reference='nonexistent_DEV.png'))
            q=requests(rows,root);self.assertEqual(len(q),2)
            for key,r in q.items():
                self.assertEqual(key,hashlib.sha256(b'input pixels\0'+r['prompt'].encode()+b'\0Qwen256').hexdigest())
                self.assertFalse(set(r)&{'target','target_latent','source_product_id'})
                self.assertEqual(r['split'],'train');self.assertNotIn('reference',r['prompt'])
            expanded=requests(rows,root,True);self.assertEqual(len(expanded),16)
            one_reference={k:v for k,v in expanded.items() if v['category']=='lamp'}
            keys=trajectory_keys(one_reference,4);self.assertEqual(len(keys),4)
            self.assertTrue(set(keys)<=set(expanded))
            rows[0]['reference_sha256']='changed'
            with self.assertRaises(ValueError):requests(rows,root)


if __name__=='__main__':unittest.main()
