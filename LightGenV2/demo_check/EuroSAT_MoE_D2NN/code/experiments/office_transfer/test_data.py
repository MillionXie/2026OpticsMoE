import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from . import data

class DataContract(unittest.TestCase):
    def test_grouped_split_and_independent_domains(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=data.plan();p['data_root']=str(root);rng=np.random.default_rng(901)
            for dom in p['domains']:
                for cls in p['classes']:
                    folder=root/dom/cls;folder.mkdir(parents=True)
                    for i in range(35):Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(folder/f'{i}.png')
            # Exact cross-domain duplicate and one conflicting-label duplicate.
            a=root/p['domains'][0]/p['classes'][0]/'0.png'
            (root/p['domains'][1]/p['classes'][0]/'duplicate.png').write_bytes(a.read_bytes())
            (root/p['domains'][1]/p['classes'][1]/'conflict.png').write_bytes((root/p['domains'][0]/p['classes'][2]/'1.png').read_bytes())
            with patch.object(data,'plan',return_value=p):
                split=data.prepare();groups={}
                for row in split['records']:groups.setdefault(row['duplicate_group'],set()).add(row['split'])
                self.assertTrue(all(len(s)==1 for s in groups.values()))
                self.assertEqual(len(split['quarantined_label_conflicts']),2)
                self.assertEqual(split['split_sha256'],data.prepare()['split_sha256'])
                a=data.Images(split,0,'train');b=data.Images(split,1,'train')
                im,meta=next(data.train_batches(a,b,1,42,True))
                self.assertEqual(len(im),60)
                for d in (0,1):
                    self.assertEqual((meta[:,1]==d).sum().item(),30)
                    self.assertEqual(np.bincount(meta[meta[:,1]==d,0].numpy()).tolist(),[3]*10)
                self.assertTrue(all(x.size==(224,224) for x in im))

if __name__=='__main__':unittest.main()
