"""Pure CPU full-query join tests, never produce a claimed experiment result."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from LightGenV2.tasks.t08_abo_image_text_retrieval.build_lab_package import normalized_rankings


class PackageContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.titles=[dict(product_id=f'P{i:03d}',title=f'Product {i}') for i in range(100)]
        self.rows=[dict(sample_id=f'S{i:04d}',product_id=f'P{i//24:03d}',image_path=f'images/P{i//24:03d}/{i}.jpg') for i in range(2400)]
        self.ids=[[r['sample_id'] for r in self.rows[i*24:i*24+10]] for i in range(100)]
        self.pred=[dict(product_id=t['product_id'],top10=[dict(sample_id=v) for v in self.ids[i]]) for i,t in enumerate(self.titles)]
        with (self.root/'ours_10cm_predictions.csv').open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=['product_id','top10_sample_ids']);w.writeheader();w.writerows(dict(product_id=t['product_id'],top10_sample_ids=json.dumps(self.ids[i])) for i,t in enumerate(self.titles))
        (self.root/'qwen_dynamic_2048_predictions.json').write_text(json.dumps([dict(product_id=t['product_id'],top10_sample_ids=self.ids[i]) for i,t in enumerate(self.titles)]),encoding='utf8')
        (self.root/'clip_predictions.json').write_text(json.dumps(dict(text_to_image_top10=self.ids)),encoding='utf8')
    def test_all100_joined(self):
        result=normalized_rankings(self.titles,self.rows,self.root,self.pred)
        self.assertEqual(len(result),100);self.assertEqual(sum(len(v) for r in result for v in r['models'].values()),4000)
        self.assertIn('DeepSeek-VL2_full_new_A100',result[0]['models'])
        self.assertFalse(any('Tiny' in name for name in result[0]['models']))
    def test_wrong_query_rejected(self):
        self.pred[0]['product_id']='not_this_SKU'
        with self.assertRaises(ValueError):normalized_rankings(self.titles,self.rows,self.root,self.pred)
    def test_duplicate_top10_rejected(self):
        self.pred[0]['top10'][1]=self.pred[0]['top10'][0]
        with self.assertRaises(ValueError):normalized_rankings(self.titles,self.rows,self.root,self.pred)
    def test_unknown_photo_rejected(self):
        self.pred[0]['top10'][0]['sample_id']='missing'
        with self.assertRaises(ValueError):normalized_rankings(self.titles,self.rows,self.root,self.pred)


if __name__=='__main__':unittest.main()
