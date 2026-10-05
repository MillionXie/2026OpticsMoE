import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('t11_receipts',Path(__file__).resolve().parents[1]/'check_t11_data_receipts.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReceiptTests(unittest.TestCase):
    def fixture(self):
        return ({'records':[{'path':'run/metadata.json','matches_registered':True,
                 'metadata':{'command':['--data','/d.npz','--manifest','/manifest.json'],
                             'data_sha256':'abc','manifest':{'cache_sha256':'abc'}}}]},
                {'assets':[{'path':'/d.npz','exists':True,'sha256':'abc','expected_sha256':[]},
                           {'path':'/manifest.json','exists':True,'sha256':'def','manifest':{'cache_sha256':'abc'}}]})
    def test_valid(self):
        m,c=self.fixture();r=module.inspect(m,c)
        self.assertEqual(r['errors'],[]);self.assertTrue(r['npz'][0]['matches_historical'])
    def test_data_changed(self):
        m,c=self.fixture();c['assets'][0]['sha256']='wrong'
        self.assertIn('data hash conflict: /d.npz',module.inspect(m,c)['errors'])
    def test_manifest_changed(self):
        m,c=self.fixture();c['assets'][1]['manifest']['cache_sha256']='wrong'
        self.assertIn('embedded manifest differs: /manifest.json',module.inspect(m,c)['errors'])
    def test_unbound_is_not_pass(self):
        m,c=self.fixture();m['records'][0]['metadata'].pop('data_sha256')
        self.assertFalse(module.inspect(m,c)['npz'][0]['matches_historical'])

if __name__=='__main__':unittest.main()
