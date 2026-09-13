import unittest
from desktop_job import command

class JobTests(unittest.TestCase):
    def test_allowlist(self):
        with self.assertRaises(ValueError):command({'action':'delete'})
        with self.assertRaises(ValueError):command({'action':'init','session':'../oops'})
        with self.assertRaises(ValueError):command({'action':'init','session':'a','limit':-1})
    def test_prepare(self):
        cmd=command({'action':'prepare','session':'test','stage':'vision_router'})
        self.assertIn('cuda',cmd);self.assertNotIn('--yes',cmd)
    def test_config_path_escape(self):
        for path in ('../outside.json','config.json','sessions/x/config.json'):
            with self.assertRaises(ValueError):command({'action':'probe','config':path,'bmp':'a.bmp','out':'results/x'})

if __name__=='__main__':unittest.main()
