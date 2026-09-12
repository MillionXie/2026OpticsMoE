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

if __name__=='__main__':unittest.main()
