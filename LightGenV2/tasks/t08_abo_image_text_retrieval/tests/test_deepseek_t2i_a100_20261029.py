"""No CUDA/model: original catalog identities and unsafe-path rejection."""
import copy
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import os
import time
from LightGenV2.tasks.t08_abo_image_text_retrieval import deepseek_t2i_a100_20261029 as task


class Contract(unittest.TestCase):
    def setUp(self):
        self.titles=[dict(label=str(i),product_id='sku'+str(i)) for i in range(100)]
        self.rows=[dict(label=str(i),product_id='sku'+str(i),sample_id=f'{i}_{j}',split='test',image_path=f'images/sku{i}/{j}.jpg') for i in range(100) for j in range(24)]
    def test_exact_catalog(self):task.validate_rows(self.titles,self.rows)
    def test_missing_candidate(self):
        with self.assertRaises(ValueError):task.validate_rows(self.titles,self.rows[:-1])
    def test_duplicate_id(self):
        self.rows[1]['sample_id']=self.rows[0]['sample_id']
        with self.assertRaises(ValueError):task.validate_rows(self.titles,self.rows)
    def test_wrong_sku(self):
        self.rows[0]['product_id']='other'
        with self.assertRaises(ValueError):task.validate_rows(self.titles,self.rows)
    def test_unsafe_paths(self):
        for value in ('../images/a.jpg','/images/a.jpg','images/../a.jpg','weights/a.pt'):
            with self.assertRaises(ValueError):task.checked_relative(value)
    def test_no_model_import(self):self.assertNotIn('torch',sys.modules)
    def test_foreign_rejected_once(self):
        value=SimpleNamespace(returncode=0,stdout='999999, '+task.GPU+'\n')
        with patch.object(task.subprocess,'run',return_value=value) as command:
            with self.assertRaises(RuntimeError):task.Guard(time.time()+60).check(True)
            self.assertEqual(command.call_count,1)
    def test_self_only_after_admission(self):
        value=SimpleNamespace(returncode=0,stdout=str(os.getpid())+', '+task.GPU+'\n')
        with patch.object(task.subprocess,'run',return_value=value):
            task.Guard(time.time()+60).check(True)
            with self.assertRaises(RuntimeError):task.Guard(time.time()+60).check(False)


if __name__=='__main__':unittest.main()
