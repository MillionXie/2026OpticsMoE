import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
import csv
import hashlib
from types import SimpleNamespace
from maintenance.git_safety.check_t08_readout_cache import inspect, digest
from maintenance.git_safety.check_t08_readout_cache import selected_indexes


class ReadoutSelectionTests(unittest.TestCase):
    def test_unknown_split_rejected(self):
        with self.assertRaises(ValueError): selected_indexes([], 'other')
    def test_original_train_selection_is_fixed_and_balanced(self):
        rows = [{'label': str(i // 48)} for i in range(4800)]
        indexes = selected_indexes(rows, 'train')
        self.assertEqual(indexes, selected_indexes(rows, 'train'))
        self.assertEqual(len(set(indexes)), 800)
        self.assertEqual(indexes, sorted(indexes))
        for label in range(100): self.assertEqual(sum(i // 48 == label for i in indexes), 8)

    def test_short_train_rejected(self):
        with self.assertRaises(ValueError): selected_indexes([{'label': '0'}], 'train')

    def test_original_test_order(self):
        self.assertEqual(selected_indexes([{}] * 2400, 'test'), list(range(2400)))
        with self.assertRaises(ValueError): selected_indexes([{}], 'test')


class CacheInspectionTests(unittest.TestCase):
    def test_late_train_change_is_detected_after_test_inspection(self):
        # Metadata-only fake tensors: exercises guards, not Torch numerical correctness.
        class Tensor:
            dtype='float32'
            def __init__(self, count): self.shape=(count,384)
            def contiguous(self): return self
            def numpy(self): return self
            def tobytes(self): return b'fixture-feature'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); evidence=self.fixture(root); evidence['body_sha256']='body'
            feature_sha=hashlib.sha256(b'fixture-feature').hexdigest()
            for record in evidence['caches']:
                record['tensor_hashes']=dict(image_features=feature_sha,title_features=feature_sha)
            def load(path, **kwargs):
                self.assertEqual(kwargs['map_location'],'cpu')
                split=Path(path).stem
                with (root/(split+'.csv')).open(encoding='utf8') as f: rows=list(csv.DictReader(f))
                indexes=selected_indexes(rows,split)
                if split=='test': (root/'train.pt').write_bytes(b'changed-late')
                return dict(split=split,checkpoint_sha256='body',selection_seed=20260926,train_per_sku=8,
                            image_keys=[('train_' if split=='train' else 'image_')+f'{i:04d}' for i in indexes],
                            title_keys=[f'title_{i:03d}' for i in range(100)],
                            image_labels=[int(rows[i]['label']) for i in indexes],
                            image_features=Tensor(len(indexes)),title_features=Tensor(100))
            fake=SimpleNamespace(load=load,float32='float32',isfinite=lambda t:SimpleNamespace(all=lambda:True))
            with patch.dict(sys.modules,{'torch':fake}), self.assertRaisesRegex(RuntimeError,'before inspection completed'):
                inspect(root,root/'train.pt',root/'test.pt',evidence)

    def fixture(self, root):
        for split, count, per_sku in (('train', 4800, 48), ('test', 2400, 24)):
            text = 'label,product_id,sample_id\n' + ''.join(
                f'{i//per_sku},p{i//per_sku},{split}{i}\n' for i in range(count))
            (root/(split+'.csv')).write_text(text, encoding='utf8')
        (root/'titles.csv').write_text('label,product_id\n'+''.join(f'{i},p{i}\n' for i in range(100)), encoding='utf8')
        for split in ('train', 'test'): (root/(split+'.pt')).write_bytes(split.encode())
        return dict(dataset_csv_sha256={s:digest(root/(s+'.csv')) for s in ('train','test','titles')},
                    caches=[dict(split=s,cache_sha256=digest(root/(s+'.pt'))) for s in ('train','test')])

    def test_bad_cache_sha_stops_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); evidence=self.fixture(root)
            (root/'test.pt').write_bytes(b'changed')
            def forbidden(*args, **kwargs): self.fail('Unverified cache deserialized')
            with patch.dict(sys.modules, {'torch':SimpleNamespace(load=forbidden)}):
                with self.assertRaisesRegex(ValueError, 'cache SHA mismatch'):
                    inspect(root,root/'train.pt',root/'test.pt',evidence)

    def test_bad_csv_sha_stops_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); evidence=self.fixture(root)
            (root/'train.csv').write_text('changed',encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'CSV SHA mismatch'):
                inspect(root,root/'train.pt',root/'test.pt',evidence)

    def test_incomplete_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            inspect('.', 'missing', 'missing', {'dataset_csv_sha256':{}, 'caches':[]})

    def test_duplicate_split_manifest_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            inspect('.', 'missing', 'missing', dict(dataset_csv_sha256=dict.fromkeys(('train','test','titles'),'x'),
                    caches=[{'split':'train'},{'split':'train'}]))
