import json
from pathlib import Path
import tempfile
import unittest
from .locked_test_cache import file_sha, verified_cached_result


class CacheTests(unittest.TestCase):
    def test_cache_requires_matching_model_data_and_prediction_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);pred=run/'test_predictions.npz';pred.write_bytes(b'fixed predictions')
            identity=dict(checkpoint_sha256='model',checkpoint_file='best_checkpoint.pt',
                          data_sha256='data',data_manifest_sha256='manifest',test_source_sha256='test')
            payload=dict(identity,state='complete',locked_result_cache=True,
                         predictions_sha256=file_sha(pred),test=dict(accuracy=.8))
            result=run/'test_result.json';result.write_text(json.dumps(payload))
            self.assertEqual(verified_cached_result(run,'test',identity)['test']['accuracy'],.8)
            for field in identity:
                changed=dict(identity);changed[field]='different'
                with self.assertRaises(ValueError):verified_cached_result(run,'test',changed)
            pred.write_bytes(b'changed predictions')
            with self.assertRaises(ValueError):verified_cached_result(run,'test',identity)

    def test_missing_or_unmarked_result_does_not_enable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            self.assertIsNone(verified_cached_result(run,'test',{}))
            (run/'test_result.json').write_text('{}')
            self.assertIsNone(verified_cached_result(run,'test',{}))


if __name__=='__main__':unittest.main()
