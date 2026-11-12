"""Public export preserves tensors while limiting checkpoint metadata."""
from pathlib import Path
import tempfile
import unittest

import torch

from ..tools.checkpoint_utils import export_public_checkpoint, tensor_fingerprint


class PublicExport(unittest.TestCase):
    def test_export_tensor_identity_and_metadata(self):
        state={'weights':torch.arange(6,dtype=torch.bfloat16),'ready':torch.tensor(True)}
        payload={'schema':7,'architecture':'fixture','config':{'width':8},'model':state,
                 'metadata':{'command':['private-local-command'],'source_path':'private-local-path',
                             'dataset':{'manifest':{'path':'private-manifest','sha256':'ab'*32}},
                             'metrics':{'srcc':.79}}}
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source.pt';destination=Path(folder)/'public.pt'
            torch.save(payload,source)
            record=export_public_checkpoint(source,destination)
            public=torch.load(destination,weights_only=True)
        self.assertEqual(tensor_fingerprint(public['model']),tensor_fingerprint(state))
        self.assertTrue(record['model_tensors_bitwise_equal'])
        self.assertNotIn('private',repr(public['metadata']))
        self.assertEqual(public['metadata']['evaluation']['cached_srcc'],.79)


if __name__=='__main__':
    unittest.main()
