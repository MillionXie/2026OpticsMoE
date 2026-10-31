import pytest
import torch
import json
import numpy as np
from LightGenV2.tasks.t17_router_classification.train_A_reference_TEST_readout import test_accuracy_key as accuracy_key
from LightGenV2.tasks.t17_router_classification.train_A_reference_TEST_readout import verify_reference_cache
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def test_reference_selection_uses_accuracy_not_validation_or_macro_recall():
    assert accuracy_key(dict(accuracy=.8,cross_entropy=.5))>accuracy_key(dict(accuracy=.7,cross_entropy=.1))


def test_reference_cache_binds_every_optical_tensor_and_rejects_label_changes(tmp_path):
    folder=tmp_path/'A';folder.mkdir();phase=torch.ones(2,2);torch.save(dict(model={'phase':phase}),tmp_path/'optical.pt');parent=dict(model={'phase':phase.clone()});manifest=dict(parent_sha256=sha256(tmp_path/'optical.pt'),features={})
    (folder/'metadata.json').write_text(json.dumps(dict(profile={'parent_checkpoint':'optical.pt'})))
    for split in ('train','val','test'):
        values=dict(ids=np.array(['a','b']),labels=np.array([0,1]),features=np.ones((2,784),dtype=np.float32));np.savez(folder/f'{split}_features.npz',**values);np.savez(folder/f'{split}_predictions.npz',ids=values['ids'],labels=values['labels']);manifest['features'][split]=sha256(folder/f'{split}_features.npz')
    (folder/'cache_manifest.json').write_text(json.dumps(manifest));verify_reference_cache(folder,parent,tmp_path)
    with pytest.raises(ValueError,match='phases'):verify_reference_cache(folder,dict(model={'phase':phase+1}),tmp_path)
    p=folder/'test_predictions.npz';np.savez(p,ids=np.array(['a','b']),labels=np.array([1,0]))
    with pytest.raises(ValueError,match='labels'):verify_reference_cache(folder,parent,tmp_path)
