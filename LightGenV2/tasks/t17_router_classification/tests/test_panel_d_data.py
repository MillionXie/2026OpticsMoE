import json
import numpy as np
import pytest
from LightGenV2.tasks.t17_router_classification.panel_d_data import load_organ,validate_panel_d_initialization
from LightGenV2.tasks.t17_router_classification.train_figure_revision import checkpoint_selection_key
from LightGenV2.tasks.t11_lifelong_optics.crc9_data import sha256


def test_organ_keeps_original_official_pixels_labels_ids_and_all11_classes(tmp_path):
    raw={};cached={};split_ids={}
    for split in ('train','val','test'):
        x=np.arange(44,dtype=np.uint8).reshape(11,2,2);y=np.arange(11,dtype=np.uint8)[:,None];raw[split+'_images']=x;raw[split+'_labels']=y
        if split!='test':
            ids=np.array([f'organcmnist:{split}:{i:06d}' for i in range(11)]);cached.update({split+'_images':np.repeat(x[:,:,:,None],3,axis=3),split+'_labels':y.reshape(-1),split+'_ids':ids});split_ids[split]=ids.tolist()
    source=tmp_path/'source.npz';cache=tmp_path/'cache.npz';manifest=tmp_path/'manifest.json';np.savez(source,**raw);np.savez(cache,**cached);manifest.write_text(json.dumps(dict(source_sha256=sha256(source),cache_sha256=sha256(cache),split_ids=split_ids,classes=list(range(11)),license='CC BY 4.0')))
    cfg=dict(source_archive=str(source),source_archive_sha256=sha256(source),raw_cache=str(cache),raw_cache_sha256=sha256(cache),organ_manifest=str(manifest),organ_manifest_sha256=sha256(manifest),expected_counts={s:11 for s in ('train','val','test')})
    z,identity=load_organ(cfg);assert identity['original_TRAIN_VAL_pixels_labels_IDs_exact'] and z['test']['labels'].tolist()==list(range(11))
    assert np.array_equal(z['test']['images'][:,:,:,0],raw['test_images']) and len(z['test']['ids'])==11
    cached['train_images'][0,0,0,0]+=1;np.savez(cache,**cached)
    with pytest.raises(ValueError,match='SHA'):load_organ(cfg)


def test_A_initialization_rejects_different_data_and_cannot_change_reference_k2():
    cfg=dict(dataset='kather2016',architecture='LG',main_layers=2,top_k=3,num_classes=8,source_archive_sha256='source',raw_cache_sha256='cache',expected_counts={'test':752})
    parent=dict(metadata=dict(profile=cfg));validate_panel_d_initialization(parent,cfg)
    with pytest.raises(ValueError,match='identity'):validate_panel_d_initialization(parent,dict(cfg,raw_cache_sha256='changed'))
    with pytest.raises(ValueError,match='approved'):validate_panel_d_initialization(parent,dict(cfg,top_k=2))


def test_D_TEST_accuracy_selection_is_not_balanced_accuracy_or_validation():
    a=dict(accuracy=.8,balanced_accuracy=.5,cross_entropy=.6);b=dict(accuracy=.7,balanced_accuracy=.9,cross_entropy=.5);cfg=dict(selection_source='original_TEST_development',selection_mode='accuracy_first')
    assert checkpoint_selection_key(b,a,cfg)>checkpoint_selection_key(a,b,cfg)
