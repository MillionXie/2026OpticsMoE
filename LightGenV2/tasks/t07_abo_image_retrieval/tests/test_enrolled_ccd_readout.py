import copy

import pytest
import torch

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_refine import PROFILES, prepare_capacity_payload

PROTOCOL='abo200_enrolled_sku_hash8train4query_v1'


def fixture():
    return dict(metadata=dict(fusion_alpha_min=.4001,input_preprocessing='contain_white',
                              ccd_readout_modes=dict(vision='prefix_rows',language='prefix_rows')),
                state_dict={'language.optics.global_phase':torch.randn(8,8),
                            'readout.projection.weight':torch.randn(64,384)})


def test_fullfield_language_changes_only_metadata_and_no_parameters():
    src=fixture(); snapshot=copy.deepcopy(src)
    converted,audit=prepare_capacity_payload(src,PROFILES['sku_fullfield_language'],PROTOCOL)
    assert converted['state_dict'] is src['state_dict']
    assert src['metadata']==snapshot['metadata']
    before=dict(src['metadata']);after=dict(converted['metadata'])
    assert before.pop('ccd_readout_modes')==dict(vision='prefix_rows',language='prefix_rows')
    assert after.pop('ccd_readout_modes')==dict(vision='prefix_rows',language='fullfield_rows')
    assert before==after and audit['extra_parameters']==0 and audit['function_preserving'] is False
    p=dict(PROFILES['sku_fullfield_language']);p.pop('ccd_readout_modes')
    assert p==PROFILES['sku_capacity_control']


@pytest.mark.parametrize('bad',['fresh','protocol','expanded','optical_only','already_full','spatial_head'])
def test_fullfield_readout_refuses_mixed_or_old_protocol(bad):
    src=fixture();p=copy.deepcopy(PROFILES['sku_fullfield_language']);protocol=PROTOCOL
    if bad=='protocol':protocol='old'
    if bad=='expanded':p['head_expansion']='spatial2x2_64'
    if bad=='optical_only':p['optical_only']=True
    if bad=='already_full':src['metadata']['ccd_readout_modes']['language']='fullfield_rows'
    if bad=='spatial_head':src['metadata']['retrieval_head']='spatial2x2_64'
    with pytest.raises(ValueError):prepare_capacity_payload(src,p,protocol,bad=='fresh')
