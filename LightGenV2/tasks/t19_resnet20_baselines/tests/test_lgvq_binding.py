import copy

import pytest
import torch

from LightGenV2.tasks.t19_resnet20_baselines.lgvq_data import validate_payload


def fixture():
    rows=[{'sample_id':'train/a','video_path':'a.mp4','split':'train','spatial':'1','temporal':'2'},
          {'sample_id':'test/b','video_path':'b.mp4','split':'test','spatial':'3','temporal':'4'}]
    p={'sample_ids':['train/a','test/b'],'splits':['train','test'],'video_paths':['a.mp4','b.mp4'],
       'targets':torch.tensor([[1.,2.],[3.,4.]]),'target_names':['spatial','temporal'],
       'feature_contract':'raw_rgb_uint8_four_frames_center65_224_v1','crop_fraction':.65,
       'frame_size':224,'frame_fractions':[.10,.37,.63,.90],'frames':torch.zeros(2,4,3,224,224,dtype=torch.uint8)}
    return rows,p


def test_bound_video_labels_and_four_raw_frames():
    rows,p=fixture()
    validate_payload(p,rows,counts={'train':1,'test':1})


@pytest.mark.parametrize('key,value',[('sample_ids',['test/b','train/a']),('splits',['test','train']),
    ('target_names',['temporal','spatial']),('crop_fraction',1.),('frame_fractions',[0.,.3,.6,.9])])
def test_reject_shifted_labels_or_preprocessing(key,value):
    rows,p=fixture();p[key]=value
    with pytest.raises(ValueError):validate_payload(p,rows,counts={'train':1,'test':1})
