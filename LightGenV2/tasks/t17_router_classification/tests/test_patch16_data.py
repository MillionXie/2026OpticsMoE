import io,json,zipfile
import numpy as np
import pytest
from PIL import Image
from LightGenV2.tasks.t17_router_classification.patch16_data import Patch16Dataset
from LightGenV2.tasks.t17_router_classification.train_patch16 import route_summary

def test_original_zip_sample_identity_and_pixels(tmp_path):
    pixels=np.full((395,395,3),[12,80,200],dtype=np.uint8)
    b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG')
    with zipfile.ZipFile(tmp_path/'data.zip','w') as z:z.writestr('tiles/skin/sample.png',b.getvalue())
    m=dict(splits=dict(train=[dict(id='tiles/skin/sample.png',label=15)]))
    ds=Patch16Dataset(tmp_path,'train',m)
    image,label,index=ds[0]
    assert np.array_equal(image,pixels)
    assert label==15 and index==0
    ds.archive.close()

def test_actual_top2_pair_statistics_expose_collapsed_router():
    fixed=np.tile([0.,0.,.55,.45],(12,1))
    varied=np.array([[.5,0,.5,0],[0,.5,.5,0],[0,0,.5,.5],[.5,.5,0,0]])
    a=route_summary(fixed);b=route_summary(varied)
    assert a['top2_pair_fraction']=={'3,4':1.}
    assert b['effective_experts']>a['effective_experts']
    assert b['mean_selected_count']==2.
    assert len(b['top2_pair_fraction'])==4
