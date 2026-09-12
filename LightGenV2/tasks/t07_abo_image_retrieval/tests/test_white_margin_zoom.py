import copy
import random
import numpy as np
import pytest
from PIL import Image
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.high_alpha import augment,white_margin_box
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.generalization import overlay_config


def image():
    a=np.full((224,224,3),255,dtype=np.uint8);a[50:174,60:164]=[20,60,120]
    return Image.fromarray(a)


def test_white_crop_retains_every_threshold_foreground_pixel_and_guard():
    im=image();a=np.array(im);box=white_margin_box(im);x0,y0,x1,y1=box
    assert box!=(0,0,224,224)
    assert 0<=x0<=44 and 0<=y0<=44 and 180<=x1<=224 and 180<=y1<=224
    ys,xs=np.nonzero(np.any(a<254,axis=2))
    assert xs.min()-x0>=4 and ys.min()-y0>=4
    assert x1-xs.max()-1>=4 and y1-ys.max()-1>=4
    # A small dark peripheral detail must never be discarded to obtain a zoom.
    a[1,1]=[0,0,0];box=white_margin_box(Image.fromarray(a))
    assert box[0]==0 and box[1]==0


@pytest.mark.parametrize('kind',['white','dark','tiny','border'])
def test_ambiguous_or_border_content_not_cropped(kind):
    a=np.full((224,224,3),255,dtype=np.uint8)
    if kind=='dark':a[:]=0
    if kind=='tiny':a[100:110,100:110]=0
    if kind=='border':a[0]=0;a[50:174,60:164]=0
    assert white_margin_box(Image.fromarray(a))==(0,0,224,224)


def config():
    return dict(minimum_crop_side_fraction=1.,rotation_degrees=0.,horizontal_flip_probability=0.,
                brightness_min=1.,brightness_max=1.,contrast_min=1.,contrast_max=1.,blur_probability=0.,blur_radius=.3)


def test_probability_zero_preserves_pixels_and_legacy_rng():
    c=config();zero=dict(c,white_margin_zoom_probability=0.)
    for seed in range(5):
        a,b=random.Random(seed),random.Random(seed)
        np.testing.assert_array_equal(augment(image(),a,c),augment(image(),b,zero))
        assert a.getstate()==b.getstate()


def test_zoom_is_rgb224_and_roughly_half_views_keep_original_geometry():
    c=dict(config(),white_margin_zoom_probability=.5);old=np.array(image());changed=0
    for seed in range(100):
        out=augment(image(),random.Random(seed),c)
        assert out.size==(224,224) and out.mode=='RGB'
        changed+=not np.array_equal(old,np.array(out))
    assert 30<changed<70


@pytest.mark.parametrize('p',[True,-.1,1.1,float('nan'),'0.5'])
def test_invalid_probability_rejected(p):
    with pytest.raises(ValueError):augment(image(),random.Random(1),dict(config(),white_margin_zoom_probability=p))


def test_training_only_profile_changes_no_inference_or_teacher_contract():
    base=overlay_config({},'domain_distill_joint_restart')
    candidate=overlay_config({},'domain_distill_joint_whitezoom');saved=copy.deepcopy(candidate)
    assert candidate['augmentation'].pop('white_margin_zoom_probability')==.5
    for c in (base,candidate):c.pop('protocol')
    assert candidate==base
    assert saved['augmentation']['minimum_crop_side_fraction']==1 and saved['augmentation']['rotation_degrees']==0
    for extra in ({'rotation_degrees':1},{'minimum_crop_side_fraction':.9},{'contain_jitter_min_scale':.9}):
        with pytest.raises(ValueError):augment(image(),random.Random(1),dict(config(),white_margin_zoom_probability=.5,**extra))
