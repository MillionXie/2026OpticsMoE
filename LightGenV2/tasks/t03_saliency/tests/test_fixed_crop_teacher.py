from copy import deepcopy
import hashlib
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch
from LightGenV2.tasks.t03_saliency.fixed_crop_teacher import (
    BOX, VIEW, crop_image, rgb_sha256, validate_payload, check_pixels, save_new_payload)


def fixture():
    images=[Image.fromarray(np.random.default_rng(seed).integers(0,256,(224,224,3),dtype=np.uint8)) for seed in [1,2]]
    ids=['train/a','train/b']; records=[SimpleNamespace(sample_id=s) for s in ids]
    payload=dict(manifest=dict(samples=2,split='train_only',view=deepcopy(VIEW),
        checkpoint_sha256='a'*64,train_annotations_sha256='b'*64,
        image_manifest_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()),
        sample_ids=ids,logits=torch.randn(2,1,224,224,dtype=torch.float16),
        original_rgb_sha256=[rgb_sha256(x) for x in images],
        crop_rgb_sha256=[rgb_sha256(crop_image(x)) for x in images])
    return images,records,payload


def test_exact_pixels_and_crop_do_not_mutate_source():
    images,records,payload=fixture();original=images[0].tobytes()
    cropped=crop_image(images[0])
    assert cropped.size==(224,224) and cropped.mode=='RGB'
    assert cropped.tobytes()==images[0].crop(BOX).resize((224,224),Image.Resampling.BICUBIC).tobytes()
    assert images[0].tobytes()==original
    check_pixels(payload,0,images[0],cropped)
    with pytest.raises(ValueError,match='pixels differ'):
        check_pixels(payload,0,images[1],cropped)
    with pytest.raises(ValueError,match='pixels differ'):
        check_pixels(payload,0,images[0],images[0].crop(BOX).resize((224,224),Image.Resampling.BILINEAR))
    with pytest.raises(ValueError):crop_image(images[0].resize((225,224)))
    with pytest.raises(ValueError):crop_image(images[0].convert('L'))


def test_cache_rejects_identity_view_dtype_and_nonfinite():
    _,records,payload=fixture();validate_payload(payload,records,'a'*64,'b'*64)
    with pytest.raises(ValueError):validate_payload(payload,records[::-1],'a'*64,'b'*64)
    for field,value in [('checkpoint_sha256','c'*64),('split','test'),('samples',3),
                        ('train_annotations_sha256','c'*64),('view',{})]:
        bad=deepcopy(payload);bad['manifest'][field]=value
        with pytest.raises(ValueError):validate_payload(bad,records,'a'*64,'b'*64)
    for field,value in [('logits',payload['logits'].float()),('logits',torch.full_like(payload['logits'],float('nan'))),
                        ('crop_rgb_sha256',['a'*64]),('original_rgb_sha256',['z'*64]*2)]:
        bad=dict(payload);bad[field]=value
        with pytest.raises(ValueError):validate_payload(bad,records,'a'*64,'b'*64)


def test_atomic_cache_save_refuses_existing_and_partial(tmp_path):
    _,records,payload=fixture();output=tmp_path/'cache.pt'
    report=save_new_payload(payload,output)
    assert report['cache_sha256']==hashlib.sha256(output.read_bytes()).hexdigest()
    validate_payload(torch.load(output,weights_only=False),records,'a'*64,'b'*64)
    assert output.with_suffix('.json').exists() and not output.with_suffix('.partial').exists()
    original=output.read_bytes()
    with pytest.raises(FileExistsError):save_new_payload(payload,output)
    assert output.read_bytes()==original
    other=tmp_path/'unfinished.pt';other.with_suffix('.partial').write_bytes(b'keep existing partial')
    with pytest.raises(FileExistsError):save_new_payload(payload,other)
    assert other.with_suffix('.partial').read_bytes()==b'keep existing partial'
