import numpy as np
from LightGenV2.tasks.t12_text_to_image.scene_lab_hardware import native, direction, field_inverse


def test_native_preserves_zero_and_phase_codes():
    a=np.arange(478*478,dtype=np.uint32).reshape(478,478).astype(np.uint8)
    assert native(a,'amplitude').shape==(1080,1920)
    assert np.count_nonzero(native(a,'amplitude')[:32])==0
    phase=native(a,'phase');assert phase.shape==(1200,1920)
    assert np.all(phase[:92]==255)
    from PIL import Image
    decoded=np.asarray(Image.fromarray(255-phase[92:1108,452:1468]).resize((478,478),Image.Resampling.NEAREST))
    assert np.array_equal(np.flipud(np.fliplr(decoded)),a)


def test_direction_isolated_axes_and_repeat():
    bg=np.zeros((478,478),np.float32);images=[]
    for x,y in ((150,150),(330,150),(150,330),(150,150)):
        a=bg.copy();a[y-22:y+22,x-22:x+22]=10000;images.append(a)
    assert direction(images,bg)[0]=='identity'
    assert direction([np.flipud(x)for x in images],bg)[0]=='flip_v'


def test_inverse_quadratic_identity():
    grays=list(range(256));means=500+20000*(np.arange(256)/255.)**2
    drive,response=field_inverse(grays,means)
    assert drive==grays
