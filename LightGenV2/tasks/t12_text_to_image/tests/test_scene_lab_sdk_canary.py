import numpy as np
from LightGenV2.tasks.t12_text_to_image.scene_lab_sdk_canary import signal_pair


def test_identical_and_fresh_noise_are_not_optical_signal():
    dark=np.full((50,50),500,np.uint16)
    assert not signal_pair(dark,dark)['passed']
    fresh=dark.copy();fresh[::2]+=20
    assert not signal_pair(dark,fresh)['passed']
    assert not signal_pair(dark,fresh)['exactly_same_array']


def test_optical_response_gate():
    dark=np.full((50,50),500,np.uint16);white=dark.copy();white[5:45,5:45]=10000
    assert signal_pair(dark,white)['passed']
