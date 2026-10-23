import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from lab_caltech20_TRAIN_metric_20261024 import family_paired_order


def test_every_TRAIN_once_and_no_augmented_sibling_positive():
    y=np.tile(np.arange(20),80);families=np.array([str(k)for k in range(1600)])
    for k in range(92):families[k+800]=families[k]
    order=family_paired_order(y,families,np.random.default_rng(73))
    assert np.array_equal(np.sort(order),np.arange(1600))
    for ix in order.reshape(-1,8):
        values,counts=np.unique(y[ix],return_counts=True);assert len(values)==4 and (counts==2).all()
        for label in values:assert len(set(families[ix[y[ix]==label]].tolist()))==2


def test_training_metric_not_an_inference_layer():
    s=(HERE/'lab_caltech20_TRAIN_metric_20261024.py').read_text()
    assert 'nn.Module' not in s and 'nn.Parameter' not in s and 'logsumexp' in s and 'TEST_used=False' in s
    f=(HERE/'lab_caltech20_TRAIN_frozen_readout_20261024.py').read_text()
    assert 'TRAIN_weighted_metric' in f and 'TRAIN_independent_family_pairs=800' in f and '.25*ce.detach()/pair_loss.detach()' in f
