import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data.dsprites_multitask import split_indices_from_config


def test_dsprites_sampling_total_size_controls_train_val_test():
    cfg = {
        "sampling_protocol": {
            "enabled": True,
            "total_size": 12000,
            "train_test_ratio": [4, 1],
            "class_balanced": False,
            "seed_offset": 0,
        }
    }
    splits = split_indices_from_config(737280, cfg, val_split=0.1, test_split=0.1, seed=7)
    assert len(splits["train"]) == 8640
    assert len(splits["val"]) == 960
    assert len(splits["test"]) == 2400
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 12000

    train = set(map(int, splits["train"]))
    val = set(map(int, splits["val"]))
    test = set(map(int, splits["test"]))
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
