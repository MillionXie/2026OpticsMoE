"""CPU checks for a generic class count, tail evaluation, and test-read barrier."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='Use native Linux Torch; shared environment is unchanged')


def test_pilot_never_loads_test_arrays(tmp_path):
    import medicinal_pilot as pilot
    np = pilot.np
    labels = np.arange(30, dtype=np.int64)
    arrays, supports, ids = {}, {}, {}
    for split in ['train', 'val']:
        arrays[split+'_images'] = np.zeros((30, 150, 150, 3), dtype=np.uint8)
        arrays[split+'_labels'] = labels
        arrays[split+'_ids'] = np.array([split+str(k) for k in labels])
        supports[split] = [1]*30; ids[split] = arrays[split+'_ids'].tolist()
    path = tmp_path/'data.npz'
    np.savez(path, **arrays)  # There are intentionally no test arrays.
    manifest = dict(classes=list(range(30)), supports=supports, split_ids=ids)
    for split in ['train', 'val']:
        result = pilot.load_fit_data(path, split, manifest)
        assert len(result[1]) == 30 and result[2].tolist() == ids[split]
    with pytest.raises(AssertionError, match='never materialize test'):
        pilot.load_fit_data(path, 'test', manifest)


def test_full_evaluation_tail_and_class_support(monkeypatch):
    import medicinal_pilot as pilot
    torch, np = pilot.torch, pilot.np
    labels = torch.arange(30)
    data = (labels.reshape(30, 1, 1, 1), labels, np.array([f'id{k}' for k in range(30)]))
    calls = []
    monkeypatch.setattr(pilot.o, 'encode', lambda images: images)
    def forward(model, images):
        calls.append(len(images))
        prob = torch.nn.functional.one_hot(images.flatten(), 30).float()
        return prob, torch.ones(len(images))
    monkeypatch.setattr(pilot.o, 'forward', forward)
    class Model:
        def eval(self): return self
    metric, rows = pilot.evaluate(Model(), data, 4)
    assert calls == [4]*7+[2]
    assert metric['accuracy'] == metric['balanced_accuracy'] == 1.
    assert metric['support'] == [1]*30 and len(rows) == 30
    assert all(row['sample_id'] == f'id{k}' and row['label_pred'] == k for k, row in enumerate(rows))
