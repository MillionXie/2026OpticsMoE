import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data.foundation_distillation import CachedTeacherFeatureDataset, IndexedGrayDataset
from foundation_distillation.scripts.build_teacher_feature_cache import encode_split


class MockTeacher(torch.nn.Module):
    def forward(self, images):
        means = images.mean(dim=(-2, -1))
        return F.normalize(torch.cat([means, means[:, :1]], dim=1), dim=-1)


def test_mock_teacher_cache_payload_round_trip(tmp_path):
    dataset = IndexedGrayDataset(TensorDataset(torch.rand(6, 1, 16, 16), torch.arange(6) % 3))
    payload = encode_split(dataset, MockTeacher(), torch.device("cpu"), {"batch_size": 2, "num_workers": 0})
    payload["split"] = "train"
    path = tmp_path / "train_features.pt"
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=False)
    joined = CachedTeacherFeatureDataset(dataset, loaded)
    image, label, feature, index = joined[2]
    assert image.shape == (1, 16, 16)
    assert label == int(payload["labels"][2])
    assert feature.shape == (4,)
    assert index == 2
    metadata = {"teacher_feature_dim": int(payload["features"].shape[1])}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert json.loads((tmp_path / "metadata.json").read_text())["teacher_feature_dim"] == 4
