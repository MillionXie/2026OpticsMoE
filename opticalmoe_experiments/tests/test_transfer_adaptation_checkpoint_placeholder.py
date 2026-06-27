import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transfer_adaptation.scripts import transfer_utils as tu


def test_pretrained_backbone_placeholder_exists():
    root = ROOT / "transfer_adaptation" / "pretrained_backbones" / "dataset_switching_moe_mnist_fashion_emnist_letters"
    assert (root / "README.md").exists()
    assert (root / "PUT_SOURCE_CHECKPOINT_HERE.txt").exists()


def test_missing_source_checkpoint_message_is_clear(tmp_path):
    cfg = {
        "source": {
            "checkpoint_dir": str(tmp_path),
            "checkpoint_name": "source_best.pt",
            "config_name": "source_config.yaml",
        }
    }
    with pytest.raises(FileNotFoundError) as exc:
        tu.validate_source_artifacts(cfg)
    assert "Please place the pretrained dataset-switching OpticalMoE checkpoint at:" in str(exc.value)
    assert "source_best.pt" in str(exc.value)

