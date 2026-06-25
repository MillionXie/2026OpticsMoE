import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_switching.scripts.train_dataset_switching import build_model
from test_dataset_switching_model import tiny_config


def test_electronic_parameter_count_is_sum_of_task_readouts():
    tasks = ["mnist", "fashionmnist", "emnist_letters"]
    model = build_model(tiny_config("learnable_route_moe"), tasks, {"mnist": 10, "fashionmnist": 10, "emnist_letters": 26})
    counts = model.task_readout_parameter_counts()
    assert set(counts) == set(tasks)
    assert model.electronic_parameter_count() == sum(counts.values())
    assert model.task_detector_configs()["mnist"]["detector_size"] == model.task_head_configs["mnist"]["detector_size"]
