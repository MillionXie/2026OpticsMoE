import sys

import pytest

from LightGenV2.tasks.t19_resnet20_baselines.queue_easy100 import main


@pytest.mark.parametrize('uuids', [
    ['GPU-a', 'GPU-a'], ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'],
])
def test_queue_rejects_duplicate_or_excess_devices_before_any_launch(monkeypatch, uuids):
    monkeypatch.setattr(sys, 'argv', ['queue', '--gpu-uuids', *uuids,
                                    '--data-root', '.', '--suite-id', 'test'])
    with pytest.raises(ValueError, match='One to three distinct'):
        main()


def test_queue_rejects_run_path_escape(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['queue', '--gpu-uuids', 'GPU-a',
                                    '--data-root', '.', '--suite-id', '../escape'])
    with pytest.raises(ValueError, match='single directory'):
        main()
