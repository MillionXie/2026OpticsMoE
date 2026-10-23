import copy

import pytest

from LightGenV2.tasks.t19_resnet20_baselines import shared_idle_gpu as policy


def state():
    return dict(gpu_uuid='GPU-test', active={99: 2500}, total_mib=40960, used_mib=2600,
                utilization_percent=40, observed_monotonic=100.)


def test_requires_stable_low_util_and_sixteen_gib_headroom_before_sharing():
    first = state()
    with pytest.raises(RuntimeError, match='ten seconds'):
        policy.validate(first, [])
    now = copy.deepcopy(first)
    now['observed_monotonic'] += 10
    result = policy.validate(now, [], first)
    assert result['foreign_pids'] == [99] and result['other_processes_modified'] is False
    now['utilization_percent'] = 51
    with pytest.raises(RuntimeError, match='headroom'):
        policy.validate(now, [], first)
    now['utilization_percent'] = 40
    now['used_mib'] = 25000
    with pytest.raises(RuntimeError, match='headroom'):
        policy.validate(now, [], first)


def test_refuses_growing_new_or_second_t19_allocation_on_shared_card():
    first = state()
    now = copy.deepcopy(first)
    now['observed_monotonic'] += 10
    now['active'][99] += 129
    with pytest.raises(RuntimeError, match='not stable'):
        policy.validate(now, [], first)
    now['active'][100] = 500
    with pytest.raises(RuntimeError, match='process set changed'):
        policy.validate(now, [], first)
    with pytest.raises(RuntimeError, match='one T19'):
        policy.validate(now, [100], first)
    now = copy.deepcopy(first)
    now['observed_monotonic'] += 10
    with pytest.raises(RuntimeError, match='one T19'):
        policy.validate(now, [12345], first)  # A registered peer still in CPU preflight.


def test_empty_card_still_rejects_unattributed_memory_and_no_headroom():
    now = state()
    now['active'] = {}
    with pytest.raises(RuntimeError, match='Unattributed'):
        policy.validate(now, [])
    now['used_mib'] = 10
    assert policy.validate(now, [])['foreign_pids'] == []
    now['used_mib'] = 40000
    with pytest.raises(RuntimeError, match='room'):
        policy.validate(now, [])
