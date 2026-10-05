"""Planning arithmetic only: no models, datasets, generated tables or devices."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = 'LightGenV2/tasks/t10_expert_scaling/plan.py'


def load_plan():
    spec = importlib.util.spec_from_file_location('reviewed_t10_plan', ROOT / PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_plan():
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show',
        '38afd43c1e28aeaff28e33bad58177ae09a0f156:' + PATH])
    module = types.ModuleType('historical_t10_plan')
    module.__file__ = str(ROOT / PATH)
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def test_repair_preserves_all_planning_arithmetic():
    current, old = load_plan(), historical_plan()
    config = current.load_protocol()
    snapshot = copy.deepcopy(config)
    current.check(config)
    assert config == snapshot
    for n in config['expert_counts']:
        assert current.geometry(config, n) == old.geometry(config, n)
        assert current.router_layout(config, n) == old.router_layout(config, n)
        assert current.router_regions(config, n) == old.router_regions(config, n)
        for pilot in (False, True):
            assert current.k_values(config, n, pilot) == old.k_values(config, n, pilot)
    for datasets in (config['datasets']['primary'], config['datasets']['optional']):
        for kwargs in ({}, {'pilot': True}, {'fixed_global': True}):
            assert current.matrix(config, datasets, **kwargs) == old.matrix(config, datasets, **kwargs)


@pytest.mark.parametrize('field,value', [
    ('wavelength_nm', 633.0), ('pixel_pitch_um', 8.0),
    ('propagation_distance_m', 0.2), ('propagation_distance_m', 0.0),
])
def test_planning_rejects_different_physical_contract(field, value):
    plan = load_plan()
    config = plan.load_protocol()
    config['geometry'][field] = value
    with pytest.raises(ValueError):
        plan.check(config)


def test_old_undefined_variable_is_observable_not_rewritten():
    old = historical_plan()
    with pytest.raises(UnboundLocalError):
        old.check(old.load_protocol())
