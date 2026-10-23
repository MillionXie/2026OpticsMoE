import importlib.util
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_group_ridge_20261024.py'
spec=importlib.util.spec_from_file_location('TRAIN_group_ridge_test',P);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_family_deletion_exact_and_no_extra_validation_split():
    assert module.selftest()['passed']
    s=P.read_text();assert 'all_siblings_deleted_together=True' in s and 'final_fits_all1600=True' in s
    assert 'external_VALID_samples=0,TEST_used=False' in s and 'continuous_TRAIN_only_analytic_leave_source_family_out_not_TEST_grid' in s
