"""CPU checks for the all-eight lock and exclusive test claim barriers."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[1]/'coil20_finalize.py'
spec=importlib.util.spec_from_file_location('finalizer',PATH)
finalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(finalizer)


def test_incomplete_global_lock_rejected_before_data_access(tmp_path):
    (tmp_path/'global_selection_lock.json').write_text(json.dumps({'entries':{'L2_rho0':{}}}))
    with pytest.raises(AssertionError,match='All eight'):
        finalizer.verify_global(tmp_path,tmp_path/'nonexistent_test_cache.npz')


def test_exclusive_claim_survives_failed_evaluation(tmp_path):
    finalizer.claim_test(tmp_path,'L16_rho03','locked-sha')
    with pytest.raises(FileExistsError):
        finalizer.claim_test(tmp_path,'L16_rho03','locked-sha')
    receipt=json.loads((tmp_path/'L16_rho03/test_started.json').read_text())
    assert receipt['global_selection_lock_sha256']=='locked-sha'
