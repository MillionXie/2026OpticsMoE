"""Final CPU auditor rejects incomplete test results before label cache access."""
import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_coil20_final import completion_barrier


def test_audit_requires_all_test_results(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'state':'tests_complete_awaiting_cpu_audit'}))
    with pytest.raises(FileNotFoundError): completion_barrier(tmp_path)


def test_audit_rejects_test_selection(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'state':'tests_complete_awaiting_cpu_audit'}))
    arm=tmp_path/'L2_rho0';arm.mkdir()
    (arm/'result.json').write_text(json.dumps({'test_evaluations':1,'test_used_for_selection':True}))
    with pytest.raises(AssertionError): completion_barrier(tmp_path)
