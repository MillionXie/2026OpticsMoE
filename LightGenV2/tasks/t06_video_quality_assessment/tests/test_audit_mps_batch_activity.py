import importlib
import sys
import numpy as np


def test_offline_metrics_do_not_require_torch_or_cuda():
    torch_preexisting = 'torch' in sys.modules
    module = importlib.import_module('LightGenV2.tasks.t06_video_quality_assessment.reports.audit_mps_batch_activity')
    assert torch_preexisting or 'torch' not in sys.modules
    r = module.metrics(np.array([1.,2.,3.,4.]),np.array([2.,4.,6.,8.]))
    assert abs(r['srcc']-1) < 1e-10 and abs(r['plcc']-1) < 1e-10
    assert abs(r['mae']-2.5) < 1e-10
