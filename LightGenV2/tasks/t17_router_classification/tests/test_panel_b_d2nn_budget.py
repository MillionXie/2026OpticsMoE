import json
from pathlib import Path
import torch
from LightGenV2.tasks.t17_router_classification.train_linear_ccd_8class import build_model
from LightGenV2.tasks.t17_router_classification.model_depth_ccd import DepthD2NN
from LightGenV2.tasks.t17_router_classification.model_linear_ccd import LinearD2NNCCD


def test_formal_D2NN_profiles_match_panel_A_training_budget_and_recipe():
    configs = Path(__file__).resolve().parents[1] / 'configs'
    original = json.loads((configs/'kather2016_d2nn_linear3_ccd_s17_20261030.json').read_text())
    keys = ['seed', 'epochs', 'batch', 'learning_rate', 'optimizer', 'gradient_clip',
            'training_evaluate_every', 'selection', 'augmentation', 'ema', 'class_weighting',
            'weight_decay', 'expected_counts', 'source_archive_sha256', 'input']
    for depth in (3, 5, 7):
        cfg = json.loads((configs/f'panel_b_d{depth}_s17.json').read_text())
        assert {k: cfg[k] for k in keys} == {k: original[k] for k in keys}
        assert cfg['epochs'] == 60 and cfg['intermediate_OEO'] is True
        model = build_model(cfg)
        assert isinstance(model, DepthD2NN) and len(model.phases) == depth
        assert sum(p.numel() for p in model.parameters()) == depth * 478**2
        assert all(torch.count_nonzero(p) == 0 for p in model.parameters())
    assert type(build_model(original)) is LinearD2NNCCD
