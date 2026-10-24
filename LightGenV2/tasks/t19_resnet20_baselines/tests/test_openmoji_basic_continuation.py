import copy
import json
from pathlib import Path
import pytest
from LightGenV2.tasks.t19_resnet20_baselines.continue_openmoji_basic import validate_plan


def test_only_latest_human_selected_parent_and_fixed_budget():
    plan=json.loads((Path(__file__).parents[1]/'configs/openmoji_basic_resume60to80_20261024.json').read_text())
    validate_plan(plan)
    for key,value in [('parent_epoch',30),('final_epoch',90),('effective_batch',16),('gradient_clip',2),
                      ('parent_SHA','old-other-weight'),('parent','runs/simulation/old/last_checkpoint.pt')]:
        changed=copy.deepcopy(plan);changed[key]=value
        with pytest.raises(ValueError):validate_plan(changed)
