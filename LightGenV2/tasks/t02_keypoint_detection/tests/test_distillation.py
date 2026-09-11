from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from LightGenV2.tasks.t02_keypoint_detection.distillation import train_identity
from LightGenV2.tasks.t02_keypoint_detection.refine import polish_config, profile_epochs, stage_spec
from LightGenV2.tasks.t02_keypoint_detection.settings import load_settings
from LightGenV2.tasks.t02_keypoint_detection.modeling import architecture_label
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_distillation, warp_cached_heatmaps


def test_train_identity_rejects_test_overlap_and_preserves_order():
    def record(i):return SimpleNamespace(sample_id=str(i),image_path=Path(f'{i}.png'),keypoints=np.zeros((14,2)))
    records=[record(0),record(1)]
    ids,sha=train_identity(records,[record(2)])
    assert ids==['0','1'] and len(sha)==64
    assert train_identity(records[::-1],[])[1]!=sha
    with pytest.raises(RuntimeError):train_identity(records,[record(1)])
    with pytest.raises(RuntimeError):train_identity([records[0],records[0]],[])


def test_distillation_gradient_and_identity_warp():
    teacher=torch.rand(2,14,56,56)
    box=torch.tensor([[0,0,224,224]]*2)
    warped=warp_cached_heatmaps(teacher,box,box,torch.zeros(2,dtype=torch.bool),224,56)
    torch.testing.assert_close(warped,teacher,atol=1e-5,rtol=1e-5)
    student=torch.zeros_like(teacher,requires_grad=True)
    visible=torch.ones(2,14,dtype=torch.bool);visible[:,0]=False
    loss=masked_heatmap_distillation(student,warped,visible)
    loss.backward()
    assert teacher.grad is None and student.grad[:,1:].abs().sum()>0
    assert student.grad[:,0].abs().sum()==0
    assert masked_heatmap_distillation(student.detach(),teacher,torch.zeros_like(visible))==0


def test_distillation_does_not_change_inference_contract():
    configs=Path(__file__).resolve().parents[1]/'configs'
    old=load_settings(configs/'moe_alpha40.yaml')
    new=load_settings(configs/'moe_alpha40_distill.yaml')
    assert architecture_label(old)==architecture_label(new)
    assert new.fusion_alpha_min==.4 and new.coordinate_loss_weight==0
    assert profile_epochs('alpha40_distill')==40
    assert polish_config('alpha40_distill')['source_sha256'].startswith('651989')
    for e in range(1,41):assert stage_spec('alpha40_distill',e)==stage_spec('alpha40_polish',e)
