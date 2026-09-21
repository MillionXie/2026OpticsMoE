from __future__ import annotations

import torch

from LightGenV2.tasks.t12_text_to_image.decoder_optical_generation import DecoderOpticalGenerator,architecture_report,load_decoder_optical_config
from LightGenV2.tasks.t12_text_to_image.product_instruction_data import apply_backpack_style,prompt_rows
from LightGenV2.tasks.t12_text_to_image.settings import TASK_DIR


def _model(name:str):
    config=load_decoder_optical_config(TASK_DIR/f"configs/{name}");return DecoderOpticalGenerator(config),config


def test_decoder_optical_alpha_is_at_least_point_four_and_is_in_decoder()->None:
    model,config=_model("backpack_style_decoder_optical.yaml");report=architecture_report(model)
    assert model.decoder_generator.fusion.alpha.item()>=.4
    assert report["optics_location"]=="first_decoder_generator_block"
    reference=torch.ones(1,3,128,128);reference[:,:,30:100,35:95]=-.2
    output=model(reference,torch.randn(1,config.text_dim));output.mean().backward()
    assert model.decoder_generator.optical.expert_phase.grad is not None


def test_both_tasks_are_single_pass_and_under_ten_million()->None:
    for name in ("backpack_style_decoder_optical.yaml","chair_view_decoder_optical.yaml"):
        model,config=_model(name);report=architecture_report(model)
        assert report["generator_parameters"]<10_000_000 and report["decoder_calls"]==1
        output=model(torch.randn(1,3,128,128),torch.randn(1,config.text_dim));assert output.shape==(1,3,128,128)


def test_instruction_sets_and_style_background()->None:
    assert len(prompt_rows("style"))==18 and len(prompt_rows("view"))==12
    reference=torch.ones(6,3,128,128);reference[:,:,40:90,45:85]=-.1
    target=apply_backpack_style(reference,torch.arange(6))
    assert torch.equal(reference[:,:,:20,:20],target[:,:,:20,:20])
