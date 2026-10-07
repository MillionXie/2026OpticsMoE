from __future__ import annotations

import torch

from LightGenV2.tasks.t12_text_to_image.decoder_optical_generation import DecoderOpticalGenerator,architecture_report,load_decoder_optical_config
from LightGenV2.tasks.t12_text_to_image.product_instruction_data import TurntableViewDataset,apply_backpack_style,prompt_rows
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


def test_view_output_is_structure_preserving_at_extreme_flow()->None:
    model,config=_model("chair_view_electronic.yaml")
    assert config.view_warp_mix<=.5 and config.view_flow_limit<=.1
    reference=torch.randn(1,3,128,128).clamp(-1,1)
    with torch.no_grad():
        model.to_rgb[-1].bias[:2].fill_(20)
        output=model(reference,torch.randn(1,config.text_dim))
    assert (output-reference).abs().mean()<.55


def test_instruction_sets_and_style_background()->None:
    assert len(prompt_rows("style"))==18 and len(prompt_rows("view"))==6
    reference=torch.ones(6,3,128,128);reference[:,:,40:90,45:85]=-.1
    target=apply_backpack_style(reference,torch.arange(6))
    assert torch.equal(reference[:,:,:20,:20],target[:,:,:20,:20])


def test_turntable_dataset_uses_one_canonical_source_per_identity(tmp_path)->None:
    rows=[]
    from PIL import Image
    for identity in ("a","b"):
        for view in range(3):
            path=tmp_path/f"{identity}-{view}.png";Image.new("RGB",(8,8),(255,255,255)).save(path)
            rows.append({"sample_id":f"chair-{identity}-{view:02d}","sequence_id":identity,"category":"chair","caption":"chair","image_path":path.name,"license":"CC BY 4.0","source_url":"https://example.test"})
    (tmp_path/"train.jsonl").write_text("".join(__import__("json").dumps(row)+"\n" for row in rows),encoding="utf-8")
    torch.save({"rows":prompt_rows("view"),"text":torch.randn(6,2048)},tmp_path/"cache.pt")
    dataset=TurntableViewDataset(tmp_path,"train",128,tmp_path/"cache.pt")
    assert len(dataset)==2*6
    assert {dataset[index]["sample_id"].split(":")[1] for index in range(len(dataset))}=={"0"}
