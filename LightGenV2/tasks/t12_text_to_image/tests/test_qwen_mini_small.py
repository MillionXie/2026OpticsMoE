import torch

from LightGenV2.tasks.t12_text_to_image.qwen_mini_small import QwenMiniConfig,QwenMiniTextEncoder
from LightGenV2.tasks.t12_text_to_image.small_fullframe import SmallEditorConfig,SmallFullFrameEditor


def test_qwen_mini_has_two_blocks_and_full_editor_is_under_50m():
    text_config=QwenMiniConfig(input_width=64,width=96,intermediate_width=192,layers=2,heads=6,condition_dim=32,max_length=12)
    editor_config=SmallEditorConfig(image_size=32,widths=(16,32,48,64),condition_dim=32,control_classes=64)
    model=SmallFullFrameEditor(editor_config);model.text=QwenMiniTextEncoder(text_config)
    embeddings=torch.randn(2,12,64);mask=torch.tensor([[1]*12,[1]*8+[0]*4],dtype=torch.bool)
    output=model(torch.randn(2,3,32,32),(embeddings,mask),torch.randn(2,3,32,32),torch.tensor([0,1]))
    assert output.shape==(2,3,32,32)
    assert len(model.text.layers)==2
    assert sum(parameter.numel() for parameter in model.parameters())<50_000_000
