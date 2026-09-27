import torch
from LightGenV2.tasks.t12_text_to_image.train_pix2pix_turbo_baseline import architecture


def test_parameter_report_separates_frozen_embeddings_and_trainable():
    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(3,4);self.requires_grad_(False)
        def get_input_embeddings(self):return self.embedding
    model=torch.nn.Module();model.text_encoder=Text()
    model.unet=torch.nn.Linear(4,4);model.vae=torch.nn.Linear(4,3)
    report=architecture(model)
    assert report['total_inference_parameters']==47
    assert report['word_embedding_parameters_separately_listed']==12
    assert report['counted_inference_parameters_excluding_word_embedding']==35
    assert report['trainable_parameters']==35
    assert report['trainable_by_component']['text_encoder']==0
    assert not report['qwen_used'] and not report['pca_condition_used']
