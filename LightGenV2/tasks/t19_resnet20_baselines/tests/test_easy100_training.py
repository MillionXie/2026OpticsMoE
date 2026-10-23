import torch

from LightGenV2.tasks.t19_resnet20_baselines.train_easy100 import evaluate, learning_rate


class FixedModel(torch.nn.Module):
    def forward(self, images):
        return images

    def encode_text(self, tokens):
        return tokens


def test_evaluation_preserves_real_query_and_gallery_denominators():
    # Two titles, three image views; title a has two positives.
    loader = [{'image': torch.tensor([[1., 0.], [0., 1.], [1., 0.]]),
               'sku': ['a', 'b', 'a'], 'sample_id': ['a0', 'b0', 'a1']}]
    tokens = torch.eye(2)
    forward = evaluate(FixedModel(), loader, tokens, ['a', 'b'], 'image_to_text', 'cpu')
    reverse = evaluate(FixedModel(), loader, tokens, ['a', 'b'], 'text_to_image', 'cpu')
    assert (forward['queries'], forward['candidates']) == (3, 2)
    assert (reverse['queries'], reverse['candidates']) == (2, 3)
    assert reverse['candidate_ids'] == ['a0', 'b0', 'a1']
    assert forward['hit_at_1'] == reverse['hit_at_1'] == 1.


def test_authorized_budget_has_no_warmup_or_decay():
    from LightGenV2.tasks.t19_resnet20_baselines.train_easy100 import PROFILE
    assert PROFILE['seeds'] == [42]
    assert PROFILE['epochs'] == 30
    assert PROFILE['warmup_epochs'] == 0
    assert {learning_rate(epoch, 30, PROFILE['learning_rate']) for epoch in range(1, 31)} == {.001}
