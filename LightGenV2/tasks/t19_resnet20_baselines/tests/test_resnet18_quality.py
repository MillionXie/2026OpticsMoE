"""Verify larger backbone keeps the original temporal quality contract."""
import torch
from LightGenV2.tasks.t19_resnet20_baselines.models import build_model, parameter_counts


def test_resnet18_temporal_architecture_and_quality_scale():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = build_model('lgvq_temporal_resnet18').eval()
        assert parameter_counts(model) == dict(total=11406277, trainable=11406277,
            visual_encoder=11176512, text_encoder=0, task_heads=229765)
        assert model.encoder.conv1.kernel_size == (7, 7)
        assert model.encoder.conv1.stride == (2, 2)
        assert model.head[0].in_features == 3584
        boundaries = model.set_train_score_range(torch.tensor([10., 90.]))
        torch.testing.assert_close(boundaries, torch.tensor([26., 42., 58., 74.]))
        frames = torch.randn(1, 4, 3, 64, 64)
        output = model(frames)
        assert output['quality_logits'].shape == (1, 5)
        assert 10 <= output['quality_score'].item() <= 90
        output['quality_score'].sum().backward()
        assert model.encoder.conv1.weight.grad.isfinite().all()
    finally:
        torch.set_num_threads(previous)
