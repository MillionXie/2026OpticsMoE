"""Dense experiment overrides after the legacy dependency schema is loaded."""
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.settings import load_settings as legacy_load

def load_settings(path):
    settings=legacy_load(path)
    settings.top_k=4
    settings.router_straight_through=False
    settings.router_protocol='dense_four_expert_transfer'
    from .layout import configure_detector_windows
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.geometry import MoEGeometry
    configure_detector_windows(settings,MoEGeometry(518,478,224,254,4,2,2))
    settings.text_depth=0;settings.text_hidden_size=0;settings.vision_depth=0
    return settings
