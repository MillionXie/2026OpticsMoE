"""Panel A/B LG with the existing optical body and one final linear CCD readout."""
import torch
from torch import nn
from torch.nn import functional as F
from .model_depth_ccd import DepthLG


class LinearReadoutLG(DepthLG):
    def __init__(self, main_layers=2, num_classes=8):
        super().__init__(main_layers,num_classes)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(122)
            self.shared_head=nn.Linear(28*28,num_classes,bias=False)

    def optical_features(self, amplitude, return_dense_route=False, return_debug=False):
        # The historical body computes its CCD diagnostics, but they do not
        # classify this model. Keep the identical propagation and final OEO.
        old=super().forward(amplitude,return_debug=True,return_dense_route=return_dense_route)
        intensity=old['post_oeo_intensity']
        active=intensity[:,20:498,20:498]
        features=F.adaptive_avg_pool2d(active[:,None],(28,28)).flatten(1)
        features=features/features.mean(1,keepdim=True).clamp_min(1e-20)
        result=dict(ccd_features=features,route_power=old['route_power'],router_capture=old['router_capture'])
        if return_dense_route:result['dense_route_power']=old['dense_route_power']
        if return_debug:result['post_oeo_intensity']=intensity
        return result

    def forward(self, amplitude, return_dense_route=False, return_debug=False):
        result=self.optical_features(amplitude,return_dense_route,return_debug)
        raw=self.shared_head(result['ccd_features'])
        # Log probabilities preserve the established evaluator/CE interface.
        # Classification is solely the single Linear, without a hidden MLP.
        result.update(raw_logits=raw,logits=raw.log_softmax(1),probabilities=raw.softmax(1))
        return result


def load_optical_parent(model, state):
    """Only the deliberately added head may be absent from a legacy checkpoint."""
    status=model.load_state_dict(state,strict=False)
    if status.missing_keys!=['shared_head.weight'] or status.unexpected_keys:
        raise ValueError('Parent must match every optical tensor and lack only the new Linear head')
