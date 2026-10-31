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


def load_historical_F_phases(model, payload):
    """Training initialization only: geometry must match; no CNN or old readout.

    The historical OEO/routing/input differences are deliberately NOT hidden:
    importing phase tensors does not import that old forward function.
    """
    if model.main_layers != 2 or model.shared_head.out_features != 10:
        raise ValueError('Historical F initialization is only for two-layer ten-class LG')
    old=payload['optical_config']
    for key in ['input_size','active_size','canvas_size','pixel_size_m','wavelength_m','propagation_m']:
        if old[key] != model.cfg[key]:raise ValueError('Historical optical geometry differs: '+key)
    if set(payload['model']) != {'first_phase','global_phase','router_phase'}:
        raise ValueError('Expected exactly three original optical tensors, no CNN/head')
    load_optical_parent(model,payload['model'])
    for name,value in payload['model'].items():
        if not torch.equal(model.state_dict()[name].cpu(),value.cpu()):raise RuntimeError('Historical phase import differs')
    return dict(imported_phase_names=sorted(payload['model']),all_imported_phases_bitwise_equal=True,
        historical_oeo=old.get('oeo_activation','none'),new_oeo=model.cfg['oeo_activation'],
        old_function_or_accuracy_inherited=False,old_CNN_in_student=False)


@torch.no_grad()
def initialize_CCD_region_prior(model, temperature=.1):
    """Project fixed detector sums onto pooling bins; training initialization only.

    The resulting classifier is still exactly the same one Linear. No ROI
    classifier is blended into inference, and no sample/label is used here.
    """
    classes=model.shared_head.out_features
    if classes not in (8,9) or temperature<=0:raise ValueError('Supported CCD prior requires C8/C9 and positive scale')
    pooling=torch.zeros((28,478),dtype=torch.float64)
    for i in range(28):
        start=i*478//28;end=((i+1)*478+27)//28
        pooling[i,start:end]=1/(end-start)
    ys=(0,239,478) if classes==8 else (0,159,318,478)
    xs=(0,119,239,358,478) if classes==8 else ys
    weights=[]
    for y0,y1 in zip(ys,ys[1:]):
        sy=torch.zeros(478,dtype=torch.float64);sy[y0:y1]=1
        wy=torch.linalg.lstsq(pooling.T,sy).solution
        for x0,x1 in zip(xs,xs[1:]):
            sx=torch.zeros(478,dtype=torch.float64);sx[x0:x1]=1
            wx=torch.linalg.lstsq(pooling.T,sx).solution
            weights.append(torch.outer(wy,wx).flatten())
    value=torch.stack(weights)*(classes/(temperature*478**2))
    model.shared_head.weight.copy_(value.to(model.shared_head.weight))
