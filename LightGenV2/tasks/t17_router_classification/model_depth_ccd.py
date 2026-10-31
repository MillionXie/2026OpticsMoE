"""Panel B proposal: extend panel A optics without changing its two-layer LG.

One input router is followed by E/G pairs on the same coherent 518-square field.
The router gates the entrance only. Subsequent E planes modulate the continuing
field, including light arriving at previously unselected slots after global mixing.
No second fanout, per-expert lossless propagation, or additional gating is implied.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model_four import FourRouterClassification, CONTRACT, ccd_grid_energies, sparse_top2
from .model_linear_ccd import LinearD2NNCCD
from LightGenV2.demo_check.pure_optical.models import PhaseOnly


class DepthLG(FourRouterClassification):
    def __init__(self, main_layers=2, num_classes=8):
        if main_layers not in (2, 4, 6):
            raise ValueError('Panel B LG main depth must be 2/4/6, excluding its one router')
        super().__init__('optical', 'ccd_grid', num_classes=num_classes)
        self.main_layers = main_layers
        self.extra_expert_phases = nn.ParameterList([
            nn.Parameter(torch.zeros_like(self.first_phase)) for _ in range(main_layers//2-1)])
        self.extra_global_phases = nn.ParameterList([
            nn.Parameter(torch.zeros_like(self.global_phase)) for _ in range(main_layers//2-1)])

    def forward(self, amplitude, return_debug=False, return_dense_route=False):
        if self.main_layers == 2:
            # Exact panel-A forward and checkpoint keys; no approximate reimplementation.
            return super().forward(amplitude, return_debug=return_debug,
                                   return_dense_route=return_dense_route)
        if amplitude.ndim != 3 or amplitude.shape[-2:] != (224, 224):
            raise ValueError('Expected panel-A Bx224x224 encoded amplitude')
        amplitude = amplitude/amplitude.square().sum((-2,-1), keepdim=True).clamp_min(1e-20).sqrt()
        if return_dense_route:
            dense, capture = self.dense_route(amplitude)
            q = sparse_top2(dense)
        else:
            q, capture = self.route(amplitude)
        coefficients = torch.where(q > 0, q.clamp_min(1e-20).sqrt(), torch.zeros_like(q))
        expert_phases = [self.first_phase, *self.extra_expert_phases]
        global_phases = [self.global_phase, *self.extra_global_phases]
        fields = []
        for cycle, (expert, global_phase) in enumerate(zip(expert_phases, global_phases)):
            if cycle == 0:
                field = torch.zeros((len(amplitude),518,518), dtype=torch.complex64, device=amplitude.device)
                for i, (y,x) in enumerate(self.apertures):
                    field[:,y:y+224,x:x+224] = (amplitude*coefficients[:,i,None,None]
                                                              *self.transmission(expert[i]))
            else:
                # Phase-only masks: identity transmission in fixed gaps and borders.
                mask = torch.ones((518,518), dtype=torch.complex64, device=amplitude.device)
                for i, (y,x) in enumerate(self.apertures):
                    mask[y:y+224,x:x+224] = self.transmission(expert[i])
                field = field*mask
            field = self.oeo(self.propagator(field))
            if return_debug: fields.append(field)
            field = self.oeo(self.propagator(field*F.pad(self.transmission(global_phase),(20,)*4,value=1)))
            if return_debug: fields.append(field)
        intensity = field.abs().square()
        energies = ccd_grid_energies(intensity[:,20:498,20:498], self.num_classes)
        total = energies.sum(1,keepdim=True)
        p = (energies+1e-12*total)/(total*(1+self.num_classes*1e-12))
        result = dict(logits=p.log(), probabilities=p, class_energies=energies,
                      route_power=q, router_capture=capture)
        if return_dense_route: result['dense_route_power'] = dense
        if return_debug: result.update(post_oeo_intensity=intensity, stage_fields=fields)
        return result


class DepthD2NN(LinearD2NNCCD):
    """3/5/7 full-global-aperture phases, each followed by the exact panel-A OEO."""
    def __init__(self, layers=3, num_classes=8):
        if layers not in (3, 5, 7):
            raise ValueError('Panel B D2NN depth must be 3/5/7')
        super().__init__(layers=3, num_classes=num_classes)
        self.phases.extend([nn.Parameter(torch.zeros(478,478)) for _ in range(layers-3)])
        self.cfg = dict(CONTRACT)
        self.layers = layers

    # Directly reuse the function used by panel-A LG; no LN/ReLU/learned adapter.
    oeo = PhaseOnly.oeo

    def propagate_field(self, field):
        for phase in self.phases:
            mask = F.pad(PhaseOnly.transmission(phase),(20,)*4,value=1)
            field = self.oeo(self.propagator(field*mask))
        return field


def depth_contract(main_layers):
    if main_layers not in (2, 4, 6): raise ValueError(main_layers)
    pairs = main_layers//2
    return dict(LG_main_layers=main_layers, LG_router_layers=1, LG_total_phase_depth=main_layers+1,
                LG_experts_per_plane=4, LG_expert_side=224, LG_top_k_at_input=2,
                LG_stages=['E','G']*pairs, LG_main_OEO_count=main_layers,
                LG_phase_parameters=50176+429188*pairs,
                D2NN_layers=main_layers+1, D2NN_phase_side=478,
                D2NN_OEO_count=main_layers+1, D2NN_phase_parameters=(main_layers+1)*478**2,
                global_side=478, canvas_side=518, pixel_pitch_um=17, wavelength_nm=532,
                propagation_m=.1, activation='panel-A intensity_softsign, no LN or ReLU',
                OEO_after_final_layer=True, learned_electronic_frontend=False,
                learned_electronic_readout=False, CCD='same panel-A full coverage 2x4 grid',
                later_expert_illumination='continuing coherent field; no repeated Top2 gate')
