"""Nine continuously weighted optical experts vs a total-parameter matched D2NN."""
import torch
from torch import nn
from torch.nn import functional as F
from optical_reference.activations import activate, CANDIDATES
from optical_reference.model import DeepHeterogeneousOpticalMoENonlinearClassifier
from optical_reference.optics import AngularSpectrumPropagator, DetectorArray, PhaseLayer
from optical_reference.losses import detector_plane_mse_loss


class RelayFanoutMoE(DeepHeterogeneousOpticalMoENonlinearClassifier):
    """Ideal nine-port image relay, with total-power-normalized routing amplitudes.

    The reference quadratic-lens convolution concentrates smooth input at the
    central port. Here every physical expert receives a centered copy of the
    same input. This is an ideal lossless relay model, not a full-wave DOE design.
    No trainable parameters are added. Main-path propagation still uses ASM.
    """
    def global_fanout_convolution(self,field,prompt_transmission):
        source_aperture=self.layout.input_aperture
        source=field[:,source_aperture.y0:source_aperture.y1,source_aperture.x0:source_aperture.x1]
        weights=torch.stack([prompt_transmission[:,*ap.center].abs() for ap in self.layout.expert_apertures],dim=1)
        amplitudes=weights/weights.square().sum(1,keepdim=True).sqrt().clamp_min(1e-12)
        padding=(self.layout.expert_size-self.layout.input_size)//2
        assert padding*2+self.layout.input_size==self.layout.expert_size
        image=F.pad(source,(padding,)*4)
        output=torch.zeros_like(field)
        for i,ap in enumerate(self.layout.expert_apertures):
            output[:,ap.y0:ap.y1,ap.x0:ap.x1]=image*amplitudes[:,i,None,None]
        return output


class OpticalMoE(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        assert cfg['prompt']['routing_type']=='optical_topk'
        assert cfg['model']['num_experts']==cfg['prompt']['top_k']==9
        assert not cfg['readout']['enabled']
        self.net=RelayFanoutMoE(cfg,num_classes=2)

    @property
    def masks(self):return self.net.detector.masks

    def forward(self,images):
        energy,details=self.net(images,return_intermediates=True,capture_expert_outputs=False)
        return {'intensity':details['detector_intensity'],'energies':energy,
                'routing':details['routing_selected_mask'],'route_probabilities':details['routing_probabilities'],
                'stage_input_power':torch.stack([s['linear_input_power'] for s in details['expert_stage_details']],dim=1)}


class CenteredPhase(nn.Module):
    """Trainable central SLM, with fixed identity transmission in the surrounding canvas."""
    def __init__(self,side,canvas,optics):
        super().__init__()
        assert 0<side<=canvas and (canvas-side)%2==0
        self.start=(canvas-side)//2;self.end=self.start+side
        self.phase=PhaseLayer(side,parameterization=optics['phase_param'],init=optics['phase_init'],
                              init_std=optics['init_std'],phase_dropout_mode='none',phase_dropout_p=0.0)

    def forward(self,field):
        result=field.clone();a,b=self.start,self.end
        result[:,a:b,a:b]=self.phase(field[:,a:b,a:b])
        return result


class D2NN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        exp=cfg['experiment'];optics=cfg['optics'];d=cfg['detector'];dist=optics['distances_m']
        self.canvas=exp['baseline_canvas_size'];side=exp['baseline_phase_size'];depth=exp['baseline_layers']
        self.active=exp['oeo_active_size'];self.a=(self.canvas-self.active)//2;self.b=self.a+self.active
        self.input_size=cfg['model']['input_size'];self.pad=(self.canvas-self.input_size)//2
        assert self.canvas==cfg['model']['canvas_size'] and self.active==cfg['model']['active_size']
        assert self.canvas-self.input_size==2*self.pad and self.canvas-self.active==2*self.a
        self.phases=nn.ModuleList([CenteredPhase(side,self.canvas,optics) for _ in range(depth)])
        common=dict(wavelength_m=optics['wavelength_m'],pixel_size_m=optics['pixel_size_m'],grid_size=self.canvas,
                    evanescent_mode=optics['evanescent_mode'],k_space_constraint_enabled=optics['k_space_constraint_enabled'])
        self.propagators=nn.ModuleList([AngularSpectrumPropagator(distance_m=dist['global_fc_to_oeo'],**common) for _ in range(depth)])
        self.camera=AngularSpectrumPropagator(distance_m=dist['last_oeo_to_detector'],**common)
        self.eps=cfg['nonlinearity']['normalization']['eps']
        self.activation=cfg['nonlinearity']['activation']['type']
        assert self.activation in CANDIDATES
        self.oeo_enabled=bool(exp['oeo_enabled'])
        assert self.oeo_enabled==cfg['nonlinearity']['enabled']==cfg['global_oeo']['enabled']
        assert not cfg['nonlinearity']['normalization']['elementwise_affine']
        self.detector=DetectorArray(2,self.canvas,d['detector_size'],d['layout'],d['normalize_detector_energy'],
                                    start_pos_x=d['start_pos_x'],start_pos_y=d['start_pos_y'],
                                    n_det_sets=d['N_det_sets'],det_steps_x=d['det_steps_x'],det_steps_y=d['det_steps_y'])

    @property
    def masks(self):return self.detector.masks

    def forward(self,images):
        assert images.shape[1:]==(1,self.input_size,self.input_size)
        field=F.pad(images[:,0],(self.pad,)*4).to(torch.complex64)
        for phase,prop in zip(self.phases,self.propagators):
            field=prop(phase(field))
            if self.oeo_enabled:
                # Same per-sample full-panel, non-affine OEO as the MoE stages.
                intensity=field[:,self.a:self.b,self.a:self.b].abs().square()
                amplitude=activate(F.layer_norm(intensity,(self.active,self.active),eps=self.eps),self.activation)
                field=field.clone();field[:,self.a:self.b,self.a:self.b]=torch.complex(amplitude,torch.zeros_like(amplitude))
            # Otherwise propagate the complex field unchanged, including phase.
        field=self.camera(field)
        return {'intensity':field.abs().square(),'energies':self.detector(field),'routing':None,'route_probabilities':None}


def build(name,cfg):return {'moe':OpticalMoE,'d2nn':D2NN}[name](cfg)
def objective(output,labels,masks):
    return detector_plane_mse_loss(output['intensity'],masks[labels],scale=100.0,normalize=True,eps=1e-8)
def probabilities(output):
    energy=output['energies']
    return (energy+1e-12)/(energy.sum(1,keepdim=True)+2e-12)
