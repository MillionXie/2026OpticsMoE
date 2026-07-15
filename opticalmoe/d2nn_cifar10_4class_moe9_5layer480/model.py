import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINE_OPTICS_PATH = Path(__file__).resolve().parent.parent / "d2nn_mnist4_amp_1phase400" / "optics.py"
_spec = importlib.util.spec_from_file_location("reliable_d2nn_optics", BASELINE_OPTICS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load reliable baseline optics from {BASELINE_OPTICS_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
AngularSpectrumPropagator = _module.AngularSpectrumPropagator
DetectorArray = _module.DetectorArray
PhaseLayer = _module.PhaseLayer

from layout import MoELayout
from prompt import GlobalRouterPrompt


class SquareDetectionLayerNormReload(nn.Module):
    """Square detection and configurable expert-wise LayerNorm/reload.

    In the default MoE path, each 120x120 expert region computes independent
    per-sample statistics and owns independent gamma/beta parameters at every
    stage. Routing amplitude is not multiplied back after normalization.
    """

    def __init__(
        self, eps=1e-5, nonlinearity="relu", layout=None,
        per_expert_enabled=False, elementwise_affine=False,
        affine_sharing="per_expert", reapply_routing_amplitude=False,
    ):
        super().__init__();self.eps=float(eps);self.nonlinearity=str(nonlinearity).lower();self.layout=layout
        if self.eps<=0:raise ValueError("optoelectronic_interlayers.eps must be positive")
        if self.nonlinearity not in {"relu","softplus"}:raise ValueError("optoelectronic_interlayers.nonlinearity must be relu or softplus")
        self.per_expert_enabled=bool(per_expert_enabled)
        self.elementwise_affine=bool(elementwise_affine)
        self.affine_sharing=str(affine_sharing)
        if self.affine_sharing not in {"per_expert","per_stage"}:raise ValueError("affine_sharing must be per_expert or per_stage")
        if bool(reapply_routing_amplitude):raise ValueError("routing amplitude must not be reapplied after LayerNorm/ReLU")
        if self.per_expert_enabled and self.layout is None:raise ValueError("layout is required for per-expert LayerNorm")
        if self.elementwise_affine:
            size=self.layout.expert_size if self.per_expert_enabled else self.layout.canvas_size
            count=self.layout.num_experts if self.per_expert_enabled and self.affine_sharing=="per_expert" else 1
            self.affine_weight=nn.Parameter(torch.ones(count,size,size,dtype=torch.float32))
            self.affine_bias=nn.Parameter(torch.zeros(count,size,size,dtype=torch.float32))
        else:
            self.register_parameter("affine_weight",None);self.register_parameter("affine_bias",None)

    def _activate(self,normalized):
        return F.relu(normalized) if self.nonlinearity=="relu" else F.softplus(normalized)

    def forward(self,field,return_details=False):
        intensity=field.to(torch.complex64).abs().square().float()
        if self.per_expert_enabled:
            crops=torch.stack([
                intensity[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]
                for aperture in self.layout.expert_apertures
            ],dim=1)
            means=crops.mean((-2,-1));variances=(crops-means[...,None,None]).square().mean((-2,-1))
            normalized_crops=F.layer_norm(crops,crops.shape[-2:],weight=None,bias=None,eps=self.eps)
            if self.affine_weight is not None:
                weight=self.affine_weight if self.affine_sharing=="per_expert" else self.affine_weight[0]
                bias=self.affine_bias if self.affine_sharing=="per_expert" else self.affine_bias[0]
                normalized_crops=normalized_crops*weight+bias
            amplitude_crops=self._activate(normalized_crops)
            normalized=torch.zeros_like(intensity);amplitude=torch.zeros_like(intensity)
            for index,aperture in enumerate(self.layout.expert_apertures):
                normalized[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]=normalized_crops[:,index]
                amplitude[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]=amplitude_crops[:,index]
            normalization_mean=means;normalization_std=torch.sqrt(variances+self.eps)
        else:
            normalization_mean=intensity.mean((-2,-1));variance=(intensity-normalization_mean[...,None,None]).square().mean((-2,-1))
            normalized=F.layer_norm(intensity,intensity.shape[-2:],weight=None,bias=None,eps=self.eps)
            if self.affine_weight is not None:normalized=normalized*self.affine_weight[0]+self.affine_bias[0]
            amplitude=self._activate(normalized);normalization_std=torch.sqrt(variance+self.eps)
        reloaded=torch.complex(amplitude,torch.zeros_like(amplitude))
        if not return_details:return reloaded
        return reloaded,{
            "detector_intensity":intensity,"layer_normalized":normalized,"reloaded_amplitude":amplitude,
            "normalization_mean":normalization_mean,"normalization_std":normalization_std,
            "normalization_scope":"per_expert" if self.per_expert_enabled else "spatial_per_sample",
            "elementwise_affine":self.elementwise_affine,"affine_sharing":self.affine_sharing,
            "routing_amplitude_reapplied":False,
        }


class ExpertPhasePlane(nn.Module):
    def __init__(self, layout, optics_cfg, dropout_cfg):
        super().__init__(); self.layout = layout
        enabled = bool(dropout_cfg.get("enabled", False))
        mode = dropout_cfg.get("mode", "none") if enabled else "none"
        p = float(dropout_cfg.get("p", 0.0)) if enabled else 0.0
        self.experts = nn.ModuleList([
            PhaseLayer(
                layout.expert_size,
                parameterization=optics_cfg.get("phase_param", "sigmoid"),
                init=optics_cfg.get("phase_init", "zeros"),
                init_std=float(optics_cfg.get("init_std", 0.02)),
                phase_dropout_mode=mode,
                phase_dropout_p=p,
                phase_dropout_block_size=int(dropout_cfg.get("block_size", 8)),
                phase_dropout_batch_shared=bool(dropout_cfg.get("batch_shared", True)),
            ) for _ in range(9)
        ])

    def forward(self, field):
        output = torch.zeros_like(field, dtype=torch.complex64)
        for aperture, expert in zip(self.layout.expert_apertures, self.experts):
            local = field[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1]
            output[:, aperture.y0:aperture.y1, aperture.x0:aperture.x1] = expert(local)
        return output

    def phase_stack(self): return torch.stack([expert.get_phase_wrapped() for expert in self.experts])

    def phase_mosaic(self):
        canvas = torch.zeros(self.layout.canvas_size, self.layout.canvas_size, device=self.experts[0].raw_phase.device)
        for aperture, phase in zip(self.layout.expert_apertures, self.phase_stack()):
            canvas[aperture.y0:aperture.y1, aperture.x0:aperture.x1] = phase
        active = self.layout.active_aperture
        return canvas[active.y0:active.y1, active.x0:active.x1]

    def set_phase_dropout_active(self, active):
        for expert in self.experts: expert.set_phase_dropout_active(active)


class GlobalFCPhaseLayer(nn.Module):
    """One trainable 450x450 phase plate with transparent 15-pixel border."""

    def __init__(self,layout,optics_cfg,dropout_cfg):
        super().__init__();self.layout=layout;enabled=bool(dropout_cfg.get("enabled",False))
        self.phase=PhaseLayer(
            layout.active_size,parameterization=optics_cfg.get("phase_param","sigmoid"),init=optics_cfg.get("phase_init","zeros"),
            init_std=float(optics_cfg.get("init_std",0.02)),phase_dropout_mode=dropout_cfg.get("mode","none") if enabled else "none",
            phase_dropout_p=float(dropout_cfg.get("p",0.0)) if enabled else 0.0,phase_dropout_block_size=int(dropout_cfg.get("block_size",8)),
            phase_dropout_batch_shared=bool(dropout_cfg.get("batch_shared",True)),
        )

    def forward(self,field):
        aperture=self.layout.active_aperture;output=field.to(torch.complex64).clone();crop=field[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]
        output[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]=self.phase(crop);return output

    def get_phase(self):return self.phase.get_phase_wrapped()
    def set_phase_dropout_active(self,active):self.phase.set_phase_dropout_active(active)


class OpticalMoEClassifier(nn.Module):
    def __init__(self, config, num_classes=4):
        super().__init__(); self.config = config; self.num_classes = int(num_classes)
        model_cfg = config.get("model", {}); optics = config.get("optics", {}); detector = config.get("detector", {})
        self.layout = MoELayout(
            canvas_size=int(model_cfg.get("canvas_size",480)), active_size=int(model_cfg.get("active_size",450)),
            input_size=int(model_cfg.get("input_size",120)), image_size=int(model_cfg.get("image_size",100)),
            num_experts=int(model_cfg.get("num_experts",9)), expert_size=int(model_cfg.get("expert_size",120)),
            expert_pitch=int(model_cfg.get("expert_pitch",150)),
        ); self.layout.validate()
        self.num_layers = int(model_cfg.get("num_layers",5))
        if self.num_layers != 5: raise ValueError("This experiment requires five expert layers.")
        conversion_cfg=config.get("optoelectronic_interlayers",{})
        self.optoelectronic_enabled=bool(conversion_cfg.get("enabled",False))
        if self.optoelectronic_enabled:
            expected={"detection":"square_law","normalization":"layer_norm","reload_as":"amplitude"}
            for key,value in expected.items():
                if str(conversion_cfg.get(key,value)).lower()!=value:raise ValueError(f"optoelectronic_interlayers.{key} must be {value!r}")
        # Older completed runs predate ``per_expert_enabled`` and explicitly
        # recorded ``normalization_scope: spatial_per_sample``.  Infer the
        # implementation from that saved scope so those checkpoints can be
        # reconstructed exactly; all current configs specify both fields.
        normalization_scope=str(conversion_cfg.get("normalization_scope","spatial_per_sample"))
        if "per_expert_enabled" in conversion_cfg:
            per_expert_enabled=bool(conversion_cfg["per_expert_enabled"])
        else:
            per_expert_enabled=normalization_scope=="per_expert"
        if self.optoelectronic_enabled and normalization_scope != ("per_expert" if per_expert_enabled else "spatial_per_sample"):
            raise ValueError("normalization_scope must match per_expert_enabled")
        self.interlayer_conversions=nn.ModuleList([
            SquareDetectionLayerNormReload(
                eps=float(conversion_cfg.get("eps",1e-5)),nonlinearity=str(conversion_cfg.get("nonlinearity","relu")),
                layout=self.layout,per_expert_enabled=per_expert_enabled,
                elementwise_affine=bool(conversion_cfg.get("elementwise_affine",conversion_cfg.get("affine",True))),
                affine_sharing=str(conversion_cfg.get("affine_sharing","per_expert")),
                reapply_routing_amplitude=bool(conversion_cfg.get("reapply_routing_amplitude",False)),
            ) for _ in range(self.num_layers)
        ]) if self.optoelectronic_enabled else nn.ModuleList()
        wavelength = float(optics.get("wavelength_m",5.32e-7)); pixel = float(optics.get("pixel_size_m",16e-6))
        distances = optics.get("distances_m", {})
        prop_common = dict(
            wavelength_m=wavelength, pixel_size_m=pixel, grid_size=self.layout.canvas_size,
            evanescent_mode=optics.get("evanescent_mode","zero"),
            k_space_constraint_enabled=bool(optics.get("k_space_constraint_enabled",False)),
            theta_max_deg=float(optics.get("theta_max_deg",1.0)),
        )
        input_to_prompt_distance=float(distances.get("input_to_prompt",0.30))
        self.input_to_prompt = AngularSpectrumPropagator(distance_m=input_to_prompt_distance,**prop_common)
        prompt_to_expert_distance = float(distances.get("prompt_to_expert",0.30))
        prompt_cfg=config.get("prompt",{})
        focal_length=float(optics.get("prompt_focal_length_m",prompt_to_expert_distance))
        if bool(prompt_cfg.get("enforce_global_convolution_geometry",True)):
            expected_distance=2.0*focal_length
            tolerance=float(prompt_cfg.get("convolution_relative_tolerance",0.02))
            if abs(prompt_to_expert_distance-expected_distance)/expected_distance>tolerance:
                raise ValueError(
                    f"prompt_to_expert={prompt_to_expert_distance:.6f} m is incompatible with the global fan-out "
                    f"convolution geometry; expected 2*f={expected_distance:.6f} m."
                )
        self.prompt = GlobalRouterPrompt(
            self.layout,wavelength,pixel,input_to_prompt_distance,prompt_to_expert_distance,focal_length,
            top_k=int(prompt_cfg.get("top_k",3)),pool_size=int(prompt_cfg.get("router_pool_size",10)),temperature=float(prompt_cfg.get("temperature",1.0)),
            grating_sign_x=float(prompt_cfg.get("grating_sign_x",1.0)),grating_sign_y=float(prompt_cfg.get("grating_sign_y",1.0)),
            min_grating_period_pixels=float(prompt_cfg.get("min_grating_period_pixels",4.0)),
            mode=str(prompt_cfg.get("mode",prompt_cfg.get("type","region_amplitude_global_lens"))),
        )
        self.prompt_to_expert = AngularSpectrumPropagator(distance_m=prompt_to_expert_distance,**prop_common)
        dropout = config.get("regularization",{}).get("phase_dropout",{})
        self.expert_layers = nn.ModuleList([ExpertPhasePlane(self.layout,optics,dropout) for _ in range(self.num_layers)])
        self.inter_props = nn.ModuleList([AngularSpectrumPropagator(distance_m=float(distances.get("inter_layer",0.05)),**prop_common) for _ in range(4)])
        self.last_expert_to_global_fc=AngularSpectrumPropagator(distance_m=float(distances.get("last_expert_to_global_fc",0.05)),**prop_common)
        self.global_fc=GlobalFCPhaseLayer(self.layout,optics,dropout)
        self.to_detector = AngularSpectrumPropagator(distance_m=float(distances.get("global_fc_to_detector",0.10)),**prop_common)
        self.detector = DetectorArray(
            self.num_classes,self.layout.canvas_size,int(detector.get("detector_size",50)),detector.get("layout","fixed_2x2"),
            bool(detector.get("normalize_detector_energy",True)),start_pos_x=int(detector.get("start_pos_x",115)),
            start_pos_y=int(detector.get("start_pos_y",115)),n_det_sets=detector.get("N_det_sets",[2,2]),
            det_steps_x=detector.get("det_steps_x",[150,150]),det_steps_y=int(detector.get("det_steps_y",150)),
            start_pos_x_per_row=detector.get("start_pos_x_per_row"),
        )

    def prepare_canvas_input(self, images):
        if images.ndim==3: images=images.unsqueeze(1)
        if images.shape[1]!=1: images=images.mean(1,keepdim=True)
        if tuple(images.shape[-2:])!=(self.layout.input_size,self.layout.input_size):
            # Match the dataset path: smooth resize to 100x100, clamp possible
            # bicubic overshoot, then explicit zero padding to 120x120.
            dataset_cfg=self.config.get("dataset",{})
            resize_mode=str(dataset_cfg.get("resize_interpolation","bicubic")).lower()
            if resize_mode not in {"nearest","bilinear","bicubic"}:
                raise ValueError("dataset.resize_interpolation must be nearest, bilinear, or bicubic.")
            kwargs={"mode":resize_mode}
            if resize_mode in {"bilinear","bicubic"}:
                kwargs.update({"align_corners":False,"antialias":bool(dataset_cfg.get("resize_antialias",True))})
            images=F.interpolate(images.float(),size=(self.layout.image_size,self.layout.image_size),**kwargs).clamp_(0.0,1.0)
            pad=(self.layout.input_size-self.layout.image_size)//2
            images=F.pad(images,(pad,pad,pad,pad),mode="constant",value=0.0)
        aperture=self.layout.input_aperture
        canvas=torch.zeros(images.shape[0],self.layout.canvas_size,self.layout.canvas_size,device=images.device)
        canvas[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]=images[:,0].clamp(0,1)
        return canvas.to(torch.complex64)

    def expert_energy_ratios(self,field):
        intensity=field.to(torch.complex64).abs().square();energies=[]
        for aperture in self.layout.expert_apertures:
            energies.append(intensity[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1].sum(dim=(-2,-1)))
        energies=torch.stack(energies,dim=1);total=intensity.sum(dim=(-2,-1),keepdim=False)
        return energies/(total[:,None]+1e-12)

    @staticmethod
    def global_fanout_convolution(field, prompt_transmission):
        """4f-style convolution with the physical prompt kernel.

        The prompt amplitude cells are kernel channels.  This is intentionally
        not ``ASM -> pointwise prompt -> ASM``: that old path illuminates the
        centre cell and cannot make the nine amplitude cells act as nine
        independent routing channels.
        """
        field=field.to(torch.complex64)
        prompt_transmission=prompt_transmission.to(torch.complex64)
        flipped=torch.flip(field,dims=(-2,-1))
        return torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.fft2(flipped)*torch.fft.fft2(prompt_transmission)),
            dim=(-2,-1),
        )

    def forward(self, images, return_intermediates=False, capture_layer_fields=True):
        canvas=self.prepare_canvas_input(images)
        routing=self.prompt.routing(images)
        # The global fan-out prompt is a 4f convolution kernel.  The two maps
        # loaded on the prompt are exactly amplitude and phase; no hidden
        # combined-amplitude map participates in the forward path.
        prompt_transmission=routing["transmission"]
        field=self.global_fanout_convolution(canvas,prompt_transmission)
        entrance=field
        entrance_energy_ratios=self.expert_energy_ratios(entrance)
        layer_fields=[];interlayer_detector_intensities=[];interlayer_normalized=[];interlayer_reloaded_amplitudes=[]
        for index,layer in enumerate(self.expert_layers):
            field=layer(field)
            propagation=self.inter_props[index] if index<4 else self.last_expert_to_global_fc
            propagated=propagation(field)
            if self.optoelectronic_enabled:
                conversion_layer=self.interlayer_conversions[index]
                if return_intermediates and capture_layer_fields:
                    field,conversion=conversion_layer(propagated,return_details=True)
                    interlayer_detector_intensities.append(conversion["detector_intensity"])
                    interlayer_normalized.append(conversion["layer_normalized"])
                    interlayer_reloaded_amplitudes.append(conversion["reloaded_amplitude"])
                else:field=conversion_layer(propagated)
            else:field=propagated
            if return_intermediates and capture_layer_fields:layer_fields.append(field)
        at_global_fc=field
        after_global_fc=self.global_fc(at_global_fc)
        detector_field=self.to_detector(after_global_fc)
        intensity=torch.abs(detector_field).square()
        logits=self.detector(detector_field)
        if not return_intermediates:return logits
        return logits,{
            "input_canvas":canvas,"at_prompt":canvas,"prompt_amplitude":routing["prompt_amplitude"],
            "prompt_phase":routing["prompt_phase"],"routing_logits":routing["logits"],"routing_probabilities":routing["probabilities"],
            "routing_weights":routing["weights"],"routing_selected_mask":routing["selected_mask"],"routing_selected_indices":routing["selected_indices"],
            "router_balance_loss":routing["balance_loss"],"router_importance":routing["importance"],"router_load":routing["load"],
            "router_importance_loss":routing["importance_loss"],"router_normalized_entropy":routing["normalized_entropy"],
            "after_prompt":prompt_transmission,"prompt_transmission":prompt_transmission,
            "expert_entrance":entrance,"expert_entrance_energy_ratios":entrance_energy_ratios,
            "after_each_expert_layer":layer_fields,"detector_field":detector_field,"detector_intensity":intensity,
            "optoelectronic_interlayers_enabled":self.optoelectronic_enabled,"interlayer_detector_intensities":interlayer_detector_intensities,
            "interlayer_layer_normalized":interlayer_normalized,"interlayer_reloaded_amplitudes":interlayer_reloaded_amplitudes,
            "at_global_fc":at_global_fc,"after_global_fc":after_global_fc,"global_fc_phase":self.global_fc.get_phase(),"detector_energies":logits,
        }

    def phase_stack(self): return torch.stack([layer.phase_stack() for layer in self.expert_layers])

    def expert_phase_mosaics(self): return torch.stack([layer.phase_mosaic() for layer in self.expert_layers])

    def set_phase_dropout_active(self,active):
        for layer in self.expert_layers: layer.set_phase_dropout_active(active)
        self.global_fc.set_phase_dropout_active(active)

    def expert_phase_parameter_count(self): return sum(v.raw_phase.numel() for layer in self.expert_layers for v in layer.experts)
    def global_fc_parameter_count(self):return self.global_fc.phase.raw_phase.numel()
    def router_parameter_count(self):return sum(p.numel() for p in self.prompt.router_network.parameters())
    def prompt_parameter_count(self): return 0
    def optical_parameter_count(self): return self.expert_phase_parameter_count()+self.global_fc_parameter_count()
    def interlayer_conversion_parameter_count(self):return sum(p.numel() for p in self.interlayer_conversions.parameters() if p.requires_grad)
    def electronic_parameter_count(self): return self.router_parameter_count()+self.interlayer_conversion_parameter_count()
