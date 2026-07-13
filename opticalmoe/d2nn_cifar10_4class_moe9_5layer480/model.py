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
        wavelength = float(optics.get("wavelength_m",5.32e-7)); pixel = float(optics.get("pixel_size_m",16e-6))
        distances = optics.get("distances_m", {})
        prop_common = dict(
            wavelength_m=wavelength, pixel_size_m=pixel, grid_size=self.layout.canvas_size,
            evanescent_mode=optics.get("evanescent_mode","zero"),
            k_space_constraint_enabled=bool(optics.get("k_space_constraint_enabled",False)),
            theta_max_deg=float(optics.get("theta_max_deg",1.0)),
        )
        self.input_to_prompt = AngularSpectrumPropagator(distance_m=float(distances.get("input_to_prompt",0.05)),**prop_common)
        prompt_to_expert_distance = float(distances.get("prompt_to_expert",0.05))
        prompt_cfg=config.get("prompt",{})
        self.prompt = GlobalRouterPrompt(
            self.layout,wavelength,pixel,prompt_to_expert_distance,float(optics.get("prompt_focal_length_m",prompt_to_expert_distance)),
            top_k=int(prompt_cfg.get("top_k",3)),pool_size=int(prompt_cfg.get("router_pool_size",10)),temperature=float(prompt_cfg.get("temperature",1.0)),
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
        )

    def prepare_canvas_input(self, images):
        if images.ndim==3: images=images.unsqueeze(1)
        if images.shape[1]!=1: images=images.mean(1,keepdim=True)
        if tuple(images.shape[-2:])!=(self.layout.input_size,self.layout.input_size):
            # Match the dataset path exactly: resize the image to 100x100 with
            # nearest-neighbor, then add explicit zero padding to 120x120.
            images=F.interpolate(images.float(),size=(self.layout.image_size,self.layout.image_size),mode="nearest")
            pad=(self.layout.input_size-self.layout.image_size)//2
            images=F.pad(images,(pad,pad,pad,pad),mode="constant",value=0.0)
        aperture=self.layout.input_aperture
        canvas=torch.zeros(images.shape[0],self.layout.canvas_size,self.layout.canvas_size,device=images.device)
        canvas[:,aperture.y0:aperture.y1,aperture.x0:aperture.x1]=images[:,0].clamp(0,1)
        return canvas.to(torch.complex64)

    def forward(self, images, return_intermediates=False):
        canvas=self.prepare_canvas_input(images)
        at_prompt=self.input_to_prompt(canvas)
        after_prompt,routing=self.prompt(at_prompt,images)
        field=self.prompt_to_expert(after_prompt)
        entrance=field
        layer_fields=[]
        for index,layer in enumerate(self.expert_layers):
            field=layer(field)
            if return_intermediates: layer_fields.append(field)
            if index<4: field=self.inter_props[index](field)
        at_global_fc=self.last_expert_to_global_fc(field)
        after_global_fc=self.global_fc(at_global_fc)
        detector_field=self.to_detector(after_global_fc)
        intensity=torch.abs(detector_field).square()
        logits=self.detector(detector_field)
        if not return_intermediates:return logits
        return logits,{
            "input_canvas":canvas,"at_prompt":at_prompt,"prompt_amplitude":routing["prompt_amplitude"],
            "prompt_phase":routing["prompt_phase"],"routing_logits":routing["logits"],"routing_probabilities":routing["probabilities"],
            "routing_weights":routing["weights"],"routing_selected_mask":routing["selected_mask"],"routing_selected_indices":routing["selected_indices"],
            "router_balance_loss":routing["balance_loss"],"router_importance":routing["importance"],"router_load":routing["load"],
            "after_prompt":after_prompt,"expert_entrance":entrance,
            "after_each_expert_layer":layer_fields,"detector_field":detector_field,"detector_intensity":intensity,
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
    def electronic_parameter_count(self): return self.router_parameter_count()
