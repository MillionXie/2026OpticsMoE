"""Original LSP learned body/head fed by exact cached frozen-stem tokens."""
import math
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import pathlib
import torch
from torch import nn
from .modeling import LightGenDenseVision2Core, OpticalDetectorTopKRouter, PoseHeatmapDecoder, architecture_label
from experiments.vision2_hybrid_dense.modeling import restore_qwen_block_major_spatial
from .build_lab_package import sha
from .lab_preflight import EXPECTED_SHA


class PortableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'PosixPath':
            return pathlib.PurePosixPath
        return super().find_class(module, name)


PORTABLE_PICKLE = SimpleNamespace(__name__='pickle', Unpickler=PortableUnpickler,
                                 load=pickle.load, loads=pickle.loads)


class CachedStudent(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.core=LightGenDenseVision2Core(settings.vision_hidden_size,settings)
        branch=self.core.optical_branch
        branch.core.router=OpticalDetectorTopKRouter(branch.core.geometry,settings)
        self.head=PoseHeatmapDecoder(input_dim=settings.electronic_width,heatmap_size=settings.heatmap_size,num_joints=14)

    def forward(self,batch):
        device=next(self.parameters()).device
        grid=batch['grid'].to(device)
        tokens=batch['tokens'].to(device)
        shapes=[tuple(map(int,r)) for r in grid.tolist()]
        self.core.forward_groups(list(tokens.split([t*h*w for t,h,w in shapes])),shapes)
        spatial=restore_qwen_block_major_spatial(torch.cat(self.core.last_latent_groups),grid)
        return self.head(spatial),spatial,self.core.optical_branch.core.current_detector_readout


def load_model(root):
    root=Path(root);pt=root/'weights/best_checkpoint.pt'
    if sha(pt)!=EXPECTED_SHA:raise ValueError('Wrong LSP PT')
    cfg=SimpleNamespace(**json.loads((root/'settings.json').read_text()))
    payload=torch.load(pt,map_location='cpu',weights_only=False,pickle_module=PORTABLE_PICKLE)
    if payload['checkpoint_architecture']!=architecture_label(cfg):raise ValueError('Architecture mismatch')
    model=CachedStudent(cfg)
    model.core.load_state_dict(payload['core'],strict=True);model.head.load_state_dict(payload['head'],strict=True)
    model.core.set_phase_dropout_active(False)
    return model.eval().requires_grad_(False)


def phase_planes(model):
    branch=model.core.optical_branch;g=branch.core.geometry;a=g.active_aperture
    field=next(model.parameters()).new_ones(1,g.canvas_size,g.canvas_size)
    with torch.no_grad():
        planes={'router':branch.core.router.active_phase()}
        for stage,modulation in [('expert',branch._expert_phase_modulation(field)),('global',branch._global_phase_modulation(field))]:
            planes[stage]=torch.remainder(torch.angle(modulation[0,a.y0:a.y1,a.x0:a.x1]),2*math.pi)
    return {k:v.detach().cpu().numpy() for k,v in planes.items()}
