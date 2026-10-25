"""Same single-sensor three-channel layout for RGB and SAR, no domain selector.

T21 fuses two sensors; T20 evaluates either one. Do not turn that change into
an untrained spatial-position shift. Keeps T21's exact frozen visual frontend,
feature strip, optical graph, 8bit payload and original linear readout.
"""
import torch
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.frozen_vision import visual_tiles
from LightGenV2.tasks.t13_four_modal_lifelong.model import normalize_power
from .train_unified_static import cache_frontend

CONTRACT='T20_sensor_neutral_T21_native8_A_graph_v2'


def encode(images,feature_rows):
    if images.dtype!=torch.uint8 or images.ndim!=4 or images.shape[1:]!=(56,56,3) or feature_rows.shape!=(len(images),128):raise ValueError('Audited raw3channels and frozen128 features required')
    tiles=visual_tiles(images.cpu().numpy(),feature_rows.float(),images.device)
    zero=torch.zeros_like(tiles[:,0]);result=torch.cat((torch.cat((tiles[:,0],tiles[:,1]),-1),torch.cat((tiles[:,2],zero),-1)),-2)
    return normalize_power(result,1.)


def frontend(data,checkpoint):
    # Existing extractor selects RGB rows. Request all rows for this genuinely
    # single-sensor task; no dataset IDs, labels or stored sensor values change.
    request=dict(data,domains=__import__('numpy').zeros(len(data['images']),dtype='int64'))
    return cache_frontend(request,checkpoint)


def amplitude(data,features,ix,device):
    return encode(torch.from_numpy(data['images'][ix]),features[ix].float()).to(device)
