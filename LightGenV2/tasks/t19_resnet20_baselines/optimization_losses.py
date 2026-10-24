"""TRAIN-diagnosed objectives; original models, data and metrics stay intact."""
import torch
from torch import nn
from torch.nn import functional as F
from .frozen_resnet18 import FrozenResNet18


def pose_distribution_loss(logits, target, visible):
    """Per-visible-joint spatial cross entropy against normalized Gaussian GT."""
    if logits.shape != target.shape or logits.ndim != 4 or visible.shape != logits.shape[:2]:
        raise ValueError('Expected matched [B,J,H,W] maps and [B,J] visibility')
    target = target.float().flatten(2)
    mass = target.sum(-1)
    valid = visible.bool() & mass.gt(0)
    if not valid.any():
        return logits.sum() * 0.
    probability = target / mass.clamp_min(1e-12)[..., None]
    values = -(probability * logits.float().flatten(2).log_softmax(-1)).sum(-1)
    return values[valid].mean()


def balanced_edit_loss(output, batch):
    """Equal scene weight, explicit changed-cell category and mask objectives."""
    category = F.cross_entropy(output['category_logits'].float(), batch['target_grid'].long(), reduction='none')
    changed = batch['edit_grid'].float()
    preserve = 1. - changed
    def per_scene(values, mask):
        return ((values * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)).mean()
    edit = F.binary_cross_entropy_with_logits(output['edit_logits'].float(), changed, reduction='none')
    return (per_scene(category, changed) + .25 * per_scene(category, preserve)
            + .5 * per_scene(edit, changed) + .5 * per_scene(edit, preserve)
            + .1 * F.cross_entropy(output['task_logits'].float(), batch['task_index'].long()))


def forward_features(model, task, features, tokens=None):
    """Exactly the existing task head, with explicit frozen-encoder feature input."""
    if task == 'lsp':
        size = features[0].shape[-2:]
        fused = torch.stack([F.interpolate(project(x), size, mode='bilinear', align_corners=False)
            for x,project in zip(features,model.lateral)]).sum(0)
        return F.interpolate(model.head(fused),(model.output_size,model.output_size),mode='bilinear',align_corners=False)
    if task == 'salicon':
        return F.interpolate(model.head(features[-1]),(224,224),mode='bilinear',align_corners=False)
    x = model.image_projection(F.adaptive_avg_pool2d(features[-1],(6,6)))
    condition = model.text_projection(model.text_encoder(tokens))
    gamma,beta = model.film(condition).chunk(2,-1)
    x = x * (1+torch.tanh(gamma)[:,:,None,None]) + beta[:,:,None,None]
    axis = torch.linspace(-1,1,6,device=x.device,dtype=x.dtype)
    yy,xx = torch.meshgrid(axis,axis,indexing='ij')
    x = model.spatial(x+model.coordinates(torch.stack((xx,yy))[None]))
    return dict(category_logits=model.category(x),edit_logits=model.edit(x).squeeze(1),task_logits=model.task(condition))


class SmallSaliconHead(nn.Module):
    """Official frozen RN18 and one learned512->1 convolution:513 task parameters."""
    def __init__(self):
        super().__init__()
        self.encoder = FrozenResNet18()
        self.head = nn.Conv2d(512,1,1)

    def forward(self, images):
        return forward_features(self,'salicon',self.encoder.features(images))
