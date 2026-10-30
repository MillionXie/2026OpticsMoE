"""Training-only utilities; no extra parameters or inference operations."""
import math
import torch


def augment_rgb_tiles(amplitude,probability):
    """Apply the same D4 transform within each RGB tile, never swap colour tiles."""
    if probability==0:return amplitude
    if amplitude.ndim!=3 or amplitude.shape[-2:]!=(224,224):raise ValueError('224-square RGB packing required')
    result=amplitude.clone()
    rgb=torch.stack((amplitude[:,:112,:112],amplitude[:,:112,112:],amplitude[:,112:,:112]),1)
    selected=torch.rand(len(amplitude),device=amplitude.device)<probability
    codes=torch.randint(8,(len(amplitude),),device=amplitude.device)
    for code in range(8):
        indices=selected&(codes==code)
        if not indices.any():continue
        channels=torch.rot90(rgb[indices],code%4,(-2,-1))
        if code>=4:channels=channels.flip(-1)
        result[indices]=torch.cat((torch.cat((channels[:,0],channels[:,1]),-1),
                                  torch.cat((channels[:,2],torch.zeros_like(channels[:,2])),-1)),-2)
    return result


def actual_top2_load_loss(dense):
    """Switch-style loss: stopped actual Top-2 load times differentiable mean power."""
    if dense.ndim!=2 or dense.shape[1]!=4:raise ValueError('four experts required')
    indices=dense.topk(2,dim=1).indices
    mask=torch.zeros_like(dense).scatter(1,indices,1.)
    load=mask.detach().mean(0)/2
    return 4*(load*dense.mean(0)).sum()


def dark_expert_coverage_loss(dense):
    """KL(uniform || batch mean): a usable calibration gradient for very dark ports."""
    return -dense.mean(0).clamp_min(1e-12).log().mean()-math.log(dense.shape[1])


def inverse_sqrt_class_weights(labels,classes):
    counts=torch.bincount(labels,minlength=classes).float()
    if (counts==0).any():raise ValueError('all training classes required')
    weights=(counts.mean()/counts).sqrt()
    return weights/weights.mean()


@torch.no_grad()
def update_phase_ema(ema,model,decay):
    # Raw phase coordinates are averaged; no learned electronic module is added.
    for target,source in zip(ema.parameters(),model.parameters()):target.lerp_(source,1-decay)
