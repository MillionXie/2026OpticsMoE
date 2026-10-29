"""Minibatches for frozen-front SDXL trajectory training, not inference batching."""
import torch


def pack_features(features,device):
    lengths=[len(f['hidden']) for f in features];width=features[0]['hidden'].shape[-1]
    hidden=torch.zeros(len(features),max(lengths),width,device=device,dtype=torch.float32)
    padding=torch.ones(len(features),max(lengths),device=device,dtype=torch.bool)
    for i,(f,n) in enumerate(zip(features,lengths)):
        hidden[i,:n]=f['hidden'].to(device).float();padding[i,:n]=False
    return hidden,torch.stack([f['visual'] for f in features]).to(device).float(),padding


def interval_loss(predicted,wanted):
    if predicted.shape!=wanted.shape or predicted.ndim!=4:raise ValueError('Matched BCHW intervals required')
    return ((predicted.float()-wanted).square().mean((1,2,3))/wanted.square().mean((1,2,3)).clamp_min(1.)).mean()
