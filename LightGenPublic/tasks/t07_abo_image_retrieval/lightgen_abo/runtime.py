"""Evaluation and phase plotting; no training or large-model forward."""
import torch
import numpy as np
from torch.nn import functional as F
from .data import _gallery_centroids, _category_prototypes, _evaluate
from .io import inputs, picture, write_csv

def autocast(device):
    return torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda')

@torch.no_grad()
def encode(model, processor, samples, device, batch_size):
    model.eval()
    output = []
    for start in range(0,len(samples),batch_size):
        batch = inputs(processor,[picture(s.image_path) for s in samples[start:start+batch_size]],device)
        with autocast(device):
            output.append(model(batch).detach().cpu())
        if (start//batch_size+1)%60==0:
            print(f'encoded {min(start+batch_size,len(samples))}/{len(samples)}',flush=True)
    return torch.cat(output)

@torch.no_grad()
def evaluate(model, processor, train, test, device, batch_size, output=None):
    vtrain = encode(model,processor,train,device,batch_size)
    vtest = encode(model,processor,test,device,batch_size)
    gallery,metadata = _gallery_centroids(train,F.normalize(vtrain.float(),dim=-1))
    metrics,rows,categories = _evaluate(vtest,test,gallery,metadata,_category_prototypes(gallery,metadata))
    if output:
        write_csv(output/'retrieval_predictions.csv',rows)
        write_csv(output/'per_category_metrics.csv',categories)
        torch.save({'train_ids':[s.sample_id for s in train],'test_ids':[s.sample_id for s in test],
                    'train':vtrain,'test':vtest},output/'retrieval_features.pt')
    return metrics

def preview(model, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,6,figsize=(14,5),constrained_layout=True)
    for row,mode in enumerate(('vision','language')):
        optics=getattr(model,mode).optics
        raws=[optics.router.raw_router_phase,*optics.experts,optics.global_phase]
        for col,(raw,label) in enumerate(zip(raws,['router','expert0','expert1','expert2','expert3','global'])):
            axes[row,col].imshow((2*torch.pi*raw.detach().cpu().sigmoid()).numpy(),vmin=0,vmax=2*np.pi,cmap='twilight')
            axes[row,col].set_title(mode+' '+label);axes[row,col].set_axis_off()
    fig.savefig(output/'phase_masks.png',dpi=160);plt.close(fig)


@torch.no_grad()
def inspect_ccd(model, processor, samples, device, output):
    """One deterministic view per product; read-only energy/readout screening.

    Pooled-intensity fractions describe this numerical decoder, not physical
    detector photon fractions. No images or embeddings are corrected here.
    """
    model.eval()
    selected = {}
    for sample in samples:
        selected.setdefault(sample.product_id, sample)
    rows = []
    for sample in selected.values():
        batch = inputs(processor, [picture(sample.image_path)], device)
        with autocast(device):
            model(batch)
        for name in ('vision', 'language'):
            modality = getattr(model, name)
            length = modality.last_latent.shape[1]
            for stage, intensity in modality.optics.last_ccd.items():
                raw_pool = F.adaptive_avg_pool2d(intensity[:, None].float(), (224, 224))[:, 0]
                relative = (intensity / intensity.mean((-2, -1), keepdim=True).clamp_min(1e-6)).clamp_max(12)
                decoded = F.relu(F.layer_norm(F.adaptive_avg_pool2d(torch.log1p(relative)[:, None], (224, 224))[:, 0], (224,), eps=1e-5))
                rows.append(dict(sample_id=sample.sample_id, product_id=sample.product_id,
                                 modality=name, stage=stage, retained_rows=length,
                                 pooled_intensity_retained=float(raw_pool[:, :length].sum() / raw_pool.sum().clamp_min(1e-8)),
                                 decoded_squared_norm_retained=float(decoded[:, :length].square().sum() / decoded.square().sum().clamp_min(1e-8))))
    write_csv(output / 'ccd_readout_audit.csv', rows)
