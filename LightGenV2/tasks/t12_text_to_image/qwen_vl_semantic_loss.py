"""Optional frozen CLIP image-caption TRAIN loss, never an inference branch."""
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .qwen_vl_turbo_data import sha256


class CaptionImageLoss(nn.Module):
    def __init__(self, encoder):
        super().__init__();self.encoder=encoder.eval().requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None])

    def forward(self, image, text_features):
        self.encoder.eval()
        value=F.interpolate(image.float().clamp(-1,1).add(1).div(2),size=(224,224),mode='bicubic',align_corners=False,antialias=True)
        visual=F.normalize(self.encoder.encode_image((value-self.mean)/self.std).float(),dim=-1)
        return (1-(visual*F.normalize(text_features.float(),dim=-1)).sum(-1)).mean()


def product_caption(row,descriptions):
    if row.get('render_protocol')=='controlled_room_edit_v1' and row['mode']!='background':
        from .qwen_vl_simple_edits import LABELS
        return LABELS[row['category']][row['requested_family']]
    return row.get('target_description',descriptions[row['target']['sample_id']])


def prepare_caption_loss(repo, weights, rows, train_ids, descriptions, device):
    repo,weights=Path(repo),Path(weights)
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD']).decode().strip()
    if revision!='d05afc436d78f1c48dc0dbf8e5980a9d471f35f6' or not weights.is_file():
        raise ValueError('Unreviewed CLIP repository or missing local weights')
    if any(rows[i]['split']!='train' for i in train_ids):
        raise ValueError('Caption loss caches TRAIN labels only')
    sys.path.insert(0,str(repo));import clip
    encoder,_=clip.load(str(weights),device=device,jit=False);encoder=encoder.float().eval().requires_grad_(False)
    scene,product={},{}
    for start in range(0,len(train_ids),32):
        ids=train_ids[start:start+32]
        for dest,texts in ((scene,[rows[i]['teacher_caption'] for i in ids]),
                           (product,[product_caption(rows[i],descriptions) for i in ids])):
            tokens=clip.tokenize(texts,truncate=False).to(device)
            with torch.no_grad():features=F.normalize(encoder.encode_text(tokens).float(),dim=-1).cpu()
            dest.update(zip(ids,features))
    module=CaptionImageLoss(encoder).to(device)
    return module,scene,product,{'enabled':True,'TRAIN_only':True,'inference_branch':False,
        'frozen_parameters':sum(p.numel() for p in encoder.parameters()),'repo_sha':revision,
        'weights_sha256':sha256(weights),'preprocess':'Differentiable 256 RGB -> 224 bicubic, standard CLIP channel normalization; no target image enters encoder in this loss.',
        'reporting':'Training loss only; separate official-preprocess metrics and visual checks required.'}
