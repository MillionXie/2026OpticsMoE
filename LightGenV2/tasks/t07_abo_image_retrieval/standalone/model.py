"""Explicit six-capture model. No Qwen model, attention, or hook replacement.

V: router/expert/global; L: router/expert/global. Only the frozen front-end is
bfloat16; optical/electronic trainable parameters remain float32 as in source.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .frontend import Frontend
from .optics import OpticalPath


class Residual(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision
        self.token_norm = nn.LayerNorm(192)
        self.token_depthwise = (nn.Conv2d(192,192,3,groups=192,bias=False) if vision
                                else nn.Conv1d(192,192,5,groups=192,bias=False))
        self.token_pointwise = nn.Linear(192,192)
        self.token_dropout = nn.Dropout(.1)
        self.token_residual_logit = nn.Parameter(torch.zeros(()))
        self.residual_logit = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(192)
        self.mlp = nn.Sequential(nn.Linear(192,384),nn.GELU(),nn.Dropout(.1),nn.Linear(384,192),nn.Dropout(.1))

    def forward(self, x):
        n = self.token_norm(x)
        if self.vision:
            # Match original per-image 2D convolution (not folded into a new batch).
            outputs = []
            for row in n:
                grid = row.view(1,7,7,2,2,192).permute(0,5,1,3,2,4).reshape(1,192,14,14)
                grid = self.token_depthwise(F.pad(grid,(1,1,1,1)))
                outputs.append(grid.view(1,192,7,2,7,2).permute(0,2,4,3,5,1).reshape(196,192))
            update = torch.stack(outputs)
        else:
            update = self.token_depthwise(F.pad(n.transpose(1,2),(4,0))).transpose(1,2)
        update = self.token_dropout(self.token_pointwise(F.gelu(update)))
        x = x + self.token_residual_logit.sigmoid()*update
        return x + self.residual_logit.sigmoid()*self.mlp(self.norm(x))


def rms(x):
    # Preserve the original sum/count sequence, including detached statistics.
    return (x.square().sum((1,2))[:,None,None] / (x.shape[1]*x.shape[2])).sqrt().clamp_min(1e-6)


def alpha_value(raw, bounds=(.01,.95)):
    return bounds[0]+(bounds[1]-bounds[0])*raw.sigmoid()


def fuse(e, o, raw_alpha, bounds=(.01,.95)):
    e32, o32 = e.float(), o.float()
    re, ro = rms(e32).detach(), rms(o32).detach()
    alpha = alpha_value(raw_alpha,bounds)
    mixture = (1-alpha)*(e32/re) + alpha*(o32/ro)
    return (re*mixture/rms(mixture).detach()).to(e.dtype)


class Modality(nn.Module):
    def __init__(self, vision, input_rms, alpha_bounds=(.01,.95), noise_config=None):
        super().__init__()
        self.vision = vision
        self.alpha_bounds = alpha_bounds
        hidden = 1024 if vision else 2048
        self.input_adapter = nn.Linear(hidden,192)
        self.input_norm = nn.LayerNorm(192)
        self.blocks = nn.ModuleList([Residual(vision),Residual(vision)])
        self.output_norm = nn.LayerNorm(192)
        if vision:
            self.output_adapter = nn.Linear(192,1024)
            self.residual_logit = nn.Parameter(torch.zeros(()))
        self.block1_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        self.block2_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        self.optics = OpticalPath(input_rms,noise_config)
        self.remove_optical = False
        self.last_latent = None
        self.last_optical = None

    def forward(self, inputs):
        latent = self.input_norm(self.input_adapter(inputs.float()))
        e1 = self.blocks[0](latent)
        if self.remove_optical:
            f1 = e1
            weights = None
        else:
            o1, weights = self.optics.expert(latent)
            f1 = fuse(e1,o1,self.block1_optical_fusion_logit,self.alpha_bounds)
        e2 = self.blocks[1](f1)
        self.last_optical = None if self.remove_optical else self.optics.global_stage(f1,weights)
        f2 = e2 if self.remove_optical else fuse(e2,self.last_optical,self.block2_optical_fusion_logit,self.alpha_bounds)
        output = self.output_norm(f2)
        self.last_latent = output
        if self.vision:
            return (inputs.float()+torch.sigmoid(self.residual_logit)*self.output_adapter(output)).to(inputs.dtype)
        return output


class RetrievalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.projection = nn.Linear(384,64)

    def forward(self, latent):
        pooled = torch.stack([torch.cat((row.mean(0),row.amax(0))) for row in latent])
        return F.normalize(self.projection(self.norm(pooled.float())),p=2,dim=-1)


class OpticalRetrieval(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        bounds=(float(metadata.get('fusion_alpha_min',.01)),float(metadata.get('fusion_alpha_max',.95)))
        if not 0<=bounds[0]<bounds[1]<=1:raise ValueError('Invalid alpha bounds')
        self.frontend = Frontend(metadata['token_count']).to(torch.bfloat16)
        self.vision = Modality(True, metadata['input_rms'],bounds,metadata.get('optical_training_noise'))
        self.language = Modality(False, metadata['input_rms'],bounds,metadata.get('optical_training_noise'))
        if metadata.get('input_preprocessing','center_crop') not in ('center_crop','contain_white'):
            raise ValueError('Unknown image preprocessing contract')
        modes = metadata.get('ccd_readout_modes',{})
        if set(modes) - {'vision','language'}:
            raise ValueError('Unknown modality in CCD readout contract')
        for name in ('vision','language'):
            mode = modes.get(name,'prefix_rows')
            if mode not in ('prefix_rows','fullfield_rows'):
                raise ValueError('Unknown CCD readout contract')
            getattr(self,name).optics.readout_mode = mode
        self.readout = RetrievalHead()

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval()
        return self

    def set_remove_optical(self, active):
        self.vision.remove_optical = self.language.remove_optical = bool(active)

    def forward(self, batch):
        ids = batch['input_ids']
        expected = torch.tensor(self.metadata['template_ids'],device=ids.device)
        if not torch.equal(ids,expected[None].expand(len(ids),-1)):
            raise ValueError('Fixed prompt/token layout contract changed')
        if not torch.equal(batch['image_grid_thw'],ids.new_tensor([[1,14,14]]).expand(len(ids),-1)):
            raise ValueError('Only one 224x224 image per query is supported')
        patches = self.frontend.patches(batch['pixel_values'],len(ids))
        vision = self.vision(patches)
        image_features = self.frontend.merge(vision)
        embeddings = self.frontend.embed(ids)
        image_mask = ids.eq(self.metadata['image_token_id']).unsqueeze(-1).expand_as(embeddings)
        if image_features.numel() != int(image_mask.sum()):
            raise ValueError('Image token count mismatch')
        embeddings = embeddings.masked_scatter(image_mask,image_features.to(embeddings.dtype))
        return self.readout(self.language(embeddings))

    def audit(self):
        forbidden = [name for name,module in self.named_modules()
                     if any(term in type(module).__name__.lower() for term in ('attention','transformer','qwen3vlmodel'))]
        if forbidden:
            raise RuntimeError(f'Forbidden large-model modules: {forbidden}')
        return {'architecture':'t07_standalone_six_capture_v1', 'native_transformer_modules':0,
                'attention_modules':0,'capture_count':6,'top_k':2,
                'frozen_parameters':sum(p.numel() for p in self.parameters() if not p.requires_grad),
                'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'alpha_bounds':list(self.vision.alpha_bounds),
                'input_preprocessing':self.metadata.get('input_preprocessing','center_crop'),
                'ccd_readout_modes':{m:getattr(self,m).optics.readout_mode for m in ('vision','language')},
                'alpha':{m:[float(alpha_value(getattr(getattr(self,m),f'block{i}_optical_fusion_logit'),getattr(self,m).alpha_bounds)) for i in (1,2)] for m in ('vision','language')},
                'ccd_postprocessing':'mean -> clip12 -> log1p -> avgpool224 -> rowLN -> ReLU -> Linear192'}
