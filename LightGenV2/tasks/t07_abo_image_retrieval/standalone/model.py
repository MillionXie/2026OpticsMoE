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
    def __init__(self, vision, kernel_size=None, mlp_width=384):
        super().__init__()
        self.vision = vision
        if type(mlp_width) is not int or mlp_width not in (384,768):
            raise ValueError('Electronic residual MLP width must be 384 or 768')
        self.mlp_width=mlp_width
        self.kernel_size=(3 if vision else 5) if kernel_size is None else kernel_size
        allowed=(3,5,7,13) if vision else (3,5,7)
        if type(self.kernel_size) is not int or self.kernel_size not in allowed:
            raise ValueError('Electronic residual kernel must be V3/5/7/13 or L3/5/7')
        self.token_norm = nn.LayerNorm(192)
        self.token_depthwise = (nn.Conv2d(192,192,self.kernel_size,groups=192,bias=False) if vision
                                else nn.Conv1d(192,192,self.kernel_size,groups=192,bias=False))
        self.token_pointwise = nn.Linear(192,192)
        self.token_dropout = nn.Dropout(.1)
        self.token_residual_logit = nn.Parameter(torch.zeros(()))
        self.residual_logit = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(192)
        self.mlp = nn.Sequential(nn.Linear(192,mlp_width),nn.GELU(),nn.Dropout(.1),nn.Linear(mlp_width,192),nn.Dropout(.1))

    def forward(self, x):
        n = self.token_norm(x)
        if self.vision:
            # Match original per-image 2D convolution (not folded into a new batch).
            outputs = []
            for row in n:
                grid = row.view(1,7,7,2,2,192).permute(0,5,1,3,2,4).reshape(1,192,14,14)
                pad=self.kernel_size//2
                grid = self.token_depthwise(F.pad(grid,(pad,pad,pad,pad)))
                outputs.append(grid.view(1,192,7,2,7,2).permute(0,2,4,3,5,1).reshape(196,192))
            update = torch.stack(outputs)
        else:
            update = self.token_depthwise(F.pad(n.transpose(1,2),(self.kernel_size-1,0))).transpose(1,2)
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
    def __init__(self, vision, input_rms, alpha_bounds=(.01,.95), noise_config=None, kernel_size=None, mlp_width=384):
        super().__init__()
        self.vision = vision
        self.alpha_bounds = alpha_bounds
        hidden = 1024 if vision else 2048
        self.input_adapter = nn.Linear(hidden,192)
        self.input_norm = nn.LayerNorm(192)
        self.blocks = nn.ModuleList([Residual(vision,kernel_size,mlp_width),Residual(vision,kernel_size,mlp_width)])
        self.output_norm = nn.LayerNorm(192)
        if vision:
            self.output_adapter = nn.Linear(192,1024)
            self.residual_logit = nn.Parameter(torch.zeros(()))
        self.block1_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        self.block2_optical_fusion_logit = nn.Parameter(torch.zeros(()))
        self.optics = OpticalPath(input_rms,noise_config)
        self.remove_optical = False
        self.optical_noise_blocks = frozenset()
        self.last_latent = None
        self.last_optical = None

    def _ablate_optical_output(self, value, block):
        if block not in self.optical_noise_blocks:
            return value
        # Information-destroying, energy-preserving control. ``fuse`` performs
        # per-sample RMS matching immediately afterwards, so unit Gaussian
        # noise has the same post-match branch energy without retaining any
        # image/SKU information from the optical output.
        return torch.randn_like(value.float()).to(value.dtype)

    def forward(self, inputs):
        latent = self.input_norm(self.input_adapter(inputs.float()))
        e1 = self.blocks[0](latent)
        if self.remove_optical:
            f1 = e1
            weights = None
        else:
            o1, weights = self.optics.expert(latent)
            o1 = self._ablate_optical_output(o1, 1)
            f1 = fuse(e1,o1,self.block1_optical_fusion_logit,self.alpha_bounds)
        e2 = self.blocks[1](f1)
        self.last_optical = None if self.remove_optical else self.optics.global_stage(f1,weights)
        if self.last_optical is not None:
            self.last_optical = self._ablate_optical_output(self.last_optical, 2)
        f2 = e2 if self.remove_optical else fuse(e2,self.last_optical,self.block2_optical_fusion_logit,self.alpha_bounds)
        output = self.output_norm(f2)
        self.last_latent = output
        if self.vision:
            return (inputs.float()+torch.sigmoid(self.residual_logit)*self.output_adapter(output)).to(inputs.dtype)
        return output


class RetrievalHead(nn.Module):
    def __init__(self, kind='linear64'):
        super().__init__()
        if kind not in ('linear64','relu128','linear256','spatial2x2_64'):
            raise ValueError('Unknown retrieval head contract')
        self.kind=kind
        self.output_dimension=256 if kind=='linear256' else 64
        self.norm = nn.LayerNorm(384)
        self.projection = (nn.Linear(1152,64) if kind=='spatial2x2_64' else
                           nn.Linear(384,self.output_dimension) if kind in ('linear64','linear256') else
                           nn.Sequential(nn.Linear(384,128),nn.ReLU(),nn.Linear(128,64)))

    def forward(self, latent, image_positions=None):
        pooled = torch.stack([torch.cat((row.mean(0),row.amax(0))) for row in latent])
        features = self.norm(pooled.float())
        if self.kind == 'spatial2x2_64':
            # These are image-token POSITIONS AFTER the entire L optical/electronic
            # path, not raw RGB, Qwen TF features, or an additional bypass branch.
            if (image_positions is None or image_positions.dtype != torch.bool
                    or image_positions.shape != latent.shape[:2]
                    or not (image_positions.sum(1) == 49).all()):
                raise ValueError('Spatial head requires49 ordered image-token positions per sample')
            grid = latent[image_positions].float().reshape(len(latent),7,7,192).permute(0,3,1,2)
            local = F.adaptive_avg_pool2d(grid,(2,2)).permute(0,2,3,1)
            # Fixed per-cell channel normalization; no new affine parameters.
            local = F.layer_norm(local,(192,)).reshape(len(latent),768)
            features = torch.cat((features,local),dim=1)
        return F.normalize(self.projection(features),p=2,dim=-1)


class OpticalRetrieval(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
        bounds=(float(metadata.get('fusion_alpha_min',.01)),float(metadata.get('fusion_alpha_max',.95)))
        if not 0<=bounds[0]<bounds[1]<=1:raise ValueError('Invalid alpha bounds')
        self.frontend = Frontend(metadata['token_count']).to(torch.bfloat16)
        frontend_training=metadata.get('frontend_training','frozen')
        if frontend_training not in ('frozen','merger_fc2','patch'):
            raise ValueError('Unknown compact frontend training contract')
        if frontend_training=='merger_fc2':
            self.frontend.merger_fc2.float().requires_grad_(True)
        if frontend_training=='patch':
            self.frontend.patch.float().requires_grad_(True)
        kernels=metadata.get('electronic_context_kernels',{})
        if set(kernels)-{'vision','language'}:raise ValueError('Unknown electronic kernel modality')
        mlp_width=metadata.get('electronic_mlp_width',384)
        self.vision = Modality(True, metadata['input_rms'],bounds,metadata.get('optical_training_noise'),kernels.get('vision'),mlp_width)
        self.language = Modality(False, metadata['input_rms'],bounds,metadata.get('optical_training_noise'),kernels.get('language'),mlp_width)
        if metadata.get('input_preprocessing','center_crop') not in ('center_crop','contain_white','contain_min_half'):
            raise ValueError('Unknown image preprocessing contract')
        modes = metadata.get('ccd_readout_modes',{})
        if set(modes) - {'vision','language'}:
            raise ValueError('Unknown modality in CCD readout contract')
        for name in ('vision','language'):
            mode = modes.get(name,'prefix_rows')
            if mode not in ('prefix_rows','fullfield_rows'):
                raise ValueError('Unknown CCD readout contract')
            getattr(self,name).optics.readout_mode = mode
            getattr(self,name).optics.configure_phase_dropout(metadata.get('phase_dropout',{}))
        self.readout = RetrievalHead(metadata.get('retrieval_head','linear64'))

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval()
        return self

    def set_remove_optical(self, active):
        self.vision.remove_optical = self.language.remove_optical = bool(active)

    def set_optical_noise_ablation(self, mode='none'):
        """Replace selected optical feature outputs by Gaussian controls.

        ``language_global`` is the literal final optical feature layer in the
        V1,V2,L1,L2 inference order. ``global_each_modality`` replaces V2 and
        L2. ``all_feature_layers`` replaces V1,V2,L1,L2. Routers still execute
        so capture count and Top-2 control flow remain unchanged, but their
        selected feature outputs carry no sample information in the all-layer
        condition.
        """
        mapping = {
            'none': (frozenset(), frozenset()),
            'language_global': (frozenset(), frozenset({2})),
            'global_each_modality': (frozenset({2}), frozenset({2})),
            'all_feature_layers': (frozenset({1,2}), frozenset({1,2})),
        }
        if mode not in mapping:
            raise ValueError('Unknown optical noise ablation: '+str(mode))
        self.vision.optical_noise_blocks, self.language.optical_noise_blocks = mapping[mode]

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
        return self.readout(self.language(embeddings), ids.eq(self.metadata['image_token_id']))

    def audit(self):
        forbidden = [name for name,module in self.named_modules()
                     if any(term in type(module).__name__.lower() for term in ('attention','transformer','qwen3vlmodel'))]
        if forbidden:
            raise RuntimeError(f'Forbidden large-model modules: {forbidden}')
        kernels={m:getattr(self,m).blocks[0].kernel_size for m in ('vision','language')}
        architecture='t07_standalone_six_capture_v1' if kernels=={'vision':3,'language':5} else 't07_standalone_six_capture_electronic_context'
        if self.readout.kind!='linear64':architecture+='_'+self.readout.kind
        if self.vision.blocks[0].mlp_width!=384:architecture+='_mlp'+str(self.vision.blocks[0].mlp_width)
        return {'architecture':architecture, 'native_transformer_modules':0,
                'attention_modules':0,'capture_count':6,'top_k':2,
                'frozen_parameters':sum(p.numel() for p in self.parameters() if not p.requires_grad),
                'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'frontend_training':self.metadata.get('frontend_training','frozen'),
                'frontend_trainable_parameters':sum(p.numel() for p in self.frontend.parameters() if p.requires_grad),
                'alpha_bounds':list(self.vision.alpha_bounds),
                'input_preprocessing':self.metadata.get('input_preprocessing','center_crop'),
                'ccd_readout_modes':{m:getattr(self,m).optics.readout_mode for m in ('vision','language')},
                'training_phase_dropout':self.metadata.get('phase_dropout',{}),
                'electronic_context_kernels':kernels,
                'electronic_mlp_width':self.vision.blocks[0].mlp_width,
                'retrieval_head':self.readout.kind,
                'retrieval_pooling':('Global all-token mean/max plus fixed2x2 average of49 image-token positions AFTER L; single linear64 projection, no attention' if self.readout.kind=='spatial2x2_64' else 'Global all-token mean/max'),
                'descriptor_dimension':self.readout.output_dimension,
                'alpha':{m:[float(alpha_value(getattr(getattr(self,m),f'block{i}_optical_fusion_logit'),getattr(self,m).alpha_bounds)) for i in (1,2)] for m in ('vision','language')},
                'ccd_postprocessing':'mean -> clip12 -> log1p -> adaptive_avg_pool (see ccd_readout_modes) -> rowLN -> ReLU -> Linear192'}
