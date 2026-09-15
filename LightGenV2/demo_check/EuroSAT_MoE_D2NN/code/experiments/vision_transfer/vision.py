"""Vision two-block hybrid with direct mean/max classification readout."""
import torch
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import VisionTwoBlockOpticalReplacement,VisionTwoBlockOpticalCore
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.modeling import _install_router
from experiments.qwen3_vl_embedding_2b_cifar10_four_layer_optical_router_classification.modeling import CIFAR10ClassificationHead
from .layout import configure_detector_windows


class ClassificationVisionCore(VisionTwoBlockOpticalCore):
    def forward_groups(self,groups,*,causal,spatial_shapes=None):
        if causal or spatial_shapes is None:raise RuntimeError('Vision needs noncausal spatial shapes')
        padded,mask,lengths=self._pad_groups(groups)
        z=self.input_norm(self.input_adapter(padded.float()))
        optical1,routing,optical_lengths=self.optical_branch.run_expert_block(z,mask)
        electronic1=self.blocks[0](z,padding_mask=mask,causal=False,spatial_shapes=spatial_shapes)
        fused1=(electronic1+self.block1_optical_fusion*optical1).masked_fill(mask.unsqueeze(-1),0.)
        global_input=self.optical_branch.encode_global_input(fused1,mask,routing)
        electronic2=self.blocks[1](fused1,padding_mask=mask,causal=False,spatial_shapes=spatial_shapes)
        optical2=self.optical_branch.run_global_block(global_input,optical_lengths,mask,fused1.dtype)
        latent=self.output_norm(electronic2+self.block2_optical_fusion*optical2).masked_fill(mask.unsqueeze(-1),0.)
        self.last_latent_groups=[latent[i,:n] for i,n in enumerate(lengths)]
        self.last_routing=routing
        return torch.cat(self.last_latent_groups),latent


class VisionReplacement:
    def __init__(self,vision,loaded):
        self.vision_surrogate=vision
        self.backbone_metadata=loaded.source_metadata
        self.checkpoint_architecture='vision_only_rows_dense_v1'

    def set_phase_dropout_active(self,active):
        self.vision_surrogate.set_phase_dropout_active(active)

    def close(self):
        # No hooks or mutation of the frozen stem; free per-batch caches.
        self.vision_surrogate.core.last_latent_groups=[]


def build_classification_student(loaded,settings):
    settings.vision_hidden_size=int(loaded.model.config.hidden_size)
    settings.vision_depth=0;settings.text_depth=0;settings.text_hidden_size=0
    settings.detector_output_size=2*int(settings.electronic_width)
    if settings.detector_output_size!=384:raise RuntimeError('Reviewed head expects width 192 with mean+max')
    vision=VisionTwoBlockOpticalReplacement(settings.vision_hidden_size,settings)
    vision.core.__class__=ClassificationVisionCore
    # These only reconstructed Qwen hidden tokens for the deleted language path.
    del vision.core.output_adapter
    del vision.core.residual_logit
    configure_detector_windows(settings,vision.core.optical_branch.core.geometry)
    _install_router(vision,settings)
    vision.to(loaded.device).requires_grad_(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(settings.classification_head_seed))
        head=CIFAR10ClassificationHead(settings.detector_output_size,settings.num_classes)
    return VisionReplacement(vision,loaded),head.to(loaded.device)


def classification_logits(model,replacement,head,inputs):
    if set(inputs)!={'pixel_values','image_grid_thw'}:
        raise RuntimeError('Vision-only forward accepts only image tensors')
    grid=inputs['image_grid_thw'];lengths=grid.prod(1).detach().cpu().tolist()
    if not bool((grid[:,0]==1).all()):raise RuntimeError('Only still images are supported')
    with torch.no_grad():hidden=model(inputs['pixel_values'],grid)
    groups=list(hidden.split(lengths))
    core=replacement.vision_surrogate.core
    core.forward_groups(groups,causal=False,spatial_shapes=[tuple(row) for row in grid.detach().cpu().tolist()])
    features=torch.stack([torch.cat((z.float().mean(0),z.float().amax(0))) for z in core.last_latent_groups])
    return head(features),features
