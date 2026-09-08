"""Fix the short-language top-strip illumination without changing physical geometry."""
import torch
from torch.nn import functional as F
from .embedding_model import EmbeddingOnlyEditor
from .shared_readout import create_shared_readout, head_signature
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import OpticalDetectorTopKRouter
from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.modeling import restore_block_major


def full_aperture_language_amplitude(fields):
    """Expand valid token rows across the existing square; no extra learned router."""
    if fields.ndim != 3 or fields.shape[-1] != fields.shape[-2]:
        raise ValueError('Expected square amplitude fields')
    size = fields.shape[-1]
    active = fields.detach().abs().sum(-1).gt(0)
    lengths = (active * torch.arange(1, size + 1, device=fields.device)[None]).amax(-1)
    if bool((lengths == 0).any()):
        raise ValueError('Empty optical Router input')
    expanded = torch.cat([F.interpolate(field[None, None, :int(n)], size=(size, size),
                                       mode='bilinear', align_corners=False)[:, 0]
                          for field, n in zip(fields, lengths)])
    old_rms = fields.square().mean((-2, -1), keepdim=True).sqrt()
    new_rms = expanded.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
    return expanded * (old_rms / new_rms)


class FullApertureLanguageRouter(OpticalDetectorTopKRouter):
    implementation_name = 'phase_detector_top2_full_aperture_language_v1'

    def forward(self, input_fields):
        # This is the actual amplitude sent to Router SLM, not CCD postprocessing.
        return super().forward(full_aperture_language_amplitude(input_fields))


class RepairedOpticalEditor(EmbeddingOnlyEditor):
    def __init__(self, settings):
        super().__init__(settings)
        if settings.editor_depth != 2:
            raise ValueError('Shared-readout contract requires exactly two groups')
        if self.router_backend == 'optical':
            core = self.language_core.optical_branch.core
            core.router = FullApertureLanguageRouter(core.geometry, self.compact_optical_settings)
        for name in ('position_readout', 'language_pool', 'post_film', 'coordinate_projection', 'editor', 'decoder', 'task_head'):
            delattr(self, name)
        self.shared_readout = create_shared_readout(settings)
        self.readout_signature = head_signature(self.shared_readout)
        self.checkpoint_architecture += '_routerfill_sharedhead_v2'
        self.assert_contract()

    def summarize_language(self, latent_groups):
        return self.shared_readout.summarize(latent_groups)

    def forward(self, source_images, prompt_hidden):
        groups = [g.to(source_images.device, dtype=torch.float32) for g in prompt_hidden]
        condition = self._language_condition(groups)
        visual = self.vision_stem(source_images)
        visual = visual + torch.sigmoid(self.prompt_vision_gate) * torch.tanh(self.prompt_to_vision(condition))[:, None]
        _, latent = self.vision_core.forward_groups([row for row in visual], causal=False,
                                                   spatial_shapes=[(1, 14, 14)] * len(visual))
        result = self.shared_readout(restore_block_major(latent[:, :196]), condition)
        result.update(ccd_operating_loss=self.ccd_operating_loss(), router_balance_loss=self.router_balance_loss())
        return result

    def architecture_report(self):
        report = super().architecture_report()
        report.update(shared_readout=self.readout_signature,
                      language_router_input='valid token rows stretched into existing central 224 square; power preserved',
                      editor_depth=2)
        return report
