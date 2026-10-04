"""Nine-class pathology optics used by the immutable-hardware comparison.

Both architectures have two phase planes, the same propagation distance and
the same OEO/readout.  A D2NN illuminates two dense full-aperture phase planes.
The MoE uses a routed bank of first-plane experts followed by one shared dense
phase plane.  Once task A has been learned, the shared phase plane and every
old expert remain immutable; later stages add four new experts.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .optics import AngularSpectrumPropagator


def normalize_power(value, power=1.0):
    norm = value.square().sum((-2, -1), keepdim=True).clamp_min(1e-20)
    return value * (power / norm).sqrt()


def encode_rgb(images):
    """NHWC uint8 RGB -> 224x224 amplitude [R,G;B,0], at unit power."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("expected NHWC uint8 RGB images")
    rgb = F.interpolate(images.permute(0, 3, 1, 2).float() / 255.0,
                        (112, 112), mode="bicubic", align_corners=False,
                        antialias=True).clamp(0, 1)
    amplitude = torch.cat((torch.cat((rgb[:, 0], rgb[:, 1]), -1),
                           torch.cat((rgb[:, 2], torch.zeros_like(rgb[:, 2])), -1)), -2)
    return normalize_power(amplitude)


class CRC9Optics(nn.Module):
    """Two-layer MoE or full-aperture D2NN with identical OEO and MLP readout."""

    def __init__(self, architecture="moe", seed=17, phase_dropout=0.05,
                 num_classes=9, oeo="centered_leaky_softsign"):
        super().__init__()
        if architecture not in {"moe", "d2nn"}:
            raise ValueError(architecture)
        if num_classes != 9:
            raise ValueError("CRC protocol requires exactly nine classes")
        if oeo != "centered_leaky_softsign":
            raise ValueError(oeo)
        self.architecture = architecture
        self.num_classes = num_classes
        self.oeo_name = oeo
        self.expert_size, self.gap, self.border = 224, 30, 20
        self.height = 4 * self.expert_size + 3 * self.gap + 2 * self.border
        self.width = self.height
        self.active_height = self.height - 2 * self.border
        self.active_width = self.width - 2 * self.border
        order = [(0, 0), (0, 3), (3, 0), (3, 3),
                 (0, 1), (0, 2), (3, 1), (3, 2),
                 (1, 0), (1, 3), (2, 0), (2, 3),
                 (1, 1), (1, 2), (2, 1), (2, 2)]
        self.slots = [(self.border + r * (self.expert_size + self.gap),
                       self.border + c * (self.expert_size + self.gap)) for r, c in order]
        self.router_centers = [(y + self.expert_size // 2, x + self.expert_size // 2)
                               for y, x in self.slots]
        self.phase_dropout = float(phase_dropout)
        # The electronic tail is deliberately small and is identical in both arms.
        self.readout = nn.Sequential(nn.LayerNorm(16 * 16), nn.Linear(16 * 16, 64),
                                     nn.GELU(), nn.Linear(64, num_classes))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 101)
            if architecture == "moe":
                self.first_phase = nn.ParameterList([
                    nn.Parameter(torch.randn(self.expert_size, self.expert_size) * .02)
                    for _ in range(16)])
                torch.manual_seed(seed + 103)
                self.router_phase = nn.Parameter(torch.randn(self.expert_size, self.expert_size) * .02)
            else:
                self.first_phase = nn.Parameter(torch.randn(self.active_height, self.active_width) * .02)
                self.register_parameter("router_phase", None)
            torch.manual_seed(seed + 102)
            self.global_phase = nn.Parameter(torch.randn(self.active_height, self.active_width) * .02)
        self.register_buffer("active_count", torch.tensor(4))
        self.propagator = AngularSpectrumPropagator(
            wavelength_m=5.32e-7, pixel_size_m=1.7e-5,
            grid_size=(self.height, self.width), distance_m=.1)

    @staticmethod
    def transmission(raw):
        return torch.exp(2j * torch.pi * torch.sigmoid(raw))

    @staticmethod
    def detect(intensity, centers, side=60):
        half = side // 2
        return torch.stack([intensity[:, y-half:y+half, x-half:x+half].sum((-2, -1))
                            for y, x in centers], 1)

    def configure_moe_stage(self, task_index, warmup=False):
        """Freeze old optical memory; add only the next four expert phases.

        The shared second phase is learned on A and then treated as fabricated.
        The optical router is the reconfigurable controller and the common MLP
        remains adaptable at every stage.
        """
        if self.architecture != "moe" or task_index not in range(4):
            raise ValueError(task_index)
        self.active_count.fill_(4 * (task_index + 1))
        start = 4 * task_index
        for i, parameter in enumerate(self.first_phase):
            parameter.requires_grad_(start <= i < start + 4)
        self.router_phase.requires_grad_(not warmup)
        self.global_phase.requires_grad_(task_index == 0 and not warmup)
        self.readout.requires_grad_(True)

    def configure_d2nn_source(self):
        if self.architecture != "d2nn":
            raise ValueError("not a D2NN")
        self.first_phase.requires_grad_(True)
        self.global_phase.requires_grad_(True)
        self.readout.requires_grad_(True)

    def configure_d2nn_head_adaptation(self):
        if self.architecture != "d2nn":
            raise ValueError("not a D2NN")
        self.first_phase.requires_grad_(False)
        self.global_phase.requires_grad_(False)
        self.readout.requires_grad_(True)

    def phase_mask(self, raw):
        value = self.transmission(raw)
        if self.training and self.phase_dropout:
            block = 8
            height, width = raw.shape[-2:]
            mask = torch.rand((1, 1, (height + block - 1) // block,
                               (width + block - 1) // block), device=raw.device) < self.phase_dropout
            mask = mask.repeat_interleave(block, -2).repeat_interleave(block, -1)[0, 0, :height, :width]
            value = torch.where(mask, torch.ones_like(value), value)
        return value

    def oeo(self, field):
        """Intensity -> spatial LN -> LeakyReLU -> Softsign -> coherent reload.

        The signed response is represented by amplitude magnitude and a 0/pi
        phase bit.  There are no trainable OEO parameters.  Power is normalized
        after every conversion, so the two architectures receive the same OEO.
        """
        border = self.border
        intensity = field[:, border:-border, border:-border].abs().square()
        intensity = intensity / intensity.mean((-2, -1), keepdim=True).clamp_min(1e-20)
        centered = F.layer_norm(intensity, intensity.shape[-2:], eps=1e-5)
        response = F.softsign(F.leaky_relu(centered, negative_slope=.1))
        response = normalize_power(response)
        return F.pad(response, (border,) * 4).to(torch.complex64)

    def route(self, amplitude, warmup=False):
        count = int(self.active_count)
        allowed = torch.arange(16, device=amplitude.device) < count
        if warmup:
            allowed &= torch.arange(16, device=amplitude.device) >= count - 4
            return allowed.float().expand(len(amplitude), -1) / allowed.sum(), None
        delta = self.height - self.expert_size
        routed = F.pad(amplitude.to(torch.complex64) * self.transmission(self.router_phase),
                       (delta // 2, delta - delta // 2, delta // 2, delta - delta // 2))
        capture = self.detect(self.propagator(routed).abs().square(), self.router_centers)
        weights = (capture + 1e-12) * allowed
        return weights / weights.sum(1, keepdim=True), capture.sum(1)

    def optical_features(self, images, warmup=False):
        amplitude = encode_rgb(images)
        if self.architecture == "d2nn":
            expanded = F.interpolate(amplitude[:, None], (self.active_height, self.active_width),
                                     mode="bilinear", align_corners=False)[:, 0]
            expanded = normalize_power(expanded)
            field = F.pad(expanded * self.phase_mask(self.first_phase), (self.border,) * 4)
            routes, router_capture = None, None
        else:
            routes, router_capture = self.route(amplitude, warmup=warmup)
            field = torch.zeros((len(amplitude), self.height, self.width),
                                device=amplitude.device, dtype=torch.complex64)
            for i in range(int(self.active_count)):
                y, x = self.slots[i]
                field[:, y:y+self.expert_size, x:x+self.expert_size] = (
                    amplitude * routes[:, i, None, None].sqrt() * self.phase_mask(self.first_phase[i]))
        # OEO is applied after both phase-modulation/propagation layers.
        field = self.oeo(self.propagator(field))
        global_mask = F.pad(self.phase_mask(self.global_phase), (self.border,) * 4, value=1)
        field = self.oeo(self.propagator(field * global_mask))
        intensity = field[:, self.border:-self.border, self.border:-self.border].abs().square()
        features = F.adaptive_avg_pool2d(intensity[:, None], (16, 16))[:, 0].flatten(1)
        features = torch.log1p(features / features.mean(1, keepdim=True).clamp_min(1e-20))
        return features, routes, router_capture, intensity.sum((-2, -1))

    def forward(self, images, warmup=False):
        features, routes, router_capture, capture = self.optical_features(images, warmup=warmup)
        logits = self.readout(features)
        return {"logits": logits, "probabilities": logits.softmax(1),
                "ccd_features": features, "route_power": routes,
                "router_capture": router_capture, "capture": capture}
