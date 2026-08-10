"""Numerical validation of warp_cached_heatmaps coordinate math."""
import torch

from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import (
    warp_cached_heatmaps,
)
from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import (
    FLIP_PERMUTATION,
)


def gaussian_peak(heatmap_size, px, py, sigma=2.0):
    ys, xs = torch.meshgrid(
        torch.arange(heatmap_size, dtype=torch.float32),
        torch.arange(heatmap_size, dtype=torch.float32), indexing="ij",
    )
    return torch.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * sigma ** 2))


def argmax(h):
    idx = int(h.reshape(-1).argmax())
    return torch.tensor([idx % h.shape[-1], idx // h.shape[-1]], dtype=torch.float32)


def check(name, got, expect, tol=0.8):
    ok = float(torch.norm(got - expect)) < tol
    print(f"{'OK ' if ok else 'FAIL'} {name}: got={got.tolist()} expect~{expect.tolist()}")
    return ok


results = []
H = W = 56

# Test 1: identity
cached = gaussian_peak(H, 28.0, 14.0).unsqueeze(0).unsqueeze(0)
out = warp_cached_heatmaps(
    cached,
    torch.tensor([[100.0, 100.0, 380.0, 380.0]]),
    torch.tensor([[100.0, 100.0, 380.0, 380.0]]),
    torch.tensor([False]), 224, H,
)
results.append(check("identity", argmax(out[0, 0]), torch.tensor([28.0, 14.0]), tol=0.4))

# Test 2: scale + translate
out = warp_cached_heatmaps(
    cached,
    torch.tensor([[100.0, 100.0, 380.0, 380.0]]),
    torch.tensor([[120.0, 90.0, 360.0, 330.0]]),
    torch.tensor([False]), 224, H,
)
results.append(check("scale+translate", argmax(out[0, 0]), torch.tensor([28.0, 18.667])))

# Test 3: flip identity, channel permutation
cached3 = torch.zeros(14, H, W)
cached3[0] = gaussian_peak(H, 28.0, 14.0)
out = warp_cached_heatmaps(
    cached3.unsqueeze(0),
    torch.tensor([[100.0, 100.0, 380.0, 380.0]]),
    torch.tensor([[100.0, 100.0, 380.0, 380.0]]),
    torch.tensor([True]), 224, H,
)
results.append(check("flip: src channel0 empty", argmax(out[0, 0]), torch.tensor([-100.0, -100.0]), tol=1e9))
ok = FLIP_PERMUTATION[5] == 0 and torch.allclose(argmax(out[0, 5]), torch.tensor([27.75, 14.0]), atol=0.6)
print(f"{'OK ' if ok else 'FAIL'} flip: joint0->joint5 peak={argmax(out[0,5]).tolist()} expect~[27.75,14]")
results.append(ok)

# Test 4: flip + scale/translate
out = warp_cached_heatmaps(
    cached3.unsqueeze(0),
    torch.tensor([[100.0, 100.0, 380.0, 380.0]]),
    torch.tensor([[120.0, 90.0, 360.0, 330.0]]),
    torch.tensor([True]), 224, H,
)
results.append(check("flip+scale", argmax(out[0, 5]), torch.tensor([27.75, 18.667])))

# Test 5: mixed batch (14 joints)
batch = torch.zeros(2, 14, H, W)
batch[0, 3] = gaussian_peak(H, 20.0, 10.0)
batch[1, 3] = gaussian_peak(H, 30.0, 40.0)  # joint 3 (left_hip) -> permuted to joint 2 (right_hip) on flip
out = warp_cached_heatmaps(
    batch,
    torch.tensor([[100.0, 100.0, 380.0, 380.0], [50.0, 60.0, 330.0, 340.0]]),
    torch.tensor([[100.0, 100.0, 380.0, 380.0], [50.0, 60.0, 330.0, 340.0]]),
    torch.tensor([False, True]), 224, H,
)
ok = torch.allclose(argmax(out[0, 3]), torch.tensor([20.0, 10.0]), atol=0.5) and \
     torch.allclose(argmax(out[1, 2]), torch.tensor([25.75, 40.0]), atol=0.6)
print(f"{'OK ' if ok else 'FAIL'} mixed batch: unflipped={argmax(out[0,3]).tolist()} flipped_perm={argmax(out[1,2]).tolist()}")
results.append(ok)

print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'}")
