"""Extract CLIP features for MNIST and save to npy files.

Usage:
    python extract_clip_features.py                          # default RN50
    python extract_clip_features.py --model ViT-B/32         # ViT
    python extract_clip_features.py --model RN101 --batch-size 512
"""

import argparse
import os

import numpy as np
import torch
import clip
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import resolve_path


def build_transform(input_size, preprocess_fn):
    """Build MNIST-to-RGB transform with CLIP preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),
            preprocess_fn,
        ]
    )


def load_clip_model(model_name="RN50", device="cuda"):
    """Load a CLIP model (OpenAI version), returns model, preprocess, visual_dim."""
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    visual_dim = model.visual.output_dim
    print(f"Loaded CLIP {model_name}, visual dim = {visual_dim}")
    return model, preprocess, visual_dim


@torch.no_grad()
def extract_features(loader, model, device):
    """Extract CLIP image features and labels from a DataLoader."""
    all_features = []
    all_labels = []
    for images, targets in loader:
        images = images.to(device)
        features = model.encode_image(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(targets.numpy())
    return (
        np.concatenate(all_features, axis=0).astype(np.float32),
        np.concatenate(all_labels, axis=0).astype(np.int64),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP RN50 features for MNIST"
    )
    parser.add_argument(
        "--model",
        default="RN50",
        choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Batch size for extraction"
    )
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. cuda or cpu"
    )
    parser.add_argument(
        "--save-dir",
        default="./data/mnist_clip_features",
        help="Output directory for .npy files",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root directory for MNIST downloads",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load CLIP model
    model, preprocess_fn, visual_dim = load_clip_model(args.model, device)

    # Build MNIST datasets with RGB transform
    transform = build_transform(input_size=224, preprocess_fn=preprocess_fn)
    train_set = datasets.MNIST(
        root=resolve_path(args.data_root),
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=resolve_path(args.data_root),
        train=False,
        download=True,
        transform=transform,
    )
    print(
        f"MNIST: train={len(train_set)}, test={len(test_set)}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory="cuda" in str(device),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory="cuda" in str(device),
    )

    # Extract features
    print("Extracting train features...")
    train_feat, train_lbl = extract_features(train_loader, model, device)
    print("Extracting test features...")
    test_feat, test_lbl = extract_features(test_loader, model, device)

    # Save
    save_dir = resolve_path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "train_clip_features.npy"), train_feat)
    np.save(os.path.join(save_dir, "train_labels.npy"), train_lbl)
    np.save(os.path.join(save_dir, "test_clip_features.npy"), test_feat)
    np.save(os.path.join(save_dir, "test_labels.npy"), test_lbl)

    print(f"Saved to {save_dir}/")
    print(f"  train features: {train_feat.shape}, labels: {train_lbl.shape}")
    print(f"  test features:  {test_feat.shape}, labels: {test_lbl.shape}")


if __name__ == "__main__":
    main()
