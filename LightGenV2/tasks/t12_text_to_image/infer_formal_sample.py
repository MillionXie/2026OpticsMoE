"""One clean-simulation sample of fixed 5b4f 17M; no CCD/SLM or ground truth input."""
import argparse
from pathlib import Path


def predict(model, reference, lookup, prompt, index):
    import torch
    device = reference.device
    embeddings, mask, _ = lookup.batch([prompt], device)
    noise = torch.randn(reference.shape, device='cpu',
                        generator=torch.Generator().manual_seed(1042 + index)).to(device)
    with torch.inference_mode():
        generated = model(reference, embeddings, mask, noise)[0].float().cpu()
    return generated.add(1).mul(127.5).clamp(0, 255).byte().permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--instruction-cache', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.index < 0:
        parser.error('Use a new output path and nonnegative index')
    # Identity gate runs before deserialization and accepts original 5b4f only.
    from .verify_formal_checkpoint import PIN, digest, load_sealed_cpu
    if digest(args.checkpoint) != PIN:
        parser.error('Not the original fixed 5b4f checkpoint')
    import torch
    from PIL import Image
    from .product_unified_edit_data_v2 import ExpandedUnifiedProductEditDataset
    from .qwen_mini_small import PromptEmbeddingLookup
    torch.set_num_threads(4)
    dataset = ExpandedUnifiedProductEditDataset(
        args.dataset_root, args.split, 256, args.instruction_cache)
    if args.index >= len(dataset):
        parser.error(f'index must be below {len(dataset)}')
    row = dataset[args.index]
    model = load_sealed_cpu(args.checkpoint)
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    pixels = predict(model, row['reference'][None], lookup, row['prompt'], args.index)
    if digest(args.checkpoint) != PIN:
        raise RuntimeError('Checkpoint changed during inference')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output creation also protects against a concurrent writer.
    with args.output.open('xb') as stream:
        Image.fromarray(pixels).save(stream, format='PNG')
    print(args.split, args.index, row['sample_id'], row['prompt'], args.output, sep='\n')


if __name__ == '__main__':
    main()
