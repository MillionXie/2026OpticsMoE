"""Train only the final electronic retrieval readout on measured TRAIN CCDs.

Six measured optical stages are frozen into the cached features. Checkpoint
selection uses a fixed 600/200 split of the captured TRAIN images; the 2,400
measured TEST images are evaluated once after selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import ElectronicRetrievalReadout


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def metrics(readout, images, titles, labels) -> dict:
    image_z = readout(images).float()
    title_z = readout(titles).float()
    scores = title_z @ image_z.T
    order = scores.argsort(dim=1, descending=True, stable=True)
    relevant = labels[order].eq(torch.arange(100, device=labels.device)[:, None])
    total = relevant.sum(1)
    if not bool(total.eq(total[0]).all()):
        raise RuntimeError('Each SKU must have the same image count')
    rank = torch.arange(1, len(images) + 1, device=labels.device)[None]
    first = torch.where(relevant, rank, len(images) + 1).amin(1)
    precision = relevant.cumsum(1) / rank
    return {
        'hit_at_1': float(relevant[:, :1].any(1).float().mean()),
        'hit_at_5': float(relevant[:, :5].any(1).float().mean()),
        'hit_at_10': float(relevant[:, :10].any(1).float().mean()),
        'mrr': float(first.float().reciprocal().mean()),
        'map': float(((precision * relevant).sum(1) / total).mean()),
        'gallery_count': len(images),
        'relevant_images_per_query': int(total[0]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--train-cache', type=Path, required=True)
    p.add_argument('--test-cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=0.0002)
    p.add_argument('--anchor-weight', type=float, default=0.1)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--seed', type=int, default=20260926)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    train = torch.load(args.train_cache, map_location='cpu', weights_only=False)
    test = torch.load(args.test_cache, map_location='cpu', weights_only=False)
    source_sha = sha(args.checkpoint)
    for data, name in ((train, 'train'), (test, 'test')):
        if data['split'] != name or data['checkpoint_sha256'] != source_sha:
            raise RuntimeError(f'{name} cache checkpoint/split mismatch')
    labels = train['image_labels']
    by_sku = {label: [] for label in range(100)}
    for index, label in enumerate(labels):
        by_sku[label].append(index)
    if any(len(indices) != 8 for indices in by_sku.values()):
        raise RuntimeError('Expected exactly eight measured TRAIN images per SKU')
    generator = torch.Generator().manual_seed(args.seed)
    fit_idx, val_idx = [], []
    for label in range(100):
        indices = by_sku[label]
        permutation = torch.randperm(8, generator=generator).tolist()
        fit_idx += [indices[i] for i in permutation[:6]]
        val_idx += [indices[i] for i in permutation[6:]]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim = train['image_features'].shape[1]
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    output_dim = source['retrieval_readout']['projection.weight'].shape[0]
    readout = ElectronicRetrievalReadout(dim, output_dim).to(device)
    readout.load_state_dict(source['retrieval_readout'])
    image_all = train['image_features'].to(device)
    title_all = train['title_features'].to(device)
    train_labels = torch.tensor(labels, dtype=torch.long, device=device)
    fit = torch.tensor(fit_idx, dtype=torch.long, device=device)
    val = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_images = test['image_features'].to(device)
    test_titles = test['title_features'].to(device)
    test_labels = torch.tensor(test['image_labels'], dtype=torch.long, device=device)
    with torch.no_grad():
        original_fit = readout(image_all[fit]).detach()
        original_title = readout(title_all).detach()
        baseline = {
            'train_fit': metrics(readout, image_all[fit], title_all, train_labels[fit]),
            'train_validation': metrics(readout, image_all[val], title_all, train_labels[val]),
            'untouched_test': metrics(readout, test_images, test_titles, test_labels),
        }
    if abs(baseline['untouched_test']['hit_at_1'] - 0.79) > 1e-5:
        raise RuntimeError('Measured TEST baseline no longer matches the audited 0.79')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=args.learning_rate,
                                  weight_decay=0.01)
    history = []
    initial_val = baseline['train_validation']
    best_key = (initial_val['hit_at_1'], initial_val['mrr'], initial_val['map'])
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone()
                  for name, value in readout.state_dict().items()}
    for epoch in range(1, args.epochs + 1):
        readout.train()
        optimizer.zero_grad(set_to_none=True)
        image_z = readout(image_all[fit])
        title_z = readout(title_all)
        logits = image_z @ title_z.T / args.temperature
        loss_image_to_title = F.cross_entropy(logits, train_labels[fit])
        title_to_image = logits.T
        positive = train_labels[fit][None, :].eq(
            torch.arange(100, device=device)[:, None])
        loss_title_to_image = (
            torch.logsumexp(title_to_image, dim=1)
            - torch.logsumexp(title_to_image.masked_fill(~positive, -torch.inf), dim=1)
        ).mean()
        anchor = (1 - F.cosine_similarity(image_z, original_fit, dim=-1)).mean()
        anchor += (1 - F.cosine_similarity(title_z, original_title, dim=-1)).mean()
        loss = loss_image_to_title + 0.5 * loss_title_to_image + args.anchor_weight * anchor
        loss.backward()
        optimizer.step()
        readout.eval()
        fit_metrics = metrics(readout, image_all[fit], title_all, train_labels[fit])
        val_metrics = metrics(readout, image_all[val], title_all, train_labels[val])
        row = {'epoch': epoch, 'loss': float(loss.detach()),
               'fit_hit_at_1': fit_metrics['hit_at_1'],
               'val_hit_at_1': val_metrics['hit_at_1'],
               'val_mrr': val_metrics['mrr'], 'val_map': val_metrics['map']}
        history.append(row)
        key = (val_metrics['hit_at_1'], val_metrics['mrr'], val_metrics['map'])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in readout.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'epoch {epoch:03d} loss={row["loss"]:.4f} '
                  f'fit={row["fit_hit_at_1"]:.3f} val={row["val_hit_at_1"]:.3f} '
                  f'best={best_epoch}', flush=True)
    last_state = {name: value.detach().cpu().clone()
                  for name, value in readout.state_dict().items()}
    last_train = metrics(readout, image_all, title_all, train_labels)
    last_test = metrics(readout, test_images, test_titles, test_labels)
    readout.load_state_dict(best_state)
    selected_train = metrics(readout, image_all, title_all, train_labels)
    selected_validation = metrics(readout, image_all[val], title_all, train_labels[val])
    selected_test = metrics(readout, test_images, test_titles, test_labels)
    with (args.output_dir / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for suffix, state, epoch in (('best', best_state, best_epoch),
                                 ('last', last_state, args.epochs)):
        payload = dict(source)
        payload['retrieval_readout'] = state
        metadata = dict(source.get('metadata', {}))
        metadata['physical_readout_finetune'] = {
            'source_checkpoint_sha256': source_sha,
            'epoch': epoch, 'selection': 'measured TRAIN 200-image validation hit_at_1, mrr, map',
            'test_used_for_selection': False,
            'only_trainable_module': 'retrieval_readout',
            'fit_images': 600, 'validation_images': 200, 'untouched_test_images': 2400,
        }
        payload['metadata'] = metadata
        payload['optimizer'] = None
        torch.save(payload, args.output_dir / f'{suffix}_readout_checkpoint.pt')
    report = {
        'schema': 1, 'source_checkpoint_sha256': source_sha,
        'training': {'captured_train_images': 800, 'fit_images': 600,
                     'validation_images': 200, 'test_images': 2400,
                     'only_trainable_module': 'retrieval_readout',
                     'epochs': args.epochs, 'selected_epoch': best_epoch,
                     'learning_rate': args.learning_rate,
                     'anchor_weight': args.anchor_weight,
                     'temperature': args.temperature, 'seed': args.seed},
        'baseline': baseline,
        'selected': {'all_train': selected_train,
                     'train_validation': selected_validation,
                     'untouched_test': selected_test},
        'last': {'all_train': last_train, 'untouched_test': last_test},
        'best_checkpoint_sha256': sha(args.output_dir / 'best_readout_checkpoint.pt'),
        'last_checkpoint_sha256': sha(args.output_dir / 'last_readout_checkpoint.pt'),
    }
    (args.output_dir / 'finetune_report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
