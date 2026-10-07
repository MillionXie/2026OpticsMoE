"""One supported training workflow: high-alpha, cross-product retrieval.

This is a fresh-optimizer continuation, not exact Adam/RNG resume. Frozen Qwen
input weights stay frozen. No teacher, external repository, or hidden branch.
"""
import random
import math
import torch
from torch.nn import functional as F
from .io import inputs, picture, write_json
from .runtime import autocast, encode, evaluate
from .objectives import (
    CategoryProxies, make_groups, sampled_indices, group_kind, optical_heads,
    optical_classification_loss, augment, product_bank, gallery_loss,
    readout_polish, supcon, regularization, phase_change,
)


def snapshot(model, auxiliary, epoch, variant):
    return {
        'metadata': model.metadata,
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'selection_variant': variant,
        'auxiliary_training_head': auxiliary.state_dict(),
        'auxiliary_head_not_used_at_inference': True,
    }


def train(model, processor, train_samples, test_samples, device, config, output, batch_size):
    if model.metadata.get('fusion_alpha_min', 0) <= .4:
        raise ValueError('Training requires a previously trained alpha>0.4 checkpoint')
    for key in ('fusion_alpha_min', 'fusion_alpha_max'):
        if config[key] != model.metadata[key]:
            raise ValueError(f'{key} differs from checkpoint; do not silently change the model')
    cfg = config['adapt']
    model.metadata['optical_training_noise'] = config['optical_training_noise']
    for modality in (model.vision, model.language):
        modality.optics.noise_config.update(config['optical_training_noise'])
        modality.optics.router.noise_config.update(config['optical_training_noise'])
    groups = make_groups(train_samples)
    labels = torch.tensor([s.category_id for s in train_samples], device=device)
    auxiliary = CategoryProxies(len(groups)).to(device)
    auxiliary.optical = optical_heads(len(groups)).to(device)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    initial = {name: p.detach().cpu().clone() for name, p in parameters}
    optimizer_groups = []
    for name, p in parameters:
        kind = group_kind(name)
        rate = cfg[kind + '_lr']
        optimizer_groups.append(dict(params=[p], lr=rate, base_lr=rate, kind=kind))
    optimizer_groups.append(dict(params=list(auxiliary.parameters()), lr=cfg['auxiliary_lr'],
                                 base_lr=cfg['auxiliary_lr'], kind='auxiliary'))
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0)
    ema = {name: p.detach().clone() for name, p in parameters}
    base = evaluate(model, processor, train_samples, test_samples, device, batch_size)
    best = (base['hit_at_1'], base['map_at_10'])
    torch.save(snapshot(model, auxiliary, 0, 'initial'), output / 'best.pt')
    features = F.normalize(encode(model, processor, train_samples, device, batch_size).float(), dim=-1).to(device)
    with torch.no_grad():
        auxiliary.weight.copy_(torch.stack([
            F.normalize(features[labels == c].mean(0), dim=0) for c in range(len(groups))
        ]))
    del features
    history = [dict(epoch=0, test=base, variant='initial')]
    write_json(output / 'history.json', history)
    for epoch in range(1, cfg['epochs'] + 1):
        bank, bank_labels, own = product_bank(
            encode(model, processor, train_samples, device, batch_size).to(device), train_samples)
        model.train()
        auxiliary.train()
        rng = random.Random(42 + epoch)
        progress = (epoch - 1) / max(1, cfg['epochs'] - 1)
        scale = min(1., epoch / 2) * (.1 + .45 * (1 + math.cos(math.pi * progress)))
        polish = readout_polish(epoch, cfg['epochs'], cfg)
        for group in optimizer.param_groups:
            frozen = polish and group['kind'] not in ('readout', 'auxiliary')
            group['lr'] = 0. if frozen else group['base_lr'] * scale
        totals = dict(loss=0., gallery_nll=0., gallery_margin=0., train_gallery_hit1=0.)
        counts = {m: torch.zeros(4, device=device) for m in ('vision', 'language')}
        for step in range(cfg['steps']):
            chosen = sampled_indices(groups, cfg['classes_per_batch'], cfg['products_per_class'], rng)
            clean = step % cfg['clean_every'] != 0
            for modality in (model.vision, model.language):
                modality.optics.train(not clean)
            images = [augment(picture(train_samples[i].image_path), rng, config['augmentation']) for i in chosen]
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                z = model(inputs(processor, images, device))
                ce = F.cross_entropy(auxiliary(z), labels[chosen], label_smoothing=.05)
                nll, margin, hit = gallery_loss(z, labels[chosen], own[chosen], bank, bank_labels)
                optical = optical_classification_loss(model, auxiliary.optical, labels[chosen])
                loss = (cfg['proxy_ce_weight'] * ce + cfg['supcon_weight'] * supcon(z, labels[chosen])
                        + cfg['gallery_nll_weight'] * nll + cfg['gallery_margin_weight'] * margin
                        + cfg['optical_auxiliary_weight'] * optical
                        + config['regularization_weight'] * regularization(model))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if polish:
                for name, p in parameters:
                    if group_kind(name) != 'readout':
                        p.grad = None
            torch.nn.utils.clip_grad_norm_([p for _, p in parameters] + list(auxiliary.parameters()), 1.)
            optimizer.step()
            with torch.no_grad():
                for name, p in parameters:
                    if p.grad is None:
                        ema[name].copy_(p)
                    else:
                        ema[name].mul_(config['ema']).add_(p, alpha=1 - config['ema'])
            for key, value in [('loss', loss), ('gallery_nll', nll), ('gallery_margin', margin), ('train_gallery_hit1', hit)]:
                totals[key] += float(value.detach()) / cfg['steps']
            for modality in counts:
                counts[modality] += getattr(model, modality).optics.router.last['selected_mask'].detach().sum(0)
        audit = model.audit()
        if not all(.4 < a <= .8 for values in audit['alpha'].values() for a in values):
            raise RuntimeError('High-alpha constraint violated')
        row = dict(epoch=epoch, readout_polish=polish, losses=totals, alpha=audit['alpha'],
                   router_selected_fraction={m: (c / (cfg['steps'] * cfg['classes_per_batch'] * cfg['products_per_class'])).cpu().tolist() for m, c in counts.items()})
        torch.save(snapshot(model, auxiliary, epoch, 'live'), output / 'last.pt')
        if epoch % config['test_every'] == 0 or epoch == cfg['epochs']:
            live = {name: p.detach().clone() for name, p in parameters}
            for variant, values in [('ema', ema), ('live', live)]:
                with torch.no_grad():
                    for name, p in parameters:
                        p.copy_(values[name])
                metrics = evaluate(model, processor, train_samples, test_samples, device, batch_size)
                row['test_' + variant] = metrics
                score = (metrics['hit_at_1'], metrics['map_at_10'])
                if score > best:
                    best = score
                    torch.save(snapshot(model, auxiliary, epoch, variant), output / 'best.pt')
            del live
        history.append(row)
        write_json(output / 'history.json', history)
        write_json(output / 'parameter_updates.json', {name: float((p.detach().cpu() - initial[name]).square().mean().sqrt()) for name, p in parameters})
        print(row, flush=True)
    selected = torch.load(output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(selected['state_dict'], strict=True)
    write_json(output / 'selected.json', dict(epoch=selected['epoch'], variant=selected['selection_variant'],
                                             test_selected=True, phase_change=phase_change(model, initial)))
