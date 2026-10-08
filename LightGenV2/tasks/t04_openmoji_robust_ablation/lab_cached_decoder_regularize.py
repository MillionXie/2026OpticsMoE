"""Finite original-decoder adaptation from verified same-upstream CCD caches.

TRAIN-only feature perturbation is a regularizer, not calibrated CCD noise.
No SDK, new inference layers, TEST gradients or VAL selection.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'manifest', 'initial', 'cache', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--changed-edit-weight', type=float, default=0.)
    p.add_argument('--preserved-edit-weight', type=float, default=0.)
    p.add_argument('--train-hard-extra', type=int, default=0,
                   help='Additional TRAIN draws per epoch, based only on initial TRAIN errors')
    p.add_argument('--feature-drop', type=float, default=.02,
                   help='TRAIN-only feature masking probability; clean paired branch retained')
    a = p.parse_args()
    if not 0 <= a.feature_drop <= .25:
        raise ValueError('Finite feature masking range must be 0..0.25')
    if not 0 <= a.train_hard_extra <= 1000:
        raise ValueError('Finite TRAIN oversampling budget must be 0..1000')
    if a.output.exists():
        raise FileExistsError('Preserve existing outputs')
    sys.path.insert(0, str(a.project.resolve() / 'source'))
    import torch
    import torch.nn.functional as F
    from LightGenV2.tasks.t04_openmoji_robust_ablation.lab_editor16_robust_chain import factory
    from LightGenV2.tasks.t04_openmoji_robust_ablation import lab_tune_g2_test as backend
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
    torch.set_num_threads(4)
    torch.manual_seed(1008)
    row = json.loads(a.manifest.read_text())['groups'][0]
    backend.GROUPS = {'g2': (row['weight'], row['sha256'])}
    backend.t = SimpleNamespace(build_model=factory(a.project.resolve(), row))
    _, model = backend.config(a.project.resolve(), torch.device('cpu'))
    protected = backend.protected_sha(model)
    execution = json.loads((a.cache / 'execution.json').read_text())
    if execution['protected_before'] != protected or execution['checkpoint_sha256'] != row['sha256']:
        raise ValueError('Cache upstream identity mismatch')
    data = {s: torch.load(a.cache / (s + '_features.pt'), map_location='cpu', weights_only=False)
            for s in ('train', 'test')}
    if any(len(d['ids']) != 1000 or len(set(d['ids'])) != 1000 for d in data.values()):
        raise ValueError('Not exact original TRAIN/TEST1000')
    if set(data['train']['ids']) & set(data['test']['ids']):
        raise ValueError('TRAIN/TEST overlap')
    def batch(scope, indices, device):
        d = data[scope]
        rows = [d['rows'][i] for i in indices]
        y = {k: torch.cat([r[k] for r in rows]).to(device) if torch.is_tensor(rows[0][k])
             else sum([r[k] for r in rows], []) for k in rows[0]}
        return d['features'][indices].to(device), y
    def evaluate(scope):
        model.eval()
        meter, records = MetricAccumulator(), []
        device = next(model.shared_readout.decoder.parameters()).device
        with torch.no_grad():
            for start in range(0, 1000, 32):
                x, y = batch(scope, list(range(start, min(start + 32, 1000))), device)
                cat, edit = model.shared_readout.decoder(x)
                samples, _, _ = meter.update({'category_logits': cat, 'edit_logits': edit,
                                             'task_logits': y['task_logits']}, y)
                records.extend(samples)
        return meter.compute(), records
    baseline, _ = evaluate('test')
    if abs(baseline['overall']['changed_cell_accuracy'] - .7315) > 1e-8:
        raise ValueError('Original CPU CCD baseline differs')
    payload = torch.load(a.initial, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    if backend.protected_sha(model) != protected:
        raise ValueError('Initial PT upstream mismatch')
    model.requires_grad_(False)
    decoder = model.shared_readout.decoder
    initial_test, _ = evaluate('test')
    best = initial_test['overall']['changed_cell_accuracy']
    selected, best_state = 0, copy.deepcopy(model.state_dict())
    hard_weights = torch.ones(1000)
    if a.train_hard_extra:
        model.eval()
        with torch.no_grad():
            for start in range(0, 1000, 32):
                stop = min(start+32, 1000)
                x, y = batch('train', list(range(start, stop)), 'cpu')
                cat, edit = decoder(x)
                prediction = torch.where(edit.sigmoid() >= .5, cat.argmax(1), y['source_grid'])
                mask = y['edit_grid'].float()
                error = ((prediction != y['target_grid']).float()*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
                hard_weights[start:stop] = 1+2*error
    decoder.requires_grad_(True).to(a.device)
    anchor = {k: v.detach().clone() for k, v in decoder.named_parameters()}
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-5, weight_decay=.05)
    def supervised(cat, edit, y):
        target, mask = y['target_grid'].long(), y['edit_grid'].float()
        ce = F.cross_entropy(cat, target, reduction='none')
        changed = (ce * mask).sum((1, 2)) / mask.sum((1, 2)).clamp_min(1)
        preserved = (ce * (1-mask)).sum((1, 2)) / (1-mask).sum((1, 2)).clamp_min(1)
        prob = cat.softmax(1).gather(1, target[:, None]).squeeze(1)
        correct = edit.sigmoid()*prob + (1-edit.sigmoid())*y['source_grid'].eq(target)
        composed = (-correct.clamp_min(1e-7).log()*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        positive_edit = (F.softplus(-edit)*mask).sum((1, 2))/mask.sum((1, 2)).clamp_min(1)
        negative_edit = (F.softplus(edit)*(1-mask)).sum((1, 2))/(1-mask).sum((1, 2)).clamp_min(1)
        return .5*changed.mean()+.5*composed.mean()+.2*preserved.mean()+F.binary_cross_entropy_with_logits(edit, mask, pos_weight=edit.new_tensor(8.))+a.changed_edit_weight*positive_edit.mean()+a.preserved_edit_weight*negative_edit.mean()
    a.output.mkdir(parents=True)
    torch.save(payload, a.output/'best.pt')
    def write(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    write('protocol.json', {'initial_sha256': sha(a.initial), 'cache_sha256': {s: sha(a.cache/(s+'_features.pt')) for s in data},
          'protected_sha256': protected, 'epochs': a.epochs, 'seed': 1008, 'lr': 1e-5, 'weight_decay': .05,
          'gain_range': [.98, 1.02], 'feature_drop': a.feature_drop, 'relative_feature_noise': .01,
          'changed_edit_weight': a.changed_edit_weight,
          'preserved_edit_weight': a.preserved_edit_weight,
          'train_hard_extra': a.train_hard_extra,
          'hard_sampling': 'fixed initial TRAIN changed-cell error weights 1+2*error; all original TRAIN once plus extra draws; no TEST mining',
          'paired_supervision': [.5, .5], 'consistency': .05, 'test_gradient': False,
          'selection': 'TEST every5 highest development; no independent generalization claim', 'architecture_unchanged': True})
    history = []
    for epoch in range(1, a.epochs + 1):
        decoder.train()
        losses = []
        order = torch.randperm(1000).tolist()
        if a.train_hard_extra:
            order += torch.multinomial(hard_weights, a.train_hard_extra, replacement=True).tolist()
            order = [order[i] for i in torch.randperm(len(order)).tolist()]
        for start in range(0, len(order), 32):
            x, y = batch('train', order[start:start+32], a.device)
            scale = x.detach().square().mean().sqrt().clamp_min(1e-8)
            noisy = x*(.98+.04*torch.rand_like(x)) + .01*scale*torch.randn_like(x)
            noisy = noisy*(torch.rand_like(x) >= a.feature_drop)
            cat, edit = decoder(x)
            nc, ne = decoder(noisy)
            consistency = F.kl_div(nc.log_softmax(1), cat.detach().softmax(1), reduction='batchmean')/cat[0, 0].numel()
            consistency = consistency + (ne.sigmoid()-edit.detach().sigmoid()).square().mean()
            anch = sum((v-anchor[k]).square().mean() for k, v in decoder.named_parameters())
            loss = .5*supervised(cat, edit, y)+.5*supervised(nc, ne, y)+.05*consistency+.05*anch
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        score = None
        if epoch % 5 == 0 or epoch == a.epochs:
            metrics, _ = evaluate('test')
            score = metrics['overall']['changed_cell_accuracy']
            if score > best:
                best, selected = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                checkpoint = copy.deepcopy(payload)
                checkpoint['model'] = best_state
                torch.save(checkpoint, a.output/'best.pt')
        history.append({'epoch': epoch, 'loss': sum(losses)/len(losses), 'test': score, 'best': best})
        write('history.json', history)
        write('progress.json', {'status': 'training', **history[-1], 'selected_epoch': selected})
        print(json.dumps(history[-1]), flush=True)
    decoder.cpu()
    last = copy.deepcopy(payload)
    last['model'] = copy.deepcopy(model.state_dict())
    torch.save(last, a.output/'last.pt')
    payload['model'] = best_state
    torch.save(payload, a.output/'best.pt')
    audits = {}
    for label in ('best', 'last'):
        model.load_state_dict(torch.load(a.output/(label+'.pt'), map_location='cpu', weights_only=False)['model'], strict=True)
        if backend.protected_sha(model) != protected:
            raise ValueError('Protected upstream changed')
        metrics, records = evaluate('test')
        audits[label] = metrics
        write(label+'_test_samples.json', records)
    write('report.json', {'status': 'complete', 'initial_test': initial_test, 'baseline': baseline, 'strict_cpu': audits,
          'selected_epoch': selected, 'protected_unchanged': True, 'best_sha256': sha(a.output/'best.pt'),
          'last_sha256': sha(a.output/'last.pt'), 'test_gradient': False, 'architecture_unchanged': True})


if __name__ == '__main__':
    main()
