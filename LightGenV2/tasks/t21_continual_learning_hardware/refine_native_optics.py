"""Independent bounded A continuation: small LR, original teacher on TRAIN only.

No active capture/model sources are modified. Candidate optics cannot be mixed
with parent CCD, and B-D must inherit a confirmed new A sequentially.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .model import build_model
from .refine_native_readout import fixed_data, upstream_state
from .train_native import sha, put, tensor_sha, gpu_gate, evaluate, summary, score, frozen
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_lifelong_moe import load_dataset
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.model import DirectCCDOptics


def TRAIN_objective(student, labels, teacher, cfg):
    ce = F.cross_entropy(student, labels, label_smoothing=cfg['label_smoothing'])
    temperature = cfg['distillation_temperature']
    kd = F.kl_div(F.log_softmax(student / temperature, 1),
                  F.softmax(teacher.detach() / temperature, 1),
                  reduction='batchmean') * temperature ** 2
    return ce + cfg['distillation_weight'] * kd


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config', 'checkpoint', 'train_protocols', 'val', 'test', 'out'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg = json.loads(args.config.read_text())
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    try:
        gpu = gpu_gate(cfg, args.gpu_uuid)
        device = torch.device('cuda:0')
        if sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('Parent A checkpoint SHA mismatch')
        parent = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if parent['architecture'] != 'moe' or parent['stage'] != 1:
            raise ValueError('Only native LightGen A is authorized')
        model = build_model('moe').to(device)
        model.load_state_dict(parent['model'], strict=True)
        model.configure_stage(1)
        immutable = frozen(model)
        gain_sha = tensor_sha(model.display_gains)
        parent_upstream = upstream_state(model)
        identity = json.loads((Path(__file__).parent / 'source_identity.json').read_text())
        reference = identity['models'][0]
        if sha(reference['checkpoint']) != reference['sha256']:
            raise ValueError('Original teacher checkpoint changed')
        original_cfg = json.loads(Path(reference['checkpoint']).with_name('config.json').read_text())
        keys = ('router_side', 'router_pitch', 'output_side', 'x_pitch', 'y_pitch',
                'routing_temperature', 'activation_order')
        teacher = DirectCCDOptics('moe', **{k: original_cfg[k] for k in keys if k in original_cfg}).to(device)
        teacher.load_state_dict(torch.load(reference['checkpoint'], map_location=device,
                                           weights_only=False)['model'], strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        paths = {k: Path(v) for k, v in json.loads(args.train_protocols.read_text()).items()}
        expected = identity['models'][3]['config']['protocol_sha256']['eurosat']
        val = fixed_data(args.val, 'val')
        if sha(paths['eurosat']) != expected or sha(paths['vision']) != val['manifest']['vision_checkpoint_sha256']:
            raise ValueError('Original TRAIN protocol/frozen frontend mismatch')
        train = load_dataset(paths, 'eurosat', 'train', vision_checkpoint=paths['vision'], device=device)
        if len(train) != 15998:
            raise ValueError('Original full TRAIN count changed')
        labels = torch.as_tensor(np.asarray(train.labels), dtype=torch.long)
        put(args.out / 'protocol.json', dict(config=cfg, parent_sha256=args.checkpoint_sha256,
            teacher_sha256=reference['sha256'], teacher_scope='original TRAIN only, no TEST inputs',
            TRAIN_protocol_sha256=expected, VAL_manifest=val['manifest'], TEST_gradients=False,
            TEST_selection=False, config_sha256=sha(args.config), gpu=gpu,
            command=__import__('sys').argv,
            source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            requires_new_CCD_if_upstream_changed=True, chain_scope='A-only continuation candidate'))
        teacher_logits = []
        with torch.no_grad():
            for start in range(0, len(train), cfg['batch_size']):
                indices = np.arange(start, min(len(train), start + cfg['batch_size']))
                teacher_logits.append(teacher(train.get_batch(indices, device))['logits'].cpu())
                if start % 2000 == 0:
                    put(args.out / 'status.json', dict(state='TRAIN_teacher_cache',
                        completed=start + len(indices), total=len(train), updated=time.time()))
        teacher_logits = torch.cat(teacher_logits)
        torch.save(dict(logits=teacher_logits, TRAIN_protocol_sha256=expected,
                        teacher_checkpoint_sha256=reference['sha256']), args.out / 'TRAIN_teacher_logits.pt')
        del teacher
        torch.cuda.empty_cache()
        validation = evaluate(model, {'eurosat': val}, ['eurosat'], device, cfg['batch_size'])
        best, best_epoch, wait = score(validation), 0, 0
        history = [dict(epoch=0, validation=summary(validation), selection_score=best)]
        def save(name, epoch):
            value = dict(parent)
            value.update(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                         parent_checkpoint_sha256=args.checkpoint_sha256,
                         selection='fixed VAL only; small-LR native A with original TRAIN teacher',
                         refinement_epoch=epoch, best_epoch=best_epoch)
            torch.save(value, args.out / name)
        save('best_checkpoint.pt', 0)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optical = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('shared_head.')]
        optimizer = torch.optim.AdamW([dict(params=optical, lr=cfg['optical_lr']),
            dict(params=model.shared_head.parameters(), lr=cfg['head_lr'])], weight_decay=cfg['weight_decay'])
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            begun = time.time()
            multiplier = cfg['minimum_lr_fraction'] + .5 * (1 - cfg['minimum_lr_fraction']) * (
                1 + math.cos(math.pi * (epoch - 1) / max(1, cfg['epochs'] - 1)))
            for group, rate in zip(optimizer.param_groups, (cfg['optical_lr'], cfg['head_lr'])):
                group['lr'] = rate * multiplier
            permutation = np.random.default_rng(cfg['seed'] + epoch).permutation(len(train))
            loss_sum, seen, correct = 0., 0, 0
            for start in range(0, len(train), cfg['batch_size']):
                indices = permutation[start:start + cfg['batch_size']]
                target = labels[indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(train.get_batch(indices, device))['logits']
                loss = TRAIN_objective(logits, target, teacher_logits[indices].to(device), cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite TRAIN optical objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                seen += len(indices)
                correct += int((logits.argmax(1) == target).sum())
                loss_sum += float(loss.detach()) * len(indices)
                if seen % 4000 < len(indices):
                    put(args.out / 'status.json', dict(state='optical_training', epoch=epoch,
                        TRAIN_seen=seen, TRAIN_total=len(train), loss=loss_sum / seen, updated=time.time()))
            validation = evaluate(model, {'eurosat': val}, ['eurosat'], device, cfg['batch_size'])
            value = score(validation)
            if frozen(model) != immutable or tensor_sha(model.display_gains) != gain_sha:
                raise RuntimeError('Inactive expert or fixed display gain changed')
            if value > best + 1e-10:
                best, best_epoch, wait = value, epoch, 0
                save('best_checkpoint.pt', epoch)
            else:
                wait += 1
            save('last_checkpoint.pt', epoch)
            row = dict(epoch=epoch, TRAIN_loss=loss_sum / seen, TRAIN_accuracy=correct / seen,
                       validation=summary(validation), selection_score=value,
                       best_score=best, best_epoch=best_epoch, seconds=time.time() - begun)
            history.append(row)
            put(args.out / 'history.json', history)
            put(args.out / 'status.json', dict(state='optical_training', updated=time.time(), **row))
            print(json.dumps(row), flush=True)
            if epoch >= cfg['minimum_epochs'] and wait >= cfg['patience']:
                break
        selected = torch.load(args.out / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
        model.cpu().load_state_dict(selected['model'], strict=True)
        torch.cuda.empty_cache()
        if frozen(model) != immutable or tensor_sha(model.display_gains) != gain_sha:
            raise RuntimeError('Saved frozen state failed')
        put(args.out / 'status.json', dict(state='strict_CPU_TEST_after_VAL_selection', updated=time.time()))
        test = fixed_data(args.test, 'test')
        scores = evaluate(model, {'eurosat': test}, ['eurosat'], torch.device('cpu'), 4)
        put(args.out / 'selected_test.json', dict(scores=scores, strict_CPU=True,
            TEST_gradients=False, TEST_selection=False, selected_epoch=best_epoch))
        put(args.out / 'frozen_audit.json', dict(passed=True, frozen_parameters=immutable,
            gain_sha256=gain_sha, selected_upstream_SHA256=upstream_state(model),
            parent_upstream_SHA256=parent_upstream))
        report = dict(passed=True, best_epoch=best_epoch, best_is_parent=(best_epoch == 0),
            validation=best, TEST=summary(scores), strict_CPU=True,
            parent_sha256=args.checkpoint_sha256, teacher_sha256=reference['sha256'],
            best_sha256=sha(args.out / 'best_checkpoint.pt'), last_sha256=sha(args.out / 'last_checkpoint.pt'),
            new_CCD_required=(upstream_state(model) != parent_upstream), seconds=time.time() - started)
        put(args.out / 'report.json', report)
        put(args.out / 'status.json', dict(state='complete', **report))
        print(json.dumps(report), flush=True)
    except BaseException:
        put(args.out / 'failure.json', dict(traceback=traceback.format_exc(), updated=time.time()))
        raise


if __name__ == '__main__':
    main()
