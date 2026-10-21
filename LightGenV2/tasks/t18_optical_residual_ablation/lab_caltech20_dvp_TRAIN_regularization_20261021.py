"""TRAIN-only CCD perturbations/dropout, reusing the published paired fit.

Hooks implement stochastic training transforms, not inference modules. Saved
PTs contain only the original model state; all assessments use eval mode and
final best/last are reloaded as the original, hook-free MeasuredPrefix class.
"""
import argparse
import json
from pathlib import Path
import traceback

PAIR_SHA = 'db5b2da2093ec7ef396346b59ce0d09aa88a0cac5466366db02a7e9fc12f3020'
PARENT_SHA = '76109ba9e4b4003213897acd4d95993bdff8087dfb4f20d19b148301b0352ff6'


class TRAINTransforms:
    """Only current recorded intensity and original Linear inputs are perturbed."""
    def __init__(self, model, stage, cfg):
        import torch
        from torch.nn import functional as F
        self.counts = dict(noise_TRAIN_batches=0, dropout_TRAIN_batches=0,
                           noise_TRAIN_samples=0, eval_calls=0)
        self.stage = stage
        self.cfg = cfg

        def noise(module, inputs):
            if not module.training:
                self.counts['eval_calls'] += 1
                return
            value = module.measured[stage].float()
            # Synthetic robustness assumptions, NOT a fitted camera noise model.
            gain = 1 + cfg['CCD_gain_sigma'] * torch.randn(
                (len(value), 1, 1), device=value.device, dtype=value.dtype)
            changed = value * gain + cfg['CCD_additive_sigma_DN'] * torch.randn_like(value)
            module.measured = {**module.measured, stage: changed.clamp(0, 255)}
            self.counts['noise_TRAIN_batches'] += 1
            self.counts['noise_TRAIN_samples'] += len(value)

        def dropout(module, inputs):
            if not module.training:
                return
            self.counts['dropout_TRAIN_batches'] += 1
            return (F.dropout(inputs[0], p=cfg['feature_dropout_p'], training=True),)

        self.handles = [model.register_forward_pre_hook(noise),
                        model.linear.register_forward_pre_hook(dropout)]

    def remove(self):
        for handle in self.handles:
            handle.remove()


def run(a):
    import numpy as np
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    import lab_caltech20_dvp_restored_pair_20261021 as paired
    torch.set_num_threads(2)
    torch.manual_seed(73)
    cfg = m.read(a.profile)
    assert cfg['schema'] == 'DVP_TRAIN_noise_dropout_suffix_v1'
    assert m.sha(paired.__file__) == PAIR_SHA
    assert m.session_hardware_digest(a.sequence) == loop.HARDWARE and not m.LOCK.exists()
    assert not a.out.exists()
    data, dm = m.data_for(a.sequence)
    assert dm['sha256'] == cfg['data_SHA256']
    parent = a.sequence / cfg['parent_relative']
    assert m.sha(parent) == PARENT_SHA == cfg['parent_SHA256']
    prior_report = m.read(parent.parent / 'report.json')
    assert prior_report['selected_correct'] == 278 and prior_report['state'] == 'complete'
    session = a.sequence / 'L4_response_20261020'
    reference = session / 'source_reference.ts'
    old, identities = m.fields_for(session, 4, data, dm, loop.HARDWARE)
    cap = a.sequence / 'TRAIN_restored_same_mask_pair_20261021/L4/capture'
    capture = m.read(cap / 'report.json')
    assert capture['state'] == 'complete' and capture['hardware_SHA256'] == loop.HARDWARE
    assert m.read(cap / 'independent_audit.json')['report_SHA256'] == m.sha(cap / 'report.json')
    assert m.read(cap / 'same_stimulus_audit.json')['matched_samples'] == 2011
    new = {k: v for k, v in old.items() if k != '2'}
    new['2'] = {}
    for role, n in (('train', 1600), ('test', 411)):
        path = cap / role / 'ccd.npy'
        assert m.sha(path) == capture['arrays'][role]['SHA256']
        array = np.load(path, mmap_mode='r')
        assert array.shape == (n, 478, 478) and array.dtype == np.uint8
        new['2'][role] = array
    model, _ = m.model_for(parent, PARENT_SHA, 4, reference)
    original = m.state_cpu(model)
    # Demonstrate transforms are exact identity during original-model eval.
    ix = np.arange(8)
    with torch.no_grad():
        before = model(loop.put_fields(model, data, new, 'test', ix, 'cpu'))[0]
    transforms = TRAINTransforms(model, '2', cfg)
    with torch.no_grad():
        after = model(loop.put_fields(model, data, new, 'test', ix, 'cpu'))[0]
    assert torch.equal(before, after) and transforms.counts['noise_TRAIN_batches'] == 0
    audit_path = a.out.with_suffix('.admission.json')
    assert not audit_path.exists()
    m.save(audit_path, dict(source_git=m.git_head(), source_SHA256=m.sha(__file__),
        profile_SHA256=m.sha(a.profile), paired_engine_SHA256=PAIR_SHA,
        parent_SHA256=PARENT_SHA, parent_TEST_correct=278, parent_TRAIN_correct=1136,
        eval_augmentation_identity_exact=True, hardware_SHA256=loop.HARDWARE,
        current_CCD_SHA256=capture['arrays'], upstream_SHA256=identities,
        user_revised_gate=288, previous_gate=314, next_layer_requires_strict_CPU=True,
        full_physical=False, SDK_used=False, noise_model_is_calibrated=False))
    fit_cfg = {**m.read(Path(__file__).with_name(cfg['base_profile'])), **cfg}
    fit_cfg['TRAIN_transform_source_SHA256'] = m.sha(__file__)
    spec = {**loop.SPECS[4], 'target': 288}
    assert cfg['stage_gate_correct'] == spec['target']
    try:
        # Existing engine: full1600 TRAIN both views, frozen TRAIN-only teacher,
        # original suffix gradients, clean TEST selection, original CPU replay.
        result = paired.fit(m, loop, data, dm, old, new, model, spec,
                            fit_cfg, a.out, reference)
        assert transforms.counts['noise_TRAIN_batches'] > 0
        assert transforms.counts['noise_TRAIN_batches'] == transforms.counts['dropout_TRAIN_batches']
        transforms.remove()
        loop.validate_state(m, model, original, m.allowed_keys(model, 4))
        m.save(a.out / 'TRAIN_transform_audit.json', dict(
            passed=True, counts=transforms.counts, config=cfg,
            eval_identity_exact=True, saved_PT_hook_free_original_model=True,
            all_input_CCD_arrays_unmodified=True, source_SHA256=m.sha(__file__),
            selected_correct=result, next_layer_started=False, full_physical=False))
    except BaseException as error:
        transforms.remove()
        m.save(a.out.with_suffix('.failure.json'), dict(
            error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


def selftest(profile):
    import os
    import torch
    import lab_caltech20_dvp_layerwise_20261019 as m
    import lab_caltech20_dvp_closedloop_20261021 as loop
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    torch.manual_seed(73)
    cfg = m.read(profile)
    model = m.MeasuredPrefix(4).eval()
    with torch.no_grad():
        for p in model.fusion_logits:
            p.fill_(.8)
        x = torch.rand(2, 224, 224) * .25
        model(x)
        fields = {k: model.last[k]['intensity'].clone() * 173 for k in ('router', '0', '1', '2')}
        model.measured = {k: v.clone() for k, v in fields.items()}
        before = model(x)[0].clone()
    state = m.state_cpu(model)
    transforms = TRAINTransforms(model, '2', cfg)
    model.measured = {k: v.clone() for k, v in fields.items()}
    with torch.no_grad():
        after = model(x)[0]
    assert torch.equal(before, after) and transforms.counts['noise_TRAIN_batches'] == 0
    assert all(torch.equal(model.measured[k], fields[k]) for k in fields)
    model.train().requires_grad_(False)
    allowed = m.allowed_keys(model, 4)
    assert not allowed & loop.mask_keys(model, '2')
    for n, p in model.named_parameters():
        p.requires_grad_(n in allowed)
    model.measured = {k: v.clone() for k, v in fields.items()}
    output = model(x)[0]
    loss = -output[:, 0].clamp_min(1e-12).log().mean()
    loss.backward()
    assert not torch.equal(model.measured['2'], fields['2'])
    assert all(torch.equal(model.measured[k], fields[k]) for k in fields if k != '2')
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in model.named_parameters() if n in allowed)
    assert all(p.grad is None for n, p in model.named_parameters() if n not in allowed)
    assert set(model.state_dict()) == set(state)
    loop.validate_state(m, model, state, allowed)
    transforms.remove()
    model.eval(); model.measured = fields
    with torch.no_grad():
        assert torch.equal(before, model(x)[0])
    print(json.dumps(dict(passed=True, eval_identity_exact=True,
        only_original_suffix_gradients=True, no_new_state_keys=True,
        source_SHA256=m.sha(__file__), profile_SHA256=m.sha(profile),
        counts=transforms.counts, TEST_used=False, SDK_used=False, GPU_used=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('run', 'selftest'), required=True)
    p.add_argument('--sequence', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--profile', type=Path, required=True)
    args = p.parse_args()
    selftest(args.profile) if args.mode == 'selftest' else run(args)
