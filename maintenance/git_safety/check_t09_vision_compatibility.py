"""Compare pinned old and server-overlay vision defaults on CPU, without data."""
import argparse
import hashlib
import json
import subprocess
import sys
import types


def check(repository, old_ref, new_ref):
    import torch
    torch.set_num_threads(2)
    def blob(ref, name):
        return subprocess.check_output(['git', '-C', repository, 'show', ref+':LightGenV2/tasks/t09_multimodal_matching/'+name])
    prepare = blob(old_ref, 'prepare.py')
    if prepare != blob(new_ref, 'prepare.py'):
        raise RuntimeError('Prepare dependency changed')
    models, identities = [], []
    for i, ref in enumerate((old_ref, new_ref)):
        prefix = 't09_compat_'+str(i)
        pkg = types.ModuleType(prefix); pkg.__path__ = []
        sys.modules[prefix] = pkg
        prep = types.ModuleType(prefix+'.prepare'); prep.__package__ = prefix
        sys.modules[prep.__name__] = prep
        exec(compile(prepare, prep.__name__, 'exec'), prep.__dict__)
        content = blob(ref, 'vision.py')
        mod = types.ModuleType(prefix+'.vision'); mod.__package__ = prefix
        exec(compile(content, mod.__name__, 'exec'), mod.__dict__)
        models.append(mod.VisionEncoder)
        identities.append(hashlib.sha256(content).hexdigest())
    count = 0
    for seed in (17, 29):
        torch.manual_seed(seed); old = models[0]()
        torch.manual_seed(seed); new = models[1]()
        state = old.state_dict()
        if list(state) != list(new.state_dict()) or any(not torch.equal(v, new.state_dict()[k]) for k, v in state.items()):
            raise RuntimeError('Default initialization/state keys changed')
        new.load_state_dict(state, strict=True)
        for training in (False, True):
            old.train(training); new.train(training)
            for size in (16, 32, 64):
                inputs = torch.randint(0, 256, (2, size, size, 3), dtype=torch.uint8)
                a, b = old(inputs), new(inputs)
                if any(not torch.equal(x, y) for x, y in zip(a, b)):
                    raise RuntimeError('Default output not bit-identical')
                if any(not torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()):
                    raise RuntimeError('BatchNorm state changed')
                count += 1
    narrow = models[1](widths=(8, 16, 32)).eval()
    narrow.load_state_dict(narrow.state_dict(), strict=True)
    feature, logits = narrow(torch.zeros(2, 32, 32, 3, dtype=torch.uint8))
    assert feature.shape == (2, 128) and logits.shape == (2, 24)
    return dict(old_vision_sha256=identities[0], overlay_vision_sha256=identities[1],
                exact_default_output_cases=count, default_initialization_and_buffers_identical=True,
                strict_default_state_load=True, declared_narrow_widths_shape_pass=True,
                device='cpu', data_read=False, checkpoint_evaluated=False,
                t16_runtime_metrics_revalidated=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', required=True)
    p.add_argument('--old-ref', required=True)
    p.add_argument('--new-ref', required=True)
    a = p.parse_args()
    print(json.dumps(check(a.repository, a.old_ref, a.new_ref), indent=2))
