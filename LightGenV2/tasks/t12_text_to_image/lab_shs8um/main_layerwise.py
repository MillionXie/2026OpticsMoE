"""Explicit main-bench adapter for the preserved formal T12 layerwise runner.

Default inspection never imports Torch/SDK or creates output. Capture is an
explicit operation, not part of repository cleanup or a deployment certification.
"""
import argparse
import ast
import functools
import hashlib
import json
from pathlib import Path
import sys

FORMAL_SHA = '5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac'
OLD_ARGUMENT = "p.add_argument('--abo-project',type=Path,required=True);"
OLD_BINDING = "    sys.path.insert(0,str(a.abo_project/'lab_dvp8um'))\n    import four_image_flow as flow\n    from shs_physical2400 import SHSBench,CORNERS\n"
NEW_BINDING = "    flow = _main_geometry\n    SHSBench = _main_bench_factory\n    CORNERS = _main_geometry.BASE_CORNERS.copy()\n"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def adapted_source(train=False):
    folder = Path(__file__).resolve().parent
    source = (folder / 'run_layerwise.py').read_text(encoding='utf8').replace('\r\n', '\n')
    manifest = json.loads((folder / 'SOURCE_IDENTITY_20261006.json').read_text(encoding='utf8'))
    identity = next(r for r in manifest['sources'] if r['path'].endswith('/run_layerwise.py'))
    if hashlib.sha256(source.encode()).hexdigest() != identity['sha256_lf']:
        raise ValueError('Preserved formal runner SHA changed; adapter must be reviewed')
    if source.count(OLD_ARGUMENT) != 1 or source.count(OLD_BINDING) != 1:
        raise ValueError('Preserved runner binding changed')
    if train:
        adapter = (folder / 'run_train_capture.py').read_text(encoding='utf8').replace('\r\n', '\n')
        expected = next(r for r in manifest['sources'] if r['path'].endswith('/run_train_capture.py'))
        if hashlib.sha256(adapter.encode()).hexdigest() != expected['sha256_lf']:
            raise ValueError('Preserved TRAIN adapter SHA changed')
        changes_node = next(n.value for n in ast.walk(ast.parse(adapter))
                            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'changes' for t in n.targets))
        changes = ast.literal_eval(changes_node)
        for old, new in changes.items():
            if old not in source:
                raise ValueError('Pinned TRAIN substitution no longer matches')
            source = source.replace(old, new)
    return source.replace(OLD_ARGUMENT, '').replace(OLD_BINDING, NEW_BINDING)


def inspect(args):
    train = getattr(args, 'split', 'test') == 'train'
    compile(adapted_source(train), 't12_formal_main_adapter', 'exec')
    if train:
        if args.selection is None or args.max_samples is not None:
            raise ValueError('TRAIN requires original --selection and no --max-samples')
        selected = json.loads(args.selection.read_text(encoding='utf8'))
        indices = selected['indices']
        if selected['split'] != 'train' or selected['test_product_overlap'] != 0 or selected['test_source_hash_overlap'] != 0:
            raise ValueError('TRAIN selection declares TEST overlap or wrong split')
        if not indices or len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < 20736 for i in indices):
            raise ValueError('Invalid TRAIN selection indices')
    contract_path = args.project / 'assets/contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf8'))
    if contract.get('checkpoint_sha256') != FORMAL_SHA or contract.get('counted_parameters') != 17026642:
        raise ValueError('Only the original formal 17M model contract is supported')
    if digest(args.project / 'assets/small.pt') != FORMAL_SHA:
        raise ValueError('Formal checkpoint SHA mismatch')
    for name in ('machine_config', 'phase_lut'):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.phase_sdk.is_dir():
        raise NotADirectoryError(args.phase_sdk)
    if not (args.phase_sdk / 'Blink_C_wrapper.dll').is_file():
        raise FileNotFoundError(args.phase_sdk / 'Blink_C_wrapper.dll')
    if args.amplitude_sdk is not None and not args.amplitude_sdk.is_dir():
        raise FileNotFoundError(args.amplitude_sdk)
    if not args.reuse.is_dir():
        raise FileNotFoundError(args.reuse)
    if args.output.exists():
        raise FileExistsError('Use a new output directory; verified old CCD belong in --reuse')
    return {'formal_checkpoint_sha256': FORMAL_SHA, 'parameters': 17026642,
            'legacy_abo_source_required': False, 'devices_opened': False,
            'model_loaded': False, 'output_created': False,
            'full_environment_or_capture_regression_verified': False,
            'limitations': ['Preserves original CUDA model execution',
                           'TRAIN uses the pinned original selection adapter; decoder evaluation is separate',
                           'SDK file presence is not device availability or full binary validation']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('inspect', 'capture'), default='inspect')
    for name in ('project', 'output', 'reuse', 'machine-config', 'phase-sdk', 'phase-lut'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--amplitude-sdk', type=Path)
    p.add_argument('--split', choices=('test', 'val', 'train'), default='test')
    p.add_argument('--selection', type=Path)
    p.add_argument('--max-samples', type=int)
    args = p.parse_args()
    result = inspect(args)
    if args.mode == 'inspect':
        print(json.dumps(result, indent=2))
        return
    # Deferred imports: no SDK initialization during --help or inspection.
    from LightGenV2.tasks.t07_abo_image_retrieval.hardware import geometry
    from LightGenV2.tasks.t07_abo_image_retrieval.hardware.bench import SHSBench
    factory = functools.partial(SHSBench, machine_config=args.machine_config,
                                phase_sdk=args.phase_sdk, phase_lut=args.phase_lut,
                                amplitude_sdk=args.amplitude_sdk)
    argv = [sys.argv[0], '--project', str(args.project), '--output', str(args.output),
            '--reuse', str(args.reuse), '--split', args.split]
    if args.max_samples is not None:
        argv += ['--max-samples', str(args.max_samples)]
    if args.split == 'train':
        argv += ['--selection', str(args.selection)]
    previous = sys.argv
    try:
        sys.argv = argv
        exec(compile(adapted_source(args.split == 'train'), str(Path(__file__).with_name('run_layerwise.py')), 'exec'),
             {'__name__': '__main__', '__package__': __package__,
              '_main_geometry': geometry, '_main_bench_factory': factory})
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
