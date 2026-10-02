"""New split-rank candidate over the SHA-pinned actual bench capture backend.

No edits to old G2/G5 modules or GROUPS on disk. Only this process substitutes
the exact initial-trained factory and weight. New upstream means new CCD runs.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from . import split_rank_head
from .train import sha

WEIGHT = 'splitrank_editor48_decoder64_alpha80_e40.pt'
WEIGHT_SHA = '782e608ba914064cd8a10bd0dbc7d3da63dae35a0646cacb893657608954cab8'
BACKEND_SHA = {
    'lab_shs_capture': '605d17590c5947fedec030fd913c19a4ee31be08b1ae832c99377ebb015fe5c6',
    'lab_shs_capture_g2_saturation': '8d08a13b0cfca0bec28b5a531535b45420caf257aea91f46d49ef162ea92b89b',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--mode', choices=('selftest', 'pilot', 'test', 'train'), required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    project = args.project.resolve()
    scope = 'train' if args.mode == 'train' else 'test'
    backend = 'lab_shs_capture_g2_saturation' if scope == 'train' else 'lab_shs_capture'
    backend_path = Path(__file__).with_name(backend+'.py')
    assert sha(backend_path) == BACKEND_SHA[backend], 'Actual bench backend changed; audit before running'
    payload = torch.load(project/'weights'/WEIGHT, map_location='cpu', weights_only=False)
    assert sha(project/'weights'/WEIGHT) == WEIGHT_SHA
    assert payload['group'] == 'r0_base' and payload['settings']['editor_rank'] == 48
    assert payload['settings']['shared_readout_variant'] == 'lowrank64'
    assert abs(payload['fixed_fusion_alpha']-.8) < 1e-9
    capture = importlib.import_module('.'+backend, __package__)
    capture.GROUPS = {'g2': (WEIGHT, WEIGHT_SHA)}
    outputs = {'selftest': 'splitrank48_selftest', 'pilot': 'splitrank48_pilot4',
               'test': 'splitrank48_test1000', 'train': 'splitrank48_train1000'}
    output = project/'runs'/outputs[args.mode]
    if output.exists() and not args.resume:
        raise ValueError('Output exists; preserve it and explicitly request same-contract resume')
    output.mkdir(parents=True, exist_ok=True)
    original_write = capture.write
    def write(path, value):
        if path.name == 'contract.json':
            value = dict(value, candidate='editor48_decoder64_alpha80', editor_rank=48,
                         decoder_rank=64, four_fixed_alpha=.8,
                         factory_sha256=sha(Path(split_rank_head.__file__)),
                         wrapper_sha256=sha(Path(__file__)),
                         actual_capture_backend_sha256=BACKEND_SHA[backend])
        return original_write(path, value)
    capture.write = write
    def build(cfg, device):
        assert device.type == 'cpu', 'Hardware model must remain CPU'
        cfg.editor_rank = 48
        cfg.optical_fusion_initial = .8
        model = split_rank_head.build_model(cfg, device)
        model.load_state_dict(payload['model'], strict=True)
        gates = {f'{name}.block{block}': float(getattr(getattr(model,name),f'block{block}_optical_fusion'))
                 for name in ('language_core','vision_core') for block in (1,2)}
        assert all(abs(v-.8)<1e-6 for v in gates.values())
        assert sum(p.numel() for p in model.shared_readout.parameters()) == 240664
        assert sum(p.numel() for p in model.shared_readout.decoder.parameters()) == 30162
        write(output/'strict_model.json', {'status':'pass','weight_sha256':WEIGHT_SHA,
                                          'gates':gates,'head_parameters':240664,'decoder_parameters':30162})
        return model
    # Capture module only uses t.build_model. Avoid modifying the shared module
    # object used by split_rank_head itself (which would recurse).
    capture.t = SimpleNamespace(build_model=build)
    sys.argv = [str(backend_path),'--project',str(project),'--group','g2','--scope',scope,
                '--output',str(output),'--limit','4' if args.mode in ('selftest','pilot') else '1000',
                '--exposure-us','2000','--device','cpu']
    if args.mode == 'selftest':
        sys.argv += ['--selftest']
    if args.resume:
        sys.argv += ['--resume']
    try:
        capture.main()
    except Exception as exc:
        write(output/'progress.json', {'status':'failed','error':repr(exc),'candidate_weight_sha256':WEIGHT_SHA})
        raise


if __name__ == '__main__':
    main()
