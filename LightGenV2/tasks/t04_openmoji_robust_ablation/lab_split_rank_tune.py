"""Same pre-trained split-rank architecture, only original decoder adaptation.

The actual bench adapter stays untouched and is SHA-pinned. TEST selects the
checkpoint every five epochs by explicit authorization, never provides gradients.
"""
import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
from . import split_rank_head
from .lab_split_rank_capture import WEIGHT, WEIGHT_SHA
from .train import sha

BACKEND_SHA = 'd4c0c4fd59096bd627cc621ce7019a0f989841885b2c198659efd622812c9a0f'


def prepare(project):
    path = Path(__file__).with_name('lab_tune_g2_test.py')
    assert sha(path) == BACKEND_SHA, 'Actual adapter backend changed; audit first'
    assert sha(project/'weights'/WEIGHT) == WEIGHT_SHA
    payload = torch.load(project/'weights'/WEIGHT, map_location='cpu', weights_only=False)
    assert payload['settings']['editor_rank'] == 48
    assert payload['settings']['shared_readout_variant'] == 'lowrank64'
    assert payload['group'] == 'r0_base' and abs(payload['fixed_fusion_alpha']-.8)<1e-9
    tune = importlib.import_module('.lab_tune_g2_test', __package__)
    tune.GROUPS = {'g2': (WEIGHT, WEIGHT_SHA)}

    def build(cfg, device):
        assert device.type == 'cpu', 'Cache must use the captured CPU model'
        cfg.editor_rank = 48
        cfg.optical_fusion_initial = .8
        model = split_rank_head.build_model(cfg, device)
        model.load_state_dict(payload['model'], strict=True)
        assert sum(p.numel() for p in model.shared_readout.parameters()) == 240664
        assert sum(p.numel() for p in model.shared_readout.decoder.parameters()) == 30162
        for name in ('language_core','vision_core'):
            for block in (1,2):
                gate = getattr(getattr(model,name),f'block{block}_optical_fusion')
                assert abs(float(gate.detach())-.8)<1e-6
        return model

    # Replace only the adapter's module reference, not the shared training module.
    tune.t = SimpleNamespace(build_model=build)
    return tune, payload


def strict_saved_audit(tune, project, output):
    execution = json.loads((output/'execution.json').read_text())
    report = json.loads((output/'report.json').read_text())
    assert execution['checkpoint_sha256'] == WEIGHT_SHA
    assert execution['trainable_prefix'] == 'shared_readout.decoder'
    assert execution['trainable_parameters'] == 30162
    assert report['protected_unchanged'] and not report['test_gradient']
    _, model = tune.config(project, torch.device('cpu'))
    before = tune.protected_sha(model)
    assert before == execution['protected_before']
    saved = torch.load(output/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    assert tune.protected_sha(model) == before
    model.eval().requires_grad_(False)
    cache = torch.load(output/'test_features.pt', map_location='cpu', weights_only=False)
    assert len(cache['ids']) == 1000
    meter = MetricAccumulator()
    with torch.no_grad():
        for start in range(0,1000,32):
            rows = cache['rows'][start:start+32]
            batch = {k: torch.cat([row[k] for row in rows]) if torch.is_tensor(rows[0][k])
                     else sum([row[k] for row in rows],[]) for k in rows[0]}
            category, edit = model.shared_readout.decoder(cache['features'][start:start+32])
            meter.update({'category_logits':category,'edit_logits':edit,
                          'task_logits':batch['task_logits']},batch)
    metrics = meter.compute()
    expected = report['physical_test']['overall']['changed_cell_accuracy']
    assert abs(metrics['overall']['changed_cell_accuracy']-expected)<1e-8
    tune.write(output/'strict_reload.json',{
        'status':'pass','device':'cpu','strict_state_dict':True,
        'best_sha256':sha(output/'best.pt'),'last_sha256':sha(output/'last.pt'),
        'protected_before':before,'protected_after':tune.protected_sha(model),
        'architecture_unchanged':True,'decoder_parameters':30162,
        'metrics':metrics,'development_test_selection':True,'test_gradient':False,
        'target_original_simulation':.8835,'target_97percent':.856995,
        'target_met':metrics['overall']['changed_cell_accuracy']>=.856995})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--epochs',type=int,default=160)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    project = args.project.resolve()
    torch.set_num_threads(4)
    tune, _ = prepare(project)
    if args.preflight:
        _, model = tune.config(project,torch.device('cpu'))
        print(json.dumps({'status':'pass','strict_initial_weight_sha256':WEIGHT_SHA,
                          'protected_before':tune.protected_sha(model),
                          'decoder_parameters':sum(p.numel() for p in model.shared_readout.decoder.parameters()),
                          'no_sdk':True,'no_gradient':True}),flush=True)
        return
    output = project/'runs/splitrank48_decoder_train1000_testselected'
    assert not output.exists(), 'Existing output must be preserved, not restarted'
    # Backend fully validates counts, scope, disjoint manifests, each receipt,
    # phase SHA and exact CPU baseline before any decoder gradient.
    sys.argv = [str(Path(tune.__file__)),'--project',str(project),
                '--train-run',str(project/'runs/splitrank48_train1000'),
                '--test-run',str(project/'runs/splitrank48_test1000'),
                '--output',str(output),'--epochs',str(args.epochs),'--device',args.device]
    tune.main()
    strict_saved_audit(tune,project,output)


if __name__ == '__main__':
    main()
