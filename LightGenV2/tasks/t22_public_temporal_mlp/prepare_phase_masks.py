"""Create phase-initialized weights and a resolved private training profile."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import subprocess

import torch

if __package__:
    from .data import sha256
    from .model import LightGenTemporal, load_lightgen
    from .phase_regularization import initialize_phase, phase_to_raw, phase_statistics
    from .run import ROOT, save_checkpoint
else:
    from data import sha256
    from model import LightGenTemporal, load_lightgen
    from phase_regularization import initialize_phase, phase_to_raw, phase_statistics
    from run import ROOT, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    parent, metadata = load_lightgen(args.source)
    config = json.loads(args.config.read_text('utf-8'))
    cfg = replace(parent.cfg, **config.get('model',{}))
    model = LightGenTemporal(cfg)
    model.load_state_dict(parent.state_dict(),strict=True)
    audit = {}
    with torch.no_grad():
        for name in ('frame_global','video_global'):
            stage = getattr(model,name)
            audit[name] = {'before':phase_statistics(stage)}
            phase = stage.phase()
            initialization = config['phase_initialization'].get(name,config['phase_initialization'])
            if initialization['kind'] == 'raw_zero':
                stage.raw_global_phase.zero_()
            elif initialization['kind'] != 'keep':
                phase = initialize_phase(phase,initialization,cfg)
                stage.raw_global_phase.copy_(phase_to_raw(phase))
            audit[name]['after'] = phase_statistics(stage)
    parent_state = parent.state_dict()
    changed = [name for name,value in model.state_dict().items()
               if not torch.equal(value,parent_state[name])]
    allowed = {'frame_global.raw_global_phase','video_global.raw_global_phase'}
    if not changed or not set(changed).issubset(allowed):
        raise ValueError('Unexpected tensors changed during phase initialization: '+str(changed))
    config['model'] = asdict(cfg)
    args.out.mkdir(parents=True)
    record = {'role':'UNMEASURED_PHASE_INITIALIZATION',
              'dataset':metadata['dataset'], 'parent_SHA256':sha256(args.source),
              'changed_tensors':changed, 'phase_statistics':audit,
              'initialization':config['phase_initialization'], 'measured_metrics_available':False}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    record['source_commit'] = os.environ.get('LIGHTGEN_SOURCE_COMMIT',commit)
    record['source_SHA256'] = {name:sha256(ROOT/name)
                              for name in ('model.py','phase_regularization.py','prepare_phase_masks.py')}
    save_checkpoint(args.out/'initialization.pt',model.checkpoint(**record))
    (args.out/'config.json').write_text(json.dumps(config,indent=2)+'\n','utf-8')
    (args.out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n','utf-8')
    print(str(args.out),flush=True)


if __name__=='__main__':
    main()
