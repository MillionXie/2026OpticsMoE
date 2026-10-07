"""Read-only sealed rank72 binding gate; never starts acquisition or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from .ccd_store import CHECKPOINT
from maintenance.git_safety.check_t07_machine_paths import inspect as inspect_machine

PROTOCOL_SHA='f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde'
GEOMETRY_SHA='bde57c1be8eae6f8362cae5d52b370eeedf410e3fae6e5a05a19ce820745ad2e'
PATH_FIELDS=('checkpoint','processor','protocol','data_root','geometry','run_dir',
             'machine_config','amplitude_sdk','phase_sdk','phase_lut')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_paths(args):
    # Geometry/OpenCV is required only for a real asset inspection, not schema tests.
    from .layerwise import inspect_paths as implementation
    return implementation(args)


def load_binding(path):
    row=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if row.get('schema_version')!=1 or row.get('task')!='t07_rank72_sealed':
        raise ValueError('Require the sealed rank72 binding schema')
    if row.get('mode','inspect')!='inspect':
        raise ValueError('This gate cannot authorize capture, training or evaluation')
    if row.get('checkpoint_sha256')!=CHECKPOINT:
        raise ValueError('The sealed checkpoint cannot be substituted')
    for field in PATH_FIELDS:
        if not isinstance(row.get(field),str) or not row[field].strip():
            raise ValueError('Missing binding path: '+field)
    return SimpleNamespace(**{field:Path(row[field]).resolve() for field in PATH_FIELDS},
                           checkpoint_sha256=CHECKPOINT,mode='inspect')


def inspect_binding(path,load_cpu=False):
    args=load_binding(path)
    if digest(args.protocol)!=PROTOCOL_SHA or digest(args.geometry)!=GEOMETRY_SHA:
        raise ValueError('Original sealed protocol/geometry SHA mismatch')
    config_before=digest(args.machine_config)
    entry=inspect_paths(args)
    machine=inspect_machine(args.machine_config,args.phase_sdk,args.phase_lut,args.amplitude_sdk)
    result={'passed':machine['passed'],'entry':entry,'machine':machine,
            'config_sha256':config_before,'model_loaded':False,'devices_opened':False,
            'dataset_evaluated':False,'outputs_created':False,'read_only':True}
    if load_cpu:
        import torch
        from transformers import AutoProcessor
        from ..standalone.model import OpticalRetrieval
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model=OpticalRetrieval(payload['metadata']).to('cpu').eval()
        model.load_state_dict(payload['state_dict'],strict=True)
        processor=AutoProcessor.from_pretrained(str(args.processor),local_files_only=True)
        result.update(model_loaded=True,strict_cpu_load=True,
                      processor_class=type(processor).__name__,model_class=type(model).__name__,
                      model_device='cpu')
    if digest(args.machine_config)!=config_before:raise RuntimeError('Machine config changed concurrently')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--load-cpu',action='store_true',help='Strict CPU PT and local processor load only; no queries')
    args=parser.parse_args()
    try:result=inspect_binding(args.binding,args.load_cpu)
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ImportError) as error:
        result={'passed':False,'read_only':True,'devices_opened':False,
                'error_type':type(error).__name__,'error':str(error)}
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
