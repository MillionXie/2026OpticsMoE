"""Original LSP/SALICON optical-student shape metadata, never Torch loading."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import subprocess
from torch_metadata import MetadataUnpickler,first_pickle

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
GROUPS={
 'LSP':('t02_keypoint_detection','a100_input_inclusive_20261024_02','head',
        '495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518'),
 'SALICON':('t03_saliency','a100_input_inclusive_20261024_01','saliency_head',
        '036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe')}


def inert_posix_path(*parts):return dict(__inert_posix_path_parts__=list(parts))


class DenseMetadataUnpickler(MetadataUnpickler):
    def find_class(self,module,name):
        # These two original metadata values are inert descriptions, not imports
        # of Torch or checkpoint/model code. Unknown globals still hard-fail.
        if (module,name)==('pathlib','PosixPath'):return inert_posix_path
        if (module,name)==('torch.torch_version','TorchVersion'):return str
        return super().find_class(module,name)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def extract_metadata(task):
    directory,run,head_key,pin=GROUPS[task]
    identity=read(REPO/'LightGenV2/tasks'/directory/'runs/simulation'/run/'ours_repeat01/identity.json')
    # Original recorded server path is only a DATA alias; no old source imported.
    marker='/2026OpticsMoE/'
    checkpoint=REPO/identity['checkpoint'].replace('\\','/').split(marker,1)[1]
    if identity['checkpoint_sha256']!=pin or sha(checkpoint)!=pin:raise ValueError('Original approved checkpoint differs')
    with checkpoint.open('rb') as f:raw=first_pickle(f)
    payload=DenseMetadataUnpickler(io.BytesIO(raw)).load()
    if not isinstance(payload,dict):raise ValueError('Original tensor metadata mapping required')
    return dict(task=task,checkpoint_path=checkpoint.relative_to(REPO).as_posix(),checkpoint_sha256=pin,
        checkpoint_bytes=checkpoint.stat().st_size,metadata_pickle_sha256=hashlib.sha256(raw).hexdigest(),metadata_pickle_bytes=len(raw),
        state_maps=dict(core=payload['core'],head=payload[head_key]),epoch=payload.get('epoch'),
        architecture=payload.get('architecture',payload.get('checkpoint_architecture')),
        no_Torch_import_or_model_forward=True,metadata_only_not_tensor_content_or_numerical_proof=True,
        source_sha256=sha(__file__),reader_source_sha256=sha(HERE/'torch_metadata.py'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extract-metadata',choices=tuple(GROUPS),required=True)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior metadata')
    value=extract_metadata(a.extract_metadata);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(task=value['task'],file=str(a.output),sha256=sha(a.output),bytes=a.output.stat().st_size)))
