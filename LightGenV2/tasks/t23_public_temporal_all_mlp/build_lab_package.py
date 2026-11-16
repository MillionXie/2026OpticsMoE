"""Build the standalone inference archive after verifying its data and results."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import zipfile

import numpy as np
import torch
from scipy.stats import spearmanr

from LightGen2_MLP.data import read_input_metadata, read_reference_scores
from LightGen2_MLP.model import ARCHITECTURE, load_lightgen2
from .checkpoints import sha256


TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
PROJECT = REPO/'LightGen2_MLP'
FILES = (
    '.gitattributes', '.gitignore', 'model.py', 'infer.py', 'data.py', 'plots.py',
    'requirements.txt', 'LICENSE', 'licenses/NOTICE.txt', 'licenses/QWEN_LICENSE.txt',
    'README.pdf', 'weights/LightGen2.pt', 'data/input_for_review.npz', 'data/reference_mos.csv',
    'results/01_predictions.csv', 'results/02_metrics.json',
    'results/03_prediction_vs_mos.png', 'results/04_input_frames.png',
)


def build():
    present = {path.relative_to(PROJECT).as_posix() for path in PROJECT.rglob('*') if path.is_file()}
    if present - set(FILES) - {'README.docx'} or set(FILES) - present:
        raise ValueError('Standalone project contains unexpected or missing files')
    if not (PROJECT/'README.pdf').read_bytes().startswith(b'%PDF-'):
        raise ValueError('The guide must be a PDF')
    model, _ = load_lightgen2(PROJECT/'weights/LightGen2.pt')
    weight = torch.load(PROJECT/'weights/LightGen2.pt', map_location='cpu', weights_only=True)
    if weight['architecture'] != ARCHITECTURE or 'training' in weight:
        raise ValueError('Expected inference-only MLP weights')
    if any(any(word in name for word in ('phase', 'propagation', 'transfer')) for name in model.state_dict()):
        raise ValueError('Unexpected propagation state in the MLP project')
    with np.load(PROJECT/'data/input_for_review.npz', allow_pickle=False) as archive:
        identities = read_input_metadata(archive)
        if len(identities) != 558 or 'target_mos' in archive.files:
            raise ValueError('Expected 558 distinct video inputs with separate labels')
        if archive['rgb_0035'].shape != (14,4,3,448,448):
            raise ValueError('Tail input batch differs')
    targets = read_reference_scores(PROJECT/'data/reference_mos.csv', identities)
    with (PROJECT/'results/01_predictions.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['sample_id','target_MOS','predicted_MOS']:
            raise ValueError('Prediction columns differ')
        rows = list(reader)
    if [row['sample_id'] for row in rows] != identities:
        raise ValueError('Predictions and input identities differ')
    np.testing.assert_array_equal([float(row['target_MOS']) for row in rows], targets)
    score = float(spearmanr([float(row['predicted_MOS']) for row in rows], targets).statistic)
    metrics = json.loads((PROJECT/'results/02_metrics.json').read_text('utf-8'))
    if metrics != dict(videos=558, SRCC=score) or not .790 <= score <= .799:
        raise ValueError('Result summary or target SRCC range differs')
    expected_input = '65209b7381614cf28d08cbdd537a0a61db51b91a0fde2dc20a9d40e18bc0acd5'
    if sha256(PROJECT/'data/input_for_review.npz') != expected_input:
        raise ValueError('Video input archive differs from the approved release')
    for name in FILES:
        if Path(name).suffix in ('.py','.txt','.json'):
            text = (PROJECT/name).read_text('utf-8-sig')
            private = ('Xml12','OneDrive','/DATA/','/home/','xwechat_files','model_parameters')
            if any(marker in text for marker in private):
                raise ValueError('Private path or metadata in '+name)
    checksums = {name:sha256(PROJECT/name) for name in FILES}
    release = TASK/'releases'
    release.mkdir(parents=True, exist_ok=True)
    temporary = release/'.LightGen2_MLP.zip.building'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for name in FILES:
            compression = zipfile.ZIP_STORED if name.endswith('.npz') else zipfile.ZIP_DEFLATED
            archive.write(PROJECT/name, 'LightGen2_MLP/'+name, compress_type=compression)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != {'LightGen2_MLP/'+name for name in FILES}:
            raise ValueError('Archive membership differs')
        for name, expected in checksums.items():
            digest = hashlib.sha256()
            with archive.open('LightGen2_MLP/'+name) as stream:
                for block in iter(lambda:stream.read(8*1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError('Archive bytes differ for '+name)
    target = release/'LightGen2_MLP.zip'
    temporary.replace(target)
    digest = sha256(target)
    (release/'LightGen2_MLP.zip.sha256').write_text(digest+'  LightGen2_MLP.zip\n', encoding='utf-8')
    receipt = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
                   files=checksums, archive_sha256=digest, metrics=metrics)
    (release/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(archive=str(target), files=len(FILES), SRCC=score), indent=2))


if __name__ == '__main__':
    build()
