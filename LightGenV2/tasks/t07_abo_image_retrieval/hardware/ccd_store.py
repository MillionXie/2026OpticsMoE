"""Read-only sealed-rank72 CCD/receipt validation; no model or device imports."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
CHECKPOINT = '25f23260864a00a0f32b5de209c27f5876166018c8d8ab0e880de81e0d2c2c22'
CONTRACT_SHA256 = 'bdc0d96745c144bd5dbe2865534ff40aceb9d4d3764e3d7b506f2ddfaed07df9'
ENCODING = 'bounded tanh(abs/.5) before phase; round255a without peak scaling'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CCDStore:
    """Validate a separately pinned session identity, never repair its contents.

    Old receipts do not contain a PNG hash. Computed hashes describe present bytes;
    matching statistics do not prove an unchanged historical image or illumination.
    Deleted temporary amplitude BMPs likewise cannot be rehashed from a receipt.
    """
    def __init__(self, root, *, contract_sha256=CONTRACT_SHA256):
        self.root = Path(root).resolve()
        if not re.fullmatch('[0-9a-f]{64}', contract_sha256):
            raise ValueError('Require independently pinned contract SHA256')
        path = self.root / 'contract.json'
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Session contract escapes run directory')
        if sha(path) != contract_sha256:
            raise ValueError('Session contract SHA mismatch')
        self.contract_sha256 = contract_sha256
        self.contract = json.loads(path.read_text(encoding='utf-8'))
        c = self.contract
        if (c['checkpoint_sha256'] != CHECKPOINT or c['encoding'] != ENCODING
                or c['exposure_us'] != 400 or c['gain'] != 'Gain_X4' or c['wait_ms'] != 240
                or c['geometry'] != [[586,147],[1379,159],[1369,946],[573,932]]):
            raise ValueError('Not the sealed rank72 optical contract')
        if set(c['phase_sha256']) != set(STAGES) or set(c['stage_orientation']) != set(STAGES):
            raise ValueError('Contract must cover exactly six stages')
        self.ids = tuple(c['ids'])
        if (not self.ids or len(set(self.ids)) != len(self.ids)
                or any(not isinstance(sid, str) or not re.fullmatch('[0-9a-f]{16}', sid) for sid in self.ids)):
            raise ValueError('Unsafe or duplicated sample identities')
        self.allowed_ids = set(self.ids)
        for stage in STAGES:
            if c['stage_orientation'][stage] != ['hv_inverse', 'flip_v']:
                raise ValueError('Sealed phase/camera orientation changed')
            phase = self.root / 'phase' / (stage + '.bmp')
            if not phase.resolve().is_relative_to(self.root):
                raise ValueError('Phase file escapes run directory')
            if sha(phase) != c['phase_sha256'][stage]:
                raise ValueError('Phase file SHA mismatch: ' + stage)
            with Image.open(phase) as im:
                if im.format != 'BMP' or im.mode != 'L' or im.size != (1920, 1200):
                    raise ValueError('Require native sealed phase BMP')

    def pair(self, stage, sid):
        if stage not in STAGES or sid not in self.allowed_ids:
            raise ValueError('Unknown stage or identity')
        folder = self.root / 'ccd' / stage
        image, receipt = folder / (sid + '.png'), folder / (sid + '.json')
        if not image.is_file() or not receipt.is_file():
            raise FileNotFoundError('Missing or partial CCD pair: ' + stage + '/' + sid)
        # Resolve symlinks as well as filenames before reading.
        if not image.resolve().is_relative_to(self.root) or not receipt.resolve().is_relative_to(self.root):
            raise ValueError('CCD pair escapes session directory')
        row = json.loads(receipt.read_text(encoding='utf-8'))
        exposure = row['exposure']
        if (row['stage'] != stage or row['sample_id'] != sid
                or row['phase_sha256'] != self.contract['phase_sha256'][stage]
                or row['wait_ms'] != 240 or exposure['exposure_us'] != 400
                or exposure['gain'] != 'Gain_X4' or row['canonical_orientation'] != 'flip_v'
                or row['no_photometric_normalization'] is not True
                or not re.fullmatch('[0-9a-f]{64}', row['amplitude_sha256'])):
            raise ValueError('CCD receipt contract mismatch: ' + stage + '/' + sid)
        with Image.open(image) as im:
            if im.format != 'PNG' or im.mode != 'L' or im.size != (478, 478):
                raise ValueError('Require native canonical Mono8 CCD PNG')
            array = np.array(im)
        p99 = float(np.percentile(array, 99))
        if p99 < 15:
            raise ValueError('Dark CCD cannot enter formal replay: ' + stage + '/' + sid)
        statistics = dict(mean=float(array.mean()), p99=p99, maximum=int(array.max()),
                          saturation_fraction=float(np.mean(array == 255)))
        if any(not np.isfinite(row[key]) or abs(row[key] - value) > 1e-9
               for key, value in statistics.items()):
            raise ValueError('CCD image disagrees with receipt statistics')
        if statistics['saturation_fraction'] > .01:
            raise ValueError('Outside the sealed ABO saturation contract')
        return array, dict(png_sha256=sha(image), receipt_sha256=sha(receipt), **statistics)

    def read(self, stage, ids):
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Require nonempty unique batch identities')
        return np.stack([self.pair(stage, sid)[0] for sid in ids])

    def audit(self):
        result = {}
        for stage in STAGES:
            folder = self.root / 'ccd' / stage
            png_ids = {path.stem for path in folder.glob('*.png')}
            json_ids = {path.stem for path in folder.glob('*.json')}
            if png_ids != json_ids or not png_ids.issubset(self.allowed_ids):
                raise ValueError('Partial or unexpected CCD identities: ' + stage)
            p99 = [self.pair(stage, sid)[1]['p99'] for sid in self.ids if sid in png_ids]
            result[stage] = dict(valid_pairs=len(p99), missing_pairs=len(self.ids) - len(p99),
                                 minimum_p99=min(p99) if p99 else None)
        return dict(checkpoint_sha256=CHECKPOINT, contract_sha256=self.contract_sha256,
                    images_per_layer=len(self.ids), stages=result,
                    complete=all(row['missing_pairs'] == 0 for row in result.values()),
                    read_only=True, dataset_metrics_computed=False, devices_opened=False,
                    historical_png_sha_available=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--contract-sha256', default=CONTRACT_SHA256)
    args = parser.parse_args()
    print(json.dumps(CCDStore(args.run, contract_sha256=args.contract_sha256).audit(), indent=2))


if __name__ == '__main__':
    main()
