"""Reconstruct an explicitly limited train/validation subset using the archived split."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from affine import Affine

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'EuroSAT_MoE_D2NN/code'))
from download_archives import SOURCES, digest


def decode_pair(pid, archives, maps):
    with Image.open(io.BytesIO(archives['rgb'].read(maps['rgb'][pid]))) as im:
        assert im.mode == 'RGB' and im.size == (64,64)
        rgb = np.array(im)[4:60,4:60]
    with MemoryFile(archives['ms'].read(maps['ms'][pid])) as mf, MemoryFile(archives['sar'].read(maps['sar'][pid])) as sf, mf.open() as ms, sf.open() as sar:
        assert ms.width == ms.height == 64 and ms.count == 13 and sar.count == 2
        raw = sar.read().astype(np.float32)
        out = np.full((2,56,56), np.nan, dtype=np.float32)
        reproject(raw, out, src_transform=sar.transform, src_crs=sar.crs, src_nodata=sar.nodata,
                  dst_transform=ms.transform*Affine.translation(4,4), dst_crs=ms.crs,
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
        assert np.isfinite(out).mean() >= .99
        channels = []
        for channel, mean, std in zip(out, [-12.59,-20.26], [5.26,5.91]):
            channel = np.where(np.isfinite(channel), channel, np.nanmedian(channel))
            low, high = np.quantile(channel, [.01,.99])
            channels.append(np.clip((np.clip(channel,low,high)-(mean-2*std))/(4*std),0,1))
        sar_image = np.round(np.stack([channels[0],channels[1],(channels[0]+channels[1])/2],2)*255).astype(np.uint8)
    return rgb, sar_image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--config', type=Path, default=HERE/'config.json')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    split_path = a.root/'SPLIT.json'
    assert digest(split_path, 'sha256') == 'cfe8373dd33cc0fe64f083b9ca32377e767c21b078c3f1d91f2dacecc25cb776'
    records = json.loads(split_path.read_text())['records']
    source_hashes = {}
    archives = {}
    maps = {}
    for name, (url, filename, size, algorithm, expected) in SOURCES.items():
        path = a.root/'archives'/filename
        assert path.stat().st_size == size and digest(path, algorithm) == expected, name
        source_hashes[name] = dict(url=url, sha256=digest(path, 'sha256'), algorithm=algorithm, expected=expected)
        archives[name] = zipfile.ZipFile(path)
        suffix = '.jpg' if name == 'rgb' else '.tif'
        maps[name] = {Path(x.filename).stem:x for x in archives[name].infolist()
                      if Path(x.filename).suffix.lower() == suffix and not x.filename.startswith('__MACOSX/')}
        assert len(maps[name]) == 27000
    arrays = {}
    manifest = []
    for partition, count in [('train', cfg['train_pairs_per_class']), ('validation', cfg['validation_pairs_per_class'])]:
        selected = []
        for label in range(10):
            candidates = [r for r in records if r['domain']=='A' and r['split']==partition and r['label']==label]
            candidates.sort(key=lambda r:hashlib.sha256(('phase_only_v1|'+r['pair_id']).encode()).hexdigest())
            assert len(candidates) >= count
            selected.extend(candidates[:count])
        images, labels, domains, ids = [], [], [], []
        for index, row in enumerate(selected):
            pid = row['pair_id']
            rgb, sar_image = decode_pair(pid, archives, maps)
            for domain, image in enumerate([rgb,sar_image]):
                assert image.shape == (56,56,3) and image.max() > 0
                images.append(image); labels.append(row['label']); domains.append(domain); ids.append(pid+':'+str(domain))
                manifest.append(dict(pair_id=pid, domain=domain, split=partition, label=row['label'],
                                     spatial_group=row['spatial_group'], pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
            if (index+1)%250 == 0:
                print(json.dumps(dict(partition=partition, pairs=index+1, total=len(selected))), flush=True)
        arrays.update({partition+'_images':np.stack(images), partition+'_labels':np.array(labels),
                       partition+'_domains':np.array(domains), partition+'_ids':np.array(ids)})
    assert not {r['spatial_group'] for r in manifest if r['split']=='train'} & {r['spatial_group'] for r in manifest if r['split']=='validation'}
    dest = a.root/'phase_only_v1'
    dest.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(dest/'data.npz', **arrays)
    (dest/'manifest.json').write_text(json.dumps(dict(protocol=cfg, original_split_sha256=digest(split_path,'sha256'),
        data_sha256=digest(dest/'data.npz','sha256'), source_archives=source_hashes, records=manifest,
        rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__, test_used=False),indent=2))
    print(json.dumps(dict(state='complete', output=str(dest), train=len(arrays['train_labels']), validation=len(arrays['validation_labels']))),flush=True)


if __name__ == '__main__':
    main()
