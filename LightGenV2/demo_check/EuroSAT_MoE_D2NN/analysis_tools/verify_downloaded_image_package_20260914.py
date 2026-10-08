"""Independently verify the downloaded EuroSAT images without extracting 53k files."""
import collections
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'.codex_plot_deps'))
from PIL import Image

OUT=Path(json.loads((ROOT/'IMAGE_EXPORT_LOCAL.json').read_text(encoding='utf-8'))['directory'])
report=json.loads((OUT/'SERVER_EXPORT_RESULT.json').read_text(encoding='utf-8'))
assert report['passed'] and report['state']=='complete'
temp=OUT/'EuroSAT_experiment_53784_images.zip.partial'
final=OUT/'EuroSAT_experiment_53784_images.zip'
path=temp if temp.exists() else final
assert path.stat().st_size==report['archive_bytes']
h=hashlib.sha256()
with path.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==report['archive_sha256']
started=time.time()
with zipfile.ZipFile(path) as z:
    assert len(z.namelist())==len(set(z.namelist()))
    split=json.loads(z.read('SPLIT.json'))
    records=split['records']
    assert records==json.loads((ROOT/'SPLIT.json').read_text(encoding='utf-8'))['records']
    manifest_bytes=z.read('IMAGE_MANIFEST.json')
    assert hashlib.sha256(manifest_bytes).hexdigest()==report['image_manifest_sha256']
    assert manifest_bytes==(OUT/'IMAGE_MANIFEST.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    expected={r['path'] for r in records}
    actual={n for n in z.namelist() if n.startswith('images/') and n.endswith('.png')}
    assert actual==expected and len(expected)==53784
    counts=collections.Counter((r['domain'],r['split']) for r in records)
    for i,r in enumerate(records,1):
        raw=z.read(r['path'])
        assert hashlib.sha256(raw).hexdigest()==manifest[r['path']]['sha256'],r['path']
        with Image.open(io.BytesIO(raw)) as im:
            assert im.size==(56,56) and im.mode=='RGB'
            assert hashlib.sha256(im.tobytes()).hexdigest()==manifest[r['path']]['pixel_sha256'],r['path']
        if i%10000==0:print(f'Locally verified {i}/{len(records)} images',flush=True)
    # Reading all remaining members also verifies each CRC through ZipFile.read.
    for n in z.namelist():
        if n not in actual:z.read(n)
    for n in ['README.md','SPLIT.json','DATA_PREPROCESSING.json','ATTRIBUTION.txt','EXPORT_VERIFICATION.json']:
        (OUT/n).write_bytes(z.read(n))
if path==temp:os.replace(temp,final)
verification={'passed':True,'archive':str(final),'archive_sha256':h.hexdigest(),'archive_bytes':final.stat().st_size,
              'images':len(records),'pairs':26892,'image_file_sha256_checks':len(records),'image_pixel_sha256_checks':len(records),
              'all_png_sizes':[56,56],'all_png_mode':'RGB','zip_member_crc_checks_passed':True,
              'split_matches_training':True,'counts':{d:{s:counts[d,s] for s in ['train','validation','test']} for d in ['A','B']},
              'local_verification_seconds':round(time.time()-started,2),
              'example_package':str(OUT/'EuroSAT_AB配对样例与示意图.zip')}
(OUT/'LOCAL_DOWNLOAD_VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(verification,ensure_ascii=False,indent=2))
