"""Package the exact EuroSAT experiment PNGs after file and pixel verification."""
import collections
import hashlib
import io
import json
import os
import time
import traceback
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path('/root/autodl-tmp/review/eurosat_expert_merge_20260912')
DATA = Path('/root/autodl-tmp/data/eurosat_optical_sar')
OUT = ROOT / 'exports/images53784_20260914'
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def atomic(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)

def status(**kwargs):
    atomic(OUT / 'STATUS.json', {'elapsed_seconds':round(time.time()-START, 2), **kwargs})

def main():
    plan = json.loads((ROOT/'PLAN.json').read_text())
    split_bytes = (ROOT/'SPLIT.json').read_bytes()
    split = json.loads(split_bytes)
    records = split['records']
    signature = sha(json.dumps(records, sort_keys=True, separators=(',', ':')).encode())
    assert signature == plan['split_records_sha256'] == '46e42dad5f19bff7ad638d13a359f118348ccd3e6eab4d76dc4b97be0340dc59'
    manifest_bytes = (DATA/'IMAGE_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert len(records) == 53784 and len({r['path'] for r in records}) == len(records)
    counts = collections.Counter((r['domain'], r['split']) for r in records)
    pairs = collections.defaultdict(list)
    for r in records:
        pairs[r['pair_id']].append(r)
    assert len(pairs) == 26892
    for pair in pairs.values():
        assert {r['domain'] for r in pair} == {'A','B'}
        assert len({r['label'] for r in pair}) == len({r['split'] for r in pair}) == 1
    count_dict = {d:{s:counts[d,s] for s in ('train','validation','test')} for d in ('A','B')}
    assert count_dict == plan['counts']
    temp = OUT/'EuroSAT_experiment_53784_images.zip.partial'
    target = OUT/'EuroSAT_experiment_53784_images.zip'
    image_bytes = 0
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for i, r in enumerate(records, 1):
            rel = r['path']
            p = (DATA/rel).resolve()
            assert p.is_relative_to(DATA) and rel.startswith('images/')
            raw = p.read_bytes()
            assert sha(raw) == manifest[rel]['sha256'], rel
            with Image.open(io.BytesIO(raw)) as im:
                assert im.mode == 'RGB' and im.size == (56,56), (rel,im.mode,im.size)
                assert sha(im.tobytes()) == manifest[rel]['pixel_sha256'], rel
            z.writestr(rel, raw)
            image_bytes += len(raw)
            if i % 2000 == 0:
                status(state='verifying_and_packaging',images_done=i,total_images=len(records),bytes_done=image_bytes)
        z.writestr('SPLIT.json', split_bytes)
        z.writestr('IMAGE_MANIFEST.json', manifest_bytes)
        for name in ['PLAN.json','DATA_PREPROCESSING.json','DATA_CHECKS.json','DATA_LICENSES.json','ATTRIBUTION.txt','SPLIT_AUDIT.json']:
            z.writestr(name, (ROOT/name).read_bytes())
        for p in sorted((ROOT/'licenses').iterdir()):
            if p.is_file(): z.writestr('licenses/'+p.name,p.read_bytes())
        readme = '''# EuroSAT 光学/SAR 本次实验图片包

包含本次 seed 42 实验实际保留的 26,892 对同地块图片，共53,784张无损PNG，全部为56×56、RGB三通道。
每个域：训练15,998张、验证5,465张、测试5,429张。SPLIT.json记录精确划分，A/B配对和地理分组保持一致。

A任务：光学RGB，来自官方EuroSAT的64×64 JPEG，取中心56×56区域。
B任务：SAR雷达图像，VV/VH重投影到对应光学地理网格，再按训练前固定的辐射归一化编码为三通道：(VV,VH,(VV+VH)/2)。B是用于模型输入的伪彩编码，不是自然光照片。
A/B同名PNG对应同一地块和类别。两种传感器的获取时间可能不同，不表示同一时刻拍摄。
训练读取这些PNG后统一缩放到224×224，训练阶段再做预设随机增强。包内保留增强前的56×56图像，以避免重复采样的随机增强混入数据存档。
本包是实验实际使用的预处理图片，不包含原始多光谱或SAR浮点GeoTIFF。原始来源地址和归档哈希保存在DATA_PREPROCESSING.json。
原始27,000对中实验保留26,892对；具体排除项及原因见DATA_CHECKS.json。

目录 images/A/类别/地块ID.png 与 images/B/类别/地块ID.png。
IMAGE_MANIFEST.json含文件SHA256、像素SHA256、原始来源文件名；EXPORT_VERIFICATION.json记录逐图校验结果。
10类别：AnnualCrop、Forest、HerbaceousVegetation、Highway、Industrial、Pasture、PermanentCrop、Residential、River、SeaLake。
作者署名、数据集许可快照及Sentinel声明完整保留于ATTRIBUTION.txt与licenses/。
'''
        z.writestr('README.md',readme.encode('utf-8'))
        report = dict(passed=True,images=len(records),pairs=len(pairs),counts=count_dict,
                      size=[56,56],mode='RGB',all_file_sha256_match_training_manifest=True,
                      all_pixel_sha256_match_training_manifest=True,paired_split_and_label_match=True,
                      split_records_sha256=signature,image_manifest_sha256=sha(manifest_bytes),
                      source_image_bytes=image_bytes,excluded_source_pairs=108)
        z.writestr('EXPORT_VERIFICATION.json',json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8'))
    status(state='checking_zip',images_done=len(records),total_images=len(records))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith('images/') and n.endswith('.png')]) == len(records)
    os.replace(temp,target)
    h=hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    result=dict(report,state='complete',archive=str(target),archive_bytes=target.stat().st_size,
                archive_sha256=h.hexdigest(),elapsed_seconds=round(time.time()-START,2))
    atomic(OUT/'EXPORT_RESULT.json',result)
    status(**result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        status(state='failed',error=traceback.format_exc())
        raise
