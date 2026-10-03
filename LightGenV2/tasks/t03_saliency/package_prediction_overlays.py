"""Figure-only repack: unchanged predictions/metrics, no GT or fixation assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();src=a.source.resolve();out=a.output.resolve()
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError(out)
    report=json.loads((src/'metrics_all.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    sheet=Image.new('RGB',(1304,len(report['rows'])*530),'white')
    for i,row in enumerate(report['rows']):
        folder=out/row['directory'];folder.mkdir()
        source=src/row['directory']
        for name in ('original.jpg','ours_density_224.npy'):
            shutil.copy2(source/name,folder/name)
            assert sha(source/name)==sha(folder/name)
        original=Image.open(folder/'original.jpg').convert('RGB')
        density=np.load(folder/'ours_density_224.npy',allow_pickle=False)
        # DISPLAY ONLY: per-prediction peak scale, inferno, saliency-dependent
        # opacity over a dimmed original. No GT consulted, no inference changes.
        value=np.asarray(Image.fromarray(density).resize(original.size,Image.Resampling.BILINEAR))
        value=np.clip(value/max(float(value.max()),1e-12),0,1)
        heat=colormaps['inferno'](value)[...,:3]
        alpha=(.82*value**.8)[...,None]
        base=np.asarray(original,dtype=np.float32)/255*.65
        mixed=np.rint(np.clip(base*(1-alpha)+heat*alpha,0,1)*255).astype(np.uint8)
        overlay=Image.fromarray(mixed);overlay.save(folder/'ours_overlay.png')
        Image.fromarray(np.rint(heat*255).astype(np.uint8)).save(folder/'ours_heatmap.png')
        row.pop('gt_sha256',None);row.pop('fixation_sha256',None)
        (folder/'metrics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        preview=Image.new('RGB',(1304,530),'white');draw=ImageDraw.Draw(preview)
        preview.paste(original.resize((640,480)),(8,42));preview.paste(overlay.resize((640,480)),(656,42))
        draw.text((8,10),f"Original | COCO {row['image_id']} | {row['split']}",fill='black',font=font)
        draw.text((656,10),f"Ours | CC={row['metrics']['cc_float64']:.4f}",fill='black',font=font)
        preview.save(folder/'preview.png');sheet.paste(preview,(0,i*530))
    sheet.save(out/'00_preview_all.png')
    report['visualization']=dict(colormap='inferno',original_brightness=.65,opacity='.82 * (prediction / prediction.max()) ** .8',
        display_only=True,metrics_unchanged=True,gt_assets_included=False)
    (out/'metrics_all.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(src/'ATTRIBUTION_AND_REVIEW.md',out/'ATTRIBUTION_AND_REVIEW.md')
    (out/'01_README.md').write_text('# 预测叠加展示包（不含GT）\n\n'
        '先看00_preview_all.png；每图目录包含原始JPG、预测叠加PNG、纯预测热图、原始预测NPY、指标和双栏预览。\n'
        '本包无GT图、GT数组、fixation文件或含GT的旧预览，也未嵌套旧结果包。\n\n'
        '叠加采用inferno色系、65%亮度原图和随预测显著程度变化的透明度；所有图统一公式。'
        '按预测峰值缩放仅用于显示，没有修改模型输出或重新计算指标。\n\n'
        'CC/PCC、SIM、NSS、KLD、AUC、MAE为此前使用GT计算的原始指标，不代表无GT评估。'
        '只有715属于public-test，其余五张是训练样例，论文中请区分。\n'
        '署名来源见ATTRIBUTION_AND_REVIEW.md；使用叠加图请注明saliency overlay added。\n',encoding='utf-8')
    shutil.copy2(Path(__file__),out/'repackage_source.py')
    manifest={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file()}
    assert not any(Path(n).name.startswith(('gt_','fixation_')) for n in manifest)
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(out.with_suffix('.zip')) as z:
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    print(out.with_suffix('.zip'),sha(out.with_suffix('.zip')))


if __name__=='__main__':main()
