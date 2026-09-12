"""Invert the hash-pinned, previously exported MNIST v2 phase, without resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
SOURCE='post_robust_best_epoch012_1920x1200.bmp'
SOURCE_SHA='9668b759717673096ed874dc46c451ec13e0ea88e73ba47827fa302e13f43dae'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,default=ROOT/'assets/mnist_v2_original')
    p.add_argument('--out',type=Path,default=ROOT/'generated/phase_inverted/mnist_v2')
    args=p.parse_args()
    source=args.source_dir/SOURCE
    if sha(source)!=SOURCE_SHA:raise ValueError('Source BMP hash does not match original MNIST v2 export')
    metadata=json.loads((args.source_dir/'mask_candidates.json').read_text(encoding='utf-8'))
    candidate=next(x for x in metadata['candidates'] if x['name']=='post_robust_best')
    if candidate['sha256']!=SOURCE_SHA:raise ValueError('Candidate metadata disagrees with source')
    with Image.open(source) as im:
        if im.format!='BMP' or im.mode!='L' or im.size!=(1920,1200):
            raise ValueError('Expected original 1920x1200 8-bit grayscale BMP')
        normal=np.array(im)
    inverted=255-normal
    args.out.mkdir(parents=True,exist_ok=False)
    output=args.out/'mnist4_v2_post_robust_best_epoch012_1920x1200_inverted.bmp'
    Image.fromarray(inverted).save(output)
    with Image.open(output) as im:
        restored=np.array(im)
        assert im.format=='BMP' and im.mode=='L' and im.size==(1920,1200)
        assert np.array_equal(normal.astype(np.uint16)+restored.astype(np.uint16),np.full(normal.shape,255,np.uint16))
    manifest=dict(source_file=str(source.resolve()),source_sha256=SOURCE_SHA,source_candidate=candidate,
        file=output.name,sha256=sha(output),size_wh=[1920,1200],mode='L',
        transformation='uint8 output[y,x] = 255 - original[y,x], including background',
        additional_spatial_flip=False,resampling=False,
        original_export_flip_vertical=metadata['phase_flip_vertical'],
        original_export_flip_horizontal=metadata['phase_flip_horizontal'],
        preserved_phase_center_xy=metadata['phase_slm_center_xy'],
        verified_pixels=int(normal.size),original_bmp_preserved=True)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.out/'README.md').write_text(
        '# MNIST-4 v2 已训练相位（本实验室反向 LUT）\n\n'
        f'加载 `{output.name}`：1920×1200，8-bit 灰度 BMP。\n\n'
        '来自 post_robust_best，epoch 12；原记录 validation=88.1212%，不是本次硬件测试准确率。\n'
        '只进行 g_new=255-g_old；全画面反灰度，包括背景。未缩放、未平移、未额外上下/左右翻转。\n'
        '旧导出已经包含垂直翻转，本文件保留该空间方向，不再翻第二次。\n'
        '沿用原中心(980,590)、有效矩形[x0,y0,x1,y1)=[472,82,1488,1098)，不是新ABO中心。\n'
        '相位面板8μm；旧逻辑相位478×478、17μm对应约1016×1016面板像素。\n'
        '只提供MNIST相位文件，不能用ABO的输入、ROI或分类器直接替代配套MNIST实验。\n'
        '正常LUT的面板请用assets/mnist_v2_original中的原图，不要加载此反向版本。\n'
        '原文件与输出SHA、checkpoint来源、逐像素校验记录见manifest.json。\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
