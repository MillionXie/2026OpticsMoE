"""Show six actual optical/SAR test pairs without per-image enhancement."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'.codex_plot_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

OUT=Path(json.loads((ROOT/'IMAGE_EXPORT_LOCAL.json').read_text(encoding='utf-8'))['directory'])
EXAMPLES=OUT/'examples'
manifest=json.loads((OUT/'IMAGE_MANIFEST.json').read_text(encoding='utf-8'))
selection=json.loads((ROOT/'AB_EXAMPLE_SELECTION.json').read_text(encoding='utf-8'))
names={'AnnualCrop':'一年生作物','Forest':'森林','Industrial':'工业区','Residential':'居民区','River':'河流','SeaLake':'海洋 / 湖泊'}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
fig,axes=plt.subplots(2,6,figsize=(15.5,6.2))
fig.subplots_adjust(left=.08,right=.99,top=.77,bottom=.18,wspace=.09,hspace=.22)
sources=[]
for col,pair in enumerate(selection['pairs']):
    assert pair['A']['pair_id']==pair['B']['pair_id'] and pair['A']['split']==pair['B']['split']=='test'
    for row,domain in enumerate(('A','B')):
        record=pair[domain]
        path=EXAMPLES/record['path']
        raw=path.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==manifest[record['path']]['sha256']
        with Image.open(path) as im:
            assert im.size==(56,56) and im.mode=='RGB'
            assert hashlib.sha256(im.tobytes()).hexdigest()==manifest[record['path']]['pixel_sha256']
            pixels=np.asarray(im).copy()
        ax=axes[row,col]
        ax.imshow(pixels,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_color('#b7c6d2')
        if row==0:ax.set_title(names[pair['class']],fontsize=12,pad=12)
        else:ax.set_xlabel(record['pair_id'],fontsize=9,labelpad=9,color='#4c5c6d')
        sources.append({**record,**manifest[record['path']],'archive_image_path':record['path'],
                        'local_image':str(path),'original_size':[56,56],'display_interpolation':'nearest'})
axes[0,0].set_ylabel('A 任务\n光学 RGB',fontsize=12,labelpad=16,rotation=0,ha='right',va='center')
axes[1,0].set_ylabel('B 任务\nSAR 编码',fontsize=12,labelpad=16,rotation=0,ha='right',va='center')
fig.suptitle('EuroSAT · A 光学 / B SAR 同地块配对示意',fontsize=18,fontweight='bold',y=.965)
fig.text(.54,.88,'每列为同一地块、同一类别的两种观测；样例均来自本次实验测试集',ha='center',fontsize=11,color='#536b7b')
fig.text(.53,.075,'B 为训练使用的 VV / VH 三通道编码（伪彩），不代表自然颜色；原始图像为 56×56，按最近邻放大。',ha='center',fontsize=10,color='#536b7b')
fig.text(.53,.032,'样例按类别中的地块编号顺序选取；光学与 SAR 的采集时间可能不同。',ha='center',fontsize=10,color='#536b7b')
stem='EuroSAT_A光学_B_SAR_同地块配对示意'
for ext in ('png','svg'):fig.savefig(EXAMPLES/f'{stem}.{ext}',dpi=300,facecolor='white')
plt.close(fig)
report={'dataset':'EuroSAT optical/SAR experiment','pairs':6,'images':12,'all_file_and_pixel_sha256_verified':True,
        'selection_policy':selection['selection_policy'],'no_per_image_contrast_or_color_transform':True,'sources':sources}
(EXAMPLES/'SOURCES.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
(EXAMPLES/'README.md').write_text('''# EuroSAT A/B 配对样例

6个类别、6对同地块图像，共12张。A为光学RGB；B为模型实际使用的SAR三通道编码：(VV,VH,(VV+VH)/2)，不是自然光照片。
所有样例来自本次实验测试集，每个指定类别按地块编号从小到大取第一对，没有根据模型预测、性能或路由偏好筛选。
相同地块、相同类别，不保证同一时刻采集。SAR已按实验预处理对齐到对应光学图像的中心56×56区域。
图片未单独增强对比度、伪造纹理或改变色彩；汇总图只做最近邻放大，images/下保留原生56×56 PNG。
每张图片的文件与像素SHA256均与训练IMAGE_MANIFEST.json核对一致，详见SOURCES.json。
完整作者署名及许可见ATTRIBUTION.txt和licenses/。
''',encoding='utf-8')
shutil.copyfile(ROOT/'ATTRIBUTION.txt',EXAMPLES/'ATTRIBUTION.txt')
for p in (ROOT/'licenses').iterdir():
    if p.is_file():
        (EXAMPLES/'licenses').mkdir(exist_ok=True)
        shutil.copyfile(p,EXAMPLES/'licenses'/p.name)
shutil.copyfile(Path(__file__),EXAMPLES/'render_examples.py')
package=OUT/'EuroSAT_AB配对样例与示意图.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(EXAMPLES.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(EXAMPLES).as_posix())
with zipfile.ZipFile(package) as z:assert z.testzip() is None
print(json.dumps({'preview':str(EXAMPLES/f'{stem}.png'),'zip':str(package),'bytes':package.stat().st_size,'verified_images':len(sources)},ensure_ascii=False,indent=2))
