"""Render EuroSAT masks using the same shared asinh colorbar as CORe50.

The training archive stores physical phase in radians even though NPZ keys
retain the model parameter names ending in raw_phase. Do not apply sigmoid again.
"""
import hashlib, json, re, shutil, sys, tarfile, zipfile
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
if not (ROOT / 'completed').exists():
    ROOT = Path('C:/Users/Z/Documents/服务器代码/EuroSAT_Training_20260912')
sys.path.insert(0, str(ROOT.parent / '.codex_plot_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.cm import ScalarMappable
from PIL import Image

ARCHIVE = ROOT / 'completed/archive'
OUT = Path('D:/陈课题组/LGII_f2_code/图2d代码/EuroSAT_相位板非线性色标_20260914')
OUT.mkdir(parents=True, exist_ok=True)
def read_json(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p, data): p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

verification = read_json(ROOT / 'completed/LOCAL_VERIFICATION.json')
assert verification['passed']
tar_path = ROOT / 'completed/completed_results.tar.gz'
assert sha(tar_path) == verification['archive_sha256']
perf = read_json(ARCHIVE / 'moe_root/results/PERFORMANCE.json')
seal = read_json(ARCHIVE / 'moe_root/runs/SELECTION_SEAL.json')
architecture = read_json(ARCHIVE / 'moe_root/runs/architecture.json')
source_hashes = {}

A = .01
def forward(phi): return np.arcsinh((phi - np.pi) / A)
def inverse(value): return np.pi + A * np.sinh(value)
norm = colors.FuncNorm((forward, inverse), vmin=0, vmax=2*np.pi, clip=False)
linear_norm = colors.Normalize(0, 2*np.pi)
cmap = plt.get_cmap('twilight_shifted').copy()
cmap.set_bad('#D8DDE3')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path'})
probe = np.unique(np.r_[np.linspace(0,2*np.pi,1001), np.pi+np.array([-.1,-.01,-1e-6,0,1e-6,.01,.1])])
mapped = np.asarray(norm(probe))
assert np.all(np.diff(mapped)>0) and np.allclose(norm([0,np.pi,2*np.pi]),[0,.5,1],atol=1e-14)
inverse_error = float(np.max(np.abs(norm.inverse(mapped)-probe)))
assert inverse_error < 1e-12

maps = {}
plate_info = []
meta = {'dataset':'EuroSAT optical/SAR','seed':42,'source_sha256':seal['source_sha256'],
        'split_sha256':seal['split_sha256'],'original_archive_sha256':verification['archive_sha256'],
        'array_semantics':'physical phase in radians; raw_phase in keys is only the original parameter name',
        'checkpoint_verification':'Selected-stage final.json hashes match PERFORMANCE.json and SELECTION_SEAL.json. NPZ bytes match the verified training archive; remote PT files were not reread for this rendering.',
        'models':{}}
with tarfile.open(tar_path, 'r:gz') as tar:
    members = {m.name.lstrip('./'):m for m in tar.getmembers() if m.isfile()}
    def verify_member(p):
        rel = p.relative_to(ARCHIVE).as_posix()
        assert tar.extractfile(members[rel]).read() == p.read_bytes(), rel
        return sha(p)
    for model in ['moe','A_only','B_only','AB']:
        checkpoint = perf['models'][model]['checkpoint']
        for key in ['sha256','selected_stage','selected_epoch']:
            assert checkpoint[key] == seal['models'][model][key]
        stage = checkpoint['selected_stage']
        rel = Path('moe_root/runs/moe')/stage if model=='moe' else Path('d2nn_root/runs')/model/stage
        stage_dir = ARCHIVE/rel
        final = read_json(stage_dir/'final.json')
        verify_member(stage_dir/'final.json')
        assert final['checkpoint_sha256']==checkpoint['sha256']
        assert final['selected_epoch']==checkpoint['selected_epoch']
        path = stage_dir/'diagnostics/selected/phase_masks.npz'
        digest = verify_member(path)
        dest = OUT/'source_data'/model/'phase_masks.npz'
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        assert sha(dest)==digest
        source_hashes[dest.relative_to(OUT).as_posix()] = digest
        model_meta = {'checkpoint_path':checkpoint['path'],'checkpoint_sha256':checkpoint['sha256'],
                      'selected_stage':stage,'selected_epoch':checkpoint['selected_epoch'],
                      'array_source':str(path),'array_file':dest.relative_to(OUT).as_posix(),
                      'array_sha256':digest,'plates':[]}
        meta['models'][model] = model_meta
        maps[model] = {}
        with np.load(dest,allow_pickle=False) as z:
            for key in z.files:
                phase = z[key].copy()
                assert phase.ndim==2 and phase.dtype==np.float32 and np.isfinite(phase).all()
                assert phase.min()>=0 and phase.max()<=2*np.pi
                if key.endswith('raw_router_phase'): short='router'; title='路由相位板'; shape=(224,224)
                elif '.experts.' in key:
                    n=int(re.search(r'\.experts\.(\d+)\.',key).group(1))
                    short=f'E{n}'; title=f'专家 E{n} · {"A 光学" if n<2 else "B SAR"} 组'; shape=(224,224)
                elif '.global_phase.' in key:
                    short='global'; title='共享相位板' if model=='moe' else '第 2 层相位板'; shape=(478,478)
                else:
                    assert model!='moe' and '.expert_layers.0.phase.' in key
                    short='layer1'; title='第 1 层相位板'; shape=(478,478)
                assert phase.shape==shape and short not in maps[model]
                maps[model][short]={'phase':phase,'title':title}
                plate_dir=OUT/'individual_masks'/model
                plate_dir.mkdir(parents=True,exist_ok=True)
                rgba=cmap(norm(phase.astype(np.float64)),bytes=True)
                png=plate_dir/f'{short}_nonlinear_color.png'
                Image.fromarray(rgba).save(png)
                with Image.open(png) as im:
                    assert im.size==(phase.shape[1],phase.shape[0]) and np.array_equal(np.asarray(im),rgba)
                gray=np.rint(phase.astype(np.float64)/(2*np.pi)*65535).astype(np.uint16)
                gray_path=plate_dir/f'{short}_linear_phase_gray16.png'
                Image.fromarray(gray).save(gray_path)
                with Image.open(gray_path) as im: assert np.array_equal(np.asarray(im),gray)
                quantization_error=float(np.max(np.abs(gray.astype(np.float64)/65535*(2*np.pi)-phase)))
                assert quantization_error<=np.pi/65535+1e-12
                plate={'model':model,'plate':short,'key':key,'shape':list(shape),'dtype':str(phase.dtype),
                       'min_rad':float(phase.min()),'max_rad':float(phase.max()),
                       'mean_rad':float(phase.astype(np.float64).mean()),'std_rad':float(phase.astype(np.float64).std()),
                       'linear_gray16_max_quantization_error_rad':quantization_error,'phase_unchanged':True}
                plate_info.append(plate);model_meta['plates'].append(plate)
        assert len(maps[model])==(6 if model=='moe' else 2)
    for rel in ['moe_root/runs/SELECTION_SEAL.json','moe_root/runs/architecture.json',
                'moe_root/experiments/vision_transfer/engine.py','moe_root/train_eurosat.py']:
        p=ARCHIVE/rel
        digest=verify_member(p)
        target=OUT/'provenance'/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target)
        source_hashes[target.relative_to(OUT).as_posix()]=digest
assert len(plate_info)==12

# Physical aperture coordinates come from the archived architecture mapping.
# The 478-pixel active aperture starts at (20,20) inside the 518-pixel canvas.
expert_canvas=np.full((518,518),np.nan,dtype=np.float32)
global_canvas=np.full((518,518),np.nan,dtype=np.float32)
for item in architecture['moe']['spatial_mapping']:
    x0,y0,x1,y1=item['phase_xyxy']
    phase=maps['moe'][f"E{item['expert']}"]['phase']
    assert (y1-y0,x1-x0)==phase.shape
    expert_canvas[y0:y1,x0:x1]=phase
global_canvas[20:498,20:498]=maps['moe']['global']['phase']
assert np.isfinite(expert_canvas).sum()==4*224**2
assert np.isfinite(global_canvas).sum()==478**2
np.savez_compressed(OUT/'source_data/MoE_physical_layout.npz',expert_phase_rad=expert_canvas,
                    global_phase_rad=global_canvas,expert_trainable_mask=np.isfinite(expert_canvas),
                    global_trainable_mask=np.isfinite(global_canvas))
write_json(OUT/'source_data/PHASE_EXPORT_METADATA.json',meta)
write_json(OUT/'PHASE_STATISTICS.json',plate_info)
print('Verified 12 phase arrays and selected checkpoint provenance; rendering figures.',flush=True)

def draw(ax,phase,title,linear=False):
    ax.imshow(np.ma.masked_invalid(phase.astype(np.float64)),cmap=cmap,norm=linear_norm if linear else norm,interpolation="nearest",origin="upper")
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_title(title,fontsize=10,pad=10)
    for spine in ax.spines.values():spine.set_color("#B6C0CA")
def cb(fig,rect,linear=False):
    cax=fig.add_axes(rect)
    bar=fig.colorbar(ScalarMappable(norm=linear_norm if linear else norm,cmap=cmap),cax=cax)
    if linear:
        ticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]
        labels=["0","π/2","π","3π/2","2π"]
    else:
        ticks=[0,np.pi-1,np.pi-.1,np.pi-.01,np.pi,np.pi+.01,np.pi+.1,np.pi+1,2*np.pi]
        labels=["0","π−1","π−0.1","π−0.01","π","π+0.01","π+0.1","π+1","2π"]
    bar.set_ticks(ticks);bar.set_ticklabels(labels)
    bar.ax.tick_params(labelsize=9)
    bar.set_label("实际相位 φ（rad） · "+("线性" if linear else "asinh 非线性"),fontsize=10,labelpad=10)
def save(fig,name):
    for ext in ["png","svg"]:fig.savefig(OUT/f"{name}.{ext}",dpi=250,facecolor="white")
    plt.close(fig)
# Side-by-side with the same colormap and identical values: only the norm differs.
fig,axs=plt.subplots(2,2,figsize=(10.7,8.3))
fig.subplots_adjust(left=.06,right=.84,top=.85,bottom=.10,wspace=.72,hspace=.22)
for row,(model,short,title) in enumerate([("moe","E0","MoE 专家 E0"),("AB","layer1","D2NN A+B 第 1 层")]):
    p=maps[model][short]["phase"]
    draw(axs[row,0],p,title+" · 线性色标",True)
    draw(axs[row,1],p,title+" · asinh 非线性")
cb(fig,[.397,.20,.016,.51],True)
cb(fig,[.87,.20,.016,.51])
fig.suptitle("EuroSAT 光学/SAR 相位纹理 · 线性与非线性色标对比",fontsize=14,y=.96,fontweight="bold")
fig.text(.48,.905,"左右相位数值和配色完全一致，仅改变相位到颜色的位置映射",ha="center",fontsize=10,color="#566A7A")
fig.text(.49,.035,"非线性参数 a=0.01 rad；色标刻度表示真实相位，π附近的细小变化占用更多颜色范围。",ha="center",fontsize=9,color="#566A7A")
save(fig,"00_线性与非线性色标对比")
fig,axs=plt.subplots(1,3,figsize=(12.7,5.0))
fig.subplots_adjust(left=.025,right=.875,top=.78,bottom=.19,wspace=.14)
for ax,p,title in [(axs[0],maps["moe"]["router"]["phase"],"路由相位板 · 224×224"),(axs[1],expert_canvas,"第 1 特征相位板 · 4 个专家"),(axs[2],global_canvas,"第 2 特征相位板 · 478×478")]:draw(ax,p,title)
for item in architecture["moe"]["spatial_mapping"]:
    x0,y0,x1,y1=item["phase_xyxy"]
    axs[1].text(x0+5,y0+5,f"E{item['expert']} / {item['task']}",ha="left",va="top",fontsize=8,color="white",bbox={"facecolor":"black","alpha":.55,"pad":1,"edgecolor":"none"})
cb(fig,[.905,.24,.015,.48])
fig.suptitle("EuroSAT 光学/SAR · MoE 相位板空间布局（非线性色标）",fontsize=14,y=.965,fontweight="bold")
fig.text(.45,.88,"全部相位板共用以 π 为中心的 asinh 映射 · a=0.01 rad · 实际相位范围 0～2π",ha="center",fontsize=10,color="#566A7A")
fig.text(.47,.09,"灰色为无可训练相位像素；仅增强颜色对比，相位矩阵和像素位置保持不变。",ha="center",fontsize=9,color="#566A7A")
save(fig,"01_MoE相位板空间布局_非线性色标")
fig,axs=plt.subplots(2,3,figsize=(11.4,7.5))
fig.subplots_adjust(left=.035,right=.865,top=.83,bottom=.10,wspace=.17,hspace=.24)
for ax,short in zip(axs.flat,["E0","E1","router","E2","E3","global"]):
    entry=maps["moe"][short];n=entry["phase"].shape[0]
    draw(ax,entry["phase"],entry["title"]+f" · {n}×{n}")
cb(fig,[.905,.20,.018,.56])
fig.suptitle("EuroSAT 光学/SAR · MoE 全部相位 mask（非线性色标）",fontsize=14,y=.96,fontweight="bold")
fig.text(.45,.90,"同一测试检查点 · E0/E1 为 A 组，E2/E3 为 B 组 · 共用同一映射与刻度",ha="center",fontsize=10,color="#566A7A")
fig.text(.48,.035,"asinh，中心 π，a=0.01 rad；显示的是当前相位纹理，未计算训练前后差分。",ha="center",fontsize=9,color="#566A7A")
save(fig,"02_MoE全部相位mask_非线性色标")
fig,axs=plt.subplots(3,2,figsize=(9.5,11))
fig.subplots_adjust(left=.11,right=.815,top=.865,bottom=.075,wspace=.14,hspace=.25)
names={"A_only":"D2NN A-only","B_only":"D2NN B-only","AB":"D2NN A+B"}
for row,model in enumerate(["A_only","B_only","AB"]):
    for col,short in enumerate(["layer1","global"]):
        draw(axs[row,col],maps[model][short]["phase"],f"{names[model]} · 第 {col+1} 层 · 478×478")
    info=meta["models"][model]
    axs[row,0].set_ylabel(f"{info['selected_stage']} / 轮次 {info['selected_epoch']}",fontsize=9,labelpad=10,color="#566A7A")
cb(fig,[.865,.205,.023,.575])
fig.suptitle("EuroSAT 光学/SAR · D2NN 三组相位 mask（非线性色标）",fontsize=14,y=.96,fontweight="bold")
fig.text(.47,.915,"三组采用相同 asinh 映射：中心 π，a=0.01 rad；刻度仍为实际相位",ha="center",fontsize=10,color="#566A7A")
fig.text(.48,.025,"全部相位来自各模型最终测试使用的选定检查点；不对各图单独拉伸或做直方图均衡化。",ha="center",fontsize=9,color="#566A7A")
save(fig,"03_D2NN三组相位mask_非线性色标")

# Individual publication figures include their own real-phase colorbar.
individual_figures=[]
model_names={'moe':'MoE A+B','A_only':'D2NN A-only','B_only':'D2NN B-only','AB':'D2NN A+B'}
for model,entries in maps.items():
    for short,entry in entries.items():
        fig=plt.figure(figsize=(6.6,5.5))
        ax=fig.add_axes([.06,.17,.66,.68])
        draw(ax,entry['phase'],entry['title']+f" · {entry['phase'].shape[0]}×{entry['phase'].shape[1]}")
        cb(fig,[.77,.22,.026,.57])
        fig.suptitle('EuroSAT · '+model_names[model],fontsize=14,y=.965,fontweight='bold')
        fig.text(.47,.06,'asinh 非线性色标 · 中心 π · a=0.01 rad',ha='center',fontsize=10,color='#566A7A')
        p=OUT/'individual_masks'/model/f'{short}_with_colorbar.png'
        fig.savefig(p,dpi=300,facecolor='white')
        plt.close(fig)
        individual_figures.append(p.relative_to(OUT).as_posix())

normalization={'name':'asinh_centered_at_pi','a_rad':A,'vmin_rad':0,'vmax_rad':2*np.pi,
    'formula':'c(phi) = 0.5 * [1 + asinh((phi - pi)/a) / asinh(pi/a)]',
    'inverse':'phi(c) = pi + a*sinh((2*c-1)*asinh(pi/a))',
    'colormap':'twilight_shifted','shared_across_all_12_masks':True,
    'clipping':False,'per_image_equalization':False,'interpolation':'nearest',
    'colorbar_ticks':'actual phase in radians; nonuniform tick positions',
    'monotonic':True,'inverse_max_error_rad':inverse_error,'original_source_sha256':source_hashes}
write_json(OUT/'COLOR_NORMALIZATION.json',normalization)
readme="""# EuroSAT 光学/SAR 相位板：非线性色标可视化

数据来自 seed 42 实验最终测试所用的选定检查点。MoE：router 阶段第15轮；D2NN A-only：第69轮；B-only：第65轮；A+B：router 阶段第13轮。
A 为光学图像，B 为 SAR 图像。MoE 的 E0/E1 是 A 组，E2/E3 是 B 组，这表示训练分组，不代表已验证的域专化。

## 数据和显示

共有12份原始float32相位矩阵：MoE的路由板、4个专家板、共享板，以及3组D2NN各2层。
训练代码 export_visuals 已将参数转换成物理相位（rad）。NPZ键名虽然仍以raw_phase/raw_router_phase结尾，数组本身已经是相位，不要再次应用sigmoid。
原始NPZ逐字节复制，并与训练结果tar.gz中的同名文件对比。检查点哈希由阶段final.json、PERFORMANCE.json及SELECTION_SEAL.json交叉核对。本次绘图没有重新读取服务器上的PT文件。

沿用之前CORe50图的同一映射：以π为中心的asinh，a=0.01 rad，完整范围0～2π，周期配色twilight_shifted。
c(φ)=0.5×[1+asinh((φ−π)/a)/asinh(π/a)]。
colorbar标注实际相位rad，刻度间距非线性。全部12份mask共用同一函数、范围和配色，不逐图自动拉伸或做直方图均衡化。
仅改变相位到颜色的映射，原始相位矩阵和像素位置保持不变；无平滑或锐化。显示的是最终相位纹理，不是训练前后相位差分。

## 文件

- 00_线性与非线性色标对比：同一相位矩阵的并排对比。
- 01_MoE相位板空间布局：路由板、四专家拼接板、共享板；灰色表示没有可训练相位像素。
- 02_MoE全部相位mask：MoE全部6份相位矩阵。
- 03_D2NN三组相位mask：A-only、B-only、A+B各两层。
- 上述汇总图均提供PNG和SVG。
- individual_masks/：12张带colorbar的独立PNG、12张原生尺寸非线性彩色PNG、12张16位线性相位编码PNG。
- 16位灰度编码为round(φ/(2π)×65535)，没有应用非线性映射；其量化误差在VERIFICATION.json中记录。
- source_data/：4份原始NPZ、相位元数据、MoE拼接布局及可训练像素mask。图像的非线性颜色值不是器件的相位驱动灰度。
- provenance/：选定检查点记录、模型空间布局、原训练导出函数和本次重绘脚本。
- VERIFICATION.json、FILES_SHA256.json：数值检查和文件完整性校验。
"""
(OUT/'README.md').write_text(readme,encoding='utf-8')
check={'passed':True,'phase_matrices':12,'source_npz_files':4,
       'source_npz_bytes_equal_verified_training_archive':True,'checkpoint_hashes_match_stage_final_performance_and_seal':True,
       'phase_arrays_are_physical_radians_already':True,'no_second_sigmoid':True,
       'all_masks_share_same_nonlinear_mapping':True,'norm_roundtrip_max_error_rad':inverse_error,
       'normalization_a_rad':A,'source_sha256':meta['source_sha256'],
       'checkpoint_sha256':{m:i['checkpoint_sha256'] for m,i in meta['models'].items()},
       'remote_checkpoints_reread':False,'rendering_only':True,'retraining':False,
       'masks':plate_info,'individual_colorbar_figures':individual_figures}
write_json(OUT/'VERIFICATION.json',check)
target_script=OUT/'provenance/render_nonlinear_masks.py'
if Path(__file__).resolve()!=target_script.resolve():shutil.copyfile(Path(__file__),target_script)
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name not in ('FILES_SHA256.json','PACKAGE_SHA256.json'))
hashes={p.relative_to(OUT).as_posix():sha(p) for p in files}
write_json(OUT/'FILES_SHA256.json',hashes)
package=OUT/'EuroSAT_MoE与D2NN_相位mask_非线性色标.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[OUT/'FILES_SHA256.json']:z.write(p,p.relative_to(OUT).as_posix())
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    for rel,digest in hashes.items():assert hashlib.sha256(z.read(rel)).hexdigest()==digest
summary={'passed':True,'zip':str(package),'bytes':package.stat().st_size,'files':len(files)+1,
         'sha256':sha(package),'normalization_a_rad':A,'phase_matrices':12}
write_json(OUT/'PACKAGE_SHA256.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
