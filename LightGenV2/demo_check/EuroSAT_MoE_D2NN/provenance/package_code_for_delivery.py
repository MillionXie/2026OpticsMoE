"""Create a verified ZIP of the server-matched EuroSAT code and analysis tools."""
import ast
import hashlib
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
OUT=Path('D:/陈课题组/LGII_f2_code/图2d代码/EuroSAT_MoE_D2NN_代码包_20260915')
OUT.mkdir(parents=True,exist_ok=True)
def sha(data):return hashlib.sha256(data).hexdigest()
def jread(p):return json.loads(p.read_text(encoding='utf-8'))
seal=jread(ROOT/'SEALED_SOURCE.json')
manifest=jread(ROOT/'SOURCE_MANIFEST.json')
expected='1743b41517518e2a1b3869e0e884351eb3f9c40da9dde109cce4ddf5321b5641'
assert sha(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())==seal['source_sha256']==expected
assert len(manifest)==420
bundle=ROOT/'source_bundle.tar.gz'
assert sha(bundle.read_bytes())==seal['archive_sha256']
entries={}
python_count=0
with tarfile.open(bundle,'r:gz') as t:
    for member in t.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk()
        path=PurePosixPath(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        name=path.as_posix()
        assert name in manifest or name=='SOURCE_MANIFEST.json'
        data=t.extractfile(member).read()
        if name in manifest:
            assert sha(data)==manifest[name],name
            assert sha((ROOT/name).read_bytes())==manifest[name],name
        else:assert json.loads(data)==manifest
        if name.endswith('.py'):
            ast.parse(data.decode('utf-8'),filename=name)
            python_count+=1
        key='code/'+name
        assert key not in entries
        entries[key]=data
assert len(entries)==421

# Queue startup needs this unsealed completion marker in addition to the source manifest.
progress=jread(ROOT/'data_progress.json')
assert progress['state']=='complete' and progress['images']==53784
entries['code/data_progress.json']=(ROOT/'data_progress.json').read_bytes()
entries['provenance/SEALED_SOURCE.json']=(ROOT/'SEALED_SOURCE.json').read_bytes()
entries['provenance/LOCAL_RESULT_VERIFICATION.json']=(ROOT/'completed/LOCAL_VERIFICATION.json').read_bytes()
archive=ROOT/'completed/archive/moe_root'
for rel in ['runs/architecture.json','runs/SELECTION_SEAL.json','results/PERFORMANCE.json']:
    entries['reference_results/'+Path(rel).name]=(archive/rel).read_bytes()
entries['reference_results/ROUTING_TEST_VERIFICATION.json']=(ROOT/'completed/ROUTING_TEST_VERIFICATION.json').read_bytes()
entries['reference_results/EuroSAT性能报告.md']=(ROOT/'completed/EuroSAT性能报告.md').read_bytes()

analysis=['analyze_completed_results.py','verify_completed_results.py','plot_three_models_20260914.py',
          'render_nonlinear_phase_package_20260914.py','export_experiment_images_20260914.py',
          'render_ab_image_examples_20260914.py','verify_downloaded_image_package_20260914.py']
for name in analysis:
    data=(ROOT/name).read_bytes()
    ast.parse(data.decode('utf-8'),filename=name)
    entries['analysis_tools/'+name]=data
entries['analysis_tools/AB_EXAMPLE_SELECTION.json']=(ROOT/'AB_EXAMPLE_SELECTION.json').read_bytes()

readme='''# EuroSAT 光学/SAR：MoE 与 D2NN 实验代码包

本包保存本次 seed42 实验的原始训练代码，以及后续绘图、相位可视化和数据打包工具。
2026-09-15 已连接 connect.bjb1.seetacloud.com:25382 核对：420个封存文件全部存在，逐文件SHA256均与本地归档相同。
原服务器主机名：autodl-container-b61844b5bd-030a9ea3。

## 文件内容

- code/：420个原始封存文件、SOURCE_MANIFEST.json，以及队列读取所需的data_progress.json。
- code/experiments/：391个复用依赖文件，包含MoE、D2NN、冻结视觉前端、电子支路和分类头实现。
- code/PLAN.json、SPLIT.json、SPLIT_AUDIT.json：训练计划、53,784张图片的完整划分和空间分组审计。
- code/IMPLEMENTATION.md：架构、训练流程、预算和公平对比条件。
- code/licenses/、ATTRIBUTION.txt：数据来源、署名与许可快照。
- analysis_tools/：性能图、非线性相位mask、A/B配对样例和图片归档工具。工具依赖原实验结果文件，部分路径使用原Windows工作目录。
- reference_results/：四模型性能、架构记录、检查点选择记录及报告，不含PT权重。
- provenance/、PACKAGE_VERIFICATION.json、FILES_SHA256.json：封存、原结果复算及本次打包校验信息。

## 主要代码入口（相对code/）

| 用途 | 文件 |
|---|---|
| 顺序运行四模型的原实验队列 | run_eurosat.py |
| 训练、专家合并、D2NN AB检查点选择 | train_eurosat.py |
| 同一固定检查点的A/B评估与路由诊断 | evaluate_eurosat.py |
| 模型创建、损失与评估接口 | eurosat_runtime.py |
| 数据读取、56→224双三次处理、固定采样和增强种子 | eurosat_data.py |
| MoE专家合并与路由模式 | experiments/expert_merge/core.py |
| MoE/D2NN切换、D2NN的224→478双线性加载 | experiments/vision_transfer/model.py |
| 两个光电混合模块和自建十分类头 | experiments/vision_transfer/vision.py |
| 冻结的Qwen图像patch与位置嵌入前端 | experiments/vision_transfer/backbone.py |
| SAR编码与同地块对齐 | prepare_data.py |

## 运行环境与外部资源

原运行环境为Linux/CUDA，Python 3.11.16，解释器为/root/miniconda3/envs/opticsmoe/bin/python。
代码包含PyTorch、torchvision、transformers、safetensors、NumPy、Pillow、Matplotlib、PyYAML等依赖；数据准备还使用rasterio、pyproj、scipy和scikit-learn。
部分上游requirements文件保留在相应experiments目录中；本ZIP是源码归档，不是可直接部署到任意机器的容器或完整环境镜像。

原服务器路径：
- 主代码：/root/autodl-tmp/review/eurosat_expert_merge_20260912
- D2NN输出：/root/autodl-tmp/review/eurosat_d2nn_baselines_20260912
- 预处理图片：/root/autodl-tmp/data/eurosat_optical_sar
- Qwen缓存：/root/autodl-tmp/huggingface-cache

原队列入口为code/run_eurosat.py，需要已准备好的图片、IMAGE_MANIFEST.json、Qwen本地缓存和对应环境；检查点评估还需要选定的PT文件。
本包保留封存源码原样，配置中的绝对路径未改写。迁移机器或更改路径需要同步处理配置与封存校验，不能直接把修改后的代码视为原实验同一版本。
code/USER_AUTHORIZATION.json保留的是原实验授权记录。

## 数据与模型

MoE：四个224×224专家、224×224路由相位、478×478共享相位；D2NN：两层478×478相位。两者均有冻结Qwen视觉前端和自建十分类头，没有完整Qwen语言模型或视觉Transformer。
实验图片包与相位mask可视化已单独交付；本代码ZIP不重复包含数据集图片、训练PT权重或Qwen预训练权重。
同目录旁的下载资源说明可在EXTERNAL_ARTIFACTS.json中查看。
'''
entries['README_使用说明.md']=readme.encode('utf-8')
external={
    'dataset_zip':'D:/陈课题组/LGII_f2_code/图2d代码/EuroSAT_images_53784_20260914/EuroSAT_experiment_53784_images.zip',
    'phase_mask_zip':'D:/陈课题组/LGII_f2_code/图2d代码/EuroSAT_相位板非线性色标_20260914/EuroSAT_MoE与D2NN_相位mask_非线性色标.zip',
    'server_code':'/root/autodl-tmp/review/eurosat_expert_merge_20260912',
    'server_data':'/root/autodl-tmp/data/eurosat_optical_sar',
    'server_qwen_cache':'/root/autodl-tmp/huggingface-cache',
    'checkpoint_paths':{m:v['checkpoint']['path'] for m,v in jread(archive/'results/PERFORMANCE.json')['models'].items()}}
entries['EXTERNAL_ARTIFACTS.json']=json.dumps(external,ensure_ascii=False,indent=2).encode('utf-8')
check={'passed':True,'sealed_files':420,'sealed_python_files_parsed':python_count,
       'source_sha256':expected,'source_tar_sha256':seal['archive_sha256'],
       'sealed_source_files_byte_identical':True,'syntax_checks':'ast.parse only; original training was not rerun',
       'server_source_verified_on':'2026-09-15','training_source_modified':False,
       'additional_queue_marker':'code/data_progress.json','analysis_scripts':analysis,
       'includes_dataset_images':False,'includes_trained_checkpoint_weights':False,'includes_qwen_weights':False}
entries['PACKAGE_VERIFICATION.json']=json.dumps(check,ensure_ascii=False,indent=2).encode('utf-8')
entries['provenance/package_code_for_delivery.py']=Path(__file__).read_bytes()
hashes={name:sha(data) for name,data in entries.items()}
entries['FILES_SHA256.json']=json.dumps(hashes,ensure_ascii=False,indent=2).encode('utf-8')
package=OUT/'EuroSAT_MoE_D2NN_完整实验代码.zip'
with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,data in sorted(entries.items()):z.writestr('EuroSAT_MoE_D2NN/'+name,data)
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(entries)==len(set(z.namelist()))
    for name,data in entries.items():assert z.read('EuroSAT_MoE_D2NN/'+name)==data
(OUT/'README_使用说明.md').write_text(readme,encoding='utf-8')
(OUT/'PACKAGE_VERIFICATION.json').write_bytes(entries['PACKAGE_VERIFICATION.json'])
summary={'passed':True,'zip':str(package),'files':len(entries),'sealed_source_files':420,
         'bytes':package.stat().st_size,'sha256':sha(package.read_bytes())}
(OUT/'PACKAGE_SHA256.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
