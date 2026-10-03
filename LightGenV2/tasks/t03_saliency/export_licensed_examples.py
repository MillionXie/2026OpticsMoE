"""Export six supplied license-reviewed SALICON examples using pinned 0.8625 weights.

This is figure preparation, not a new test-set evaluation or legal review.
The release runtime is imported explicitly to preserve the accepted model.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--license-zip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();root=a.release.resolve();out=a.output.resolve()
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError(out)
    sys.path.insert(0,str(root/'runtime'))
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from LightGenV2.tasks.t03_saliency.lab_runtime import CHECKPOINT_SHA,load_model
    from LightGenV2.tasks.t03_saliency.settings import load_settings
    from LightGenV2.tasks.t03_saliency.modeling import load_vision_backbone,build_student
    from LightGenV2.tasks.t03_saliency.reproduce_baseline import independent_cc
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import SALICONRecord,SALICONSaliencyDataset
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.modeling import preprocess_vision
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import density_from_logits,SaliencyAccumulator
    torch.set_num_threads(4);torch.manual_seed(42)
    checkpoint=root/'weights/best_checkpoint.pt'
    if digest(checkpoint)!=CHECKPOINT_SHA:raise ValueError('Wrong checkpoint')
    resolved=json.loads((root/'settings.json').read_text())
    settings=load_settings(resolved['config_path'])
    for key,value in resolved.items():
        if hasattr(settings,key):setattr(settings,key,Path(value) if isinstance(getattr(settings,key),Path) and value is not None else value)
    settings.local_files_only=True;settings.download=False
    z=zipfile.ZipFile(a.license_zip);prefix='salicon_ccby2_candidates/'
    candidates=json.loads(z.read(prefix+'metadata/recommended_images.json'))
    if {r['image_id'] for r in candidates}!={715,508,450,6730,292271,562382}:raise ValueError('Review changed selection')
    out.mkdir(parents=True)
    shutil.copy2(a.license_zip,out/'license_review_original.zip')
    (out/'ATTRIBUTION_AND_REVIEW.md').write_bytes(z.read(prefix+'README.md'))
    (out/'source_export.py').write_bytes(Path(__file__).read_bytes())
    loaded=load_vision_backbone(settings,torch.device(a.device))
    model=build_student(loaded,settings)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.core.load_state_dict(payload['core'],strict=True);model.head.load_state_dict(payload['saliency_head'],strict=True)
    model.core.set_phase_dropout_active(False);model.eval()
    release=json.loads((root/'release.json').read_text())
    cached_items={r['sample_id']:r for r in release['fields']}
    rows=[];panels=[]
    with torch.inference_mode():
        for index,r in enumerate(candidates):
            iid=r['image_id'];split='validation' if '_val2014_' in r['file_name'] else 'train'
            folder=out/f'{iid:012d}_{split}';folder.mkdir()
            raw=z.read(prefix+'images/'+r['file_name'])
            if hashlib.sha256(raw).hexdigest()!=r['image_sha256']:raise ValueError('Image identity mismatch')
            original=folder/'original.jpg';original.write_bytes(raw)
            maps=Path(settings.artifact_cache_dir)/'prepared_maps'/split
            density=maps/f'{iid:012d}_density.png';fixation=maps/f'{iid:012d}_fixation.png'
            record=SALICONRecord(index,split,iid,original,density,fixation)
            sample=SALICONSaliencyDataset([record],settings,training=False)[0]
            batch=preprocess_vision(loaded.processor,[sample['image']],loaded.device)
            logits=model(batch['pixel_values'],batch['image_grid_thw'])[0]
            gt=sample['density'][None].to(a.device);fix=sample['fixation'][None].to(a.device)
            pred=density_from_logits(logits);acc=SaliencyAccumulator();acc.update(logits,gt,fix)
            metrics=acc.compute();metrics['cc_float64']=float(independent_cc(pred.cpu().numpy(),gt.cpu().numpy())[0])
            if record.sample_id in cached_items:
                metrics['original_test_reference_cc']=cached_items[record.sample_id]['simulation_cc']
                if abs(metrics['cc_float64']-metrics['original_test_reference_cc'])>2e-4:
                    raise ValueError('Test example does not reproduce original reference')
            row=dict(**r,split=split,sample_id=record.sample_id,metrics=metrics,
                     gt_sha256=digest(density),fixation_sha256=digest(fixation),directory=folder.name)
            rows.append(row);(folder/'metrics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
            shutil.copy2(density,folder/'gt_source_224.png');shutil.copy2(fixation,folder/'fixation_source_224.png')
            sample['image'].save(folder/'model_input_224.png')
            rgb=Image.open(io.BytesIO(raw)).convert('RGB');views=[]
            arrays=[gt.cpu().numpy().squeeze(),pred.cpu().numpy().squeeze()]
            # One shared display scale per image pair. No fitting, smoothing,
            # registration, gamma or separate per-map contrast manipulation.
            vmax=max(float(x.max()) for x in arrays)
            for label,value in zip(['gt','ours'],arrays):
                np.save(folder/(label+'_density_224.npy'),value)
                display=np.clip(value/vmax,0,1)
                gray=Image.fromarray(np.rint(display*255).astype(np.uint8))
                gray.save(folder/(label+'_gray_224.png'))
                gray.resize(rgb.size,Image.Resampling.BILINEAR).save(folder/(label+'_gray_original_size.png'))
                color=Image.fromarray(np.rint(plt.get_cmap('turbo')(display)[...,:3]*255).astype(np.uint8))
                color.save(folder/(label+'_heatmap_224.png'))
                color=color.resize(rgb.size,Image.Resampling.BILINEAR)
                color.save(folder/(label+'_heatmap_original_size.png'))
                Image.blend(rgb,color,.45).save(folder/(label+'_overlay.png'))
                views.append(color)
            fig,axes=plt.subplots(1,3,figsize=(12,3.8))
            for ax,im,title in zip(axes,[rgb,*views],['Original','GT','Ours']):ax.imshow(im);ax.set_title(title);ax.axis('off')
            fig.suptitle(f"COCO {iid} | {split} | CC {metrics['cc_float64']:.4f} | SIM {metrics['sim']:.4f} | NSS {metrics['nss']:.4f}")
            fig.tight_layout();fig.savefig(folder/'preview.png',dpi=170);plt.close(fig)
            panels.append((rgb,*views,iid,split,metrics['cc_float64']))
            print('EXPORTED',iid,split,metrics,flush=True)
    model.restore_native()
    fig,axes=plt.subplots(len(panels),3,figsize=(12,3.4*len(panels)))
    for rr,(original,gt,pred,iid,split,cc) in enumerate(panels):
        for ax,im,title in zip(axes[rr],[original,gt,pred],['Original','GT','Ours']):
            ax.imshow(im);ax.axis('off');ax.set_title(f'{title} | {iid} | {split}'+(f' | CC={cc:.4f}' if title=='Ours' else ''))
    fig.tight_layout();fig.savefig(out/'00_preview_all.png',dpi=160);plt.close(fig)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent,text=True).strip()
    report=dict(checkpoint_sha256=CHECKPOINT_SHA,full_test_reference_cc=release['simulation_cc_float64'],
        export_source_commit=commit,model_runtime_commit=release['source_commit'],license_zip_sha256=digest(a.license_zip),
        command=sys.argv,device=a.device,torch=torch.__version__,rows=rows)
    (out/'metrics_all.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    table='\n'.join(f"|{r['image_id']}|{r['split']}|{r['metrics']['cc_float64']:.6f}|{r['metrics']['sim']:.6f}|{r['metrics']['nss']:.6f}|{r['metrics']['kld']:.6f}|{r['metrics']['auc_judd']:.6f}|{r['metrics']['mae']:.6f}|" for r in rows)
    (out/'01_README.md').write_text('# SALICON 0.8625：论文插图候选\n\n先看00_preview_all.png，然后每图文件夹preview.png。各文件夹保留原始JPG、GT、预测、热图、叠加图与原始浮点NPY。\n\n'
        '**仅715属于本项目public-test，另外五张属于训练集，必须标注为training examples，不能用六图均值冒充测试成绩。没有重新训练或挑选最高指标图片。**\n\n'
        '0.8624925是该权重完整5000图测试CC，不是每张图的保证分数。以下为本次逐图推理。CC为float64 Pearson；其他指标沿用项目原实现。均在224×224概率密度图上计算，AUC无jitter；MAE按项目各图峰值归一化口径。\n\n'
        '|COCO ID|划分|CC↑|SIM↑|NSS↑|KLD↓|AUC↑|MAE↓|\n|---|---|---|---|---|---|---|---|\n'+table+
        '\n\nGT为原协议fixation投影与高斯密度构建后的224图，并非新生成标签。预览的GT/预测共用两者最大值作线性显示，turbo色系；原尺寸结果仅双线性放大方便排版，不用于指标。没有裁剪、平移或调图提高指标。\n\n'
        '版权信息沿用同学核验包，不构成新的法律保证；请保留照片作者、标题、来源和许可，GT来源另行署名，叠加/缩放应说明改动。详见ATTRIBUTION_AND_REVIEW.md和原核验ZIP。\n',encoding='utf-8')
    manifest={f.relative_to(out).as_posix():digest(f) for f in out.rglob('*') if f.is_file()}
    (out/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as bundle:
        for f in out.rglob('*'):
            if f.is_file():bundle.write(f,f.relative_to(out))
    print('ZIP',out.with_suffix('.zip'),'SHA256',digest(out.with_suffix('.zip')),flush=True)


if __name__=='__main__':main()
