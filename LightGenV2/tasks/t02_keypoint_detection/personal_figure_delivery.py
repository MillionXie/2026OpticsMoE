"""LightGen2-only figure handoff with auditable per-person metrics; no inference changes."""
import argparse,html,json,math,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .personal_prepare import sha256

EDGES=[(0,1),(1,2),(2,3),(3,4),(4,5),(6,7),(7,8),(8,9),(9,10),(10,11)]
JOINTS=['right_ankle','right_knee','right_hip','left_hip','left_knee','left_ankle','right_wrist','right_elbow','right_shoulder','left_shoulder','left_elbow','left_wrist']
GT_COLOR='#0072B2';MODEL_COLOR='#D55E00'

def measurements(points,prediction,box,offset=1.5):
    box=[math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])]
    pts=np.asarray(points,dtype=np.float32)[:12];gt=pts[:,:2].copy()
    gt[:,0]=(gt[:,0]-box[0])*(224/(box[2]-box[0]));gt[:,1]=(gt[:,1]-box[1])*(224/(box[3]-box[1]))
    valid=(pts[:,2]>0)&(gt>=0).all(1)&(gt<224).all(1)
    # Historic preview JSON stores argmax_index*4, whereas evaluation uses pixel centres.
    # For 56 -> 224: (index+.5)*4-.5 = historic_preview + 1.5.
    pred=np.asarray(prediction,dtype=np.float32)[:12]+offset
    scales=[float(np.linalg.norm(gt[a]-gt[b])) for a,b in [(8,3),(9,2)] if valid[a] and valid[b]]
    torso=float(np.mean(scales));errors=np.linalg.norm(pred-gt,axis=1);hits=(errors<=.2*torso)&valid
    metrics={'pck_at_0.2_torso':int(hits.sum())/int(valid.sum()),'correct_joints':int(hits.sum()),
        'evaluated_joints':int(valid.sum()),'mean_pixel_error_224':float(errors[valid].mean()),
        'normalized_mean_error_torso':float((errors[valid]/torso).mean()),'torso_scale_224':torso,
        'threshold_pixels_224':.2*torso,'PCKh':None,
        'joints':[{'name':JOINTS[j],'valid':bool(valid[j]),'error_px_224':float(errors[j]) if valid[j] else None,
                   'pck_hit':bool(hits[j]) if valid[j] else None} for j in range(12)]}
    return box,gt,pred,valid,metrics

def skeleton(image,xy,valid,color,cross=False):
    d=ImageDraw.Draw(image);xy=xy*(image.width/224)
    for a,b in EDGES:
        if valid[a] and valid[b]:d.line([tuple(xy[a]),tuple(xy[b])],fill=color,width=6)
    for (x,y),v in zip(xy,valid):
        if not v:continue
        if cross:d.line((x-9,y,x+9,y),fill=color,width=4);d.line((x,y-9,x,y+9),fill=color,width=4)
        else:d.ellipse((x-7,y-7,x+7,y+7),fill=color,outline='white',width=2)

def run(dataset,source_run,output,evaluated_rows=None):
    dataset=Path(dataset);source_run=Path(source_run);output=Path(output)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    annotations=load(dataset/'annotations_provisional.json');manifest=load(source_run/'run_manifest.json')
    report=load(source_run/'final_report.json');predictions=load(source_run/'after_images/predictions.json')['samples']
    if evaluated_rows:
        evaluated={}
        for r in load(Path(evaluated_rows)):evaluated.setdefault(r['sample_id'],{})[int(r['joint_index'])]=[r['pred_x'],r['pred_y']]
        for p in predictions:p['prediction_xy_224']=[evaluated[p['id']][j] for j in range(14)]
    if sha256(dataset/'annotations_provisional.json')!=manifest['data']['annotation_sha256']:raise ValueError('Annotation identity mismatch')
    lookup={r['id']+'_'+p['id']:(r,p) for r in annotations['images'] if r['split']=='test' for p in r['people'] if p['include']}
    if {p['id'] for p in predictions}!=set(lookup):raise ValueError('Prediction/test identity mismatch')
    records=[]
    for item in predictions:
        row,person=lookup[item['id']];box,gt,pred,valid,m=measurements(person['keypoints'],item['prediction_xy_224'],item['crop_box'],0 if evaluated_rows else 1.5)
        records.append((item,row,person,box,gt,pred,valid,m))
    total=sum(x[-1]['evaluated_joints'] for x in records);hits=sum(x[-1]['correct_joints'] for x in records)
    mean_error=sum(x[-1]['mean_pixel_error_224']*x[-1]['evaluated_joints'] for x in records)/total
    if hits/total!=report['after']['pck_at_0.2_torso'] or total!=report['after']['evaluated_joints'] or abs(mean_error-report['after']['mean_pixel_error'])>1e-4:
        raise ValueError(f'Per-person recomputation differs from source metrics: {hits}/{total}, mean={mean_error}')
    output.mkdir(parents=True,exist_ok=False)
    dump=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    sections=[];all_metrics=[]
    fontpath=Path('C:/Windows/Fonts/arial.ttf');font=ImageFont.truetype(str(fontpath),30) if fontpath.exists() else ImageFont.load_default()
    for item,row,person,box,gt,pred,valid,m in records:
        ident=item['id'];dest=output/ident;dest.mkdir()
        im=Image.open(dataset/row['image']).convert('RGB').crop(box).resize((224,224),Image.Resampling.BILINEAR)
        im=im.resize((896,896),Image.Resampling.LANCZOS)
        gt_im=im.copy();skeleton(gt_im,gt,valid,GT_COLOR)
        ours_im=im.copy();skeleton(ours_im,pred,np.ones(12,dtype=bool),MODEL_COLOR)
        overlay=im.copy();skeleton(overlay,gt,valid,GT_COLOR);skeleton(overlay,pred,np.ones(12,dtype=bool),MODEL_COLOR,True)
        panels=[im,gt_im,ours_im,overlay];names=['input','GT','LightGen2','overlay']
        for name,panel in zip(names,panels):panel.save(dest/(name+'.png'))
        sheet=Image.new('RGB',(896*4,950),'white');d=ImageDraw.Draw(sheet)
        for i,(panel,title) in enumerate(zip(panels,['Input','GT','LightGen2','GT + LightGen2'])):
            sheet.paste(panel,(i*896,54));d.text((i*896+20,10),title,fill='black',font=font)
        sheet.save(dest/'preview.png')
        m.update(sample_id=ident,original_sample_id=person.get('source_id',ident),capture_group=row['group_id'],
                 label_status='automatic RTMPose preannotation; NOT manually verified GT',result_type='simulation')
        dump(dest/'metrics.json',m);all_metrics.append(m)
        (dest/'metrics.md').write_text(f'# {ident}\n\nLightGen2 仿真；GT为自动预标注，非人工真值。\n\nPCK@0.2：{m["pck_at_0.2_torso"]:.4%}（{m["correct_joints"]}/{m["evaluated_joints"]}）\n\n平均定位误差：{m["mean_pixel_error_224"]:.4f} px（224输入坐标）\n\n躯干归一化误差：{m["normalized_mean_error_torso"]:.6f}\n',encoding='utf-8')
        dump(dest/'keypoints.json',{'joint_order':JOINTS,'gt_xy_224':gt.tolist(),'GT_valid':valid.tolist(),'LightGen2_xy_224':pred.tolist(),'crop_box_source_xyxy':box,'source_image':row['image']})
        sections.append(f'<section><h2>{html.escape(ident)} · PCK {m["pck_at_0.2_torso"]:.2%} · error {m["mean_pixel_error_224"]:.2f}px</h2><img loading="lazy" src="{ident}/preview.png"><p><a href="{ident}/metrics.md">样本指标</a></p></section>')
    summary={'model':'LightGen2','photos':len({x[1]['id'] for x in records}),'person_samples':len(records),
             'correct_joints':hits,'evaluated_joints':total,'pck_at_0.2_torso':hits/total,'mean_pixel_error_224':mean_error,
             'normalized_mean_error_torso':report['after']['normalized_mean_error_torso'],
             'PCKh':None,'GT_status':'automatic RTMPose preannotation, not human-reviewed',
             'test_used_for_selection':report['test_used_for_selection'],'result_type':'simulation',
             'checkpoint_sha256':report['checkpoint_sha256'],'source_run':source_run.name,
             'annotation_sha256':sha256(dataset/'annotations_provisional.json'),
             'prediction_sha256':sha256(Path(evaluated_rows) if evaluated_rows else source_run/'after_images/predictions.json'),
             'export_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'display_joints':'12 limbs; unreliable neck/head excluded from both GT and LightGen2 display',
             'coordinate_correction':'historic preview coordinate +1.5px to match original evaluator; metric recomputation verified'}
    dump(output/'SUMMARY.json',summary);dump(output/'SAMPLE_METRICS.json',all_metrics)
    (output/'SUMMARY.md').write_text(f'# LightGen2 总体性能\n\n54张测试照片，55个人体样本。仿真，非CCD实测。\n\n- PCK@0.2：**{hits/total:.4%}**（{hits}/{total}）\n- 平均定位误差：{mean_error:.4f} px（224坐标）\n- 躯干归一化误差：{summary["normalized_mean_error_torso"]:.6f}\n- PCKh：不适用，头颈未标真值\n\nGT仅自动预标注；测试集参与周期选模；用户筛选数据集，不代表原始全集或标准LSP性能。\n',encoding='utf-8')
    notice='GT 为自动预标注，尚未人工审核；这里只展示和评价12个四肢关节。蓝色圆点：GT；橙色十字：LightGen2（叠加图）。测试集参与选模，结果为筛选数据集上的仿真。'
    (output/'PREVIEW.html').write_text('<!doctype html><meta charset="utf-8"><title>LightGen2</title><style>body{font:16px system-ui;background:#f5f6f8;margin:24px}img{width:100%}section{background:white;padding:16px;margin:24px 0}p{line-height:1.6}</style><h1>LightGen2 · 54 photos / 55 persons</h1><p>'+notice+'</p><p>PCK@0.2: 83.33% · <a href="SUMMARY.md">总体指标</a></p>'+''.join(sections),encoding='utf-8')
    (output/'README.md').write_text('# 画图交付说明\n\n打开 PREVIEW.html 查看全部样本。SUMMARY.md/JSON是总指标；SAMPLE_METRICS.json是逐样本索引。\n\n每个人体一个photo_xxx_pxx目录：input.png、GT.png、LightGen2.png、overlay.png为独立无标题图，preview.png为四列组合；metrics.md/JSON为指标，keypoints.json可用于重绘。\n\n'+notice+'\n\n图片896×896由实际224×224模型输入放大，仅供排版，不代表模型分辨率提高。只做原模型人体裁剪，不增强照片或修改预测；所有测试人体固定顺序展示，未按分数筛选。旧预览坐标缺少热图中心偏移，本交付统一采用原评估器坐标并逐样本复算核对总体指标。\n',encoding='utf-8')
    dump(output/'FILES_SHA256.json',{str(p.relative_to(output)).replace('\\','/'):sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluated-rows',type=Path)
    a=p.parse_args();run(a.dataset,a.source_run,a.output,a.evaluated_rows)
