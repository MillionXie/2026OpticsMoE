"""Curated real-CCD LGVQ examples, never simulated Ours predictions.

select joins four frozen sources; fetch downloads videos read-only over SFTP;
render decodes four audited temporal landmarks and exports publication panels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
SOURCES = {
    'temporal_ours': TASK/'runs/hardware/temporal_full_20260914/results.json',
    'spatial_ours': TASK/'runs/hardware/spatial_train2250_20260914/readout_partial_test20/weighted_target5/train2250_plus_test112_holdout446/results.json',
    'temporal_qwen': REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/raw_remote/frozen_batch_20261013/qwen_batch01/predictions.csv',
    'spatial_qwen': REPO/'LightGenV2/reports/20260914_latest_qwen_baselines_a100_FIRSTBLOCK_FINAL/evidence/T06_spatial/per_video_predictions_and_timing.csv',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def indexed(rows,key):
    result={r[key]:r for r in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate sample identity')
    return result


def load_joined():
    import numpy as np
    from scipy.stats import spearmanr
    t=json.loads(SOURCES['temporal_ours'].read_text(encoding='utf-8'))
    s=json.loads(SOURCES['spatial_ours'].read_text(encoding='utf-8'))
    temporal=indexed(t['rows'],'video')
    spatial=indexed([r for r in s['rows'] if r['partition']!='original_train'],'video')
    qt=indexed(list(csv.DictReader(SOURCES['temporal_qwen'].open(encoding='utf-8-sig'))),'sample_id')
    qs=indexed(list(csv.DictReader(SOURCES['spatial_qwen'].open(encoding='utf-8-sig'))),'sample_id')
    if not len(temporal)==len(spatial)==len(qt)==len(qs)==558 or not set(temporal)==set(spatial)==set(qt)==set(qs):
        raise ValueError('Four sources must match exactly 558 unique test videos')
    rows=[]
    for sample in sorted(temporal):
        tr,sr=temporal[sample],spatial[sample]
        row={'sample_id':sample,'generator':sample.split('/')[0],'spatial_partition':sr['partition']}
        for metric,ours,qwen,target in (('temporal',tr['prediction'],qt[sample],tr['target']),('spatial',sr['after'],qs[sample],sr['target'])):
            if abs(float(qwen['target_mos'])-target)>.001:raise ValueError('MOS differs across sources')
            row.update({metric+'_true_mos':target,metric+'_ours_measured':ours,metric+'_qwen':float(qwen['prediction']),metric+'_ours_abs_error':abs(ours-target),metric+'_qwen_abs_error':abs(float(qwen['prediction'])-target)})
            row[metric+'_error_advantage']=row[metric+'_qwen_abs_error']-row[metric+'_ours_abs_error']
        p1=json.loads(qt[sample]['selected_frame_positions']);p2=json.loads(qs[sample]['selected_frame_positions'])
        if p1!=p2:raise ValueError('Temporal/Spatial baseline sampled different frames')
        row['selected_frame_positions']=p1
        rows.append(row)
    metrics={}
    for metric in ('temporal','spatial'):
        target=np.array([r[metric+'_true_mos'] for r in rows])
        metrics[metric]={}
        for method in ('ours_measured','qwen'):
            pred=np.array([r[metric+'_'+method] for r in rows]);metrics[metric][method]={'srcc':float(spearmanr(pred,target).statistic),'mae':float(np.abs(pred-target).mean()),'count':558}
    if abs(metrics['temporal']['ours_measured']['srcc']-.7977138739203681)>1e-9 or abs(metrics['spatial']['ours_measured']['srcc']-.6380783491502683)>1e-9:
        raise ValueError('Wrong measured model version')
    return rows,metrics


def select(root):
    if root.exists():raise FileExistsError('Refusing to overwrite output')
    rows,metrics=load_joined()
    eligible=[r for r in rows if r['spatial_partition']=='test_holdout'
              and min(r['temporal_error_advantage'],r['spatial_error_advantage'])>0
              and max(r['temporal_ours_abs_error'],r['spatial_ours_abs_error'])<=5]
    # Joint improvement in both tasks; avoid only selecting one generator.
    eligible.sort(key=lambda r:(min(r['temporal_error_advantage'],r['spatial_error_advantage']),-max(r['temporal_ours_abs_error'],r['spatial_ours_abs_error'])),reverse=True)
    chosen=[];counts={}
    for row in eligible:
        if counts.get(row['generator'],0)>=3:continue
        chosen.append(row);counts[row['generator']]=counts.get(row['generator'],0)+1
        if len(chosen)==10:break
    if len(chosen)!=10:raise ValueError('Not enough eligible examples; do not silently relax holdout rule')
    root.mkdir(parents=True)
    for i,row in enumerate(chosen,1):row['folder']=f'{i:02d}_'+hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8]
    write(root/'selection.json',{'selection_rule':'curated examples: spatial holdout only, Ours absolute MOS error <=5 and below Qwen in both tasks; rank by min joint error reduction; max3 per generator','eligible_count':len(eligible),'metrics_all558':metrics,'examples':chosen})
    write(root/'sources.json',{k:{'path':str(v),'sha256':sha(v)} for k,v in SOURCES.items()})
    with (root/'all558_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k not in ('folder','selected_frame_positions')]);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    print(json.dumps({'metrics':metrics,'eligible_count':len(eligible),'chosen':chosen},ensure_ascii=False,indent=2))


def fetch(root):
    import paramiko,shlex
    password=os.environ.get('LGVQ_SSH_PASSWORD')
    if not password:raise ValueError('Set LGVQ_SSH_PASSWORD; credential is never stored in outputs')
    chosen=json.loads((root/'selection.json').read_text(encoding='utf-8'))['examples']
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('202.120.62.181',port=24096,username='guest3',password=password,timeout=20)
    try:
        sftp=client.open_sftp();video_root=root/'source_videos';video_root.mkdir(exist_ok=True)
        for row in chosen:
            remote='/DATA/DATA1/lixinyue/xyli/data/LGVQ/videos/'+row['sample_id']
            local=video_root/(row['folder']+'.mp4')
            _,stdout,stderr=client.exec_command('sha256sum -- '+shlex.quote(remote));line=stdout.read().decode();error=stderr.read().decode()
            if not line:raise RuntimeError(error)
            expected=line.split()[0]
            if not local.exists():sftp.get(remote,str(local))
            if sha(local)!=expected:raise ValueError('Video transfer SHA mismatch')
            row.update(remote_video_path=remote,video_sha256=expected,video_bytes=local.stat().st_size)
            print('verified',row['folder'],local.stat().st_size,flush=True)
        write(root/'video_sources.json',chosen)
        sftp.close()
    finally:client.close()


def render(root):
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rows=json.loads((root/'video_sources.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'Arial','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    previews=[]
    for row in rows:
        out=root/row['folder']
        if out.exists() and any(out.iterdir()):
            if not (out/'scores.json').exists() or json.loads((out/'scores.json').read_text(encoding='utf-8'))['sample_id']!=row['sample_id']:
                raise FileExistsError('Existing example has a different identity')
        out.mkdir(exist_ok=True)
        video=root/'source_videos'/(row['folder']+'.mp4')
        cap=cv2.VideoCapture(str(video),cv2.CAP_MSMF if os.name=='nt' else cv2.CAP_ANY);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=[]
        if n<=0:raise ValueError('Video decoder could not read frame count')
        positions=[round((n-1)*(.1+i*.8/3)) for i in range(4)]
        baseline_positions=[round((n-1)*f) for f in (.1,.37,.63,.9)]
        if baseline_positions!=row['selected_frame_positions']:raise ValueError('Qwen record disagrees with source frame count')
        row.update(ours_frame_positions=positions,qwen_frame_positions=baseline_positions,
                   four_frame_sampling_identical=positions==baseline_positions)
        for i,pos in enumerate(positions,1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,pos);ok,bgr=cap.read()
            if not ok:raise RuntimeError('Cannot decode selected frame')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];side=max(2,round(min(h,w)*.65));top,left=(h-side)//2,(w-side)//2
            crop=cv2.resize(rgb[top:top+side,left:left+side],(448,448),interpolation=cv2.INTER_AREA)
            Image.fromarray(crop).save(out/f'frame_{i:02d}_input.png');Image.fromarray(rgb).save(out/f'frame_{i:02d}_original.png');frames.append(crop)
        if positions!=baseline_positions:
            alternate=out/'qwen_sampling';alternate.mkdir(exist_ok=True)
            for i,pos in enumerate(baseline_positions,1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,pos);ok,bgr=cap.read()
                if not ok:raise RuntimeError('Cannot decode Qwen frame')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];side=max(2,round(min(h,w)*.65));top,left=(h-side)//2,(w-side)//2
                Image.fromarray(cv2.resize(rgb[top:top+side,left:left+side],(448,448),interpolation=cv2.INTER_AREA)).save(alternate/f'frame_{i:02d}_input.png')
        cap.release();write(out/'scores.json',row)
        with (out/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['metric','true_MOS','ours_measured','Qwen_baseline','ours_absolute_error','Qwen_absolute_error'])
            for metric in ('temporal','spatial'):writer.writerow([metric]+[row[metric+'_'+key] for key in ('true_mos','ours_measured','qwen','ours_abs_error','qwen_abs_error')])
        fig=plt.figure(figsize=(8,5.2));grid=fig.add_gridspec(2,4,height_ratios=[1,1.2],hspace=.35,wspace=.025)
        for i,frame in enumerate(frames):
            ax=fig.add_subplot(grid[0,i]);ax.imshow(frame);ax.axis('off');ax.set_title(('10%','37%','63%','90%')[i],fontsize=10)
        ax=fig.add_subplot(grid[1,:]);colors=['#a7bbdf','#ddb8ba','#e3bd80']
        for j,metric in enumerate(('temporal','spatial')):
            for k,key in enumerate(('true_mos','ours_measured','qwen')):
                value=row[metric+'_'+key];y=(1-j)*1.3+.25-.25*k
                ax.barh(y,value,height=.2,color=colors[k],label=('True MOS','LightGen-II (measured)','Qwen baseline')[k] if j==0 else None)
                ax.text(value+1,y,f'{value:.1f}',va='center',fontsize=10)
        ax.set_yticks([1.3,0],['Temporal','Spatial']);ax.set_xlim(0,100);ax.set_xlabel('Quality score (MOS)');ax.set_ylim(-.5,1.8)
        ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,frameon=False,fontsize=9)
        fig.subplots_adjust(left=.12,right=.96,top=.92,bottom=.2)
        fig.suptitle(f"Example {row['folder'][:2]} | {row['generator']}",fontsize=11,y=.985)
        for ext in ('png','pdf','svg'):fig.savefig(out/('comparison.'+ext),dpi=220)
        plt.close(fig);previews.append(out/'comparison.png')
        text=f"视频：{row['sample_id']}\nOurs四帧：10%、36.67%、63.33%、90%，实际帧索引{positions}。图标题37%/63%为四舍五入。\nQwen四帧索引{baseline_positions}，使用10%、37%、63%、90%。若整数索引不一致，qwen_sampling含其实际输入图，不将两者宣称为完全相同的帧。\ninput图：与模型一致的65%短边中心裁剪及448×448缩放；original图：原始帧。无亮度增强。\nOurs为六层实测CCD进入网络的分数，不是仿真。空间样本属于未参与部署微调的test_holdout。\n这是按两任务绝对MOS误差优于Qwen筛选的定性示例，不代表全测试集。\n"
        (out/'说明.txt').write_text(text,encoding='utf-8')
    contact=Image.new('RGB',(1600,520*5),'white')
    for i,path in enumerate(previews):
        im=Image.open(path).convert('RGB');im.thumbnail((800,520));contact.paste(im,((i%2)*800,(i//2)*520))
    contact.save(root/'10_examples_overview.jpg',quality=94)
    columns=['folder','sample_id','spatial_partition']+[metric+'_'+key for metric in ('temporal','spatial') for key in ('true_mos','ours_measured','qwen','ours_abs_error','qwen_abs_error')]
    with (root/'selected10_scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in rows)
    metadata=json.loads((root/'selection.json').read_text(encoding='utf-8'));metrics=metadata['metrics_all558']
    (root/'说明.txt').write_text('10个定性精选示例，按两任务绝对MOS误差优于Qwen且Ours误差均不超过5分筛选，空间样本均来自未参与适配的446条留出视频。不能用精选样本均值替代总体指标。\n时间实测SRCC=0.7977138739；空间完整558混合SRCC=0.6380783492，其中112条参与适配，446条留出SRCC=0.6092739512。\nQwen时间baseline SRCC='+str(metrics['temporal']['qwen']['srcc'])+'；空间baseline SRCC='+str(metrics['spatial']['qwen']['srcc'])+'。整体空间Qwen优于Ours，示例只表明局部优势。\n每个视频文件夹包含四帧模型输入图、四帧原始图、原始分数JSON/CSV以及PNG/PDF/SVG对照图。三组柱状图统一MOS 0–100，不截断坐标。\nOurs采样10/36.67/63.33/90%，Qwen采样10/37/63/90%；若取整后索引不同，额外Qwen图在qwen_sampling，scores.json记录双方索引。\n优先展示建议：01麦田、02薰衣草、04蒲公英路边、06服装设计师、07奶昔。05网球视频主体缺失，适合展示低质量失败画面而非美观示例。\n',encoding='utf-8')
    write(root/'SHA256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='SHA256.json'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('select','fetch','render'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve()
    if TASK/'runs'/'hardware' not in root.parents:raise ValueError('Output belongs under T06/runs/hardware')
    {'select':select,'fetch':fetch,'render':render}[a.phase](root)


if __name__=='__main__':main()
