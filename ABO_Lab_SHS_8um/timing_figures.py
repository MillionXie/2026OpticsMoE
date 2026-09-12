"""Regenerate timing evidence figures from unchanged diagnostic JSON/PNG files.

No device access, no enhancement in metrics, no generated/AI image content.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    out=ROOT/'reports/timing_20260912';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/digit_timing_20260912';d=read(source/'report.json')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11})
    keys=['bmp_validate_ms','slm_preload_ms','slm_show_to_visible_ms','settle_actual_ms','final_fresh_ms']
    labels=['BMP检查','SLM预加载','请求显示 → Visible','持续取走过渡帧','丢6帧 + 留1帧']
    vals=[d['timing_ms'][k]['mean'] for k in keys]
    fig,ax=plt.subplots(figsize=(12,3.4))
    fig.subplots_adjust(left=.04,right=.99,top=.82,bottom=.38)
    colors=['#8092a6','#77aadd','#225588','#eeaa44','#55aa99'];left=0
    for i,(v,label,col) in enumerate(zip(vals,labels,colors)):
        ax.barh(0,v,left=left,color=col,height=.4,label=f'{label}：{v:.2f} ms')
        if v>30:ax.text(left+v/2,0,f'{v:.2f} ms',ha='center',va='center')
        left+=v
    ax.set(xlim=(0,300),ylim=(-.5,.5),yticks=[],xlabel='从提交已准备好的 BMP 开始（ms）',
           title=f'当前 Holoeye + SHS 软件周期：均值 {sum(vals):.2f} ms，约 {d["raw_ready_cycles_per_s"]:.2f} 次/秒')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.46),ncol=3,frameon=False)
    fig.savefig(out/'01_cycle_timeline.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
    for i in range(4):
        a=np.array(Image.open(source/f'{i:02d}_digit{i}.png'))
        for row in range(2):
            # Same display limits for every image; raw bytes remain untouched.
            axes[row,i].imshow(a,cmap='gray',vmin=0,vmax=255 if row==0 else 85)
            axes[row,i].set_title(f'输入 {i} / 匹配 {d["rows"][i]["predicted_reference_digit"]}')
            axes[row,i].axis('off')
    fig.suptitle('数字切换证据（相位保持全黑）\n上排：原始灰度0–255；下排：统一×3线性显示，仅辅助观察，不用于PCC')
    fig.savefig(out/'02_digit_captures.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,3.5),layout='constrained')
    for i in range(4):
        idx=[j for j,r in enumerate(d['rows']) if r['digit']==i]
        ax.plot([j+1 for j in idx],[d['rows'][j]['same_pcc'] for j in idx],'o-',label=f'数字{i}')
    ax.set(xlabel='切换序号',ylabel='与独立400 ms参考光场的PCC',ylim=(.97,1.002),xticks=range(1,21),
           title='200 ms持续排空 + 最后取帧：20/20匹配正确；无饱和像素')
    ax.legend(ncol=4);ax.grid(alpha=.2)
    fig.savefig(out/'03_digit_pcc.png',dpi=170);plt.close(fig)
    evidence={'digit_test':{'source_commit':d['source_commit'],'report_relative_path':'results/digit_timing_20260912/report.json',
                           'report_sha256':hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
                           'correct':d['correct_count'],'n':d['measured_frames'],'minimum_pcc':d['minimum_same_pcc'],
                           'cycles_per_s':d['raw_ready_cycles_per_s'],'timing_ms':d['timing_ms']},'camera_benchmarks':[]}
    for path in sorted((ROOT/'results').glob('camera_steady2250_*/benchmark.json')):
        b=read(path)
        evidence['camera_benchmarks'].append({'path':path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:b[k] for k in ['complete','restored','mode','received_frames','elapsed_s','host_received_fps','skipped_sensor_frame_ids','same_mode_warmup']}})
    if len(evidence['camera_benchmarks'])!=6:raise RuntimeError('Need six steady camera benchmark reports')
    (out/'evidence_summary.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    print(out)

if __name__=='__main__':main()
