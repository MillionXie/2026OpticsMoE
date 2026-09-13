"""Serial hardware trials and ONE fixed, human-readable result directory."""
import argparse,html,json,shutil,subprocess,sys,time
from pathlib import Path
from guarded_workflow import read,write
ROOT=Path(__file__).resolve().parent
CURRENT=ROOT/'reports/00_current'
TRIALS=[('01_400us_250ms',400,250),('02_350us_200ms',350,200),('03_350us_250ms',350,250)]


def write_report(path,value):
    # Windows/OneDrive may briefly hold the destination during atomic replace.
    for attempt in range(10):
        try:
            write(path,value);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.2*(attempt+1))


def publish(out,state):
    CURRENT.mkdir(parents=True,exist_ok=True)
    write_report(CURRENT/'02_summary.json',state)
    lines=['# 本次结果：只看这个文件夹即可','',f"状态：{state['status']}；当前：{state.get('active','—')}",
           '',f'原始数据相对工程目录：`{out.relative_to(ROOT)}`','',
           '|曝光μs|等待ms|数字正确数|最低目标PCC|灰度255饱和比例|状态|',
           '|---|---|---|---|---|---|']
    body=[]
    for i,t in enumerate(state['trials'],1):
        if t['status']!='complete':
            lines.append(f"|{t['exposure_us']}|{t['wait_ms']}|—|—|—|{t['status']}|");continue
        r=read(out/t['name']/'analysis.json');tim=r['timing'];sat=next(s['saturation_fraction'] for s in r['sweep'] if s['gray']==255)
        lines.append(f"|{t['exposure_us']}|{t['wait_ms']}|{tim['correct']}/{tim['n']}|{tim['minimum_target_pcc']:.6f}|{sat*100:.3f}%|{'图案短测通过' if tim['pattern_timing_passed'] else '未通过'}|")
        for src,label in [('gray_response.png','灰度曲线'),('gray_previews.png','原始ROI预览'),('timing_components.png','周期耗时'),('timing_failure.png','错帧证据')]:
            if not (out/t['name']/src).exists():continue
            filename=f"{i+3:02d}_{t['name']}_{src}";shutil.copy2(out/t['name']/src,CURRENT/filename)
            body.append(f'<h2>{html.escape(t["name"])}：{label}</h2><img style="max-width:100%" src="{filename}">')
        if r.get('timing_ms'):
            labels={'bmp_validate_ms':'BMP校验','slm_preload_ms':'SLM上传','slm_show_to_visible_ms':'显示请求到Visible',
                    'settle_actual_ms':'持续排帧等待','final_fresh_ms':'最后排帧并取新帧','capture_total_ms':'BMP就绪到原始帧就绪合计'}
            parts=['环节 | 平均ms | P95 ms']+[f"{labels[k]} | {v['mean']:.3f} | {v['p95']:.3f}" for k,v in r['timing_ms'].items()]
            body.append('<h2>实际周期耗时</h2><pre>'+html.escape('\n'.join(parts))+'</pre><p>不含模型推理、PNG保存、SSH回传和跨层相位切换；100fps是相机流帧率，不是网络每秒处理100张。</p>')
        for f in tim['rows']:
            if f['digit']!=f['prediction']:body.append('<p>错帧：'+html.escape(json.dumps(f,ensure_ascii=False))+'</p>')
    if any(t['status']=='complete' and not t['timing']['pattern_timing_passed'] for t in state['trials']):
        lines+=['','本轮存在明确错帧：不改正式曝光/等待配置，不启动六层全量；先解决取帧对应性。']
    lines+=['',f"口径：每组13灰度×3帧、{state.get('switch_count',40)}次数字切换及6张独立参考。固定ROI/增益X4/100fps/平相位。",
            '图案时序通过不代表没有饱和，更不代表六层全量已获批准；失败帧不删除、不按准确率挑样本。',
            '饱和比例取同档3帧的最大值；数字正确数是显示/取帧对应性，不是MNIST识别准确率。',
            '200ms此前出现过明确旧帧，不能因为本轮短测通过就推翻旧证据；其他等待也不保证全量零错帧。',
            '', '## 已有重要结果','',
            '- MNIST：旧mask实测24/40，原生8μm实测37/40；不是全量准确率。原记录 results/mnist_pair_wait400_20260913。',
            '- ABO：仅4查询六层流程完成；2400查询全量尚未启动。原记录 results/smoke_runs/smoke_abo_newroi_20260913_161905。',
            '- 旧400μs+200ms：39/40，存在明确旧帧。原记录 results/gray400_wait200_20260913_02。',
            '', '## 实时监督','', '每组的日志在本次原始数据目录下同名.log文件，suite.json记录整个进度。']
    network=out/'02_network_scan/report.json'
    if network.exists():
        nr=read(network);lines+=['','## 六层实际输入曝光复核','',f"状态：{nr['status']}；已记录阶段：{len(nr.get('stages',[]))}/6"]
        for s in nr.get('stages',[]):
            d=s['data'];lines.append(f"- {s['stage']}：{json.dumps(d['recommendation'],ensure_ascii=False)}；同相位原始ROI PCC={d['phase_hold_pcc']:.6f}")
        if nr.get('error'):lines+=['',nr['error']]
        if (network.parent/'exposure_range.png').exists():
            shutil.copy2(network.parent/'exposure_range.png',CURRENT/'07_network_exposure.png')
            body.append('<h2>六层实际输入</h2><img style="max-width:100%" src="07_network_exposure.png">')
    text='\n'.join(lines)+'\n';(CURRENT/'00_READ_ME.md').write_text(text,encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>SHS 当前实验结果</title><style>body{max-width:1300px;margin:30px auto;font-family:Arial,sans-serif}pre{white-space:pre-wrap;background:#f3f5f8;padding:20px}</style>'
    page+='<h1>SHS 当前实验结果</h1><p>此目录是固定入口；刷新页面查看最新已完成组。详细指标见02_summary.json及各组analysis.json。</p><pre>'+html.escape(text)+'</pre>'+'\n'.join(body)
    page+='<hr><p>旧结果分类入口：<a href="99_history_index.html">99_history_index.html</a>。清理清单：<a href="90_cleanup.json">90_cleanup.json</a>（如已执行清理）。</p>'
    (CURRENT/'01_summary.html').write_text(page,encoding='utf-8')
    groups={k:[] for k in ['本次与重要保留','MNIST与方向标定','旧曝光和时序','旧SDK与光路排障','其他历史记录']}
    for p in sorted((ROOT/'results').iterdir()):
        if not p.is_dir():continue
        name=p.name
        if p==out or name in ['smoke_runs','smoke_configs','mnist_pair_wait400_20260913','mnist_markers_20260913_154939']:key='本次与重要保留'
        elif any(x in name for x in ['mnist','marker','orientation','geometry']):key='MNIST与方向标定'
        elif any(x in name for x in ['timing','exposure','gray','gain','digits','benchmark']):key='旧曝光和时序'
        elif any(x in name for x in ['phase','sdk','checker','check64','camera','display','joint','digital','serial','handle']):key='旧SDK与光路排障'
        else:key='其他历史记录'
        groups[key].append(name)
    index='<!doctype html><meta charset="utf-8"><title>历史记录分类</title><h1>历史记录分类</h1><p>历史目录原位保留，避免破坏配置中的路径。下列分类是浏览索引，不表示每个旧run通过验收。日常看01_summary.html即可。</p>'
    for group,names in groups.items():
        index+=f'<details><summary>{group}（{len(names)}）</summary><ul>'
        for name in names:index+=f'<li><a href="../../results/{html.escape(name,quote=True)}/">{html.escape(name)}</a></li>'
        index+='</ul></details>'
    (CURRENT/'99_history_index.html').write_text(index,encoding='utf-8')


def run(a):
    out=a.out.resolve()
    trials=TRIALS if not a.trial else [(f'{i:02d}_{e:g}us_{w:g}ms',e,w) for i,(e,w) in enumerate(a.trial,1)]
    if not out.is_relative_to(ROOT/'results'):raise ValueError('Suite outside results')
    if a.publish_only:
        publish(out,read(out/'suite.json'));return
    if a.append_trial:
        if a.resume or a.trial:raise ValueError('Append is separate from resume/custom initial trials')
        state=read(out/'suite.json')
        if state['status']!='complete' or any(t['status']!='complete' for t in state['trials']):raise ValueError('Only append to a completed suite')
        if state.get('switch_count',40)!=a.switch_count:raise ValueError('Keep identical switching count for comparison')
        e,w=a.append_trial
        from gray_response_scan import validate_settings
        validate_settings(e,w)
        n=f"{len(state['trials'])+1:02d}_{e:g}us_{w:g}ms"
        if (out/n).exists() or (out/(n+'.log')).exists():raise FileExistsError('Appended trial output exists')
        state['trials'].append(dict(name=n,exposure_us=e,wait_ms=w,status='pending'))
        state.update(status='running',extended=time.strftime('%Y-%m-%dT%H:%M:%S'));state.pop('finished',None)
    elif a.resume:
        state=read(out/'suite.json')
        if [(t['name'],t['exposure_us'],t['wait_ms']) for t in state['trials']]!=trials or state.get('switch_count',40)!=a.switch_count:raise ValueError('Trial contract mismatch')
        for t in state['trials']:
            if t['status']=='complete':
                if not (out/t['name']/'analysis.json').is_file():raise ValueError('Completed trial data missing')
            elif (out/t['name']).exists() or (out/(t['name']+'.log')).exists():
                raise ValueError('Partial trial exists; inspect it, never automatically overwrite/reacquire')
        state.update(status='running',resumed=time.strftime('%Y-%m-%dT%H:%M:%S'))
    else:
        out.mkdir(parents=True,exist_ok=False)
        state=dict(status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S'),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            trials=[dict(name=n,exposure_us=e,wait_ms=w,status='pending') for n,e,w in trials],switch_count=a.switch_count,full_dataset_qualified=False)
    def save():write_report(out/'suite.json',state);publish(out,state)
    save()
    for t in state['trials']:
        if t['status']=='complete':continue
        t['status']='running';state['active']=t['name'];save()
        cmd=[sys.executable,'-u',str(ROOT/'gray_response_scan.py'),'--link',str(a.link.resolve()),'--source-config',a.source_config,
             '--out',str(out/t['name']),'--exposure-us',str(t['exposure_us']),'--wait-ms',str(t['wait_ms']),'--switch-count',str(a.switch_count)]
        t['command']=cmd;t['log']=str(out/(t['name']+'.log'));save()
        print('START',t['name'],'LOG',t['log'],flush=True)
        with Path(t['log']).open('x',encoding='utf-8') as log:
            proc=subprocess.run(cmd,cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            t['status']='failed';state.update(status='stopped',error='See trial log; no automatic hardware takeover/retry');save();raise RuntimeError(t['log'])
        r=read(out/t['name']/'analysis.json');t.update(status='complete',timing=r['timing'],gray255=next(s for s in r['sweep'] if s['gray']==255));save()
        print('DONE',t['name'],r['timing']['correct'],'/',r['timing']['n'],flush=True)
    state.update(status='complete',active=None,finished=time.strftime('%Y-%m-%dT%H:%M:%S'));save()
    print('READ',CURRENT,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--link',type=Path,required=True);p.add_argument('--source-config',required=True)
    p.add_argument('--trial',type=float,nargs=2,action='append',metavar=('EXPOSURE_US','WAIT_MS'))
    p.add_argument('--append-trial',type=float,nargs=2,metavar=('EXPOSURE_US','WAIT_MS'),help='Append one candidate to a completed suite; existing captures are never rerun')
    p.add_argument('--switch-count',type=int,default=40)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--publish-only',action='store_true');p.add_argument('--resume',action='store_true');run(p.parse_args())
