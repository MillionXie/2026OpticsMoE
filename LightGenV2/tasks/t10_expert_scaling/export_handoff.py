"""Build plotting JSON and a Chinese methods note from downloaded run evidence."""
import argparse,json,statistics,hashlib
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[];curves=[];routes=[];cms=[]
    for dataset in ['kather','adrenal']:
        for f in sorted((a.source/dataset).glob('*/result.json')):
            result=json.loads(f.read_text());meta=json.loads((f.parent/'metadata.json').read_text());args=meta['arguments']
            key=dict(dataset=dataset,run=f.parent.name,architecture=args['arch'],experts=args['experts'],top_k=args['top_k'],seed=args['seed'])
            row=dict(**key,best_epoch=result['best_epoch'],git_commit=result.get('git_commit'),checkpoint_sha256=result.get('checkpoint_sha256'),source=str(f.resolve()),test_read=result.get('test_read'))
            assert row['test_read'] is False
            for split in ['train','val']:
                v=result[split]
                for metric in ['accuracy','balanced_accuracy','macro_f1','macro_nll','capture_mean']:row[split+'_'+metric]=v[metric]
                cms.append(dict(**key,split=split,matrix=v['confusion_matrix']))
                if 'route_load' in v:
                    assert abs(sum(v['route_load'])-args['top_k'])<1e-4
                    assert abs(sum(v['route_probability'])-1)<1e-4
                    for i,(load,prob) in enumerate(zip(v['route_load'],v['route_probability'])):routes.append(dict(**key,split=split,expert_index=i,hard_selection_fraction=load,soft_probability_mean=prob,distinct_selected_sets=v['distinct_selected_sets']))
            rows.append(row)
            for h in json.loads((f.parent/'history.json').read_text()):
                curves.append(dict(**key,epoch=h['epoch'],train_objective=h['train_objective'],learning_rate=h['lr'],elapsed_seconds=h['elapsed_seconds'],validation=h['val']))
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['architecture'],r['experts'],r['top_k'])].append(r)
    summary=[]
    for (ds,arch,n,k),rs in sorted(groups.items()):
        seeds=sorted(r['seed'] for r in rs);assert len(set(seeds))==len(seeds)
        v=dict(dataset=ds,architecture=arch,experts=n,top_k=k,seeds=seeds,n_seeds=len(seeds))
        for metric in ['train_accuracy','val_accuracy','val_balanced_accuracy','val_macro_f1','val_macro_nll','val_capture_mean']:
            values=[r[metric] for r in rs];v[metric+'_mean']=statistics.mean(values);v[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
        summary.append(v)
    pairs=[]
    for r in rows:
        if r['architecture']!='moe_oeo':continue
        d=next(x for x in rows if x['dataset']==r['dataset'] and x['experts']==r['experts'] and x['seed']==r['seed'] and x['architecture']=='d2nn_expert_global')
        pairs.append(dict(dataset=r['dataset'],experts=r['experts'],top_k=r['top_k'],seed=r['seed'],accuracy_difference_pp=100*(r['val_accuracy']-d['val_accuracy']),balanced_accuracy_difference_pp=100*(r['val_balanced_accuracy']-d['val_balanced_accuracy'])))
    for name,data in [('per_seed',rows),('summary',summary),('paired_differences',pairs),('learning_curves',curves),('expert_allocation',routes),('confusion_matrices',cms)]:
        (a.out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# T10 绘图交接：2026-09-21实际保留结果','',
    '本目录只汇总动态路由、增长版面、全孔径输入的Kather/Adrenal队列；不包含后续fixed478实验。全部为验证集结果，不能标为test。',
    'Kather服务器状态记录为superseded_without_N100：100专家文件已移除、seed37取消；实际保留16组（4个N×2架构×2种子）。此整理没有删除任何run。Adrenal完成30组（5个N×2架构×3种子）。',
    '', '## 文件与统计口径','',
    '- per_seed.json：每组种子的训练/验证结果、源码commit和checkpoint SHA256；是绘图原始点。',
    '- summary.json：按数据集、架构、N、k分组的均值和样本标准差（ddof=1），同时保存真实种子数量。',
    '- paired_differences.json：相同数据集/N/seed的MoE减D2NN，单位百分点。',
    '- learning_curves.json：逐轮训练总目标、验证全部指标（含路由）和耗时。训练总目标包含正则，不等于验证NLL；没有逐轮训练准确率，不得伪造训练准确率曲线。',
    '- expert_allocation.json：最佳checkpoint训练/验证专家选择比例与soft概率均值；expert_index从0开始，按端口中心向外排序，不是阵列行优先空间编号。',
    '- confusion_matrices.json：行是真实类别，列是预测类别。',
    '- 原始证据目录包含metadata/history/result/status，TRANSFER_MANIFEST.json记录远端来源、SHA256和文件大小。权重仍在服务器，本交接不含大体积权重和原始图像。',
    '', 'JSON中accuracy、balanced_accuracy、macro_f1、capture均为0–1，绘图时乘100；macro_nll不用乘100。误差棒为种子间样本标准差，不是置信区间。',
    '', '## 实验合同与限制','',
    '输入逻辑采样17μm、相位器件8μm为等物理尺寸映射；专家224²、间隔30像素；4个主干相位层，MoE为两次专家层/global层组合，D2NN为4个连续相位层。两者逐层OEO。',
    'RGB分别缩放后拼[R,G;B,mean]；MoE填满每个专家，D2NN填满连续相位孔径。相同原图但实际输入采样分辨率不同，不宣称相同输入信息带宽。没有CNN/Qwen/电子残差分类支路。',
    '路由端口按sqrt(N)×sqrt(N)生成，端口24²、间距32；N≤49路由相位224²，N100为312²。top-k只控制入口功率，后续全场传播会耦合专家，不能解释为每一层始终只有k个专家获得光。',
    'OEO：强度→均值归一化→全场无仿射LayerNorm→ReLU→t/(1+t)→单位功率、零相位重编码。t/(1+t)与非负t的Softsign数学等价，不能作为两个不同激活做消融。ReLU负值处局部导数为0不等于整个网络无梯度。',
    '路由采用hard前向、dense振幅STE反向。当前只有soft概率均衡loss=0.01；按microbatch=2计算，不是按梯度累积后的16个样本一起计算。没有执行hard均衡loss消融。',
    'hard_selection_fraction表示某专家被选择的样本比例，所有专家之和是k，不能直接画成总和100%的饼图；soft_probability_mean之和是1。soft概率不是top-k后实际入口功率。未保存逐样本路由，不可由均值恢复每个样本的功率分配。',
    'N/k组合：4/3、16/16、25/12、49/24、100/50。N、k和global尺寸一起变化；N100还改变路由尺寸。因此这不是固定k的纯专家数因果消融，也不是完整top-k遍历。',
    'D2NN匹配专家+global相位参数，不含路由相位；最近偶数边长带来少量参数误差，实际geometry在metadata。',
    '训练60轮，AdamW lr0.002，effective batch16、microbatch2、EMA0.95；按验证macro-NLL最小选择checkpoint，并非最高准确率。详细实际配置/命令见metadata。',
    'Kather为5000张8类组织图像，train3496/val752/test752；无患者ID支持患者独立性声明。Adrenal使用既有三维数据二维投影缓存，非原生RGB；灰度复制三通道并量化为uint8，没有增加颜色信息。train1188/val98/test298。既往任务看过这些测试数据，不能宣称全新盲测。',
    'Adrenal验证集76正常/22阳性；全部判正常准确率77.55%、平衡准确率50%。应同时画balanced accuracy，不能用普通准确率单独证明学会分类。',
    '', '## 建议作图','',
    '分别按任务作N–验证性能图，图例写MoE+逐层OEO及D2NN+逐层OEO；每个N标注k，叠加原始种子点及mean±SD。Kather仅2种子，Adrenal3种子。另画同一checkpoint训练/验证分组柱状图和每个seed专家负载热图。不要把不同N的差异当成固定k消融。',
    '', '## 汇总（百分比，均值±样本标准差）','',
    '|任务|架构|N|k|种子数|训练准确率|验证准确率|验证平衡准确率|','|---|---|---:|---:|---:|---:|---:|---:|']
    for v in summary:
        fmt=lambda m:f"{v[m+'_mean']*100:.2f}±{v[m+'_sd']*100:.2f}"
        lines.append(f"|{v['dataset']}|{v['architecture']}|{v['experts']}|{v['top_k']}|{v['n_seeds']}|{fmt('train_accuracy')}|{fmt('val_accuracy')}|{fmt('val_balanced_accuracy')}|")
    (a.out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file() and p.name!='SHA256.json'}
    (a.out/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\n'.join(lines[-len(summary):]));print('counts',len(rows),len(curves),len(routes))

if __name__=='__main__':main()
