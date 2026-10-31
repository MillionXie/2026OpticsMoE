"""Eight explicit source-to-target F cells, plus separately labelled historical joint runs."""
from pathlib import Path
from datetime import datetime,timezone
import json


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main():
    task=Path(__file__).resolve().parent;run=task/'runs/simulation/f_cross_domain_s17_20261031';out=task/'reports/f_cross_domain_s17_20261031';out.mkdir(exist_ok=True)
    data=dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',generated_UTC=datetime.now(timezone.utc).isoformat(),protocol='four fresh source-domain models; each evaluated on both domains; eight TEST cells',records={},historical_joint={})
    rows=[]
    for architecture in ('LG','D2NN'):
        for domain in ('RGB','SAR'):
            name=f'{architecture}_{domain}';p=run/name;result=read(p/'result.json');status=read(p/'status.json');history=read(p/'history.json') or []
            data['records'][name]=dict(run=str(p.relative_to(task)),status=status,result=result,teacher=read(run/f'teacher_{domain}/result.json') if architecture=='LG' else None)
            scores=[f"{100*result['original_test_dev']['domains'][d]['accuracy']:.2f}%" if result else '未锁权重/未测试' for d in ('RGB','SAR')]
            values=' / '.join(f"{100*result[s]['accuracy']:.2f}%" for s in ('train','val')) if result else '运行中；不冒充最终选中模型'
            state=status.get('state','尚未启动') if status else '尚未启动';epoch=result['completed_epochs'] if result else history[-1]['epoch'] if history else 0
            rows.append(f'<tr><td>{architecture}</td><td>{domain}</td><td>{state} / {epoch}轮</td><td>{values}</td><td>{scores[0]}</td><td>{scores[1]}</td></tr>')
    historical=[]
    for name,path in [('LG联合','f_backtracking_s17_20261031/paired'),('D2NN联合','figure_revision_s17_20261031/panel_f_d2nn_s17_20261031')]:
        result=read(task/'runs/simulation'/path/'result.json');data['historical_joint'][name]=dict(run=path,result=result)
        if result:
            for domain in ('RGB','SAR'):
                vals=[f"{100*result[s]['domains'][domain]['accuracy']:.2f}%" for s in ('train','val','original_test_dev')]
                historical.append(f'<tr><td>{name}</td><td>{domain}</td>'+''.join(f'<td>{v}</td>' for v in vals)+'</tr>')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图F八格交叉评估</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:0 16px}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}.note{background:#fff1d5;padding:14px}</style><h1>图F：四个单域模型、八格评估</h1><p class="note">最新人工要求覆盖此前联合训练协议。RGB与SAR各训练LG/D2NN一个新模型，每模型测试两个域，共八个TEST数值。表中未跑完的单元格保持空缺，不用联合模型结果填入。seed17；均为原TEST已暴露的开发结果，不是独立TEST泛化或等数据预算结构优势。</p><table><tr><th>模型</th><th>训练域</th><th>实际状态</th><th>获选TRAIN / 同域VAL</th><th>测试RGB</th><th>测试SAR</th></tr>'''+''.join(rows)+'''</table><p>学生各只用本域3000原TRAIN，同域1000 VAL选权重，原2000 TEST含两个域各1000，全部原空间划分/ID/像素/标签保持。D2NN三层478、每层同OEO无Linear，从头60轮，无教师。LG原四专家/Top2/E-G两层/同OEO/单Linear784→10，无CNN，亦从头初始化，不继承联合模型。</p><p>LG两个独立教师使用冻结ImageNet ResNet18与可训练分类头；各只学习本域3000 TRAIN+1000原TEST的D4增强，VAL与对方域不进教师训练。学生只在本域TRAIN接收真标签和匹配八视图软目标，没有把原TEST图像直接加入学生loss。原TEST参与教师训练必须明示，不称这些格子为独立测试；教师不进入LG推理。所有失败候选/预算、权重、完整预测与路由占比均保留。</p><h2>历史联合训练：仅用于解释当前分数与过拟合</h2><p>以下为一个RGB+SAR联合LG、一个联合D2NN各测两域，不是上方八格单域实验。</p><table><tr><th>联合模型</th><th>评价域</th><th>TRAIN</th><th>VAL</th><th>原TEST开发</th></tr>'''+''.join(historical)+'''</table><p>联合LG仍存在训练—验证差距，但D2NN训练本身也低，因此不只过拟合。旧高分版本含CNN前端，且dense路由、无OEO与窗口读出也不同；删除前端是重要差异，现有证据不能把整个降幅单独归因于它。旧图79.5/74.9为VAL，同PT TEST78.9/72.4，不能移入当前无前端版本。</p></html>'''
    page=page.replace('</html>','<p class="note">RGB/SAR为配对地块：即使教师没有接触对方域像素，源域原TEST进入教师后，对应地块的标签也已暴露。因此交叉格同样属于开发指标，不能称作未见地块的独立跨域测试。</p></html>')
    page=page.replace('</html>','<h2>各源域独立收敛曲线</h2><p>黑线为同域VAL，蓝虚线为干净原TRAIN；菱形为获选权重完整TRAIN重评。两个TEST星号仅在锁权重后推理完成才画，不将VAL或中途模型填入八格。</p><img src="convergence.svg" style="width:100%"></html>')
    (out/'results.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');(out/'results.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
