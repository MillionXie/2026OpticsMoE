"""Current four-metric fitting goal; incomplete snapshots are not model backups."""
from pathlib import Path
from datetime import datetime,timezone
import html,json


def read(p):return json.loads(p.read_text(encoding='utf-8'))if p.exists()else None

def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_four_metric_s17_20261101';out.mkdir(exist_ok=True);root=task/'runs/simulation/f_four_metric_s17_20261101';stages={};rows=[]
    for stage in ('teacher','student'):
        history=read(root/stage/'history.json')or [];result=read(root/stage/'result.json');status=read(root/stage/'status.json');last=history[-1]if history else None
        stages[stage]=dict(status=status,result=result,last_history=last)
        value=result or last
        if value:
            train=value['train'];test=value.get('original_test_dev',value.get('original_test_development_selection'));values=[train['domains'][d]['accuracy']for d in ('RGB','SAR')]+[test['domains'][d]['accuracy']for d in ('RGB','SAR')]
            rows.append('<tr><td>'+stage+'</td><td>'+html.escape(str((status or {}).get('state')))+' / E'+str(value.get('epoch',value.get('completed_epochs',0)))+'</td>'+''.join('<td>'+f'{v*100:.2f}%'+ '</td>'for v in values)+'<td>'+f'{min(values)*100:.2f}%'+ '</td></tr>')
    data=dict(generated_UTC=datetime.now(timezone.utc).isoformat(),role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',goal='Same PT original TRAIN RGB/SAR and original TEST RGB/SAR all >=65%; fixed verified balanced optical router',selection='minimum of four original group accuracies, then their mean, then CE; VAL only recorded',stages=stages,progress_only=not (out/'student_model_audit.json').exists())
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>F：四项准确率联合提升</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px}th{background:#edf2f7}</style><h1>F：同一PT的TRAIN与开发TEST均达到65%</h1><p>四项均为固定完整原图、各域无权重统计。每轮混合原TRAIN6000和原TEST2000，四组损失分别平均后侧重较弱组。教师可用已授权TEST硬标签；学生TEST只用教师软目标，继承权重已暴露TEST。VAL只记录，不设门槛。不是独立泛化结果。</p><p>冻结已经核验的原光router，保持四专家实际选择负载；训练原E/G与同一个Linear。教师最多60轮、学生最多15轮，父E0参与。17项CPU检查及精确父40张真实输入GPU更新、学生TEST硬GT拒绝、头拟合相位冻结均通过。样本准入不是准确率保证。</p><table><tr><th>阶段</th><th>状态／最新轮</th><th>TRAIN RGB</th><th>TRAIN SAR</th><th>TEST RGB</th><th>TEST SAR</th><th>四项最低</th></tr>'+''.join(rows)+'</table><p>中途最新轮不是获选PT的最终成绩。完整best/last、三划分预测路由、教师目标与源SHA核验前，不替换旧77.6/71.8开发权重。旧权重的TRAIN52.03/47.57尚未满足新增条件。</p><p><a href="../f_joint_dual_teacher_s17_20261101/results.html">主图与全部历史</a></p></html>'
    if (out/'four_metrics.svg').exists():page=page.replace('</html>','<p>教师与学生独立预算轴，TEST已用于开发选模，不当独立测试。</p><img src="four_metrics.svg" style="width:100%"></html>')
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'results.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
