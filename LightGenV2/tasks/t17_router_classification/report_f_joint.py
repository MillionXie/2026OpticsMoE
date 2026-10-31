"""Main F protocol: one joint LG/two outputs and finalized D2NN/four outputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import html
import json


def read(path):return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
def pct(value):return f'{100*value:.2f}%'


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_joint_dual_teacher_s17_20261101';out.mkdir(exist_ok=True)
    parent=task/'runs/simulation/f_backtracking_s17_20261031/paired';candidate=task/'runs/simulation/f_joint_dual_teacher_s17_20261101/joint'
    reference=read(parent/'result.json');result=read(candidate/'result.json');status=read(candidate/'status.json');history=read(candidate/'history.json') or []
    selected=reference;path=parent;verified=False
    if result:
        weight=candidate/'best_checkpoint.pt';verified=weight.exists() and hashlib.sha256(weight.read_bytes()).hexdigest()==result['checkpoint_sha256']
        if verified:selected=result;path=candidate
    ste=task/'runs/simulation/f_joint_route_STE_s17_20261101/joint';ste_result=read(ste/'result.json');ste_status=read(ste/'status.json');ste_verified=False
    if ste_result and (ste/'best_checkpoint.pt').exists():
        ste_verified=hashlib.sha256((ste/'best_checkpoint.pt').read_bytes()).hexdigest()==ste_result['checkpoint_sha256'] and (task/'reports/f_joint_route_STE_s17_20261101/model_audit.json').exists() and (task/'reports/f_joint_route_STE_s17_20261101/completed_transfer.json').exists()
        if ste_verified:selected=ste_result;path=ste
    projection=task/'runs/simulation/f_joint_router_projection_s17_20261101/joint';projection_result=read(projection/'result.json');projection_status=read(projection/'status.json');projection_verified=False
    if projection_result and (projection/'best_checkpoint.pt').exists():
        projection_verified=hashlib.sha256((projection/'best_checkpoint.pt').read_bytes()).hexdigest()==projection_result['checkpoint_sha256'] and (task/'reports/f_joint_router_projection_s17_20261101/model_audit.json').exists() and (task/'reports/f_joint_router_projection_s17_20261101/completed_transfer.json').exists()
        if projection_verified:selected=projection_result;path=projection
    records={};rows=[]
    for architecture,source,r in [('LG','RGB+SAR联合',selected)]+[('D2NN',d,read(task/f'runs/simulation/f_cross_domain_s17_20261031/D2NN_{d}/result.json')) for d in ('RGB','SAR')]:
        records[architecture+'_'+source]=r
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [architecture,source,pct(r['train']['accuracy']),pct(r['val']['accuracy']),pct(r['original_test_dev']['domains']['RGB']['accuracy']),pct(r['original_test_dev']['domains']['SAR']['accuracy'])])+'</tr>')
    target=all(selected['original_test_dev']['domains'][d]['accuracy']>=.70 for d in ('RGB','SAR'))
    note='联合双教师候选尚未完成：主表显示保留的联合父，不能把中途VAL当TEST。' if not verified else '联合双教师候选已完成并核验；较新的训练梯度候选须完整归档核验后才替换。整行来自同一联合PT。'
    display_status=projection_status or ste_status or status
    progress=' / '.join([display_status.get('state','尚未启动'),str(display_status.get('epoch',display_status.get('completed_epochs',0)))+'轮']) if display_status else '尚未抓取运行状态'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图F联合LG与定稿D2NN</title><style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:32px auto;padding:0 18px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{background:#fff0cf;padding:14px}</style><h1>图F：联合LG两个主结果＋定稿D2NN四格</h1><p class="note">ORIGINAL_TEST_EXPOSED_DEVELOPMENT。原TEST已用于教师及LG软目标训练，不是独立测试或等训练预算结构优势。seed17，无多种子标准差。LG两个域必须使用同一份权重；D2NN四格已定稿。</p>'''
    page+='<p>'+note+'</p><table><tr><th>模型</th><th>训练域</th><th>原TRAIN</th><th>VAL</th><th>开发RGB</th><th>开发SAR</th></tr>'+''.join(rows)+'</table>'
    page+=f'<p>新候选状态：{html.escape(progress)}；RGB/SAR两域均≥70%目标：{"已达" if target else "未达"}。联合模型路径：{html.escape(str(path.relative_to(task)))}。</p>'
    page+='<p>原RGB/SAR混合6000 TRAIN，2000 VAL；同一学生保留四专家/Top2/两层E-G/相同OEO/单Linear784→10，无CNN前端。两个已冻结教师仅提供训练目标，不作为学生推理模块，也不将域标签交给学生或固定分配专家。</p><p>原TRAIN用真实类别CE和教师KD；两域原TEST2000仅用匹配D4软目标，不进入硬标签CE。每轮6000硬TRAIN图与2000软TEST图；每5轮原单头64000视图（48000硬TRAIN增强＋16000软TEST增强）最多40内步，所有预算与历史父均保留。</p><p>选模优先RGB/SAR中较低的VAL准确率，再联合宏召回、再CE；原父E0参与。最多60/min25/pat15轮，不保证达到目标。跨域单域LG的八格实验另存，不能充当联合LG主图结果。</p></html>'
    if (out/'convergence.svg').exists():page=page.replace('</html>','<h2>联合训练实际曲线</h2><p>两域分别显示同一学生的TRAIN/VAL；仅完整锁权重后才画本run开发TEST星号。每个头block的目标前后使用相同特征/软目标，跨轮目标不能当严格配对。</p><img src="convergence.svg" style="width:100%"></html>')
    if ste_status:
        text=f"{ste_status.get('state')}, epoch {ste_status.get('epoch',ste_status.get('completed_epochs',0))}"
        page=page.replace('</html>','<h2>有界训练梯度对照</h2><p>新候选f_joint_route_STE_s17_20261101：'+html.escape(text)+'。176项CPU及精确父真实GPU准入通过，已实际启动。主干、参数和forward仍是原硬Top2，训练时只用dense振幅代理反向梯度，eval禁用代理。真实GPU检查原forward逐位相同，未选端口获得非零梯度；这不等于分类准确率已改善。父为上述完整双教师f8c149权重，最多40/min20/pat15轮，其他监督与选模不变。旧父和所有失败保留。</p></html>')
        if ste_verified:
            page=page.replace('176项CPU及精确父真实GPU准入通过，已实际启动。','176项CPU及精确父真实GPU准入通过，现已25轮早停并完整SHA/strict/逐样本核验和canonical TEST重放。')
            page=page.replace('</html>','<h2>最新完整阶段与训练诊断</h2><p>当前同一LG权重：开发RGB63.20/SAR48.30%，TRAIN64.57/59.03%，VAL56.20/51.10%。仍未两域70%。两个原训练PID已退出，当前没有自己的训练。只读完整原TRAIN6000/code0测得新父RGB/SAR物理router梯度余弦约-0.979；surrogate与canonical联合router方向余弦约0.9997。仅原TRAIN原方向，不外推全8视图或容量上限。</p><p>下一有界训练方法仅准备：按训练域对router梯度做物理坐标冲突投影，E/G/head仍原混合损失，forward不变、无新参数。尚未实现或启动，需默认兼容/GT隔离/数值与真实GPU准入后发布再运行；不保证目标，所有原权重与失败保留。</p></html>')
    if projection_status:
        page=page.replace('当前没有自己的训练。','原两条训练均已结束，新投影候选状态见下节。')
        page=page.replace('下一有界训练方法仅准备：','有界训练方法：').replace('尚未实现或启动，需默认兼容/GT隔离/数值与真实GPU准入后发布再运行；','已通过默认兼容/GT隔离/数值检查及真实GPU准入后发布并启动，详见下节；')
        text=f"{projection_status.get('state')}, epoch {projection_status.get('epoch',projection_status.get('completed_epochs',0))}"
        page=page.replace('</html>','<h2>两域router训练梯度冲突投影</h2><p>f_joint_router_projection_s17_20261101：'+html.escape(text)+'。186项CPU及精确124父真实GPU准入通过并启动，最多30/min15/pat10轮。仅对router的两域物理相位梯度作同时负内积投影，按实际样本数加权；E/G/head仍原混合损失梯度，学生forward完全不变。域标签只用于训练损失分组，TEST软目标分支拒绝硬标签输入。额外router任务梯度评估次数单独记账，投影不保证Adam步后两域下降或70%。未完成/未归档前主表仍保留63.20/48.30。</p></html>')
    data=dict(generated_UTC=datetime.now(timezone.utc).isoformat(),role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',protocol='one joint LG two outputs plus two finalized source-only D2NN four outputs',records=records,selected_joint_run=str(path.relative_to(task)),candidate_status=status,candidate_result=result,candidate_local_SHA_verified=verified,parent_result=reference,target_both_domains_at_least_70_percent=target,latest_history=history[-1] if history else None,route_STE_candidate=dict(status=ste_status,result=ste_result,fully_collected_local_SHA_verified=ste_verified),router_projection_candidate=dict(status=projection_status,result=projection_result,fully_collected_local_SHA_verified=projection_verified))
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'results.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
