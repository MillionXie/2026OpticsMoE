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
    sgd=task/'runs/simulation/f_joint_router_sgd_s17_20261101/joint';sgd_result=read(sgd/'result.json');sgd_status=read(sgd/'status.json');sgd_verified=False
    if sgd_result and (sgd/'best_checkpoint.pt').exists():
        sgd_verified=hashlib.sha256((sgd/'best_checkpoint.pt').read_bytes()).hexdigest()==sgd_result['checkpoint_sha256'] and (task/'reports/f_joint_router_sgd_s17_20261101/model_audit.json').exists() and (task/'reports/f_joint_router_sgd_s17_20261101/completed_transfer.json').exists()
    choices=[('Original joint parent',reference,parent)]
    for name,value,location,checked in [('Dual teacher',result,candidate,verified),('Surrogate backward',ste_result,ste,ste_verified),('Router projection',projection_result,projection,projection_verified),('Router angle SGD',sgd_result,sgd,sgd_verified)]:
        if checked:choices.append((name,value,location))
    shared=task/'runs/simulation/f_joint_teacher_student_TEST_s17_20261101/joint';shared_result=read(shared/'result.json');shared_status=read(shared/'status.json');shared_verified=False
    if shared_result and (shared/'best_checkpoint.pt').exists():
        shared_verified=hashlib.sha256((shared/'best_checkpoint.pt').read_bytes()).hexdigest()==shared_result['checkpoint_sha256'] and (task/'reports/f_joint_teacher_student_TEST_s17_20261101/model_audit.json').exists() and (task/'reports/f_joint_teacher_student_TEST_s17_20261101/completed_transfer.json').exists()
        if shared_verified:choices.append(('Joint teacher / TEST selection',shared_result,shared))
    val_floor=min(v['accuracy'] for v in reference['val']['domains'].values())
    eligible=choices # Latest human rule: original TEST only; no validation floor.
    def development_key(item):
        values=[v['accuracy'] for v in item[1]['original_test_dev']['domains'].values()]
        return min(values),sum(values)/len(values)
    _,selected,path=max(eligible,key=development_key)
    records={};rows=[]
    for architecture,source,r in [('LG','RGB+SAR联合',selected)]+[('D2NN',d,read(task/f'runs/simulation/f_cross_domain_s17_20261031/D2NN_{d}/result.json')) for d in ('RGB','SAR')]:
        records[architecture+'_'+source]=r
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [architecture,source,pct(r['train']['accuracy']),pct(r['val']['accuracy']),pct(r['original_test_dev']['domains']['RGB']['accuracy']),pct(r['original_test_dev']['domains']['SAR']['accuracy'])])+'</tr>')
    target=all(selected['original_test_dev']['domains'][d]['accuracy']>=.70 for d in ('RGB','SAR'))
    note='联合双教师候选尚未完成：主表显示保留的联合父，不能把中途VAL当TEST。' if not verified else '联合双教师候选已完成并核验；较新的训练梯度候选须完整归档核验后才替换。整行来自同一联合PT。'
    display_status=shared_status or sgd_status or projection_status or ste_status or status
    progress=' / '.join([display_status.get('state','尚未启动'),str(display_status.get('epoch',display_status.get('completed_epochs',0)))+'轮']) if display_status else '尚未抓取运行状态'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>图F联合LG与定稿D2NN</title><style>body{font:16px/1.7 sans-serif;max-width:1100px;margin:32px auto;padding:0 18px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.note{background:#fff0cf;padding:14px}</style><h1>图F：联合LG两个主结果＋定稿D2NN四格</h1><p class="note">ORIGINAL_TEST_EXPOSED_DEVELOPMENT。原TEST已用于教师及LG软目标训练，不是独立测试或等训练预算结构优势。seed17，无多种子标准差。LG两个域必须使用同一份权重；D2NN四格已定稿。</p>'''
    page+='<p>'+note+'</p><table><tr><th>模型</th><th>训练域</th><th>原TRAIN</th><th>VAL</th><th>开发RGB</th><th>开发SAR</th></tr>'+''.join(rows)+'</table>'
    page+='<p>最新人工要求：新run直接按原TEST开发的较低域准确率、再宏召回/CE选权重，VAL只记录，不设选模门槛；历史分支仍保留其原先VAL选模身份。跨完整核验候选按较低域原TEST开发准确率、再两域均值选整行，取消旧VAL下限。必须同一PT推两域，不能逐域拼权重；所有结果均为暴露开发。</p>'
    stage_rows=[]
    for name,value,location in choices:
        scores=[' / '.join(pct(value[s]['domains'][d]['accuracy']) for d in ('RGB','SAR')) for s in ('train','val','original_test_dev')]
        stage_rows.append('<tr><td>'+name+'</td>'+''.join('<td>'+v+'</td>' for v in scores)+'</tr>')
    page+='<h2>全部已核验联合候选（RGB / SAR）</h2><table><tr><th>分支</th><th>TRAIN</th><th>VAL</th><th>开发</th></tr>'+''.join(stage_rows)+'</table>'
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
        if projection_verified:
            page=page.replace('186项CPU及精确124父真实GPU准入通过并启动，最多30/min15/pat10轮。','已完整20轮早停选E10，并全量SHA/strict/原ID标签与canonical TEST重放核验。开发RGB63.10/SAR47.90%，两域均低于父63.20/48.30，保留父整行。当前没有自己的训练。')
            page=page.replace('未完成/未归档前主表仍保留63.20/48.30。','完整失败候选保留，主表仍用父63.20/48.30。')
    paired_probe=read(task/'reports/f_joint_router_projection_s17_20261101/TRAIN_paired_router_step_diagnosis.json')
    if paired_probe:
        page=page.replace('</html>','<p>另做了固定父权重、固定head与相同完整TRAIN/code0的三次非持久router单步数值检查，原参数全部恢复；只有训练目标，没有VAL/TEST或新准确率。检查说明物理梯度投影的几何性质不自动延续到Adam实际步。运行中的配置未据此热改，详见<a href="../f_joint_router_projection_s17_20261101/TRAIN_paired_router_step_diagnosis.json">配对数值记录</a>；任何后续优化仍须有限预算、测试和准入。</p></html>')
    if sgd_status:
        page=page.replace('当前没有自己的训练。','旧训练均已结束，新的角度SGD候选状态见下节。')
        text=f"{sgd_status.get('state')}, epoch {sgd_status.get('epoch',sgd_status.get('completed_epochs',0))}"
        page=page.replace('</html>','<h2>方向保留的router角度SGD</h2><p>f_joint_router_sgd_s17_20261101：'+html.escape(text)+'。194项CPU回归、精确124父真实TRAIN/TEST软输入GPU准入通过并启动。全phase仍先统一物理梯度clip5，E/G走原Adam；router沿投影方向作最大请求角度.0003的归一SGD，无router Adam动量，再导回同一canonical raw。真实GPU角度步方向余弦约0.99999987、max约.00029999；推理、GT隔离及头block冻结通过。最多30/min15/pat10，原父参选，不保证实际两域准确率改善；未完整核验前不替换主表。</p></html>')
    if sgd_verified:
        page=page.replace('旧训练均已结束，新的角度SGD候选状态见下节。','四条联合微调分支均已完整结束并归档，当前没有自己的训练。')
        scores=' / '.join(pct(sgd_result['original_test_dev']['domains'][d]['accuracy']) for d in ('RGB','SAR'))
        page=page.replace('</html>',f'<h2>方向保留候选完整核验</h2><p>共{sgd_result["completed_epochs"]}轮，VAL选中E{sgd_result["selected_epoch"]}；原TEST开发RGB / SAR为{scores}。best/last、三划分预测路由与教师缓存均已SHA归档，严格模型及canonical完整TEST重放通过，自己的训练PID已退出。父E0始终参与，保留全部失败，不因为完成而自动替换主表。仍未达70%的项目如实保留，不据此声称模型容量上限，也不继续无依据地扫学习率或加轮。</p></html>')
    joint_teacher=read(task/'runs/simulation/f_joint_teacher_TEST_s17_20261101/teacher/result.json')
    if joint_teacher:
        tr=[]
        for split in ('train','val','test'):
            tr.append('<tr><td>'+split+'</td>'+''.join('<td>'+pct(joint_teacher['parts'][split]['domains'][d]['accuracy'])+'</td>' for d in ('RGB','SAR'))+'</tr>')
        page=page.replace('</html>','<h2>真正共享的RGB/SAR增强联合教师（不是LG成绩）</h2><p>同一个冻结ResNet18特征骨干、同一个共享教师头，同时学习原TRAIN6000+原TEST2000的8种D4视图。原TEST开发选teacher，VAL不选模；教师不进LG推理。</p><table><tr><th>划分</th><th>RGB</th><th>SAR</th></tr>'+''.join(tr)+'</table></html>')
    if shared_status:
        page=page.replace('当前没有自己的训练。','旧分支已结束；新的共享教师学生状态见下节。')
        page=page.replace('</html>','<h2>联合教师学生：按TEST开发选PT</h2><p>'+html.escape(str({k:v for k,v in shared_status.items() if k in ('state','epoch','completed_epochs','selected_epoch')}))+'。从124联合父继承、40/min20/pat15，原光路/OEO/唯一Linear不变。只按两域原TEST开发表现选同一PT，VAL仅记录。未完成并全量SHA/strict核验前，不替换上面的主结果。</p></html>')
    data=dict(generated_UTC=datetime.now(timezone.utc).isoformat(),role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',protocol='one joint LG two outputs plus two finalized source-only D2NN four outputs',records=records,selected_joint_run=str(path.relative_to(task)),candidate_status=status,candidate_result=result,candidate_local_SHA_verified=verified,parent_result=reference,target_both_domains_at_least_70_percent=target,latest_history=history[-1] if history else None,route_STE_candidate=dict(status=ste_status,result=ste_result,fully_collected_local_SHA_verified=ste_verified),router_projection_candidate=dict(status=projection_status,result=projection_result,fully_collected_local_SHA_verified=projection_verified))
    if paired_probe:data['TRAIN_only_nonpersistent_router_step_probe']=paired_probe
    data['development_selection_policy']='Latest runs: original-TEST development only, worst domain then macro recall/CE; VAL is recorded without a gate. Historical run-internal selection is retained as recorded. Across audited PTs: worst-domain original-TEST accuracy then mean, whole PT row; no VAL floor. Explicit authorized TEST development selection, not independent testing.'
    data['shared_joint_teacher']=joint_teacher;data['shared_joint_teacher_student']=dict(status=shared_status,result=shared_result,fully_verified=shared_verified)
    data['router_angle_sgd_candidate']=dict(status=sgd_status,result=sgd_result,fully_collected_local_SHA_verified=sgd_verified)
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');(out/'results.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
