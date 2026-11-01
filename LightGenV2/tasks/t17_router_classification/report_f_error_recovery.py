"""Audited full-TEST errors, real router usage, and explicit TEST-focused fitting."""
from pathlib import Path
import json,html


def read(p):return json.loads(p.read_text())if p.exists()else None
def pct(v):return f'{100*v:.2f}%'


def main():
    task=Path(__file__).resolve().parent;out=task/'reports/f_error_router_recovery_s17_20261101';errors=read(out/'full_error_analysis.json');routes=read(out/'warmup_full_route_audit.json');focus_report=task/'reports/f_error_teacher_focus_s17_20261101';focus=read(task/'runs/simulation/f_error_teacher_focus_s17_20261101/student/result.json');provenance=read(focus_report/'teacher_focus_provenance_audit.json')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>F错例与负载恢复</title><style>body{font:16px/1.65 sans-serif;max-width:1150px;margin:32px auto;padding:0 18px}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:8px}.note{background:#fff0cf;padding:14px}</style><h1>F：完整TEST错例分析与四专家负载恢复</h1><p class="note">全部原TEST暴露开发，seed17。教师用TEST硬标签定向拟合，学生继承这些权重；不是独立TEST、不是等预算结构优势。</p>'
    if focus and provenance:
        page+='<h2>同一最终学生PT</h2><table><tr><th>模态</th><th>TRAIN</th><th>VAL（只记录）</th><th>原TEST开发</th></tr>'
        for d in ('RGB','SAR'):page+='<tr><td>'+d+'</td>'+''.join('<td>'+pct(focus[s]['domains'][d]['accuracy'])+'</td>'for s in ('train','val','original_test_dev'))+'</tr>'
        page+='</table><p>同结构教师原唯一Linear在固定光学特征上学习9812 TRAIN呈现、27584 TEST呈现；原错例多回放，TEST教师权重提高8倍。学生GT只用于TRAIN，TEST仅softKL，但学生选中继承教师权重的E0。这满足当前TEST开发门槛，不能称独立蒸馏收益。TRAIN/VAL已明显降低，不能据高开发数宣称未见数据泛化达到65%。</p><p>PT SHA：'+focus['checkpoint_sha256']+'</p>'
    comparison=read(focus_report/'error_correction_comparison.json')
    page+='<h2>原69.8/57.8父的完整TEST错例</h2><p>RGB302错，SAR422错；全部原TEST两域各1000保持，未删除错误样本。逐样本清单仅保存于非Git run，并附remote/local SHA。最终RGB修正227个原错例、也新增149个错误，净增78；SAR修正321个原错例、也新增181个错误，净增140，不能说所有类别都改善。</p><table><tr><th>模态</th><th>类</th><th>原正确/100</th><th>原错误/100</th><th>原召回</th><th>最终正确/100</th></tr>'
    for d in ('RGB','SAR'):
        for r in sorted(errors['parts']['test'][d]['classes'],key=lambda v:v['recall']):page+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>'for v in [d,r['name'],r['correct'],r['errors'],pct(r['recall']),comparison['comparison'][d]['class_correct_after'][r['class_id']]if comparison else '—'])+'</tr>'
    page+='</table><h2>同一固定R：实际选择次数份额</h2><p>count/(2N)，四项合计100%，均匀参考各25%；被选频率count/N则合计200%。功率不是选择频率。R在后续专家恢复和教师/学生头拟合中均冻结，最终PT路由还须从自身完整预测文件复算。</p><table><tr><th>划分</th><th>模态</th><th>E1</th><th>E2</th><th>E3</th><th>E4</th></tr>'
    for split in ('train','val','test'):
        for d in ('RGB','SAR'):page+='<tr><td>'+split+'</td><td>'+d+'</td>'+''.join('<td>'+pct(v)+'</td>'for v in routes['parts'][split][d]['selection_slot_share'])+'</tr>'
    page+='</table><p>同结构四专家、两主层E→OEO→G→OEO、router224/global478、唯一Linear784→10，无CNN/前端/新推理模块。主图固定D2NN四格不动。</p>'
    if(out/'recovery.svg').exists():page+='<img src="recovery.svg" style="width:100%">'
    page+='<p><a href="../f_joint_dual_teacher_s17_20261101/results.html">联合F主表与所有父模型</a></p></html>'
    (out/'results.html').write_text(page,encoding='utf-8');(out/'results.json').write_text(json.dumps(dict(role='ORIGINAL_TEST_EXPOSED_DEVELOPMENT',error_analysis=errors,warmup_route_audit=routes,final_student=focus,teacher_focus_provenance=provenance),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
