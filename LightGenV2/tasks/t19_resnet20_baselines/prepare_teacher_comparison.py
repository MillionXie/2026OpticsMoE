"""Collect fixed-last metrics/parameter metadata for the presentation table, without inference."""
import hashlib, json
from pathlib import Path

root=Path(__file__).resolve().parent
sources={}
def read(rel):
    p=root/rel; raw=p.read_bytes(); sources[rel]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def audit(name): return read('runs/control/'+name)
old={r['task']:r for r in audit('last_epoch_final_audit_20261023.json')['runs'] if r['task']!='lgvq_temporal_resnet18'}
refs={
 'lgvq_temporal':(.8044,.7663), 'lgvq_spatial':(.6710,.6908),
 'abo_image_to_text':(.8058,.7358), 'abo_image_to_image':(.8225,.8513),
 'abo_text_to_image':(.8600,.8200), 'lsp':(.7348,.7226), 'salicon':(.8625,.8748),
 'openmoji':(.8880,.8120), 'abo_conditional_edit':(34.28,27.26)}
names={'lgvq_temporal':'LGVQ 时间质量','lgvq_spatial':'LGVQ 空间质量','abo_image_to_text':'ABO 图搜文','abo_image_to_image':'ABO 图搜图','abo_text_to_image':'ABO 文搜图','lsp':'LSP 关键点','salicon':'SALICON 显著性','openmoji':'OpenMoji 语义编辑','abo_conditional_edit':'ABO 参考图＋指令编辑'}
metric={'lgvq_temporal':('SRCC','srcc'),'lgvq_spatial':('SRCC','srcc'),'abo_image_to_text':('H@1','hit_at_1'),'abo_image_to_image':('H@1','hit_at_1'),'abo_text_to_image':('H@1','hit_at_1'),'lsp':('PCK@0.2 torso','pck_at_0.2_torso'),'salicon':('CC','cc'),'openmoji':('修改格准确率','changed_cell_accuracy'),'abo_conditional_edit':('PSNR (dB)','psnr_db')}
rows=[]
def add(task,variant,epoch,value,counts,scope,note,source):
    m,key=metric[task]; o=old[task]
    assert o['epoch']==30 and o['parameters']['total']==o['parameters']['trainable']
    assert counts['total']-counts['trainable']==11176512 and counts['total']<15000000
    rows.append(dict(task_id=task,task=names[task],metric=m,variant=variant,epoch=epoch,value=value,total=counts['total'],trainable=counts['trainable'],pretraining='视觉：ImageNet RN18',scope=scope,rn20=o['metrics'][key],rn20_total=o['parameters']['total'],ours_sim=refs[task][0],qwen=refs[task][1],note=note,source=source))
lg=audit('frozen_lgvq_last30_independent_audit_20261023.json')
for kind in ['temporal','spatial']:
    fn=f'frozen_lgvq_{kind}_last50_audit_20261023.json'; a=audit(fn)
    add('lgvq_'+kind,'采用：累计 last50',50,a['result_metrics']['srcc'],lg[kind]['parameter_counts'],'视觉/BN 冻结；只训质量头','30＋20轮，继承模型和AdamW；不是挑选最佳轮次。','runs/control/'+fn)
budget=read('runs/simulation/retrieval_fixed_budget_cpu_20261024/results.json')['results']
for task in ['abo_image_to_text','abo_image_to_image','abo_text_to_image']:
    for r in sorted([r for r in budget if r['task']==task],key=lambda x:x['epochs']):
        epoch=r['epochs']; rd='runs/simulation/retrieval_fixed_budget_cpu_20261024/'+r['run_id']
        mf=read(rd+'/run_manifest.json')
        saved=read(rd+'/result.json')
        assert saved['checkpoint_sha256']==r['checkpoint_sha256']
        assert saved['metrics']==r['metrics']
        counts=mf['parameters']
        if task=='abo_image_to_image':
            variant='采用：官方 GAP512＋L2'
            scope='视觉/BN 冻结；无学习任务头'
            note='零任务训练。原图 Ours Sim 差距仅4.375个百分点。'
        else:
            variant='随机未对齐诊断' if epoch==0 else f'保留：固定 last{epoch} 预算'
            scope='视觉/BN 冻结；GRU＋双投影可训'
            note=('0轮实际更新0参数；64,096为可训练配置，不是已训练量。非预训练图文零样本。' if epoch==0 else f'独立CPU训练{epoch}轮（{epoch*150}次更新），不从其他预算接续。')
        add(task,variant,epoch,r['metrics']['hit_at_1'],counts,scope,note,rd+'/result.json')
    fn={'abo_image_to_text':'frozen_i2t_last30_audit_20261024.json','abo_image_to_image':'frozen_i2i_last30_audit_20261023.json','abo_text_to_image':'frozen_t2i_last30_audit_20261024.json'}[task]
    a=audit(fn)
    add(task,'保留：原正式 last30',30,a['metrics']['hit_at_1'],a['parameter_counts'],('视觉/BN 冻结；只训512→64投影' if task=='abo_image_to_image' else '视觉/BN 冻结；训练GRU＋双投影'),'原GPU30轮对照；CPU与GPU不保证位级相同。','runs/control/'+fn)
    if task=='abo_image_to_image':
        fn='abo_i2i_pca64_reference_audit_20261024.json'; a=audit(fn)
        add(task,'补充：TRAIN 拟合 PCA64',0,a['independent_metrics']['hit_at_1'],dict(total=11176512,trainable=0),'视觉/BN 冻结；PCA无梯度拟合','0梯度轮，不是零数据。固定PCA矩阵/均值为buffers，未计入神经参数。','runs/control/'+fn)
for task,fn,epoch,variant,scope,note in [
 ('lsp','lsp_optimization_cacheclone1_last90_audit_20261024.json',90,'采用：累计 last90','视觉/BN 冻结；只训姿态头','30轮MSE＋坐标，30轮MSE-only，30轮空间CE；新CE阶段重启AdamW。'),
 ('salicon','salicon_optimization_cacheclone1_last_audit_20261024.json',30,'采用：513参数小头 last30','视觉/BN 冻结；训1×1 Conv(512→1)','有学习任务头。完整5000 public-val作TEST，非私有challenge测试。'),
 ('openmoji','openmoji_grouped_validation_last60_audit_20261024.json',60,'采用：独立新头 last60','视觉/BN 冻结；训练GRU＋编辑头','原TRAIN分组3981拟合/1019验证；3候选固定60轮，VAL选dropout0.2/WD0.0001。非累计180轮。'),
 ('abo_conditional_edit','frozen_editor_last50_audit_20261024.json',50,'采用：累计 last50','视觉/BN 冻结；训练GRU＋RGB解码头','完整2304 TEST（不是VAL）；输入包含参考图。与原图latent损失不同。')]:
    a=audit(fn); metrics=a['TEST_metrics'] if task=='abo_conditional_edit' else a['metrics']
    add(task,variant,epoch,metrics[metric[task][1]],a['parameter_counts'],scope,note,'runs/control/'+fn)
assert len(rows)==17
assert len(old)==9
notes=[
 '参数为完整模型精确参数个数，包含11,176,512个冻结视觉参数；所有当前模型均小于15,000,000。',
 '当前视觉统一为官方 torchvision ResNet18 IMAGENET1K_V1；全部视觉参数、BN统计均冻结，eval/no_grad。',
 '含文字的图搜文、文搜图、OpenMoji和参考图编辑：byte-GRU 27,104参数随机初始化且可训练，不使用MiniLM。',
 '跨模态64,096可训参数＝GRU 27,104＋图像512→64投影32,832＋文字64→64投影4,160；推理保留这些组件。',
 '旧ResNet-20统一采用last30，随机初始化、无预训练，视觉/文字/头全参数训练；表中参数总量＝可训练量。',
 'Ours采用用户原图Sim列，Qwen采用原图Qwen3VL列；它们的逐任务epoch、完整/可训参数及冻结范围未从原图核实。',
 'Ours/Qwen为历史参考，不声称共同30/50/60/90末轮，也不声称同预训练、损失、分辨率或训练预算的消融。',
 '当前各预算固定使用自己的最后一轮；无按TEST选epoch。OpenMoji仅以分组VAL选候选，各候选都固定last60。',
 'ABO图搜文/文搜图未指定唯一最终训练预算，0/1/5/30完整并列；PCA只是图搜图补充对照，未做维度搜索。',
 'ABO easy100为同100商品/固定标题留视角，TRAIN每商品48图、TEST24图；T2I每标题24正例，非未见商品泛化。',
 'ABO图搜图为200商品，每商品8图库/4查询；历史跨split字节重复仅1对，不能把高分全部归因这1对。',
 'OpenMoji主指标是修改格准确率77.55%，不是整体格90.48%；保留格90.98%、IoU0.2868，较旧版存在质量取舍。',
 '历史TEST已多次观察；本表为后续开发实验汇总，不能作为全新盲测。LSP多阶段目标不同，不能直接相减跨阶段loss。',
 '空间分辨率等历史版本差异（如原Qwen448/CNN224）及编辑RGB/latent损失差异保留，不能将分差都归于骨干。',
 'SALICON小头确实训练513参数；它不同于零训练能量读出CC0.5920，也不同于原40,129参数头CC0.8675。',
 '比例指标使用百分比显示；SRCC/CC保持0—1系数；PSNR单位dB。所有原预算与强弱结果均保留。'
]
data=dict(title='T19 小模型基线：最终采用版本与 ABO 全预算',as_of='2026-10-24',rows=rows,notes=notes,sources=sources,reference_source='用户提供的蓝色对比表图片；Ours使用Sim列，Qwen使用Qwen3VL列。',reference_metadata=dict(ours='原图未给出逐任务epoch/参数量/冻结范围，未核实',qwen='原图未给出逐任务epoch/精确参数量/冻结范围，未核实'))
out=root/'runs/control/teacher_comparison_sources_20261024.json'
out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'rows':len(rows),'source_files':len(sources),'path':str(out)},ensure_ascii=False))
