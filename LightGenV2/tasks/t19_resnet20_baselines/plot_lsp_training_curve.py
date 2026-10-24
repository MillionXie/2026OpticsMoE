"""Plot saved evidence only, with independent axes for the three objectives."""
import json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
task=Path('LightGenV2/tasks/t19_resnet20_baselines');control=task/'runs/control'
old=task/'runs/simulation/lsp_finetune60_formal_s42_20261024_cleanup1'
new=task/'runs/simulation/lsp_optimization_fixed30_formal_s42_20261024_cacheclone1'
read=lambda p:json.loads(p.read_text())
if (control/'frozen_lsp_last90_curve_diagnosis_20261024.json').exists():
    print(json.dumps(read(control/'frozen_lsp_last90_curve_diagnosis_20261024.json')));raise SystemExit(0)
previous=read(old/'history.json');history=read(new/'history.json');result=read(new/'result.json')
assert result['status']=='complete' and result['selected_epoch']==90
audit=read(control/'lsp_optimization_cacheclone1_last90_audit_20261024.json');assert audit['status']=='passed'
assert [x['epoch'] for x in previous]==list(range(1,61)) and [x['epoch'] for x in history]==list(range(61,91))
stages=[previous[:30],previous[30:],history];colors=['#1c6cb3','#d18022','#7458a2']
labels=['Gaussian MSE + coordinate','Gaussian MSE only','Spatial distribution cross-entropy']
fig=plt.figure(figsize=(12,6.8),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,1.2])
for i,(rows,color,label) in enumerate(zip(stages,colors,labels)):
    ax=fig.add_subplot(grid[0,i]);ax.plot([x['epoch'] for x in rows],[x['train_loss'] for x in rows],color=color,lw=2)
    ax.set(title=label,xlabel='Epoch',ylabel='TRAIN loss');ax.grid(alpha=.2)
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
ax=fig.add_subplot(grid[1,:]);periodic={x['epoch']:x['pck_at_0.2_torso'] for x in previous[:30] if 'pck_at_0.2_torso' in x}
ax.plot(list(periodic),[100*x for x in periodic.values()],marker='o',color=colors[0],label='Original periodic TEST observations')
old60=read(old/'result.json')['metrics']['pck_at_0.2_torso'];new90=result['metrics']['pck_at_0.2_torso']
for epoch,value,color,label in [(60,old60,colors[1],'Fixed last60 TEST'),(90,new90,colors[2],'Fixed last90 TEST')]:
    ax.scatter([epoch],[100*value],s=75,color=color,label=label,zorder=4)
    ax.annotate(f'{100*value:.2f}%',(epoch,100*value),xytext=(-28,9),textcoords='offset points',color=color)
ax.axhline(50,ls='--',lw=1,color='#777777',label='Requested 50% reference')
ax.set(xlim=(0,94),ylim=(0,max(80,100*new90+12)),xlabel='Epoch',ylabel='PCK@0.2 torso (%)',title='Available TEST observations: no inferred intermediate points')
ax.grid(alpha=.2);ax.legend(loc='upper left',ncols=2,fontsize=9)
fig.suptitle('Frozen pretrained ResNet-18 / LSP: three training objectives',fontsize=14)
fig.text(.5,-.025,'Loss panels use independent scales. No TEST evaluations at epochs31–59 or61–89. Historical TEST observations are disclosed.',ha='center',fontsize=9)
prefix=control/'frozen_lsp_last90_three_objective_curve_20261024'
fig.savefig(str(prefix)+'.png',dpi=200,bbox_inches='tight');fig.savefig(str(prefix)+'.svg',bbox_inches='tight');plt.close(fig)
metadata=dict(checkpoint_SHA=result['checkpoint_SHA'],audit_SHA=hashlib.sha256((control/'lsp_optimization_cacheclone1_last90_audit_20261024.json').read_bytes()).hexdigest(),
    old_periodic_TEST_PCK=periodic,last60_TEST_PCK=old60,last90_TEST_PCK=new90,
    stage3_first_loss=history[0]['train_loss'],stage3_last_loss=history[-1]['train_loss'],
    stage3_last10_loss_min=min(x['train_loss'] for x in history[-10:]),stage3_last10_loss_max=max(x['train_loss'] for x in history[-10:]),
    note='Only one TEST point for each new objective stage; not sufficient to identify a TEST-overfitting trajectory within a stage. Loss scales across stages are incomparable.',
    new_model_forwards=0,new_optimizer_steps=0)
with (control/'frozen_lsp_last90_curve_diagnosis_20261024.json').open('x') as f:json.dump(metadata,f,indent=2)
print(json.dumps(dict(PNG=str(prefix)+'.png',SVG=str(prefix)+'.svg',metrics=metadata)))
