from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).parent
plt.rcParams.update({'font.family':'Arial', 'font.size':7, 'axes.labelsize':7,
    'xtick.labelsize':7, 'ytick.labelsize':7, 'axes.linewidth':0.6,
    'svg.fonttype':'none', 'pdf.fonttype':42, 'savefig.facecolor':'white'})
labels = ['Simulation reference', 'Hardware baseline', 'Sensor-aware training',
          'Sensor- and zero-order-aware training',
          'Sensor-, zero-order- and\ngeometry-aware training', 'Electronic adaptation†']
values = [89.50,54.90,59.80,69.15,73.15,87.05]
positions = [6.1,4.8,3.8,2.8,1.8,0.65]
colors = ['#92989E'] + ['#0072B2']*4 + ['#D55E00']
fig, ax = plt.subplots(figsize=(183/25.4,108/25.4))
fig.subplots_adjust(left=.365,right=.97,top=.86,bottom=.23)
ax.barh(positions,values,height=.52,color=colors,edgecolor='none')
for y,v in zip(positions,values):
    ax.text(v+1.3,y,f'{v:.2f}',va='center',ha='left',fontsize=7)
ax.set_yticks(positions,labels)
ax.set_xlim(0,100)
ax.set_ylim(-.05,6.8)
ax.set_xticks([0,20,40,60,80,100])
ax.set_xlabel('Classification accuracy (%)',labelpad=7)
ax.tick_params(axis='y',length=0,pad=8)
ax.tick_params(axis='x',length=3,width=.6)
for side in ['top','right','left']:
    ax.spines[side].set_visible(False)
fig.legend(handles=[Patch(facecolor='#92989E',label='Simulation'),
                    Patch(facecolor='#0072B2',label='Hardware evaluation'),
                    Patch(facecolor='#D55E00',label='Adapted historical model')],
           loc='upper center',bbox_to_anchor=(.5,.985),ncol=3,frameon=False,
           handlelength=1.1,columnspacing=1.7,fontsize=7)
fig.text(.035,.105,'† Electronic adaptation uses a historical upstream checkpoint, not the 73.15% model.',fontsize=6)
fig.text(.035,.065,'Configurations differ in training perturbation strength. TEST development set: n = 1,000.',fontsize=6)
for ext in ['png','svg','pdf']:
    fig.savefig(OUT / f'openmoji_cumulative_nature_style.{ext}',dpi=600)
plt.close(fig)
