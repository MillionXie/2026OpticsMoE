"""Schematic of encoding power and two-class readout, not measured optical data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    out=Path(__file__).parent/'reports/figures/audio_input_s17_20260918';out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,5.6))
    for ax in axes:
        ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    axes[0].set_xlim(0,224);axes[0].set_ylim(224,0)
    for x,y,label,color in [(0,0,'V: 1/6','#9dc4df'),(112,0,'V: 1/6','#9dc4df'),(0,112,'V: 1/6','#9dc4df'),(112,112,'Text: 1/2','#f1c1a3')]:
        axes[0].add_patch(Rectangle((x,y),112,112,fc=color,ec='white',lw=2));axes[0].text(x+56,y+56,label,ha='center',va='center')
    axes[0].set_title('Legacy: same V repeated 3 times\n224 x 224 amplitude input',fontsize=11)
    axes[0].set_xlabel('Fractions denote optical power, not amplitude')
    axes[1].set_xlim(0,224);axes[1].set_ylim(224,0)
    for y,label,color in [(0,'One log-mel spectrogram\n112 x 224 | power 1/2','#9dc4df'),(112,'Ordered word codes\n112 x 224 | power 1/2','#f1c1a3')]:
        axes[1].add_patch(Rectangle((0,y),224,112,fc=color,ec='white',lw=2));axes[1].text(112,y+56,label,ha='center',va='center',fontsize=10)
    axes[1].set_title('New two-band raw-audio input\nNo CNN, no repeated sensor tiles',fontsize=11)
    axes[1].set_xlabel('Power balancing precedes expert routing')
    ax=axes[2];ax.set_xlim(0,518);ax.set_ylim(518,0)
    ax.add_patch(Rectangle((20,20),478,478,fc='#eeeeee',ec='#777777'))
    for x,label,color in [(179,'No / 0','#719fc7'),(339,'Yes / 1','#d77c89')]:
        ax.add_patch(Rectangle((x-32,259-32),64,64,fc=color,ec='black'))
        ax.text(x,210,label,ha='center',fontsize=9)
    ax.text(259,340,'Two 64 x 64 energy windows',ha='center',fontsize=9)
    ax.set_title('Readout after final-layer OEO\n478 x 478 active plane',fontsize=11)
    ax.set_xlabel('p(yes) = E_yes / (E_no + E_yes)')
    fig.suptitle('Encoding and detector schematic (not an observed intensity map)',fontsize=12)
    fig.subplots_adjust(left=.03,right=.98,bottom=.20,top=.77,wspace=.14)
    fig.savefig(out/'input_and_readout.png',dpi=180,bbox_inches='tight');fig.savefig(out/'input_and_readout.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
